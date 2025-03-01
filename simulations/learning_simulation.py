import asyncio
import json
import logging
import os
import pickle
import shutil
import stat
import subprocess
import time
from argparse import Namespace
from base64 import b64encode
from binascii import hexlify
from random import Random
from statistics import median, mean
from typing import Dict, List, Optional, Tuple

import torch

import yappi

from cpfl.core.model_manager import ModelManager
from cpfl.core.model_evaluator import ModelEvaluator
from cpfl.core.session_settings import SessionSettings, dump_settings
from cpfl.dfl.community import DFLCommunity

from ipv8.configuration import ConfigBuilder
from ipv8.taskmanager import TaskManager
from ipv8_service import IPv8

from simulation.discrete_loop import DiscreteLoop
from simulation.simulation_endpoint import SimulationEndpoint

from simulations.cpfl.bypass_network_community import DFLBypassNetworkCommunity
from simulations.logger import SimulationLoggerAdapter


class LearningSimulation(TaskManager):
    """
    Base class for any simulation that involves learning.
    """

    def __init__(self, args: Namespace) -> None:
        super().__init__()
        self.args = args
        self.session_settings: Optional[SessionSettings] = None
        self.nodes = []
        self.data_dir = os.path.join("data", "n_%d_%s_sd%d" % (self.args.peers, self.args.dataset, self.args.seed))
        self.evaluator = None
        self.logger = None
        self.model_manager: Optional[ModelManager] = None

        if self.args.torch_threads:
            print("Setting number of PyTorch threads to: %d" % self.args.torch_threads)
            torch.set_num_threads(self.args.torch_threads)

        self.loop = DiscreteLoop()
        asyncio.set_event_loop(self.loop)

    def get_ipv8_builder(self, peer_id: int) -> ConfigBuilder:
        builder = ConfigBuilder().clear_keys().clear_overlays()

        key_str = chr(peer_id).encode() * 1000
        key_base = b"LibNaCLSK:%s" % key_str[:68]
        key_material = b64encode(key_base).decode()
        builder.add_key_from_bin("my peer", key_material, file_path=os.path.join(self.data_dir, f"ec{peer_id}.pem"))
        return builder

    async def start_ipv8_nodes(self) -> None:
        for peer_id in range(1, self.args.peers + 1):
            if peer_id % 100 == 0:
                print("Created %d peers..." % peer_id)  # The logger has not been setup at this point
            endpoint = SimulationEndpoint()
            instance = IPv8(self.get_ipv8_builder(peer_id).finalize(), endpoint_override=endpoint,
                            extra_communities={
                                'DFLCommunity': DFLCommunity,
                                'DFLBypassNetworkCommunity': DFLBypassNetworkCommunity,
                            })
            await instance.start()

            # Set the WAN address of the peer to the address of the endpoint
            for overlay in instance.overlays:
                overlay.max_peers = -1
                overlay.my_peer.address = instance.overlays[0].endpoint.wan_address
                overlay.my_estimated_wan = instance.overlays[0].endpoint.wan_address
                overlay.cancel_pending_task("_check_tasks")  # To ignore the warning for long-running tasks
                overlay.logger = SimulationLoggerAdapter(overlay.logger, {})
                overlay.peer_manager.logger = SimulationLoggerAdapter(overlay.peer_manager.logger, {})
                if self.args.bypass_model_transfers:
                    overlay.bw_scheduler.logger = SimulationLoggerAdapter(overlay.peer_manager.logger, {})

            self.nodes.append(instance)

    def setup_directories(self) -> None:
        if os.path.exists(self.data_dir):
            shutil.rmtree(self.data_dir)
        os.makedirs(self.data_dir, exist_ok=True)

    def setup_logger(self) -> None:
        root = logging.getLogger()
        root.handlers[0].setFormatter(logging.Formatter("%(asctime)s:%(levelname)s:%(message)s"))
        root.setLevel(getattr(logging, self.args.log_level))

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger = SimulationLoggerAdapter(self.logger, {})

    def ipv8_discover_peers(self) -> None:
        peers_list = [node.overlays[0].my_peer for node in self.nodes]
        for node in self.nodes:
            node.overlays[0].peers_list = peers_list

    def apply_traces(self):
        """
        Set the relevant traces.
        """
        if self.args.availability_traces:
            self.logger.info("Applying availability trace file %s", self.args.availability_traces)
            with open(self.args.availability_traces, "rb") as traces_file:
                data = pickle.load(traces_file)

            rand = Random(self.args.seed)
            device_ids = rand.sample(list(data.keys()), self.args.peers)
            for ind, node in enumerate(self.nodes):
                node.overlays[0].set_traces(data[device_ids[ind]])

        if self.args.capability_traces:
            self.logger.info("Applying capability trace file %s", self.args.capability_traces)
            with open(self.args.capability_traces, "rb") as traces_file:
                data = pickle.load(traces_file)

            rand = Random(self.args.seed)
            device_ids = rand.sample(list(data.keys()), self.args.peers)
            nodes_bws: Dict[bytes, int] = {}
            for ind, node in enumerate(self.nodes):
                if not node.overlays[0].did_setup:
                    continue

                node.overlays[0].model_manager.model_trainer.simulated_speed = data[device_ids[ind]]["computation"]
                if self.args.bypass_model_transfers and not self.args.instant_network:
                    if node.overlays[0].bw_scheduler.bw_limit == -1:
                        node.overlays[0].bw_scheduler.bw_limit = 10000000000000
                    else:
                        # Also apply the network latencies
                        bw_limit: int = int(data[ind + 1]["communication"]) * 1024 // 8
                        node.overlays[0].bw_scheduler.bw_limit = bw_limit
                        nodes_bws[node.overlays[0].my_peer.public_key.key_to_bin()] = bw_limit

            for node in self.nodes:
                node.overlays[0].other_nodes_bws = nodes_bws
        else:  # No traces set, but maybe we are using a fixed training/transfer time
            if self.args.fixed_training_time is not None:
                for node in self.nodes:
                    node.overlays[0].model_manager.model_trainer.fixed_simulated_training_time = \
                        self.args.fixed_training_time

        self.logger.info("Traces applied!")

    def apply_latencies(self):
        """
        If specified in the settings, add latencies between the endpoints.
        """
        if not self.args.latencies_file:
            return

        latencies = []
        with open(self.args.latencies_file) as latencies_file:
            for line in latencies_file.readlines():
                latencies.append([float(l) for l in line.strip().split(",")])

        self.logger.info("Read latency matrix with %d sites!" % len(latencies))

        # Assign nodes to sites in a round-robin fashion and apply latencies accordingly
        for from_ind, from_node in enumerate(self.nodes):
            for to_ind, to_node in enumerate(self.nodes):
                from_site_ind = from_ind % len(latencies)
                to_site_ind = to_ind % len(latencies)
                latency_ms = int(latencies[from_site_ind][to_site_ind]) / 1000
                from_node.endpoint.latencies[to_node.endpoint.wan_address] = latency_ms

        self.logger.info("Latencies applied!")

    def determine_peer_with_lowest_median_latency(self, eligible_peers: List[int]) -> int:
        """
        Based on the latencies, determine the ID of the peer with the lowest median latency to other peers.
        """
        if not self.args.latencies_file:
            return eligible_peers[0]

        latencies = []
        with open(self.args.latencies_file) as latencies_file:
            for line in latencies_file.readlines():
                latencies.append([float(l) for l in line.strip().split(",")])

        lowest_median_latency = 100000
        lowest_peer_id = 0
        avg_latencies = []
        for peer_id in range(min(len(self.nodes), len(latencies))):
            if peer_id not in eligible_peers:
                continue
            median_latency = median(latencies[peer_id])
            avg_latencies.append(mean(latencies[peer_id]))
            if median_latency < lowest_median_latency:
                lowest_median_latency = median_latency
                lowest_peer_id = peer_id

        self.logger.info("Determined peer %d with lowest median latency: %f", lowest_peer_id + 1, lowest_median_latency)
        self.logger.debug("Average latency: %f" % mean(avg_latencies))
        return lowest_peer_id

    async def setup_simulation(self) -> None:
        self.logger.info("Setting up simulation with %d peers..." % self.args.peers)
        with open(os.path.join(self.data_dir, "accuracies.csv"), "w") as out_file:
            out_file.write("dataset,group,time,peer,round,accuracy,loss,bytes_up,bytes_down,train_time,network_time\n")

        if self.args.activity_log_interval:
            with open(os.path.join(self.data_dir, "activities.csv"), "w") as out_file:
                out_file.write("time,online,offline,training,bytes_up,bytes_down,train_time,network_time,min_nodes_in_view,max_nodes_in_view,avg_nodes_in_view,median_nodes_in_view\n")
            self.register_task("check_activity", self.check_activity, interval=self.args.activity_log_interval)

        if self.args.flush_statistics_interval:
            self.register_task("flush_statistics", self.flush_statistics, interval=self.args.flush_statistics_interval)

        if self.args.bypass_model_transfers:
            with open(os.path.join(self.data_dir, "transfers.csv"), "w") as out_file:
                out_file.write("from,to,round,start_time,duration,type,success\n")

    def check_activity(self):
        """
        Count the number of online/offline peers and write it away.
        """
        online, offline, training = 0, 0, 0
        active_nodes_in_view: List[int] = []
        for node in self.nodes:
            if not node.overlays[0].did_setup:
                continue

            if node.overlays[0].is_active:
                online += 1
                active_nodes_in_view.append(len(node.overlays[0].peer_manager.get_active_peers()))
            else:
                offline += 1

            if node.overlays[0].model_manager.model_trainer.is_training:
                training += 1

        cur_time = asyncio.get_event_loop().time()
        bytes_up, bytes_down, train_time, network_time = self.get_aggregated_statistics()
        with open(os.path.join(self.data_dir, "activities.csv"), "a") as out_file:
            out_file.write("%d,%d,%d,%d,%d,%d,%f,%f,%d,%d,%f,%f\n" % (
                cur_time, online, offline, training, bytes_up, bytes_down, train_time, network_time,
                min(active_nodes_in_view), max(active_nodes_in_view),
                sum(active_nodes_in_view) / len(active_nodes_in_view), median(active_nodes_in_view)))

    async def start_simulation(self) -> None:
        active_nodes: List = []
        for ind, node in enumerate(self.nodes):
            if self.args.cohort is not None and hexlify(node.overlays[0].my_peer.public_key.key_to_bin()).decode() not in self.session_settings.participants:
                continue

            if not node.overlays[0].traces or (node.overlays[0].traces and node.overlays[0].traces["active"][0] == 0):
                node.overlays[0].start()
                active_nodes.append(node)
        self.logger.info("Started %d nodes...", len(active_nodes))

        self.start_nodes_training(active_nodes)

        dataset_base_path: str = self.args.dataset_base_path or os.environ["HOME"]
        if self.args.dataset in ["cifar10", "mnist", "fashionmnist", "google_speech"]:
            data_dir = os.path.join(dataset_base_path, "dfl-data")
        else:
            # The LEAF dataset
            data_dir = os.path.join(dataset_base_path, "leaf", self.args.dataset)

        if not self.args.bypass_training:
            self.evaluator = ModelEvaluator(data_dir, self.session_settings)

        if self.args.profile:
            yappi.start(builtins=True)

        start_time = time.time()
        if self.args.duration > 0:
            await asyncio.sleep(self.args.duration)
            self.logger.info("Simulation took %f seconds" % (time.time() - start_time))
            self.on_simulation_finished()
            self.loop.stop()
        else:
            self.logger.info("Running simulation for undefined time")

    def start_nodes_training(self, active_nodes: List) -> None:
        pass

    def on_ipv8_ready(self) -> None:
        """
        This method is called when IPv8 is started and peer discovery is finished.
        """
        pass

    def checkpoint_models(self, round_nr: int):
        """
        Dump all models during a particular round.
        """
        models_dir = os.path.join(self.data_dir, "models", "%d" % round_nr)
        shutil.rmtree(models_dir, ignore_errors=True)
        os.makedirs(models_dir, exist_ok=True)

        avg_model = self.model_manager.aggregate_trained_models()
        for peer_ind, node in enumerate(self.nodes):
            torch.save(node.overlays[0].model_manager.model.state_dict(),
                       os.path.join(models_dir, "%d.model" % peer_ind))
        torch.save(avg_model.state_dict(), os.path.join(models_dir, "avg.model"))

    def checkpoint_model(self, peer_ind: int, round_nr: int):
        """
        Checkpoint a particular model of a peer during a particular round.
        """
        models_dir = os.path.join(self.data_dir, "models", "%d" % round_nr)
        os.makedirs(models_dir, exist_ok=True)

        model = self.nodes[peer_ind].overlays[0].model_manager.model
        torch.save(model.state_dict(), os.path.join(models_dir, "%d.model" % peer_ind))

    def test_models_with_das_jobs(self) -> Dict[int, Tuple[float, float]]:
        """
        Test the accuracy of all models in the model manager by spawning different DAS jobs.
        """
        results: Dict[int, Tuple[float, float]] = {}

        dump_settings(self.session_settings)

        # Divide the clients over the DAS nodes
        client_queue = list(self.model_manager.incoming_trained_models.keys())
        while client_queue:
            self.logger.info("Scheduling new batch on DAS nodes - %d clients left", len(client_queue))

            processes = []
            all_model_ids = set()
            for job_ind in range(self.args.das_test_subprocess_jobs):
                if not client_queue:
                    continue

                clients_on_this_node = []
                while client_queue and len(clients_on_this_node) < self.args.das_test_num_models_per_subprocess:
                    client = client_queue.pop(0)
                    clients_on_this_node.append(client)

                # Prepare the input files for the subjobs
                model_ids = []
                for client_id in clients_on_this_node:
                    model_id = int(client_id)
                    model_ids.append(model_id)
                    all_model_ids.add(model_id)
                    model_file_name = "%d.model" % model_id
                    model_path = os.path.join(self.session_settings.work_dir, model_file_name)
                    torch.save(self.model_manager.incoming_trained_models[client_id].state_dict(), model_path)

                import cpfl.util as autil
                script_dir = os.path.join(os.path.abspath(os.path.dirname(autil.__file__)), "evaluate_model.py")

                # Prepare the files and spawn the processes!
                out_file_path = os.path.join(os.getcwd(), "out_%d.log" % job_ind)
                model_ids_str = ",".join(["%d" % model_id for model_id in model_ids])

                train_cmd = "python3 %s %s %s %s" % (
                script_dir, self.session_settings.work_dir, model_ids_str, self.model_manager.data_dir)
                bash_file_name = "run_%d.sh" % job_ind
                with open(bash_file_name, "w") as bash_file:
                    bash_file.write("""#!/bin/bash
module load cuda11.7/toolkit/11.7
source /home/spandey/venv3/bin/activate
cd %s
export PYTHONPATH=%s
%s
""" % (os.getcwd(), os.getcwd(), train_cmd))
                    st = os.stat(bash_file_name)
                    os.chmod(bash_file_name, st.st_mode | stat.S_IEXEC)

                cmd = "ssh fs3.das6.tudelft.nl \"prun -t 5:00 -np 1 -o %s %s\"" % (
                out_file_path, os.path.join(os.getcwd(), bash_file_name))
                self.logger.debug("Command: %s", cmd)
                p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                processes.append((p, cmd, model_ids))

            for p, cmd, model_ids in processes:
                p.wait()
                self.logger.info("Command %s completed!", cmd)
                if p.returncode != 0:
                    raise RuntimeError("Training subprocess exited with non-zero code %d" % p.returncode)

                # This batch is done! Collect the results...
                for model_id in model_ids:
                    model_file_name = "%d.model" % model_id
                    model_path = os.path.join(self.session_settings.work_dir, model_file_name)
                    os.unlink(model_path)

                    # Read the accuracy and the loss from the file
                    results_file = os.path.join(self.session_settings.work_dir, "%d_results.csv" % model_id)
                    with open(results_file) as in_file:
                        content = in_file.read().strip().split(",")
                        accuracy, loss = float(content[0]), float(content[1])
                        results[model_id] = (accuracy, loss)

                    os.unlink(results_file)

        return results

    def test_models(self) -> Dict[int, Tuple[float, float]]:
        """
        Test the accuracy of all models in the model manager locally.
        """
        results: Dict[int, Tuple[float, float]] = {}
        for ind, model in enumerate(self.model_manager.incoming_trained_models.values()):
            self.logger.warning("Testing model %d on device %s..." % (ind + 1, self.args.accuracy_device_name))
            if not self.args.bypass_training:
                accuracy, loss = self.evaluator.evaluate_accuracy(model, device_name=self.args.accuracy_device_name)
            else:
                accuracy, loss = 0, 0

            results[ind] = (accuracy, loss)
        return results

    def get_aggregated_statistics(self) -> Tuple[int, int, float, float]:
        total_bytes_up: int = 0
        total_bytes_down: int = 0
        total_train_time: float = 0
        total_network_time: float = 0

        for ind, node in enumerate(self.nodes):
            if not node.overlays[0].did_setup:
                continue

            total_bytes_up += node.overlays[0].endpoint.bytes_up
            total_bytes_down += node.overlays[0].endpoint.bytes_down
            total_train_time += node.overlays[0].model_manager.model_trainer.total_training_time
            total_network_time += node.overlays[0].bw_scheduler.total_time_transmitting

        return total_bytes_up, total_bytes_down, total_train_time, total_network_time

    def get_statistics(self) -> Dict:
        # Determine both individual and aggregate statistics.
        total_bytes_up: int = 0
        total_bytes_down: int = 0
        total_train_time: float = 0
        total_network_time: float = 0

        individual_stats = {}
        for ind, node in enumerate(self.nodes):
            if not node.overlays[0].did_setup:
                continue

            bytes_up = node.overlays[0].endpoint.bytes_up
            bytes_down = node.overlays[0].endpoint.bytes_down
            train_time = node.overlays[0].model_manager.model_trainer.total_training_time
            network_time = node.overlays[0].bw_scheduler.total_time_transmitting
            individual_stats[ind] = {
                "bytes_up": bytes_up,
                "bytes_down": bytes_down,
                "train_time": train_time,
                "network_time": network_time
            }

            total_bytes_up += bytes_up
            total_bytes_down += bytes_down
            total_train_time += train_time
            total_network_time += network_time

        aggregate_stats = {
            "bytes_up": total_bytes_up,
            "bytes_down": total_bytes_down,
            "train_time": total_train_time,
            "network_time": total_network_time
        }

        return {
            "time": asyncio.get_event_loop().time(),
            "global": aggregate_stats
        }

    def flush_statistics(self):
        """
        Flush all the statistics generated by nodes.
        """

        # Write away the model transfers between peers
        if self.args.bypass_model_transfers:
            with open(os.path.join(self.data_dir, "transfers.csv"), "a") as out_file:
                for node in self.nodes:
                    for transfer in node.overlays[0].transfers:
                        out_file.write("%s,%s,%d,%f,%f,%s,%d\n" % transfer)
                    node.overlays[0].transfers = []

        with open(os.path.join(self.data_dir, "statistics.json"), "a") as out_file:
            out_file.write(json.dumps(self.get_statistics()) + "\n")

    def on_simulation_finished(self) -> None:
        self.flush_statistics()

        if self.args.profile:
            yappi.stop()
            yappi_stats = yappi.get_func_stats()
            yappi_stats.sort("tsub")
            yappi_stats.save(os.path.join(self.data_dir, "yappi.stats"), type='callgrind')

    async def run(self) -> None:
        self.setup_directories()
        await self.start_ipv8_nodes()
        self.setup_logger()
        self.ipv8_discover_peers()
        self.apply_latencies()
        self.on_ipv8_ready()
        await self.setup_simulation()
        self.apply_traces()
        await self.start_simulation()
        self.on_simulation_finished()
