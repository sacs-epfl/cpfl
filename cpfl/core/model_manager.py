import asyncio
import logging
import os
import random
import sys
import time
from typing import Dict, Optional, List, Tuple

import torch
import torch.nn as nn

from cpfl.core.gradient_aggregation import GradientAggregationMethod
from cpfl.core.gradient_aggregation.fedavg import FedAvg
from cpfl.core.model_trainer import ModelTrainer
from cpfl.core.models import create_model, unserialize_model, serialize_model
from cpfl.core.session_settings import SessionSettings, dump_settings


class ModelManager:
    """
    This class manages the current ML model and training.
    """

    def __init__(self, model: Optional[nn.Module], settings: SessionSettings, participant_index: int):
        self.model: Optional[nn.Module] = model
        self.settings: SessionSettings = settings
        self.participant_index: int = participant_index
        self.logger = logging.getLogger(self.__class__.__name__)

        dataset_base_path: str = self.settings.dataset_base_path or os.environ["HOME"]
        if self.settings.dataset in ["cifar10", "fashionmnist", "mnist", "google_speech"]:
            self.data_dir = os.path.join(dataset_base_path, "dfl-data")
        else:
            # The LEAF dataset
            self.data_dir = os.path.join(dataset_base_path, "leaf", self.settings.dataset)

        self.model_trainer: ModelTrainer = ModelTrainer(self.data_dir, self.settings, self.participant_index)

        # Keeps track of the incoming trained models as aggregator
        self.incoming_trained_models: Dict[bytes, nn.Module] = {}

    def process_incoming_trained_model(self, peer_pk: bytes, incoming_model: nn.Module):
        if peer_pk in self.incoming_trained_models:
            # We already processed this model
            return

        self.incoming_trained_models[peer_pk] = incoming_model

    def reset_incoming_trained_models(self):
        self.incoming_trained_models = {}

    def get_aggregation_method(self):
        if self.settings.gradient_aggregation == GradientAggregationMethod.FEDAVG:
            return FedAvg

    def aggregate_trained_models(self, weights: List[float] = None) -> Optional[nn.Module]:
        models = [model for model in self.incoming_trained_models.values()]
        return self.get_aggregation_method().aggregate(models, weights=weights)

    async def train(self, round_nr) -> Tuple[int, Optional[float], Optional[float]]:
        if not self.model:
            self.logger.info("Initializing model of peer %d", self.participant_index)
            torch.manual_seed(self.settings.model_seed)
            self.model = create_model(self.settings.dataset, architecture=self.settings.model)

        train_info = await self.model_trainer.train(self.model, round_nr, device_name=self.settings.train_device_name)

        # Detach the gradients
        self.model = unserialize_model(serialize_model(self.model), self.settings.dataset, architecture=self.settings.model)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return train_info

    async def compute_accuracy(self, model: nn.Module):
        """
        Compute the accuracy/loss of the current model.
        Optionally, one can provide a custom iterator to compute the accuracy on a custom dataset.
        """
        self.logger.info("Computing accuracy of model")

        # Dump the model and settings to a file
        model_id = random.randint(1, 1000000)
        model_file_name = "%d.model" % model_id
        model_path = os.path.join(self.settings.work_dir, model_file_name)
        torch.save(model.state_dict(), model_path)
        dump_settings(self.settings)

        # Get full path to the script
        import cpfl.util as autil
        script_dir = os.path.join(os.path.abspath(os.path.dirname(autil.__file__)), "evaluate_model.py")
        cmd = "%s %s %s %d %s %d" % (sys.executable, script_dir, self.settings.work_dir, model_id,
                                     self.data_dir, torch.get_num_threads())
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)

        stdout, stderr = await proc.communicate()
        self.logger.info(f'Accuracy evaluator exited with {proc.returncode}]')

        if proc.returncode != 0:
            if stdout:
                self.logger.error(f'[stdout]\n{stdout.decode()}')
            if stderr:
                self.logger.error(f'[stderr]\n{stderr.decode()}')
            raise RuntimeError("Accuracy evaluation subprocess exited with non-zero exit code %d: %s" %
                               (proc.returncode, stderr.decode()))

        os.unlink(model_path)

        # Read the accuracy and the loss from the file
        results_file = os.path.join(self.settings.work_dir, "%d_results.csv" % model_id)
        with open(results_file) as in_file:
            content = in_file.read().strip().split(",")

        os.unlink(results_file)

        return float(content[0]), float(content[1])
