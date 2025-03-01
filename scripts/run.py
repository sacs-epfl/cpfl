import argparse
import logging
import os
import stat
import subprocess
import sys
import time
from typing import List

import torch

from cpfl.core.datasets import create_dataset
from cpfl.core.model_trainer import ModelTrainer
from cpfl.core.models import create_model
from cpfl.core.session_settings import SessionSettings, LearningSettings


logger = logging.getLogger("standalone-trainer")


def get_args(default_lr: float, default_momentum: float = 0):
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=default_lr)
    parser.add_argument('--momentum', type=float, default=default_momentum)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--local-steps', type=int, default=5)
    parser.add_argument('--peers', type=int, default=1)
    parser.add_argument('--rounds', type=int, default=100)
    parser.add_argument('--acc-check-interval', type=int, default=1)
    parser.add_argument('--partitioner', type=str, default="iid", choices=["iid", "shards", "dirichlet", "realworld"])
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--train-method', type=str, default="local")
    parser.add_argument('--das-subprocess-jobs', type=int, default=1)
    parser.add_argument('--das-peers-per-subprocess', type=int, default=10)
    parser.add_argument('--das-peers-in-this-subprocess', type=str, default="")
    parser.add_argument('--data-dir', type=str, default=os.path.join(os.environ["HOME"], "dfl-data"))
    return parser.parse_args()


async def run(args, dataset: str):
    learning_settings = LearningSettings(
        local_steps=args.local_steps,
        learning_rate=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
    )

    settings = SessionSettings(
        dataset=dataset,
        partitioner=args.partitioner,
        alpha=args.alpha,
        work_dir="",
        learning=learning_settings,
        participants=["a"],
        all_participants=["a"],
        target_participants=args.peers,
        model=args.model,
    )

    data_path = os.path.join("data", "%s_n_%d" % (dataset, args.peers))
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)

    with open(os.path.join(data_path, "accuracies.csv"), "w") as out_file:
        out_file.write("dataset,algorithm,peer,peers,round,learning_rate,accuracy,loss\n")

    if args.train_method == "local" or (args.train_method == "das" and args.das_peers_in_this_subprocess):
        await train_local(args, dataset, settings, data_path)

async def train_local(args, dataset: str, settings: SessionSettings, data_path: str):
    if settings.dataset in ["cifar10", "mnist", "fashionmnist", "movielens"]:
        test_dir = args.data_dir
    else:
        test_dir = os.path.join(args.data_dir, "data", "test")

    test_dataset = create_dataset(settings, 0, test_dir=test_dir)

    device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    print("Device to train/determine accuracy: %s" % device)

    # Determine which peers we should train for
    peers: List[int] = range(args.peers) if args.train_method == "local" else [int(n) for n in args.das_peers_in_this_subprocess.split(",")]

    for n in peers:
        model = create_model(settings.dataset, architecture=settings.model)
        trainer = ModelTrainer(args.data_dir, settings, n)
        highest_acc = 0
        for round in range(1, args.rounds + 1):
            start_time = time.time()
            print("Starting training round %d for peer %d" % (round, n))
            await trainer.train(model, device_name=device)
            print("Training round %d for peer %d done - time: %f" % (round, n, time.time() - start_time))

            if round % args.acc_check_interval == 0:
                acc, loss = test_dataset.test(model, device_name=device)
                print("Accuracy: %f, loss: %f" % (acc, loss))

                # Save the model if it's better
                if acc > highest_acc:
                    torch.save(model.state_dict(), os.path.join(data_path, "cifar10_%d.model" % n))
                    highest_acc = acc

                acc_file_name = "accuracies.csv" if args.train_method == "local" else "accuracies_%d.csv" % n
                with open(os.path.join(data_path, acc_file_name), "a") as out_file:
                    out_file.write("%s,%s,%d,%d,%d,%f,%f,%f\n" % (dataset, "standalone", n, args.peers, round, settings.learning.learning_rate, acc, loss))
