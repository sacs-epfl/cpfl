import logging
import os
from asyncio import sleep, CancelledError, get_event_loop
from typing import Optional, Dict, Tuple

import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, MSELoss, NLLLoss

from cpfl.core.datasets import create_dataset, Dataset
from cpfl.core.optimizer.sgd import SGDOptimizer
from cpfl.core.session_settings import SessionSettings

AUGMENTATION_FACTOR_SIM = 3.0


class ModelTrainer:
    """
    Manager to train a particular model.
    Runs in a separate process.
    """

    def __init__(self, data_dir, settings: SessionSettings, participant_index: int):
        """
        :param simulated_speed: compute speed of the simulated device, in ms/sample.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.settings: SessionSettings = settings
        self.participant_index: int = participant_index
        self.simulated_speed: Optional[float] = None
        self.fixed_simulated_training_time: Optional[float] = None
        self.total_training_time: float = 0
        self.is_training: bool = False
        self.training_losses: Dict[int, float] = {}
        self.validation_loss_global_model: Dict[int, float] = {}
        self.validation_loss_updated_model: Dict[int, float] = {}

        if settings.dataset in ["cifar10", "mnist", "fashionmnist", "movielens", "spambase", "google_speech"]:
            self.train_dir = data_dir
        else:
            self.train_dir = os.path.join(data_dir, "per_user_data", "train")
        self.dataset: Optional[Dataset] = None

    def get_validation_loss(self, model) -> float:
        validation_set = self.dataset.get_validationset()
        total_loss = 0.0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()  # set model to evaluation mode

        with torch.no_grad():
            for data, target in validation_set:
                data, target = data.to(device), target.to(device)
                output = model(data)

                if self.settings.dataset == "movielens":
                    loss_func = MSELoss()
                elif self.settings.dataset == "cifar10":
                    if self.settings.model in ["resnet8", "resnet18"]:
                        loss_func = CrossEntropyLoss()
                    else:
                        loss_func = NLLLoss()
                else:
                    loss_func = CrossEntropyLoss()

                loss = loss_func(output, target)
                total_loss += loss.item()

        avg_loss = total_loss / len(validation_set)
        return float(avg_loss)

    async def train(self, model, round_nr, device_name: str = "cpu") -> Tuple[int, Optional[float], Optional[float]]:
        """
        Train the model on a batch. Return an integer that indicates how many local steps we have done.
        """
        self.is_training = True

        if not self.dataset:
            self.dataset = create_dataset(self.settings, participant_index=self.participant_index, train_dir=self.train_dir)

        validation_loss_global_model: Optional[float] = None
        if self.settings.compute_validation_loss_global_model and len(self.dataset.validationset) > 0:
            validation_loss_global_model = self.get_validation_loss(model)
            self.validation_loss_global_model[round_nr] = validation_loss_global_model

        local_steps: int = self.settings.learning.local_steps
        device = torch.device(device_name)
        optimizer = SGDOptimizer(model, self.settings.learning.learning_rate, self.settings.learning.momentum, self.settings.learning.weight_decay)

        if self.settings.learning.local_steps == 0:
            # Load the train set and determine the number of local steps we should take
            train_set = self.dataset.get_trainset(batch_size=self.settings.learning.batch_size, shuffle=True)
            train_set_it = iter(train_set)
            local_steps = len(train_set.dataset) // self.settings.learning.batch_size
            if len(train_set.dataset) % self.settings.learning.batch_size != 0:
                local_steps += 1

        self.logger.info("Will perform %d local steps of training on device %s (batch size: %d, lr: %f, wd: %f)",
                         local_steps, device_name, self.settings.learning.batch_size,
                         self.settings.learning.learning_rate, self.settings.learning.weight_decay)

        if self.settings.is_simulation:
            # If we're running a simulation, we should advance the time of the DiscreteLoop with either the simulated
            # elapsed time or the elapsed real-world time for training. Otherwise,training would be considered instant
            # in our simulations. We do this before the actual training so if our sleep gets interrupted, the local
            # model will not be updated.
            start_time = get_event_loop().time()
            if self.fixed_simulated_training_time:
                elapsed_time = self.fixed_simulated_training_time
            elif self.simulated_speed:
                elapsed_time = AUGMENTATION_FACTOR_SIM * local_steps * self.settings.learning.batch_size * (self.simulated_speed / 1000)
            else:
                elapsed_time = 0

            try:
                await sleep(elapsed_time)
            except CancelledError:
                self.is_training = False
                self.total_training_time += (get_event_loop().time() - start_time)
                return 0  # Training got interrupted - don't update the model
            self.total_training_time += elapsed_time

            self.logger.info("Model training completed and took %f s.", elapsed_time)

        samples_trained_on = 0
        model = model.to(device)
        for local_step in range(local_steps):
            if self.settings.bypass_training:
                continue

            if self.settings.learning.local_steps != 0:
                # Refresh the training set
                train_set = self.dataset.get_trainset(batch_size=self.settings.learning.batch_size, shuffle=True)
                train_set_it = iter(train_set)

            try:
                data, target = next(train_set_it)

                if self.settings.dataset == "google_speech":
                    data = torch.unsqueeze(data, 1)

                model.train()
                data, target = Variable(data.to(device)), Variable(target.to(device))
                samples_trained_on += len(data)

                optimizer.optimizer.zero_grad()
                self.logger.debug('d-sgd.next node forward propagation (step %d/%d)', local_step, local_steps)
                output = model.forward(data)

                if self.settings.dataset == "movielens":
                    lossf = MSELoss()
                elif self.settings.dataset == "cifar10":
                    if self.settings.model == ["resnet8", "resnet18"]:
                        lossf = CrossEntropyLoss()
                    else:
                        lossf = NLLLoss()
                else:
                    lossf = CrossEntropyLoss()

                loss = lossf(output, target)
                self.training_losses[round_nr] = float(loss)
                self.logger.debug('d-sgd.next node backward propagation (step %d/%d)', local_step, local_steps)
                loss.backward()
                optimizer.optimizer.step()
            except StopIteration:
                pass

        validation_loss_updated_model: Optional[float] = None
        if self.settings.compute_validation_loss_updated_model and len(self.dataset.validationset) > 0:
            validation_loss_updated_model = self.get_validation_loss(model)
            self.validation_loss_updated_model[round_nr] = validation_loss_updated_model

        self.is_training = False
        model.to("cpu")

        return samples_trained_on, validation_loss_global_model, validation_loss_updated_model
