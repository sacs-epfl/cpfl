import json
import logging
import os
from collections import defaultdict

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from cpfl.core.datasets.Data import Data
from cpfl.core.datasets.Dataset import Dataset
from cpfl.core.mappings.Mapping import Mapping

NUM_CLASSES = 62
IMAGE_SIZE = (28, 28)
FLAT_SIZE = 28 * 28
PIXEL_RANGE = 256.0


class Femnist(Dataset):
    """
    Class for the FEMNIST dataset
    """

    def __read_file__(self, file_path):
        """
        Read data from the given json file

        Parameters
        ----------
        file_path : str
            The file path

        Returns
        -------
        tuple
            (users, num_samples, data)

        """
        with open(file_path, "r") as inf:
            client_data = json.load(inf)
        return (
            client_data["users"],
            client_data["num_samples"],
            client_data["user_data"],
        )

    def __read_dir__(self, data_dir):
        """
        Function to read all the FEMNIST data files in the directory

        Parameters
        ----------
        data_dir : str
            Path to the folder containing the data files

        Returns
        -------
        3-tuple
            A tuple containing list of clients, number of samples per client,
            and the data items per client

        """
        clients = []
        num_samples = []
        data = defaultdict(lambda: None)

        files = os.listdir(data_dir)
        files = [f for f in files if f.endswith(".json")]
        for f in files:
            file_path = os.path.join(data_dir, f)
            u, n, d = self.__read_file__(file_path)
            clients.extend(u)
            num_samples.extend(n)
            data.update(d)
        return clients, num_samples, data

    def load_trainset(self):
        """
        Loads the training set. Partitions it if needed.
        """
        logging.info("Loading training set.")
        files = os.listdir(self.train_dir)
        files = [f for f in files if f.endswith(".json")]
        files.sort()
        c_len = len(files)

        if not files:
            raise RuntimeError("No FEMNIST files found - is the dataset path (%s) correct?" % self.train_dir)

        self.uid = self.mapping.get_uid(self.rank, self.machine_id)
        my_train_data = {"x": [], "y": []}
        logging.debug("Clients Length: %d", c_len)

        if self.partitioner == "realworld":
            file_name = files[self.uid]
            clients, _, train_data = self.__read_file__(
                os.path.join(self.train_dir, file_name)
            )
            for cur_client in clients:
                my_train_data["x"].extend(train_data[cur_client]["x"])
                my_train_data["y"].extend(train_data[cur_client]["y"])
        else:
            raise RuntimeError("Unknown partitioner %s for FEMNIST" % self.partitioner)

        self.train_x = (
            np.array(my_train_data["x"], dtype=np.dtype("float32"))
            .reshape(-1, 28, 28, 1)
            .transpose(0, 3, 1, 2)
        )
        self.train_y = np.array(my_train_data["y"], dtype=np.dtype("int64")).reshape(-1)
        assert self.train_x.shape[0] == self.train_y.shape[0]
        assert self.train_x.shape[0] > 0

        self.trainset = Data(self.train_x, self.train_y)

    def load_testset(self):
        """
        Loads the testing set.
        """
        logging.info("Loading testing set at directory %s", self.test_dir)
        _, _, d = self.__read_dir__(self.test_dir)
        test_x = []
        test_y = []
        for test_data in d.values():
            for x in test_data["x"]:
                test_x.append(x)
            for y in test_data["y"]:
                test_y.append(y)
        self.test_x = (
            np.array(test_x, dtype=np.dtype("float32"))
            .reshape(-1, 28, 28, 1)
            .transpose(0, 3, 1, 2)
        )
        self.test_y = np.array(test_y, dtype=np.dtype("int64")).reshape(-1)
        assert self.test_x.shape[0] == self.test_y.shape[0]
        assert self.test_x.shape[0] > 0

    def __init__(
        self,
        rank: int,
        machine_id: int,
        mapping: Mapping,
        partitioner: str,
        train_dir="",
        test_dir="",
        sizes="",
        test_batch_size=1024,
        validation_size=0,
        shards=1,
        alpha: float = 1,
        seed: int = 42,
    ):
        """
        Constructor which reads the data files, instantiates and partitions the dataset

        Parameters
        ----------
        rank : int
            Rank of the current process (to get the partition).
        machine_id : int
            Machine ID
        mapping : decentralizepy.mappings.Mapping
            Mapping to convert rank, machine_id -> uid for data partitioning
            It also provides the total number of global processes
        train_dir : str, optional
            Path to the training data files. Required to instantiate the training set
            The training set is partitioned according to the number of global processes and sizes
        test_dir : str. optional
            Path to the testing data files Required to instantiate the testing set
        sizes : list(int), optional
            A list of fractions specifying how much data to alot each process. Sum of fractions should be 1.0
            By default, each process gets an equal amount.
        test_batch_size : int, optional
            Batch size during testing. Default value is 64

        """
        super().__init__(
            rank,
            machine_id,
            mapping,
            train_dir,
            test_dir,
            sizes,
            test_batch_size,
            validation_size,
        )

        self.partitioner = partitioner
        self.shards = shards
        self.alpha = alpha
        self.seed = seed

        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None

        if self.__training__:
            self.load_trainset()

        if self.__testing__:
            self.load_testset()

        if self.__training__ and self.__validating__:
            self.load_validationset()

    def load_validationset(self):
        """
        Loads the validation set
        """
        self.logger.info("Creating validation set.")

        dataset_len = len(self.trainset)
        val_len = int(self.validation_size * dataset_len) if dataset_len >= 10 else 0

        self.validationset, self.trainset = torch.utils.data.random_split(
            self.trainset, [val_len, dataset_len - val_len], torch.Generator().manual_seed(42),
        )

    def get_trainset(self, batch_size=1, shuffle=False):
        """
        Function to get the training set

        Parameters
        ----------
        batch_size : int, optional
            Batch size for learning

        Returns
        -------
        torch.utils.Dataset(decentralizepy.datasets.Data)

        Raises
        ------
        RuntimeError
            If the training set was not initialized

        """
        if self.__training__:
            return DataLoader(
                self.trainset,
                batch_size=batch_size,
                shuffle=shuffle,
            )
        raise RuntimeError("Training set not initialized!")

    def get_testset(self):
        """
        Function to get the test set

        Returns
        -------
        torch.utils.Dataset(decentralizepy.datasets.Data)

        Raises
        ------
        RuntimeError
            If the test set was not initialized

        """
        if self.__testing__:
            return DataLoader(
                Data(self.test_x, self.test_y), batch_size=self.test_batch_size
            )
        raise RuntimeError("Test set not initialized!")

    def get_validationset(self):
        """
        Function to get the validation set
        Returns
        -------
        torch.utils.Dataset(decentralizepy.datasets.Data)
        Raises
        ------
        RuntimeError
            If the test set was not initialized
        """
        if self.__validating__:
            return DataLoader(self.validationset, batch_size=self.test_batch_size)
        raise RuntimeError("Validation set not initialized!")

    def test(self, model, device_name: str = "cpu"):
        """
        Function to evaluate model on the test dataset.

        Parameters
        ----------
        model : decentralizepy.models.Model
            Model to evaluate

        Returns
        -------
        tuple
            (accuracy, loss_value)

        """
        testloader = self.get_testset()

        logging.debug("Test Loader instantiated.")
        device = torch.device(device_name)
        self.logger.debug("Device for Femnist accuracy check: %s", device)
        model.to(device)
        model.eval()

        correct_pred = [0 for _ in range(NUM_CLASSES)]
        total_pred = [0 for _ in range(NUM_CLASSES)]

        total_correct = 0
        total_predicted = 0

        with torch.no_grad():
            loss_val = 0.0
            count = 0
            for elems, labels in testloader:
                elems, labels = elems.to(device), labels.to(device)
                outputs = model(elems)
                lossf = CrossEntropyLoss()
                loss_val += lossf(outputs, labels).item()
                count += 1
                _, predictions = torch.max(outputs, 1)
                for label, prediction in zip(labels, predictions):
                    logging.debug("{} predicted as {}".format(label, prediction))
                    if label == prediction:
                        correct_pred[label] += 1
                        total_correct += 1
                    total_pred[label] += 1
                    total_predicted += 1

        logging.debug("Predicted on the test set")

        for key, value in enumerate(correct_pred):
            if total_pred[key] != 0:
                accuracy = 100 * float(value) / total_pred[key]
            else:
                accuracy = 100.0
            logging.info("Accuracy for class {} is: {:.1f} %".format(key, accuracy))

        accuracy = 100 * float(total_correct) / total_predicted
        loss_val = loss_val / count
        logging.info("Overall accuracy is: {:.1f} %".format(accuracy))
        return accuracy, loss_val

    def get_num_classes(self):
        return NUM_CLASSES
