import os
from typing import Optional

from cpfl.core.datasets.Dataset import Dataset
from cpfl.core.mappings import Linear
from cpfl.core.session_settings import SessionSettings


def create_dataset(settings: SessionSettings, participant_index: int = 0, train_dir: Optional[str] = None, test_dir: Optional[str] = None) -> Dataset:
    mapping = Linear(1, settings.target_participants)
    if settings.dataset == "cifar10":
        from cpfl.core.datasets.CIFAR10 import CIFAR10
        return CIFAR10(participant_index, 0, mapping, settings.partitioner,
                       train_dir=train_dir, test_dir=test_dir, shards=settings.target_participants,
                       alpha=settings.alpha, validation_size=settings.validation_set_fraction, seed=settings.seed)
    elif settings.dataset == "cifar100":
        from cpfl.core.datasets.CIFAR100 import CIFAR100
        return CIFAR100(participant_index, 0, mapping, settings.partitioner,
                        train_dir=train_dir, test_dir=test_dir, shards=settings.target_participants, alpha=settings.alpha)
    elif settings.dataset == "stl10":
        from cpfl.core.datasets.STL10 import STL10
        return STL10(participant_index, 0, mapping, settings.partitioner, train_dir=train_dir, test_dir=test_dir)
    elif settings.dataset == "femnist":
        from cpfl.core.datasets.Femnist import Femnist
        return Femnist(participant_index, 0, mapping, settings.partitioner,
                       train_dir=train_dir, test_dir=test_dir, shards=settings.target_participants, alpha=settings.alpha,
                       validation_size=settings.validation_set_fraction, seed=settings.seed)
    elif settings.dataset == "svhn":
        from cpfl.core.datasets.SVHN import SVHN
        return SVHN(participant_index, 0, mapping, settings.partitioner, train_dir=train_dir, test_dir=test_dir)
    else:
        raise RuntimeError("Unknown dataset %s" % settings.dataset)
