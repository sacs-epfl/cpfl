import pickle
from typing import Optional

import torch

from cpfl.core.models.Model import Model


def serialize_model(model: torch.nn.Module) -> bytes:
    return pickle.dumps(model.state_dict())


def unserialize_model(serialized_model: bytes, dataset: str, architecture: Optional[str] = None) -> torch.nn.Module:
    model = create_model(dataset, architecture=architecture)
    model.load_state_dict(pickle.loads(serialized_model))
    return model


def create_model(dataset: str, architecture: Optional[str] = None) -> Model:
    if dataset == "cifar10":
        if not architecture:
            from cpfl.core.models.cifar10 import GNLeNet
            return GNLeNet(input_channel=3, output=10, model_input=(32, 32))
        elif architecture == "resnet8":
            from cpfl.core.models.resnet8 import ResNet8
            return ResNet8()
        elif architecture == "resnet18":
            import torchvision.models as tormodels
            return tormodels.__dict__["resnet18"](num_classes=10)
        else:
            raise RuntimeError("Unknown model architecture for CIFAR10: %s" % architecture)
    elif dataset == "femnist":
        from cpfl.core.models.femnist import CNN
        return CNN()
    else:
        raise RuntimeError("Unknown dataset %s" % dataset)
