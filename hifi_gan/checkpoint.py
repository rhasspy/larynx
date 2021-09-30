"""Methods for saving/loading checkpoints"""
import logging
import typing
from dataclasses import dataclass
from pathlib import Path

import torch

from .config import TrainingConfig
from .models import TrainingModel, setup_model

_LOGGER = logging.getLogger("hifi_gan.checkpoint")

# -----------------------------------------------------------------------------


@dataclass
class Checkpoint:
    training_model: TrainingModel
    epoch: int
    global_step: int
    version: int


def get_state_dict(model):
    """Return model state dictionary whether or not distributed training was used"""
    if hasattr(model, "module"):
        return model.module.state_dict()

    return model.state_dict()


# -----------------------------------------------------------------------------


def load_checkpoint(
    generator_path: typing.Union[str, Path],
    config: TrainingConfig,
    training_model: typing.Optional[TrainingModel] = None,
    use_cuda: bool = False,
) -> Checkpoint:
    """Load models and training state from a directory of Torch checkpoints"""
    # Generator
    generator_path = Path(generator_path)

    _LOGGER.debug("Loading generator from %s", generator_path)
    generator_dict = torch.load(generator_path, map_location="cpu")

    assert "generator" in generator_dict, "Missing 'generator' in state dict"
    version = int(generator_dict.get("version", 1))
    global_step = int(generator_dict.get("global_step", 1))
    epoch = int(generator_dict.get("epoch", -1))

    # Set up the generator first
    training_model = setup_model(
        config=config,
        training_model=training_model,
        last_epoch=epoch,
        use_cuda=use_cuda,
    )

    assert training_model.generator, "No generator"
    set_state_dict(training_model.generator, generator_dict["generator"])

    return Checkpoint(
        training_model=training_model,
        epoch=epoch,
        global_step=global_step,
        version=version,
    )


def set_state_dict(model, state_dict):
    """Load state dictionary whether or not distributed training was used"""
    if hasattr(model, "module"):
        return model.module.load_state_dict(state_dict)

    return model.load_state_dict(state_dict)
