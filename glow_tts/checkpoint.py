"""Methods for saving/loading checkpoints"""
import logging
import sys
import typing
from dataclasses import dataclass
from pathlib import Path

import torch

from glow_tts.config import TrainingConfig
from glow_tts.models import ModelType, setup_model

_LOGGER = logging.getLogger("glow_tts.checkpoint")

# -----------------------------------------------------------------------------


@dataclass
class Checkpoint:
    model: ModelType
    learning_rate: float
    global_step: int
    version: int


def load_checkpoint(
    checkpoint_path: Path,
    config: TrainingConfig,
    model: typing.Optional[ModelType] = None,
    use_cuda: bool = False,
) -> Checkpoint:
    """Load model from a Torch checkpoint"""
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    version = int(checkpoint_dict.get("version", 1))
    global_step = int(checkpoint_dict.get("global_step", 1))
    learning_rate = float(checkpoint_dict.get("learning_rate", 1.0))

    # Create model/optimizer if necessary
    model = setup_model(config, model=model, use_cuda=use_cuda,)

    saved_state_dict = checkpoint_dict["model"]
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()  # type: ignore
    else:
        state_dict = model.state_dict()

    new_state_dict = {}

    for k, v in state_dict.items():
        if k in saved_state_dict:
            # Use saved value
            new_state_dict[k] = saved_state_dict[k]
        else:
            # Use initialized value
            _LOGGER.warning("%s is not in the checkpoint", k)
            new_state_dict[k] = v

    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict)  # type: ignore
    else:
        model.load_state_dict(new_state_dict)  # type: ignore

    return Checkpoint(
        model=model,
        learning_rate=learning_rate,
        global_step=global_step,
        version=version,
    )


def remove_optimizer_from_checkpoint(
    in_path: Path, out_path: Path,
):
    """Load model from a Torch checkpoint and remove optimizer weights"""
    checkpoint_dict = torch.load(in_path, map_location="cpu")

    checkpoint_dict.pop("optimizer", None)

    torch.save(checkpoint_dict, str(out_path))


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    remove_optimizer_from_checkpoint(
        in_path=Path(sys.argv[1]), out_path=Path(sys.argv[2])
    )
