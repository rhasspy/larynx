"""Configuration classes"""
import collections
import json
import typing
from dataclasses import dataclass, field
from pathlib import Path

from dataclasses_json import DataClassJsonMixin


@dataclass
class AudioConfig(DataClassJsonMixin):
    filter_length: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    segment_size: int = 8192
    num_mels: int = 80
    num_freq: int = 1025
    n_fft: int = 1024
    sampling_rate: int = 22050
    sample_bytes: int = 2
    channels: int = 1
    mel_fmin: float = 0.0
    mel_fmax: float = 8000.0
    mel_fmax_loss: typing.Optional[float] = None
    normalized: bool = True


@dataclass
class ModelConfig(DataClassJsonMixin):
    resblock: str = "1"  # 1=ResBlock1, 2=ResBlock2
    upsample_rates: typing.Tuple[int, ...] = (8, 8, 2, 2)
    upsample_kernel_sizes: typing.Tuple[int, ...] = (16, 16, 4, 4)
    upsample_initial_channel: int = 512
    resblock_kernel_sizes: typing.Tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: typing.Tuple[typing.Tuple[int, ...], ...] = (
        (1, 3, 5),
        (1, 3, 5),
        (1, 3, 5),
    )


@dataclass
class TrainingConfig(DataClassJsonMixin):
    seed: int = 1234
    epochs: int = 10000
    learning_rate: float = 0.0002
    adam_b1: float = 0.8
    adam_b2: float = 0.99
    lr_decay: float = 0.999
    batch_size: int = 32
    fp16_run: bool = False
    grad_clip: typing.Optional[float] = None
    num_workers: int = 4
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    version: int = 1
    git_commit: str = ""

    def save(self, config_file: typing.TextIO):
        """Save config as JSON to a file"""
        json.dump(self.to_dict(), config_file, indent=4)

    @staticmethod
    def load(config_file: typing.TextIO) -> "TrainingConfig":
        """Load config from a JSON file"""
        return TrainingConfig.from_json(config_file.read())

    @staticmethod
    def load_and_merge(
        config: "TrainingConfig",
        config_files: typing.Iterable[typing.Union[str, Path, typing.TextIO]],
    ) -> "TrainingConfig":
        """Loads one or more JSON configuration files and overlays them on top of an existing config"""
        base_dict = config.to_dict()
        for maybe_config_file in config_files:
            if isinstance(maybe_config_file, (str, Path)):
                # File path
                config_file = open(maybe_config_file, "r")
            else:
                # File object
                config_file = maybe_config_file

            with config_file:
                # Load new config and overlay on existing config
                new_dict = json.load(config_file)
                TrainingConfig.recursive_update(base_dict, new_dict)

        return TrainingConfig.from_dict(base_dict)

    @staticmethod
    def recursive_update(
        base_dict: typing.Dict[typing.Any, typing.Any],
        new_dict: typing.Mapping[typing.Any, typing.Any],
    ) -> None:
        """Recursively overwrites values in base dictionary with values from new dictionary"""
        for k, v in new_dict.items():
            if isinstance(v, collections.Mapping) and (base_dict.get(k) is not None):
                TrainingConfig.recursive_update(base_dict[k], v)
            else:
                base_dict[k] = v
