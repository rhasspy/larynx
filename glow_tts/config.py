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
    mel_channels: int = 80
    sample_rate: int = 22050
    sample_bytes: int = 2
    channels: int = 1
    mel_fmin: float = 0.0
    mel_fmax: typing.Optional[float] = 8000.0
    ref_level_db: float = 20.0
    spec_gain: float = 1.0

    # Normalization
    signal_norm: bool = True
    min_level_db: float = -100.0
    max_norm: float = 1.0
    clip_norm: bool = True
    symmetric_norm: bool = True
    do_dynamic_range_compression: bool = True
    convert_db_to_amp: bool = True


@dataclass
class ModelConfig(DataClassJsonMixin):
    num_symbols: int = 0
    hidden_channels: int = 192
    filter_channels: int = 768
    filter_channels_dp: int = 256
    kernel_size: int = 3
    p_dropout: float = 0.1
    n_blocks_dec: int = 12
    n_layers_enc: int = 6
    n_heads: int = 2
    p_dropout_dec: float = 0.05
    dilation_rate: int = 1
    kernel_size_dec: int = 5
    n_block_layers: int = 4
    n_sqz: int = 2
    prenet: bool = True
    mean_only: bool = True
    hidden_channels_enc: int = 192
    hidden_channels_dec: int = 192
    window_size: int = 4
    n_speakers: int = 1
    n_split: int = 4
    sigmoid_scale: bool = False
    block_length: typing.Optional[int] = None
    gin_channels: int = 0
    n_frames_per_step: int = 1


@dataclass
class TrainingConfig(DataClassJsonMixin):
    seed: int = 1234
    epochs: int = 10000
    learning_rate: float = 1e0
    betas: typing.Tuple[float, float] = field(default=(0.9, 0.98))
    eps: float = 1e-9
    grad_clip: float = 5.0
    warmup_steps: int = 4000
    scheduler: str = "noam"
    batch_size: int = 32
    fp16_run: bool = False
    min_seq_length: typing.Optional[int] = None
    max_seq_length: typing.Optional[int] = None
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
