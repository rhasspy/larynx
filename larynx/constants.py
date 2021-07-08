from __future__ import annotations

import typing
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

if typing.TYPE_CHECKING:
    # Only import here if type checking
    import numpy as np
    import onnxruntime

# -----------------------------------------------------------------------------


class TextToSpeechType(str, Enum):
    """Available text to speech model types"""

    TACOTRON2 = "tacotron2"
    GLOW_TTS = "glow_tts"


class VocoderType(str, Enum):
    """Available vocoder model types"""

    GRIFFIN_LIM = "griffin_lim"
    HIFI_GAN = "hifi_gan"
    WAVEGLOW = "waveglow"


SettingsType = typing.Dict[str, typing.Any]

# -----------------------------------------------------------------------------


@dataclass
class TextToSpeechModelConfig:
    """Configuration base class for text to speech models"""

    model_path: Path
    session_options: onnxruntime.SessionOptions


class TextToSpeechModel(ABC):
    """Base class of text to speech models"""

    def __init__(self, config: TextToSpeechModelConfig):
        pass

    def phonemes_to_mels(
        self, phoneme_ids: np.ndarray, settings: typing.Optional[SettingsType] = None
    ) -> np.ndarray:
        """Convert phoneme ids to mel spectrograms"""
        pass


# -----------------------------------------------------------------------------


@dataclass
class VocoderModelConfig:
    """Configuration base class for vocoder models"""

    model_path: Path
    session_options: onnxruntime.SessionOptions
    denoiser_strength: float = 0.0


class VocoderModel(ABC):
    """Base class of vocoders"""

    def __init__(self, config: VocoderModelConfig):
        pass

    def mels_to_audio(
        self, mels: np.ndarray, settings: typing.Optional[SettingsType] = None
    ) -> np.ndarray:
        """Convert mel spectrograms to WAV audio"""
        pass
