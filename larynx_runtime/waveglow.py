import logging
import typing

import numpy as np
import onnxruntime

from .audio import inverse, transform
from .constants import SettingsType, VocoderModel, VocoderModelConfig

_LOGGER = logging.getLogger("waveglow")

# -----------------------------------------------------------------------------


class WaveGlowVocoder(VocoderModel):
    def __init__(self, config: VocoderModelConfig):
        super(WaveGlowVocoder, self).__init__(config)
        self.config = config

        _LOGGER.debug("Loading WaveGlow model from %s", config.model_path)
        self.waveglow = onnxruntime.InferenceSession(
            str(config.model_path), sess_options=config.session_options
        )

        self.filter_length = 1024
        self.win_length = 1024
        self.hop_length = 256
        self.mel_channels = 80
        self.wn_channels = 256
        self.wn_layers = 8

        # Initialize denoiser
        self.denoiser_strength = config.denoiser_strength
        self.bias_spec: typing.Optional[np.ndarray] = None

        if self.denoiser_strength > 0:
            _LOGGER.debug("Initializing denoiser")
            mel_zeros = np.zeros(shape=(1, self.mel_channels, 88), dtype=np.float32)
            z = self.make_z(mel_zeros)
            bias_audio = self.waveglow.run(None, {"mel": mel_zeros, "z": z})[0].astype(
                np.float32
            )
            bias_spec, _ = transform(bias_audio)

            self.bias_spec = bias_spec[:, :, 0][:, :, None]

    def mels_to_audio(
        self, mels: np.ndarray, settings: typing.Optional[SettingsType] = None
    ) -> np.ndarray:
        """Convert mel spectrograms to WAV audio"""
        z = self.make_z(mels)
        audio = self.waveglow.run(None, {"mel": mels, "z": z})[0]

        if self.denoiser_strength > 0:
            _LOGGER.debug("Running denoiser (strength=%s)", self.denoiser_strength)
            audio = self.denoise(audio)

        audio = audio.squeeze(0)

        return audio

    def make_z(self, mels: np.ndarray) -> np.ndarray:
        z_size2 = (mels.shape[2] * self.wn_channels) // self.wn_layers
        z = np.random.normal(size=(1, self.wn_layers, z_size2)).astype(np.float32)

        return z

    def denoise(self, audio: np.ndarray) -> np.ndarray:
        assert self.bias_spec is not None

        audio_spec, audio_angles = transform(audio)
        audio_spec_denoised = audio_spec - (self.bias_spec * self.denoiser_strength)
        audio_spec_denoised = np.clip(audio_spec_denoised, a_min=0.0, a_max=None)
        audio_denoised = inverse(audio_spec_denoised, audio_angles)

        return audio_denoised
