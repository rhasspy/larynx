import typing

import numpy as np

from larynx.audio import dynamic_range_decompression, inverse, mel_basis, transform
from larynx.constants import (
    ARRAY_OR_TENSOR,
    SettingsType,
    VocoderModel,
    VocoderModelConfig,
)

try:
    import torch
except ImportError:
    pass


# -----------------------------------------------------------------------------


class GriffinLimVocoder(VocoderModel):
    def __init__(
        self,
        config: VocoderModelConfig,
        sample_rate: int = 22050,
        num_fft: int = 1024,
        num_mels: int = 80,
        mel_fmin: float = 0.0,
        mel_fmax: float = 8000,
        mel_scaling: float = 1000.0,
        iterations: int = 60,
    ):
        super().__init__(config)

        self.mel_basis = mel_basis(sample_rate, num_fft, num_mels, mel_fmin, mel_fmax)
        self.mel_scaling = mel_scaling
        self.iterations = iterations

    def mels_to_audio(
        self, mels: ARRAY_OR_TENSOR, settings: typing.Optional[SettingsType] = None
    ) -> np.ndarray:
        """Convert mel spectrograms to WAV audio"""
        if isinstance(mels, np.ndarray):
            mels_array = mels
        else:
            assert isinstance(mels, torch.Tensor)
            mels_array = mels.cpu().numpy()

        mel_decompress = dynamic_range_decompression(mels_array).squeeze(0)
        mel_decompress = mel_decompress.transpose()
        spec_from_mel = np.matmul(mel_decompress, self.mel_basis)
        spec_from_mel = np.expand_dims(spec_from_mel.transpose(), 0)
        spec_from_mel = spec_from_mel * self.mel_scaling

        signal = griffin_lim_iter(
            spec_from_mel[:, :, :-1], n_iters=self.iterations
        ).squeeze(0)

        return signal


# -----------------------------------------------------------------------------


def griffin_lim_iter(magnitudes, n_iters=60):
    """Create audio signal from mel spectrogram"""
    angles = np.angle(np.exp(2j * np.pi * np.random.rand(*magnitudes.shape)))
    angles = angles.astype(np.float32)
    signal = inverse(magnitudes, angles)

    for _ in range(n_iters):
        _, angles = transform(signal)
        signal = inverse(magnitudes, angles)

    return signal
