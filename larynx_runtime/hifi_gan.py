import logging

import numpy as np
import onnxruntime

from .constants import VocoderModel, VocoderModelConfig

_LOGGER = logging.getLogger("hifi_gan")

# -----------------------------------------------------------------------------


class HiFiGanVocoder(VocoderModel):
    def __init__(self, config: VocoderModelConfig):
        super(HiFiGanVocoder, self).__init__(config)
        self.config = config

        _LOGGER.debug("Loading hifi-gan model from %s", config.model_path)
        self.generator = onnxruntime.InferenceSession(
            str(config.model_path), sess_options=config.session_options
        )

        self.max_wav_value = 32768.0

    def mels_to_audio(self, mels: np.ndarray) -> np.ndarray:
        """Convert mel spectrograms to WAV audio"""
        audio = self.generator.run(None, {"mel": mels})[0]
        audio = audio * self.max_wav_value
        audio = audio.astype("int16")

        return audio
