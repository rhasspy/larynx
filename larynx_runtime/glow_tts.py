import logging
import typing

import numpy as np
import onnxruntime

from .constants import SettingsType, TextToSpeechModel, TextToSpeechModelConfig

_LOGGER = logging.getLogger("glow_tts")

# -----------------------------------------------------------------------------


class GlowTextToSpeech(TextToSpeechModel):
    def __init__(self, config: TextToSpeechModelConfig):
        super(GlowTextToSpeech, self).__init__(config)
        sess_options = config.session_options

        # Load model
        _LOGGER.debug("Loading model from %s", config.model_path)
        self.model = onnxruntime.InferenceSession(
            str(config.model_path), sess_options=sess_options
        )

        self.noise_scale = 0.333
        self.length_scale = 1.0

    # -------------------------------------------------------------------------

    def phonemes_to_mels(
        self, phoneme_ids: np.ndarray, settings: typing.Optional[SettingsType] = None
    ) -> np.ndarray:
        """Convert phoneme ids to mel spectrograms"""
        # Convert to tensors
        # TODO: Allow batches
        text = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
        text_lengths = np.array([text.shape[1]], dtype=np.int64)

        noise_scale = self.noise_scale
        length_scale = self.length_scale

        if settings:
            noise_scale = settings.get("noise_scale", noise_scale)
            length_scale = settings.get("length_scale", length_scale)

        scales = np.array([noise_scale, length_scale], dtype=np.float32)

        # Infer mel spectrograms
        mel = self.model.run(
            None, {"input": text, "input_lengths": text_lengths, "scales": scales}
        )[0]

        return mel
