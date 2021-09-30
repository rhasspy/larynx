import logging
import typing

import numpy as np
import onnxruntime
import torch

from glow_tts.checkpoint import load_checkpoint
from glow_tts.config import TrainingConfig
from glow_tts.models import FlowGenerator
from larynx.constants import SettingsType, TextToSpeechModel, TextToSpeechModelConfig

_LOGGER = logging.getLogger("glow_tts")

# -----------------------------------------------------------------------------


class GlowTextToSpeech(TextToSpeechModel):
    def __init__(self, config: TextToSpeechModelConfig):
        super(GlowTextToSpeech, self).__init__(config)

        self.use_cuda = config.use_cuda

        # Load model
        if config.model_path.is_file():
            # Model path is a file
            generator_path = config.model_path
        else:
            # Model path is a directory
            generator_path = config.model_path / "generator.pth"

        config_path = generator_path.parent / "config.json"

        _LOGGER.debug("Loading config from %s", config_path)
        with open(config_path, "r", encoding="utf-8") as config_file:
            self.config = TrainingConfig.load(config_file)

        _LOGGER.debug(
            "Loading GlowTTS model from %s (CUDA=%s, half=%s)",
            generator_path,
            config.use_cuda,
            config.half,
        )
        checkpoint = load_checkpoint(
            generator_path, self.config, use_cuda=config.use_cuda
        )
        self.model = checkpoint.model

        if config.half:
            self.model.half()

        # Do not calcuate jacobians for fast decoding
        self.model.decoder.store_inverse()
        self.model.eval()

        self.noise_scale = 0.333
        self.length_scale = 1.0

    # -------------------------------------------------------------------------

    def phonemes_to_mels(
        self, phoneme_ids: np.ndarray, settings: typing.Optional[SettingsType] = None
    ) -> torch.Tensor:
        """Convert phoneme ids to mel spectrograms"""
        # Convert to tensors
        # TODO: Allow batches
        text = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
        text_lengths = np.array([text.shape[1]], dtype=np.int64)

        noise_scale = self.noise_scale
        length_scale = self.length_scale
        speaker_id = None

        if settings:
            noise_scale = float(settings.get("noise_scale", noise_scale))
            length_scale = float(settings.get("length_scale", length_scale))
            speaker_idx = settings.get("speaker_id")

            if speaker_idx is not None:
                speaker_id = torch.LongTensor([speaker_idx])
                if self.use_cuda:
                    speaker_id = speaker_id.cuda()

        text = torch.autograd.Variable(torch.LongTensor(phoneme_ids).unsqueeze(0))
        text_lengths = torch.LongTensor([text.shape[1]])

        if self.use_cuda:
            text = text.cuda()
            text_lengths = text_lengths.cuda()

        # Infer mel spectrograms
        with torch.no_grad():
            (mel, *_), _, _ = self.model(
                text,
                text_lengths,
                noise_scale=noise_scale,
                length_scale=length_scale,
                g=speaker_id,
            )

        return mel.cpu()
