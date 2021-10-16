"""Code for GlowTTS text to speech model"""
import logging
import typing

import numpy as np
import onnxruntime

from larynx.constants import (
    ARRAY_OR_TENSOR,
    InferenceBackend,
    SettingsType,
    TextToSpeechModel,
    TextToSpeechModelConfig,
)

try:
    import torch

    torch_available = True
except ImportError:
    torch_available = False


_LOGGER = logging.getLogger("glow_tts")

# -----------------------------------------------------------------------------


class GlowTextToSpeech(TextToSpeechModel):
    def __init__(self, config: TextToSpeechModelConfig):
        super().__init__(config)

        self.onnx_model: typing.Optional[onnxruntime.InferenceSession] = None
        self.pytorch_model: typing.Optional[typing.Any] = None

        self.use_cuda = config.use_cuda

        # Load model
        onnx_path = config.model_path / "generator.onnx"
        pytorch_path = config.model_path / "generator.pth"
        backend = InferenceBackend.ONNX
        generator_path = onnx_path

        if torch_available:
            if config.backend == InferenceBackend.PYTORCH:
                # Force PyTorch
                generator_path = pytorch_path
                backend = InferenceBackend.PYTORCH
            elif config.backend == InferenceBackend.ONNX:
                # Force Onxx
                generator_path = onnx_path
                backend = InferenceBackend.ONNX
            else:
                # Choose based on settings/availability
                if self.use_cuda and pytorch_path.is_file():
                    # Prefer PyTorch model (supports CUDA)
                    generator_path = pytorch_path
                    backend = InferenceBackend.PYTORCH
                else:
                    # Prefer Onnx model (faster inference)
                    generator_path = onnx_path
                    backend = InferenceBackend.ONNX

        config_path = generator_path.parent / "config.json"

        if backend == InferenceBackend.PYTORCH:
            # Load PyTorch checkpoint
            from glow_tts.checkpoint import load_checkpoint
            from glow_tts.config import TrainingConfig

            _LOGGER.debug("Loading config from %s", config_path)
            with open(config_path, "r", encoding="utf-8") as config_file:
                self.config = TrainingConfig.load(config_file)

            # Load PyTorch model
            _LOGGER.debug(
                "Loading GlowTTS PyTorch model from %s (CUDA=%s, half=%s)",
                generator_path,
                config.use_cuda,
                config.half,
            )
            checkpoint = load_checkpoint(
                pytorch_path, self.config, use_cuda=config.use_cuda
            )

            assert checkpoint.model is not None

            self.pytorch_model = checkpoint.model

            if config.half:
                self.pytorch_model.half()

            # Do not calcuate jacobians for fast decoding
            self.pytorch_model.decoder.store_inverse()
            self.pytorch_model.eval()
        elif backend == InferenceBackend.ONNX:
            _LOGGER.debug("Loading GlowTTS Onnx from %s", generator_path)
            self.onnx_model = onnxruntime.InferenceSession(
                str(generator_path), sess_options=config.session_options
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self.noise_scale = 0.333
        self.length_scale = 1.0

    # -------------------------------------------------------------------------

    def phonemes_to_mels(
        self, phoneme_ids: np.ndarray, settings: typing.Optional[SettingsType] = None
    ) -> ARRAY_OR_TENSOR:
        """Convert phoneme ids to mel spectrograms"""
        # Convert to tensors
        noise_scale = self.noise_scale
        length_scale = self.length_scale
        speaker_idx: typing.Optional[int] = None

        if settings:
            noise_scale = float(settings.get("noise_scale", noise_scale))
            length_scale = float(settings.get("length_scale", length_scale))
            speaker_idx = settings.get("speaker_id")

        if self.pytorch_model is not None:
            # Inference with PyTorch
            speaker_id: typing.Optional[torch.Tensor] = None

            if speaker_idx is not None:
                speaker_id = torch.LongTensor([speaker_idx])
                if self.use_cuda:
                    speaker_id = speaker_id.cuda()

            text_tensor = torch.autograd.Variable(
                torch.LongTensor(phoneme_ids).unsqueeze(0)
            )
            text_lengths_tensor: torch.Tensor = torch.LongTensor([text_tensor.shape[1]])

            if self.use_cuda:
                text_tensor = text_tensor.cuda()
                text_lengths_tensor = text_lengths_tensor.cuda()

            # Infer mel spectrograms
            with torch.no_grad():
                (mel, *_), _, _ = self.pytorch_model(
                    text_tensor,
                    text_lengths_tensor,
                    noise_scale=noise_scale,
                    length_scale=length_scale,
                    g=speaker_id,
                )

            return mel.cpu()

        # Inference with Onnx
        assert self.onnx_model is not None

        text_array = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
        text_lengths_array = np.array([text_array.shape[1]], dtype=np.int64)
        scales_array = np.array([noise_scale, length_scale], dtype=np.float32)

        # Infer mel spectrograms
        mel = self.onnx_model.run(
            None,
            {
                "input": text_array,
                "input_lengths": text_lengths_array,
                "scales": scales_array,
            },
        )[0]

        return mel
