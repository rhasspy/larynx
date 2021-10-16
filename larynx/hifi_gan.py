"""Code for HiFi-GAN vocoder"""
import concurrent.futures
import json
import logging
import typing
from concurrent.futures import Executor, Future

import numpy as np
import onnxruntime

from larynx.audio import audio_float_to_int16, inverse, transform
from larynx.constants import (
    ARRAY_OR_TENSOR,
    InferenceBackend,
    SettingsType,
    VocoderModel,
    VocoderModelConfig,
)

try:
    import torch

    torch_available = True
except ImportError:
    torch_available = False


_LOGGER = logging.getLogger("hifi_gan")

# -----------------------------------------------------------------------------


class HiFiGanVocoder(VocoderModel):
    def __init__(self, config: VocoderModelConfig, executor: typing.Optional[Executor]):
        super().__init__(config)

        self.use_cuda = config.use_cuda
        self.half = config.half

        self.onnx_model: typing.Optional[onnxruntime.InferenceSession] = None
        self.pytorch_model: typing.Optional[typing.Any] = None

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
            from hifi_gan.checkpoint import load_checkpoint
            from hifi_gan.config import TrainingConfig

            _LOGGER.debug("Loading config from %s", config_path)
            with open(config_path, "r", encoding="utf-8") as config_file:
                self.config = TrainingConfig.load(config_file)

                self.mel_channels = self.config.audio.num_mels

            # Load PyTorch model
            _LOGGER.debug(
                "Loading HiFi-GAN PyTorch model from %s (CUDA=%s, half=%s)",
                generator_path,
                config.use_cuda,
                config.half,
            )
            checkpoint = load_checkpoint(
                generator_path, self.config, use_cuda=config.use_cuda
            )

            assert checkpoint.training_model.generator is not None

            self.pytorch_model = checkpoint.training_model.generator

            if self.half:
                self.pytorch_model.half()

            self.pytorch_model.eval()
            self.pytorch_model.remove_weight_norm()
        elif backend == InferenceBackend.ONNX:
            _LOGGER.debug("Loading HiFi-GAN Onnx from %s", generator_path)
            self.onnx_model = onnxruntime.InferenceSession(
                str(generator_path), sess_options=config.session_options
            )

            with open(config_path, "r", encoding="utf-8") as config_file:
                model_config = json.load(config_file)
                self.mel_channels = int(
                    model_config.get("audio", {}).get("num_mels", "80")
                )

        else:
            raise ValueError(f"Unknown backend: {backend}")

        # Initialize denoiser
        self.denoiser_strength = config.denoiser_strength
        self.bias_spec: typing.Optional[np.ndarray] = None

        self.denoiser_future: typing.Optional[Future] = None

        if self.denoiser_strength > 0:
            if executor is not None:
                # Run in executor
                self.denoiser_future = executor.submit(self.maybe_init_denoiser)
            else:
                # Run here
                self.maybe_init_denoiser()

    def mels_to_audio(
        self, mels: ARRAY_OR_TENSOR, settings: typing.Optional[SettingsType] = None,
    ) -> np.ndarray:
        """Convert mel spectrograms to WAV audio"""
        if self.pytorch_model is not None:
            # Inference with PyTorch
            assert isinstance(mels, torch.Tensor)

            if self.use_cuda:
                mels = mels.cuda()

            with torch.no_grad():
                audio = self.pytorch_model(mels)

            audio = audio.squeeze(0).cpu().numpy()
        else:
            # Inference with Onnx
            assert self.onnx_model is not None
            assert isinstance(mels, np.ndarray)

            audio = self.onnx_model.run(None, {"mel": mels})[0].squeeze(0)

        denoiser_strength = self.denoiser_strength
        if settings:
            denoiser_strength = float(
                settings.get("denoiser_strength", denoiser_strength)
            )

        if denoiser_strength > 0:
            if self.denoiser_future is not None:
                # Denoiser init is already in progress
                concurrent.futures.wait([self.denoiser_future])
                self.denoiser_future = None

            self.maybe_init_denoiser()
            _LOGGER.debug("Running denoiser (strength=%s)", denoiser_strength)
            audio = self.denoise(audio, denoiser_strength)

        audio_norm = audio_float_to_int16(audio)
        return audio_norm.squeeze()

    def denoise(self, audio: np.ndarray, denoiser_strength: float) -> np.ndarray:
        assert self.bias_spec is not None

        audio_spec, audio_angles = transform(audio)
        audio_spec_denoised = audio_spec - (self.bias_spec * denoiser_strength)
        audio_spec_denoised = np.clip(audio_spec_denoised, a_min=0.0, a_max=None)
        audio_denoised = inverse(audio_spec_denoised, audio_angles)

        return audio_denoised

    def maybe_init_denoiser(self):
        if self.bias_spec is None:
            _LOGGER.debug("Initializing denoiser")
            if self.pytorch_model is not None:
                # Inference with PyTorch
                mel_zeros = torch.zeros(
                    size=(1, self.mel_channels, 88),
                    dtype=(torch.float16 if self.half else torch.float32),
                )

                if self.use_cuda:
                    mel_zeros = mel_zeros.cuda()

                with torch.no_grad():
                    bias_audio = self.pytorch_model(mel_zeros).squeeze(0).cpu().numpy()
            else:
                # Inference with Onnx
                mel_zeros = np.zeros(shape=(1, self.mel_channels, 88), dtype=np.float32)
                bias_audio = self.onnx_model.run(None, {"mel": mel_zeros})[0].squeeze(0)

            bias_spec, _ = transform(bias_audio)

            self.bias_spec = bias_spec[:, :, 0][:, :, None]
