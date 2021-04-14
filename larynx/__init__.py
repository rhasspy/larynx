import itertools
import logging
import time
import typing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import onnxruntime

import gruut
import gruut_ipa

from .audio import AudioSettings
from .constants import (
    TextToSpeechModel,
    TextToSpeechModelConfig,
    TextToSpeechType,
    VocoderModel,
    VocoderModelConfig,
    VocoderType,
)

_LOGGER = logging.getLogger("larynx")

_DIR = Path(__file__).parent

__version__ = (_DIR / "VERSION").read_text().strip()

# -----------------------------------------------------------------------------


def text_to_speech(
    text: str,
    gruut_lang: gruut.Language,
    tts_model: TextToSpeechModel,
    vocoder_model: VocoderModel,
    audio_settings: AudioSettings,
    number_converters: bool = False,
    disable_currency: bool = False,
    word_indexes: bool = False,
    tts_settings: typing.Optional[typing.Dict[str, typing.Any]] = None,
    vocoder_settings: typing.Optional[typing.Dict[str, typing.Any]] = None,
    native_lang: typing.Optional[gruut.Language] = None,
    max_workers: typing.Optional[int] = 2,
) -> typing.Iterable[typing.Tuple[str, np.ndarray]]:
    """Tokenize/phonemize text, convert mel spectrograms, then to audio"""
    tokenizer = gruut_lang.tokenizer
    phonemizer = gruut_lang.phonemizer

    accent_map: typing.Optional[typing.Dict[str, typing.List[str]]] = None
    if native_lang:
        # Use native language phonemes
        accent_map = gruut_lang.accents[native_lang.language]
        phoneme_lang = native_lang
    else:
        # Use original language phonemes
        phoneme_lang = gruut_lang

    phoneme_to_id = getattr(phoneme_lang, "phoneme_to_id", None)
    if phoneme_to_id is None:
        phonemes_list = phoneme_lang.id_to_phonemes()
        phoneme_to_id = {p: i for i, p in enumerate(phonemes_list)}
        _LOGGER.debug(phoneme_to_id)

        setattr(phoneme_lang, "phoneme_to_id", phoneme_to_id)

    all_sentences = list(
        tokenizer.tokenize(
            text,
            number_converters=number_converters,
            replace_currency=(not disable_currency),
        )
    )

    # Process sentences in separate threads concurrently
    futures = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for sentence in all_sentences:
            # Phonemize each sentence
            sentence_prons = phonemizer.phonemize(
                sentence.tokens, word_indexes=word_indexes, word_breaks=True
            )

            # Pick first pronunciation for each word
            first_pron = []
            for word_prons in sentence_prons:
                if word_prons:
                    for phoneme in word_prons[0].phonemes:
                        if not phoneme:
                            continue

                        # Split out stress ("ˈa" -> "ˈ", "a")
                        # Loop because languages like Swedish can have multiple
                        # accents on a single phoneme.
                        while phoneme and (
                            gruut_ipa.IPA.is_stress(phoneme[0])
                            or gruut_ipa.IPA.is_accent(phoneme[0])
                        ):
                            first_pron.append(phoneme[0])
                            phoneme = phoneme[1:]

                        first_pron.append(phoneme)

            if not first_pron:
                # No phonemes
                continue

            # Ensure sentence ends with major break
            if first_pron[-1] != gruut_ipa.IPA.BREAK_MAJOR.value:
                first_pron.append(gruut_ipa.IPA.BREAK_MAJOR.value)

            # Add another major break for good measure
            first_pron.append(gruut_ipa.IPA.BREAK_MAJOR.value)

            text_phonemes: typing.Iterable[str] = first_pron

            if accent_map:
                # Map to different phoneme set
                text_phonemes = itertools.chain(
                    *[accent_map.get(p, p) for p in first_pron]
                )

            # ---------------------------------------------------------------------

            _LOGGER.debug("Words for '%s': %s", sentence.raw_text, sentence.clean_words)
            _LOGGER.debug("Phonemes for '%s': %s", sentence.raw_text, text_phonemes)

            # Convert to phoneme ids
            phoneme_ids = np.array([phoneme_to_id[p] for p in text_phonemes])

            future = executor.submit(
                _sentence_task,
                phoneme_ids,
                audio_settings,
                tts_model,
                tts_settings,
                vocoder_model,
                vocoder_settings,
            )
            futures[future] = sentence.raw_text

        # ---------------------------------------------------------------------

        for future, raw_text in futures.items():
            audio = future.result()
            yield raw_text, audio


# -----------------------------------------------------------------------------


def _sentence_task(
    phoneme_ids,
    audio_settings,
    tts_model,
    tts_settings,
    vocoder_model,
    vocoder_settings,
):
    # Run text to speech
    _LOGGER.debug("Running text to speech model (%s)", tts_model.__class__.__name__)
    tts_start_time = time.perf_counter()

    mels = tts_model.phonemes_to_mels(phoneme_ids, settings=tts_settings)
    tts_end_time = time.perf_counter()

    _LOGGER.debug(
        "Got mels in %s second(s) (shape=%s)", tts_end_time - tts_start_time, mels.shape
    )

    # Do denormalization, etc.
    if audio_settings.signal_norm:
        mels = audio_settings.denormalize(mels)

    if audio_settings.convert_db_to_amp:
        mels = audio_settings.db_to_amp(mels)

    if audio_settings.do_dynamic_range_compression:
        mels = audio_settings.dynamic_range_compression(mels)

    # Run vocoder
    _LOGGER.debug("Running vocoder model (%s)", vocoder_model.__class__.__name__)
    vocoder_start_time = time.perf_counter()
    audio = vocoder_model.mels_to_audio(mels, settings=vocoder_settings)
    vocoder_end_time = time.perf_counter()

    _LOGGER.debug(
        "Got audio in %s second(s) (shape=%s)",
        vocoder_end_time - vocoder_start_time,
        audio.shape,
    )

    audio_duration_sec = audio.shape[-1] / audio_settings.sample_rate
    infer_sec = vocoder_end_time - tts_start_time
    real_time_factor = audio_duration_sec / infer_sec

    _LOGGER.debug(
        "Real-time factor: %0.2f (audio=%0.2f sec, infer=%0.2f sec)",
        real_time_factor,
        audio_duration_sec,
        infer_sec,
    )

    return audio


# -----------------------------------------------------------------------------


def load_tts_model(
    model_type: typing.Union[str, TextToSpeechType],
    model_path: typing.Union[str, Path],
    no_optimizations: bool = False,
) -> TextToSpeechModel:
    """Load the appropriate text to speech model"""
    sess_options = onnxruntime.SessionOptions()
    if no_optimizations:
        sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        )

    config = TextToSpeechModelConfig(
        model_path=Path(model_path), session_options=sess_options
    )

    if model_type == TextToSpeechType.TACOTRON2:
        from .tacotron2 import Tacotron2TextToSpeech

        return Tacotron2TextToSpeech(config)

    if model_type == TextToSpeechType.GLOW_TTS:
        from .glow_tts import GlowTextToSpeech

        return GlowTextToSpeech(config)

    raise ValueError(f"Unknown text to speech model type: {model_type}")


# -----------------------------------------------------------------------------


def load_vocoder_model(
    model_type: typing.Union[str, VocoderType],
    model_path: typing.Union[str, Path],
    no_optimizations: bool = False,
    denoiser_strength: float = 0.0,
) -> VocoderModel:
    """Load the appropriate vocoder model"""
    sess_options = onnxruntime.SessionOptions()
    if no_optimizations:
        sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        )

    config = VocoderModelConfig(
        model_path=Path(model_path),
        session_options=sess_options,
        denoiser_strength=denoiser_strength,
    )

    if model_type == VocoderType.GRIFFIN_LIM:
        from .griffin_lim import GriffinLimVocoder

        return GriffinLimVocoder(config)

    if model_type == VocoderType.HIFI_GAN:
        from .hifi_gan import HiFiGanVocoder

        return HiFiGanVocoder(config)

    if model_type == VocoderType.WAVEGLOW:
        from .waveglow import WaveGlowVocoder

        return WaveGlowVocoder(config)

    raise ValueError(f"Unknown vocoder model type: {model_type}")
