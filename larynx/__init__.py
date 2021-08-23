import logging
import time
import typing
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from pathlib import Path

import gruut
import gruut_ipa
import numpy as np
import onnxruntime

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

# lang -> phoneme -> id
_PHONEME_TO_ID: typing.Dict[str, typing.Dict[str, int]] = {}

# True if stress is included in phonemes
_LANG_STRESS = {
    "en": True,
    "en-us": True,
    "fr": True,
    "fr-fr": True,
    "es": True,
    "es-es": True,
    "it": True,
    "it-it": True,
    "nl": True,
    "sw": True,
}


def text_to_speech(
    text: str,
    lang: str,
    tts_model: typing.Union[TextToSpeechModel, Future],
    vocoder_model: typing.Union[VocoderModel, Future],
    audio_settings: AudioSettings,
    number_converters: bool = False,
    disable_currency: bool = False,
    word_indexes: bool = False,
    inline_pronunciations: bool = False,
    phoneme_transform: typing.Optional[typing.Callable[[str], str]] = None,
    text_lang: typing.Optional[str] = None,
    phoneme_lang: typing.Optional[str] = None,
    tts_settings: typing.Optional[typing.Dict[str, typing.Any]] = None,
    vocoder_settings: typing.Optional[typing.Dict[str, typing.Any]] = None,
    max_workers: typing.Optional[int] = 2,
    executor: typing.Optional[Executor] = None,
    phonemizer: typing.Optional[gruut.Phonemizer] = None,
) -> typing.Iterable[typing.Tuple[str, np.ndarray]]:
    """Tokenize/phonemize text, convert mel spectrograms, then to audio"""
    phoneme_lang = phoneme_lang or lang
    text_lang = text_lang or lang

    phoneme_to_id = _PHONEME_TO_ID.get(phoneme_lang)
    if phoneme_to_id is None:
        no_stress = not _LANG_STRESS.get(phoneme_lang, False)
        phonemes_list = gruut.lang.id_to_phonemes(phoneme_lang, no_stress=no_stress)
        phoneme_to_id = {p: i for i, p in enumerate(phonemes_list)}
        _LOGGER.debug(phoneme_to_id)

        _PHONEME_TO_ID[phoneme_lang] = phoneme_to_id

    # -------------------------------------------------------------------------
    # Tokenization/Phonemization
    # -------------------------------------------------------------------------

    all_sentences = typing.cast(
        typing.List[gruut.Sentence],
        list(
            gruut.text_to_phonemes(
                text,
                lang=text_lang,
                return_format="sentences",
                inline_pronunciations=inline_pronunciations,
                tokenizer_args={
                    "use_number_converters": number_converters,
                    "do_replace_currency": (not disable_currency),
                },
                phonemizer=phonemizer,
                phonemizer_args={
                    "word_break": gruut_ipa.IPA.BREAK_WORD.value,
                    "use_word_indexes": word_indexes,
                },
            )
        ),
    )

    # Process sentences in separate threads concurrently
    futures = {}
    if executor is None:
        executor = ThreadPoolExecutor(max_workers=max_workers)

    for sentence in all_sentences:
        if sentence.phonemes is None:
            continue

        sentence_pron = []
        for word_pron in sentence.phonemes:
            for phoneme in word_pron:
                if not phoneme:
                    continue

                # Split out stress ("ˈa" -> "ˈ", "a")
                # Loop because languages like Swedish can have multiple
                # accents on a single phoneme.
                while phoneme and (
                    gruut_ipa.IPA.is_stress(phoneme[0])
                    or gruut_ipa.IPA.is_accent(phoneme[0])
                ):
                    sentence_pron.append(phoneme[0])
                    phoneme = phoneme[1:]

                sentence_pron.append(phoneme)

        if not sentence_pron:
            # No phonemes
            continue

        # Ensure sentence ends with major break
        if sentence_pron[-1] != gruut_ipa.IPA.BREAK_MAJOR.value:
            sentence_pron.append(gruut_ipa.IPA.BREAK_MAJOR.value)

        # Add another major break for good measure
        sentence_pron.append(gruut_ipa.IPA.BREAK_MAJOR.value)

        text_phonemes: typing.Iterable[str] = sentence_pron

        # ---------------------------------------------------------------------

        if phoneme_transform is not None:
            mapped_phonemes = []
            for phoneme in text_phonemes:
                mapped_phoneme = phoneme_transform(phoneme)
                if isinstance(mapped_phoneme, str):
                    mapped_phonemes.append(mapped_phoneme)
                else:
                    # List
                    mapped_phonemes.extend(mapped_phoneme)

            text_phonemes = mapped_phonemes

        _LOGGER.debug("Words for '%s': %s", sentence.raw_text, sentence.clean_words)
        _LOGGER.debug("Phonemes for '%s': %s", sentence.raw_text, text_phonemes)

        # Convert to phoneme ids
        phoneme_ids_list = []
        for phoneme in text_phonemes:
            phoneme_id = phoneme_to_id.get(phoneme)
            if phoneme_id is not None:
                phoneme_ids_list.append(phoneme_id)
            elif not gruut_ipa.IPA.is_stress(phoneme):
                _LOGGER.warning("%s is missing from voice phoneme inventory", phoneme)

        phoneme_ids = np.array(phoneme_ids_list)

        # Resolve TTS/vocoder model futures
        if isinstance(tts_model, Future):
            tts_model = tts_model.result()

        if isinstance(vocoder_model, Future):
            vocoder_model = vocoder_model.result()

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

    # -------------------------------------------------------------------------

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
    real_time_factor = infer_sec / audio_duration_sec if audio_duration_sec > 0 else 0.0

    _LOGGER.debug(
        "Real-time factor: %0.2f (infer=%0.2f sec, audio=%0.2f sec)",
        real_time_factor,
        infer_sec,
        audio_duration_sec,
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
    executor: typing.Optional[Executor] = None,
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

        return HiFiGanVocoder(config, executor=executor)

    if model_type == VocoderType.WAVEGLOW:
        from .waveglow import WaveGlowVocoder

        return WaveGlowVocoder(config, executor=executor)

    raise ValueError(f"Unknown vocoder model type: {model_type}")
