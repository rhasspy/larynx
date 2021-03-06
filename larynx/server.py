#!/usr/bin/env python3
"""Larynx web server"""
import argparse
import asyncio
import functools
import io
import json
import logging
import os
import platform
import signal
import time
import typing
from pathlib import Path
from urllib.parse import parse_qs
from uuid import uuid4

import hypercorn
import numpy as np
import quart_cors
from quart import (
    Quart,
    Response,
    jsonify,
    render_template,
    request,
    send_from_directory,
)
from swagger_ui import quart_api_doc

import gruut

from . import (
    TextToSpeechModel,
    VocoderModel,
    load_tts_model,
    load_vocoder_model,
    text_to_speech,
)
from .audio import AudioSettings
from .wavfile import write as wav_write

_DIR = Path(__file__).parent
_VOICES_DIR = _DIR.parent / "local"
_WAV_DIR = _DIR / "wav"

# Directory names that contain vocoders instead of voices
_VOCODER_DIR_NAMES = {"hifi_gan", "waveglow"}

_LOGGER = logging.getLogger("larynx")
_LOOP = asyncio.get_event_loop()

# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(prog="larynx")
parser.add_argument(
    "--host", default="0.0.0.0", help="Host of HTTP server (default: 0.0.0.0)"
)
parser.add_argument(
    "--port", type=int, default=5002, help="Port of HTTP server (default: 5002)"
)
parser.add_argument(
    "--voices-dir",
    help="Directory with <LANGUAGE>/<VOICE> structure (overrides LARYNX_VOICES_DIR env variable)",
)
parser.add_argument(
    "--optimizations",
    choices=["auto", "on", "off"],
    default="auto",
    help="Enable/disable Onnx optimizations (auto=disable on armv7l)",
)
parser.add_argument(
    "--quality",
    choices=["high", "medium", "low"],
    default="high",
    help="Vocoder quality used if not set in API call (default: high)",
)
parser.add_argument(
    "--denoiser-strength",
    type=float,
    default=0.001,
    help="Denoiser strength used if not set in API call (default: 0.001)",
)
parser.add_argument(
    "--noise-scale",
    type=float,
    default=0.333,
    help="Noise scale (voice volatility) used if not set in API call (default: 0.333)",
)
parser.add_argument(
    "--length-scale",
    type=float,
    default=1.0,
    help="Length scale (<1 is faster) used if not set in API call (default: 1.0)",
)
parser.add_argument(
    "--debug", action="store_true", help="Print DEBUG messages to console"
)
args = parser.parse_args()

if args.debug:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

_LOGGER.debug(args)

setattr(args, "no_optimizations", False)
if args.optimizations == "off":
    args.no_optimizations = True
elif args.optimizations == "auto":
    if platform.machine() == "armv7l":
        # Enabling optimizations on 32-bit ARM crashes
        args.no_optimizations = True

if args.voices_dir:
    # Use command-line
    _VOICES_DIR = Path(args.voices_dir)
else:
    # Use environment variable or default
    _VOICES_DIR = Path(os.environ.get("LARYNX_VOICES_DIR", _VOICES_DIR))

_LOGGER.info("Voices directory: %s", _VOICES_DIR)

# -----------------------------------------------------------------------------

app = Quart("larynx")
app.secret_key = str(uuid4())

if args.debug:
    app.config["TEMPLATES_AUTO_RELOAD"] = True

app = quart_cors.cors(app)

# -----------------------------------------------------------------------------

_DEFAULT_AUDIO_SETTINGS = AudioSettings()

# Set default vocoder based on quality
_DEFAULT_VOCODER = {
    "high": "hifi_gan/universal_large",
    "medium": "hifi_gan/vctk_medium",
    "low": "hifi_gan/vctk_small",
}[args.quality]

# Caches
_TTS_MODELS: typing.Dict[str, TextToSpeechModel] = {}
_AUDIO_SETTINGS: typing.Dict[str, AudioSettings] = {}
_VOCODER_MODELS: typing.Dict[str, VocoderModel] = {}
_GRUUT_LANGS: typing.Dict[str, gruut.Language] = {}


async def text_to_wav(
    text: str,
    voice: str,
    vocoder: str,
    denoiser_strength: typing.Optional[float] = None,
    noise_scale: typing.Optional[float] = None,
    length_scale: typing.Optional[float] = None,
    inline_phonemes: bool = False,
) -> bytes:
    """Runs TTS for each line and accumulates all audio into a single WAV."""
    language: typing.Optional[str] = None

    if "/" in voice:
        # <LANGUAGE>/<VOICE_NAME>-<TTS_SYSTEM>
        language, voice_str = voice.split("/", maxsplit=1)
    else:
        # Search for voice
        found_voice = False

        # <VOICES_DIR>/<LANGUAGE>/<VOICE>
        for lang_dir in _VOICES_DIR.iterdir():
            if not lang_dir.is_dir():
                continue

            for voice_dir in lang_dir.iterdir():
                if not voice_dir.is_dir():
                    continue

                if voice_dir.name == voice:
                    language = lang_dir.name
                    voice_str = voice_dir.name
                    found_voice = True
                    break

            if found_voice:
                break

    assert (
        language
    ), f"No language set for voice: {voice} (is it installed in {_VOICES_DIR}?)"

    _voice_name, tts_system = voice_str.split("-", maxsplit=1)

    tts_model = _TTS_MODELS.get(voice)
    if tts_model is None:
        # Load TTS model
        tts_model_path = _VOICES_DIR / language / voice_str
        tts_model = load_tts_model(
            model_type=tts_system,
            model_path=tts_model_path,
            no_optimizations=args.no_optimizations,
        )
        _TTS_MODELS[voice] = tts_model

        # Load audio settings
        tts_config_path = tts_model_path / "config.json"
        if tts_config_path.is_file():
            _LOGGER.debug("Loading audio settings from %s", tts_config_path)
            with open(tts_config_path, "r") as tts_config_file:
                tts_config = json.load(tts_config_file)
                _AUDIO_SETTINGS[voice] = AudioSettings(**tts_config["audio"])
        else:
            _LOGGER.warning(
                "No config file found at %s, using default audio settings",
                tts_config_path,
            )

    audio_settings = _AUDIO_SETTINGS.get(voice, _DEFAULT_AUDIO_SETTINGS)

    # <VOCODER_SYSTEM>/<MODEL_NAME>
    vocoder_system, vocoder_name = vocoder.split("/", maxsplit=1)

    vocoder_model = _VOCODER_MODELS.get(vocoder)
    if vocoder_model is None:
        # Load vocoder
        vocoder_model_path = _VOICES_DIR / vocoder_system / vocoder_name
        vocoder_model = load_vocoder_model(
            model_type=vocoder_system,
            model_path=vocoder_model_path,
            no_optimizations=args.no_optimizations,
        )
        _VOCODER_MODELS[vocoder] = vocoder_model

    # Load language
    gruut_lang = get_lang(language)

    # Settings
    tts_settings = None
    if (noise_scale is not None) or (length_scale is not None):
        tts_settings = {}
        if noise_scale is not None:
            tts_settings["noise_scale"] = noise_scale

        if length_scale is not None:
            tts_settings["length_scale"] = length_scale

    vocoder_settings = None
    if denoiser_strength is not None:
        vocoder_settings = {"denoiser_strength": denoiser_strength}

    # Synthesize each line separately.
    # Accumulate into a single WAV file.
    _LOGGER.info("Synthesizing with %s, %s (%s char(s))...", voice, vocoder, len(text))
    start_time = time.time()

    text_and_audios = await _LOOP.run_in_executor(
        None,
        functools.partial(
            text_to_speech,
            text=text,
            gruut_lang=gruut_lang,
            tts_model=tts_model,
            vocoder_model=vocoder_model,
            audio_settings=audio_settings,
            tts_settings=tts_settings,
            vocoder_settings=vocoder_settings,
            inline_phonemes=inline_phonemes,
        ),
    )

    audios = []
    for _, audio in text_and_audios:
        audios.append(audio)

    with io.BytesIO() as wav_io:
        wav_write(wav_io, audio_settings.sample_rate, np.concatenate(audios))
        wav_bytes = wav_io.getvalue()

    end_time = time.time()
    _LOGGER.debug(
        "Synthesized %s byte(s) in %s second(s)", len(wav_bytes), end_time - start_time
    )

    return wav_bytes


def get_voices() -> typing.Dict[str, typing.Dict[str, str]]:
    """Get dict of voices"""
    voices = {}

    # <LANGUAGE>/<VOICE>-<TTS_SYSTEM>
    for lang_dir in _VOICES_DIR.iterdir():
        if (not lang_dir.is_dir()) or (lang_dir.name in _VOCODER_DIR_NAMES):
            continue

        language = lang_dir.name

        for voice_dir in lang_dir.iterdir():
            if not voice_dir.is_dir():
                continue

            voice_name, tts_system = voice_dir.name.split("-", maxsplit=1)
            voice_id = f"{language}/{voice_dir.name}"

            voices[voice_id] = {
                "id": voice_id,
                "name": voice_name,
                "language": language,
                "tts_system": tts_system,
            }

    return voices


def get_lang(language: str) -> gruut.Language:
    """Load language from cache or disk"""
    gruut_lang = _GRUUT_LANGS.get(language)
    if gruut_lang is None:
        data_dirs = gruut.Language.get_data_dirs() + [_DIR.parent / "gruut"]
        gruut_lang = gruut.Language.load(language=language, data_dirs=data_dirs)

        assert gruut_lang, f"No support for language {language} in gruut ({data_dirs})"
        _GRUUT_LANGS[language] = gruut_lang

    return gruut_lang


# -----------------------------------------------------------------------------
# HTTP Endpoints
# -----------------------------------------------------------------------------


@app.route("/api/voices")
async def app_voices() -> Response:
    """Get available voices."""
    return jsonify(get_voices())


@app.route("/api/vocoders")
async def app_vocoders() -> Response:
    """Get available vocoders."""
    vocoders = []

    # <VOCODER_SYSTEM>/<VOCODER_MODEL>
    for vocoder_dir in _VOICES_DIR.iterdir():
        if (not vocoder_dir.is_dir()) or (vocoder_dir.name not in _VOCODER_DIR_NAMES):
            continue

        vocoder_system = vocoder_dir.name

        for model_dir in vocoder_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name
            vocoder_id = f"{vocoder_system}/{model_name}"

            vocoders.append(
                {"id": vocoder_id, "name": model_name, "vocoder_system": vocoder_system}
            )

    return jsonify(vocoders)


@app.route("/api/tts", methods=["GET", "POST"])
async def app_say() -> Response:
    """Speak text to WAV."""
    voice = request.args.get("voice", "")
    assert voice, "No voice provided"

    # If true, [[ phonemes ]] in brackets are spoken literally
    inline_phonemes = request.args.get("inlinePhonemes", "").strip().lower() in {
        "true",
        "1",
    }

    # TTS settings
    noise_scale = request.args.get("noiseScale", args.noise_scale)
    if noise_scale is not None:
        noise_scale = float(noise_scale)

    length_scale = request.args.get("lengthScale", args.length_scale)
    if length_scale is not None:
        length_scale = float(length_scale)

    # Text can come from POST body or GET ?text arg
    if request.method == "POST":
        text = (await request.data).decode()
    else:
        text = request.args.get("text")

    assert text, "No text provided"

    vocoder = request.args.get("vocoder", _DEFAULT_VOCODER)

    # Vocoder settings
    denoiser_strength = request.args.get("denoiserStrength", args.denoiser_strength)
    if denoiser_strength is not None:
        denoiser_strength = float(denoiser_strength)

    wav_bytes = await text_to_wav(
        text,
        voice,
        vocoder=vocoder,
        denoiser_strength=denoiser_strength,
        noise_scale=noise_scale,
        length_scale=length_scale,
        inline_phonemes=inline_phonemes,
    )

    return Response(wav_bytes, mimetype="audio/wav")


@app.route("/api/phonemes", methods=["GET"])
async def api_phonemes():
    """Get phonemes for language"""
    language = request.args.get("language", "en-us")
    gruut_lang = get_lang(language)

    phonemes: typing.Dict[str, typing.Dict[str, typing.Any]] = {}
    for phoneme in gruut_lang.phonemes:
        # Try to guess WAV file name for phoneme
        # Files from https://www.ipachart.com/
        wav_path: typing.Optional[Path] = None
        if phoneme.vowel:
            height_str = phoneme.vowel.height.value
            placement_str = phoneme.vowel.placement.value
            rounded_str = "rounded" if phoneme.vowel.rounded else "unrounded"
            wav_path = (
                _WAV_DIR / f"{height_str}_{placement_str}_{rounded_str}_vowel.wav"
            )
        elif phoneme.consonant:
            voiced_str = "voiced" if phoneme.consonant.voiced else "voiceless"
            place_str = phoneme.consonant.place.value.replace("-", "")
            type_str = phoneme.consonant.type.value.replace("-", "_")
            wav_path = _WAV_DIR / f"{voiced_str}_{place_str}_{type_str}.wav"
            if not wav_path.is_file():
                # Try without voicing
                wav_path = _WAV_DIR / f"{place_str}_{type_str}.wav"
        elif phoneme.schwa:
            if phoneme.schwa.r_coloured:
                # Close enough to "r" (er in corn[er])
                wav_path = _WAV_DIR / f"alveolar_approximant.wav"
            else:
                # ??
                wav_path = _WAV_DIR / f"mid-central_vowel.wav"

        phoneme_dict = {"example": phoneme.example}

        if wav_path and wav_path.is_file():
            # Augment with relative URL to WAV file
            phoneme_dict["url"] = f"wav/{wav_path.name}"
        else:
            _LOGGER.debug("No WAV for %s (%s)", phoneme.text, wav_path)

        phonemes[phoneme.text] = phoneme_dict

    # {
    #   "<PHONEME>": {
    #     "example": "...",
    #     "url": "wav/<voiced>_<place>_<type>.wav"
    #   }
    # }
    return jsonify(phonemes)


@app.route("/api/word-phonemes", methods=["GET", "POST"])
async def api_word_phonemes():
    """Get phonemes for a word"""
    if request.method == "POST":
        word = (await request.data).decode()
    else:
        word = request.args.get("word", "")

    assert word, "No word given"

    language = request.args.get("language", "en-us")
    gruut_lang = get_lang(language)

    pron_phonemes: typing.List[typing.List[str]] = []

    _LOGGER.debug("Looking up in %s lexicon: %s", language, word)
    prons = gruut_lang.phonemizer.lookup_word(word)
    if prons:
        # Use lexicon
        pron_phonemes = [p.phonemes for p in prons]
        _LOGGER.debug(
            "Found %s pronunciation(s) in lexicon for %s", len(pron_phonemes), word
        )
    else:
        # Guess pronunciations
        num_guesses = int(request.args.get("numGuesses", "5"))
        pron_phonemes = [
            phonemes
            for _word, phonemes in gruut_lang.phonemizer.predict(
                [word], nbest=num_guesses
            )
        ]
        _LOGGER.debug("Guessed %s pronunciation(s) for %s", len(pron_phonemes), word)

    return jsonify(pron_phonemes)


# -----------------------------------------------------------------------------

# MaryTTS compatibility layer
@app.route("/process", methods=["GET", "POST"])
async def api_process():
    """MaryTTS-compatible /process endpoint"""
    if request.method == "POST":
        data = parse_qs((await request.data).decode())
        text = data.get("INPUT_TEXT", [""])[0]
        voice = data.get("VOICE", [""])[0]
    else:
        text = request.args.get("INPUT_TEXT", "")
        voice = request.args.get("VOICE", "")

    # <VOICE>;<VOCODER>
    vocoder: typing.Optional[str] = None

    if ";" in voice:
        voice, vocoder = voice.split(";", maxsplit=1)

    vocoder = vocoder or _DEFAULT_VOCODER

    wav_bytes = await text_to_wav(
        text,
        voice,
        vocoder=vocoder,
        denoiser_strength=args.denoiser_strength,
        noise_scale=args.noise_scale,
        length_scale=args.length_scale,
    )

    return Response(wav_bytes, mimetype="audio/wav")


@app.route("/voices", methods=["GET"])
async def api_voices():
    """MaryTTS-compatible /voices endpoint"""
    lines = []
    for voice_id in get_voices():
        lines.append(voice_id)

    return "\n".join(lines)


# -----------------------------------------------------------------------------

_CSS_DIR = _DIR / "css"
_IMG_DIR = _DIR / "img"


@app.route("/")
async def app_index():
    """Main page."""
    return await render_template("index.html")


@app.route("/css/<path:filename>", methods=["GET"])
async def css(filename) -> Response:
    """CSS static endpoint."""
    return await send_from_directory(_CSS_DIR, filename)


@app.route("/img/<path:filename>", methods=["GET"])
async def img(filename) -> Response:
    """Image static endpoint."""
    return await send_from_directory(_IMG_DIR, filename)


@app.route("/wav/<path:filename>", methods=["GET"])
async def wav(filename) -> Response:
    """WAV audio static endpoint."""
    return await send_from_directory(_WAV_DIR, filename)


# Swagger UI
quart_api_doc(
    app, config_path=str(_DIR / "swagger.yaml"), url_prefix="/openapi", title="Larynx"
)


@app.errorhandler(Exception)
async def handle_error(err) -> typing.Tuple[str, int]:
    """Return error as text."""
    _LOGGER.exception(err)
    return (f"{err.__class__.__name__}: {err}", 500)


# -----------------------------------------------------------------------------
# Run Web Server
# -----------------------------------------------------------------------------

hyp_config = hypercorn.config.Config()
hyp_config.bind = [f"{args.host}:{args.port}"]

# Create shutdown event for Hypercorn
shutdown_event = asyncio.Event()


def _signal_handler(*_: typing.Any) -> None:
    """Signal shutdown to Hypercorn"""
    shutdown_event.set()


_LOOP.add_signal_handler(signal.SIGTERM, _signal_handler)

try:
    # Need to type cast to satisfy mypy
    shutdown_trigger = typing.cast(
        typing.Callable[..., typing.Awaitable[None]], shutdown_event.wait
    )

    _LOOP.run_until_complete(
        hypercorn.asyncio.serve(app, hyp_config, shutdown_trigger=shutdown_trigger)
    )
except KeyboardInterrupt:
    _LOOP.call_soon(shutdown_event.set)
