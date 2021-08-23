#!/usr/bin/env python3
"""Larynx web server"""
import argparse
import asyncio
import contextlib
import functools
import io
import json
import logging
import platform
import signal
import time
import typing
from collections import defaultdict
from pathlib import Path
from urllib.parse import parse_qs
from uuid import uuid4

import gruut
import gruut_ipa
import hypercorn
import numpy as np
import pidfile
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

from . import (
    TextToSpeechModel,
    VocoderModel,
    load_tts_model,
    load_vocoder_model,
    text_to_speech,
)
from .audio import AudioSettings
from .utils import (
    DEFAULT_VOICE_URL_FORMAT,
    VOCODER_DIR_NAMES,
    VOICE_DOWNLOAD_NAMES,
    download_voice,
    get_voices_dirs,
    load_voices_aliases,
    resolve_voice_name,
    valid_voice_dir,
)
from .wavfile import write as wav_write

_DIR = Path(__file__).parent
_WAV_DIR = _DIR / "wav"

_LOGGER = logging.getLogger("larynx")
_LOOP = asyncio.get_event_loop()

# language -> file name
_SAMPLE_SENTENCES = {
    "de-de": "haben_sie_ein_vegetarisches",
    "en-us": "it_took_me_quite_a_long_time_to_develop_a_voice",
    "es-es": "siga_recto",
    "fr-fr": "pourriez-vous_parler",
    "it-it": "parli_un_altra",
    "nl": "kunt_u_wat_langzamer_praten_alstublieft",
    "ru-ru": "Моё_судно_на",
    "sv-se": "den_här_damen",
    "sw": "gari_langu_linaloangama_limejaa_na_mikunga",
}

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
    "--url-format",
    default=DEFAULT_VOICE_URL_FORMAT,
    help="Format string for download URLs (accepts {voice})",
)
parser.add_argument(
    "--pidfile", help="Path to pidfile. Exit if pidfile already exists."
)
parser.add_argument(
    "--lexicon",
    nargs=2,
    action="append",
    metavar=("language", "lexicon"),
    help="Language and path to custom lexicon with format WORD PHONEME PHONEME ...",
)
parser.add_argument("--logfile", help="Path to logging file (default: stderr)")
parser.add_argument(
    "--debug", action="store_true", help="Print DEBUG messages to console"
)
args = parser.parse_args()

# Set up logging
log_args = {}

if args.debug:
    log_args["level"] = logging.DEBUG
else:
    log_args["level"] = logging.INFO

if args.logfile:
    log_args["filename"] = args.logfile

logging.basicConfig(**log_args)  # type: ignore
_LOGGER.debug(args)

setattr(args, "no_optimizations", False)
if args.optimizations == "off":
    args.no_optimizations = True
elif args.optimizations == "auto":
    if platform.machine() == "armv7l":
        # Enabling optimizations on 32-bit ARM crashes
        args.no_optimizations = True

voices_dirs = get_voices_dirs(args.voices_dir)
load_voices_aliases()


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
_GRUUT_PHONEMIZER: typing.Dict[str, gruut.Phonemizer] = {}
_PHONEME_TRANSFORM: typing.Dict[
    typing.Tuple[str, str], typing.Callable[[str], str]
] = {}


async def text_to_wav(
    text: str,
    voice: str,
    vocoder: str,
    denoiser_strength: typing.Optional[float] = None,
    noise_scale: typing.Optional[float] = None,
    length_scale: typing.Optional[float] = None,
    inline_pronunciations: bool = False,
    text_lang: typing.Optional[str] = None,
) -> bytes:
    """Runs TTS for each line and accumulates all audio into a single WAV."""
    language: typing.Optional[str] = None
    voice_str = voice

    if "/" in voice:
        # <LANGUAGE>/<VOICE_NAME>-<TTS_SYSTEM>
        language, voice_str = voice_str.split("/", maxsplit=1)

    tts_model = _TTS_MODELS.get(voice)
    if tts_model is None:
        tts_model_dir: typing.Optional[Path] = None

        if language is not None:
            # Search by exact language/name
            # <VOICES_DIR>/<LANGUAGE>/<VOICE>
            for voices_dir in voices_dirs:
                maybe_tts_model_dir = voices_dir / language / voice_str
                if valid_voice_dir(maybe_tts_model_dir):
                    tts_model_dir = maybe_tts_model_dir
                    break

        if tts_model_dir is None:
            # Search for voice in all directories
            voice_str = resolve_voice_name(voice_str)

            # <VOICES_DIR>/<LANGUAGE>/<VOICE>
            found_voice = False
            for voices_dir in voices_dirs:
                if not voices_dir.is_dir():
                    continue

                # <LANGUAGE>
                for lang_dir in voices_dir.iterdir():
                    if not lang_dir.is_dir():
                        continue

                    # <VOICE>
                    for maybe_tts_model_dir in lang_dir.iterdir():
                        if not maybe_tts_model_dir.is_dir():
                            continue

                        if (maybe_tts_model_dir.name == voice_str) and valid_voice_dir(
                            maybe_tts_model_dir
                        ):
                            tts_model_dir = maybe_tts_model_dir
                            language = lang_dir.name
                            found_voice = True
                            break

                    if found_voice:
                        break

                if found_voice:
                    break

        assert tts_model_dir is not None, f"Voice '{voice}' not found in {voices_dirs}"
        assert language is not None, f"Language not set for {voice}"

        # Load TTS model
        _voice_name, tts_system = voice_str.split("-", maxsplit=1)
        tts_model = load_tts_model(
            model_type=tts_system,
            model_path=tts_model_dir,
            no_optimizations=args.no_optimizations,
        )
        setattr(tts_model, "language", language)

        _TTS_MODELS[voice] = tts_model
        _TTS_MODELS[voice_str] = tts_model

        # Load audio settings
        tts_config_path = tts_model_dir / "config.json"
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

    # Load vocoder

    # <VOCODER_SYSTEM>/<MODEL_NAME>
    vocoder_model = _VOCODER_MODELS.get(vocoder)
    if vocoder_model is None:
        vocoder_system, vocoder_name = vocoder.split("/", maxsplit=1)

        # Load vocoder
        vocoder_model_path: typing.Optional[Path] = None
        for voices_dir in voices_dirs:
            maybe_vocoder_model_path = voices_dir / vocoder_system / vocoder_name
            if valid_voice_dir(maybe_vocoder_model_path):
                vocoder_model_path = maybe_vocoder_model_path
                break

        assert (
            vocoder_model_path is not None
        ), f"Vocoder '{vocoder}' not found in {voices_dirs}"

        vocoder_model = load_vocoder_model(
            model_type=vocoder_system,
            model_path=vocoder_model_path,
            no_optimizations=args.no_optimizations,
        )
        _VOCODER_MODELS[vocoder] = vocoder_model

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

    # Phonemizer
    phoneme_lang: str = tts_model.language  # type: ignore

    if not text_lang:
        text_lang = phoneme_lang

    phonemizer = get_phonemizer(text_lang)
    phoneme_transform: typing.Optional[typing.Callable[[str], str]] = None

    if text_lang != phoneme_lang:
        phoneme_transform = _PHONEME_TRANSFORM.get((text_lang, phoneme_lang))
        if phoneme_transform is None:
            from gruut_ipa import Phoneme, Phonemes
            from gruut_ipa.accent import guess_phonemes

            # Guess phoneme map
            from_phonemes, to_phonemes = (
                Phonemes.from_language(text_lang),
                Phonemes.from_language(phoneme_lang),
            )

            phoneme_map = {
                from_p.text: [
                    to_p.text
                    for to_p in typing.cast(
                        typing.List[Phoneme], guess_phonemes(from_p, to_phonemes)
                    )
                ]
                for from_p in from_phonemes
            }

            _LOGGER.debug(
                "Guessed phoneme map from %s to %s: %s",
                text_lang,
                phoneme_lang,
                phoneme_map,
            )

            def phoneme_map_transform(p):
                return phoneme_map.get(p, p)

            phoneme_transform = phoneme_map_transform

            # Cache for later
            _PHONEME_TRANSFORM[(text_lang, phoneme_lang)] = phoneme_transform

    # Synthesize each line separately.
    # Accumulate into a single WAV file.
    _LOGGER.info("Synthesizing with %s, %s (%s char(s))...", voice, vocoder, len(text))
    start_time = time.time()

    text_and_audios = await _LOOP.run_in_executor(
        None,
        functools.partial(
            text_to_speech,
            text=text,
            lang=phoneme_lang,
            text_lang=text_lang,
            tts_model=tts_model,
            vocoder_model=vocoder_model,
            audio_settings=audio_settings,
            tts_settings=tts_settings,
            vocoder_settings=vocoder_settings,
            inline_pronunciations=inline_pronunciations,
            phonemizer=phonemizer,
            phoneme_transform=phoneme_transform,
        ),
    )

    audios = []
    for _, audio in text_and_audios:
        audios.append(audio)

    with io.BytesIO() as wav_io:
        wav_write(wav_io, audio_settings.sample_rate, np.concatenate(audios))
        wav_bytes = wav_io.getvalue()

    end_time = time.time()
    _LOGGER.info(
        "Synthesized %s byte(s) in %s second(s)", len(wav_bytes), end_time - start_time
    )

    return wav_bytes


def get_voices() -> typing.Dict[str, typing.Dict[str, typing.Any]]:
    """Get dict of voices"""
    voices = {}

    # Search for downloaded voices/vocoders
    for voices_dir in voices_dirs:
        if not voices_dir.is_dir():
            continue

        # <LANGUAGE>/<VOICE>-<TTS_SYSTEM>
        for lang_dir in voices_dir.iterdir():
            if (not lang_dir.is_dir()) or (lang_dir.name in VOCODER_DIR_NAMES):
                continue

            # Voice
            voice_lang = lang_dir.name
            for voice_model_dir in lang_dir.iterdir():
                if not valid_voice_dir(voice_model_dir):
                    continue

                full_voice_name = voice_model_dir.name
                voice_name, tts_system = full_voice_name.split("-", maxsplit=1)
                voice_id = f"{voice_lang}/{full_voice_name}"

                voices[voice_id] = {
                    "id": voice_id,
                    "name": voice_name,
                    "language": voice_lang,
                    "tts_system": tts_system,
                    "downloaded": True,
                }

    # Add voices that haven't been downloaded
    for download_name in VOICE_DOWNLOAD_NAMES.values():
        voice_lang, full_voice_name = download_name.split("_", maxsplit=1)
        voice_name, tts_system = full_voice_name.split("-", maxsplit=1)
        voice_id = f"{voice_lang}/{full_voice_name}"

        if voice_id in voices:
            # Already downloaded
            continue

        sample_sentence = _SAMPLE_SENTENCES.get(voice_lang)
        if sample_sentence:
            sample_url = f"https://raw.githubusercontent.com/rhasspy/larynx/master/local/{voice_lang}/{full_voice_name}/samples/{sample_sentence}.wav"
        else:
            # No sample
            sample_url = ""

        voices[voice_id] = {
            "id": voice_id,
            "name": voice_name,
            "language": voice_lang,
            "tts_system": tts_system,
            "downloaded": False,
            "sample_url": sample_url,
        }

    return voices


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

    for voices_dir in voices_dirs:
        if not voices_dir.is_dir():
            continue

        # <VOCODER_SYSTEM>/<VOCODER_MODEL>
        for vocoder_dir in voices_dir.iterdir():
            if (not vocoder_dir.is_dir()) or (
                vocoder_dir.name not in VOCODER_DIR_NAMES
            ):
                continue

            vocoder_system = vocoder_dir.name

            for model_dir in vocoder_dir.iterdir():
                if not valid_voice_dir(model_dir):
                    continue

                model_name = model_dir.name
                vocoder_id = f"{vocoder_system}/{model_name}"

                vocoders.append(
                    {
                        "id": vocoder_id,
                        "name": model_name,
                        "vocoder_system": vocoder_system,
                    }
                )

    return jsonify(vocoders)


@app.route("/api/tts", methods=["GET", "POST"])
async def app_say() -> Response:
    """Speak text to WAV."""
    voice = request.args.get("voice", "")
    assert voice, "No voice provided"

    # If true, [[ phonemes ]] in brackets are spoken literally.
    # Additionally, {{ par{tial} word{s} }} can be used.
    inline_pronunciations = request.args.get(
        "inlinePronunciations", ""
    ).strip().lower() in {"true", "1"}

    # TTS settings
    noise_scale = request.args.get("noiseScale", args.noise_scale)
    if noise_scale is not None:
        noise_scale = float(noise_scale)

    length_scale = request.args.get("lengthScale", args.length_scale)
    if length_scale is not None:
        length_scale = float(length_scale)

    text_lang = request.args.get("textLanguage", "").strip()
    if text_lang:
        text_lang = gruut.resolve_lang(text_lang)

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
        inline_pronunciations=inline_pronunciations,
        text_lang=text_lang,
    )

    return Response(wav_bytes, mimetype="audio/wav")


@app.route("/api/phonemes", methods=["GET"])
async def api_phonemes():
    """Get phonemes for language"""
    language = request.args.get("language", "en-us")

    phonemes: typing.Dict[str, typing.Dict[str, typing.Any]] = {}
    lang_phonemes = gruut_ipa.Phonemes.from_language(language)
    assert lang_phonemes, f"Unsupported language: {lang_phonemes}"

    for phoneme in lang_phonemes:
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
                # ə
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
    phonemizer = get_phonemizer(language)

    pron_phonemes: typing.List[typing.List[str]] = []

    _LOGGER.debug("Looking up in %s lexicon: %s", language, word)
    prons = next(phonemizer.phonemize([word]))
    pron_phonemes = [prons]

    return jsonify(pron_phonemes)


@app.route("/api/download", methods=["GET"])
async def api_download():
    """Download voice"""
    voice_id = request.args.get("id", "")
    if "/" in voice_id:
        voice_name = voice_id.split("/", maxsplit=1)[1]
    else:
        voice_name = voice_id

    download_name = VOICE_DOWNLOAD_NAMES.get(voice_name)
    assert download_name, f"No download known for {voice_name}"

    url = args.url_format.format(voice=download_name)
    tts_model_dir = download_voice(voice_name, voices_dirs[0], url)

    return jsonify({"id": voice_id, "url": url, "dir": str(tts_model_dir)})


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


def get_phonemizer(language: str) -> gruut.Phonemizer:
    phonemizer = _GRUUT_PHONEMIZER.get(language)
    if phonemizer is None:
        # Custom lexicon
        lexicon: typing.Optional[
            typing.MutableMapping[str, typing.List[gruut.WordPronunciation]]
        ] = None

        lexicon_path: typing.Optional[str] = None
        inline_prons: typing.List[typing.Tuple[str, str]] = []

        if args.lexicon:
            # Check for a custom lexicon for this language
            for lexicon_lang, lexicon_lang_path in args.lexicon:
                if lexicon_lang == language:
                    lexicon_path = lexicon_lang_path
                    break

        if lexicon_path is not None:
            _LOGGER.debug("Loading lexicon from %s", lexicon_path)
            lexicon = defaultdict(list)

            with open(lexicon_path, "r") as lexicon_file:
                for line in lexicon_file:
                    line = line.strip()
                    if not line:
                        continue

                    word, phonemes = line.split(maxsplit=1)
                    if phonemes.startswith("{{") and phonemes.endswith("}}"):
                        # {{ inline pronunciation }}
                        inline_prons.append((word, phonemes))
                    else:
                        # Phonemes
                        lexicon[word].append(
                            gruut.WordPronunciation(phonemes=phonemes.split())
                        )

        phonemizer = gruut.get_phonemizer(
            language,
            word_break=gruut_ipa.IPA.BREAK_WORD.value,
            inline_pronunciations=True,
            lexicon=lexicon,
        )

        assert phonemizer is not None, f"Unsupported language: {language}"

        if (lexicon is not None) and inline_prons:
            # Expand {{ inline pronunciations }}
            for word, inline_pron in inline_prons:
                encoded_pron = gruut.encode_inline_pronunciations(inline_pron, None)
                word_pron = phonemizer.get_pronunciation(
                    encoded_pron, inline_pronunciations=True
                )
                if word_pron is not None:
                    lexicon[word].append(word_pron)

        _GRUUT_PHONEMIZER[language] = phonemizer

    return phonemizer


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
    if args.pidfile:
        # Create directory to pidfile
        Path(args.pidfile).parent.mkdir(parents=True, exist_ok=True)
        ctx_pid = pidfile.PIDFile(args.pidfile)
    else:
        # No pidfile
        ctx_pid = contextlib.nullcontext()

    with ctx_pid:
        if args.pidfile:
            _LOGGER.debug("pidfile: %s", args.pidfile)

        # Need to type cast to satisfy mypy
        shutdown_trigger = typing.cast(
            typing.Callable[..., typing.Awaitable[None]], shutdown_event.wait
        )

        _LOOP.run_until_complete(
            hypercorn.asyncio.serve(app, hyp_config, shutdown_trigger=shutdown_trigger)
        )
except KeyboardInterrupt:
    _LOOP.call_soon(shutdown_event.set)
except pidfile.AlreadyRunningError:
    _LOGGER.info("Daemon already running (pidfile=%s). Exiting now.", args.pidfile)
