#!/usr/bin/env python3
import argparse
import io
import json
import logging
import os
import platform
import string
import subprocess
import sys
import time
import typing
from enum import Enum
from pathlib import Path

import numpy as np

from . import load_tts_model, load_vocoder_model, text_to_speech
from .audio import AudioSettings
from .constants import TextToSpeechType, VocoderType
from .utils import (
    download_voice,
    get_voice_download_name,
    get_voices_dirs,
    resolve_voice_name,
    valid_voice_dir,
)
from .wavfile import write as wav_write

_DIR = Path(__file__).parent
_DEFAULT_URL_FORMAT = (
    "http://github.com/rhasspy/larynx/releases/download/2021-03-28/{voice}.tar.gz"
)

_LOGGER = logging.getLogger("larynx")

# -----------------------------------------------------------------------------


class OutputNaming(str, Enum):
    """Format used for output file names"""

    TEXT = "text"
    TIME = "time"
    ID = "id"


# -----------------------------------------------------------------------------


def main():
    """Main entry point"""
    args = get_args()

    # Load audio settings
    maybe_config_path: typing.Optional[Path] = None
    if args.config:
        maybe_config_path = args.config
    elif not args.no_autoload_config:
        maybe_config_path = args.tts_model / "config.json"
        if not maybe_config_path.is_file():
            maybe_config_path = None

    if maybe_config_path is not None:
        _LOGGER.debug("Loading audio settings from %s", maybe_config_path)
        with open(maybe_config_path, "r") as config_file:
            config = json.load(config_file)
            audio_settings = AudioSettings(**config["audio"])
    else:
        # Default audio settings
        audio_settings = AudioSettings()

    _LOGGER.debug(audio_settings)

    # Create output directory
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        _LOGGER.debug("Setting random seed to %s", args.seed)
        np.random.seed(args.seed)

    if args.csv:
        args.output_naming = "id"

    # Load TTS
    _LOGGER.debug(
        "Loading text to speech model (%s, %s)", args.tts_model_type, args.tts_model
    )

    tts_model = load_tts_model(
        model_type=args.tts_model_type,
        model_path=args.tts_model,
        no_optimizations=args.no_optimizations,
    )

    tts_settings: typing.Optional[typing.Dict[str, typing.Any]] = None
    if args.tts_model_type == TextToSpeechType.GLOW_TTS:
        tts_settings = {
            "noise_scale": args.noise_scale,
            "length_scale": args.length_scale,
        }

    # Load vocoder
    _LOGGER.debug(
        "Loading vocoder model (%s, %s)", args.vocoder_model_type, args.vocoder_model
    )

    vocoder_model = load_vocoder_model(
        model_type=args.vocoder_model_type,
        model_path=args.vocoder_model,
        no_optimizations=args.no_optimizations,
        denoiser_strength=args.denoiser_strength,
    )

    # Read text from stdin or arguments
    if args.text:
        # Use arguments
        texts = args.text
    else:
        # Use stdin
        texts = sys.stdin

        if os.isatty(sys.stdin.fileno()):
            print("Reading text from stdin...", file=sys.stderr)

            if not args.output_dir:
                # No where else for the audio to go
                args.interactive = True

    all_audios: typing.List[np.ndarray] = []
    wav_data: typing.Optional[bytes] = None

    for line in texts:
        line_id = ""
        line = line.strip()
        if not line:
            continue

        if args.output_naming == OutputNaming.ID:
            line_id, line = line.split(args.id_delimiter, maxsplit=1)

        text_and_audios = text_to_speech(
            text=line,
            lang=args.language,
            tts_model=tts_model,
            vocoder_model=vocoder_model,
            audio_settings=audio_settings,
            number_converters=args.number_converters,
            disable_currency=args.disable_currency,
            word_indexes=args.word_indexes,
            tts_settings=tts_settings,
            max_workers=(
                None if args.max_thread_workers <= 0 else args.max_thread_workers
            ),
        )

        text_id = ""

        for text_idx, (text, audio) in enumerate(text_and_audios):
            if args.interactive or args.output_dir:
                # Convert to WAV audio
                with io.BytesIO() as wav_io:
                    wav_write(wav_io, args.sample_rate, audio)
                    wav_data = wav_io.getvalue()

                assert wav_data is not None

                if args.interactive:

                    # Play audio
                    _LOGGER.debug("Playing audio with play command")
                    subprocess.run(
                        ["play", "-"],
                        input=wav_data,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=True,
                    )

                if args.output_dir:
                    # Determine file name
                    if args.output_naming == OutputNaming.TEXT:
                        # Use text itself
                        file_name = text.replace(" ", "_")
                        file_name = file_name.translate(
                            str.maketrans("", "", string.punctuation.replace("_", ""))
                        )
                    elif args.output_naming == OutputNaming.TIME:
                        # Use timestamp
                        file_name = str(time.time())
                    elif args.output_naming == OutputNaming.ID:
                        if not text_id:
                            text_id = line_id
                        else:
                            text_id = f"{line_id}_{text_idx + 1}"

                        file_name = text_id

                    assert file_name, f"No file name for text: {text}"
                    wav_path = args.output_dir / (file_name + ".wav")
                    with open(wav_path, "wb") as wav_file:
                        wav_write(wav_file, args.sample_rate, audio)

                    _LOGGER.debug("Wrote %s", wav_path)
            else:
                # Combine all audio and output to stdout at the end
                all_audios.append(audio)

    # -------------------------------------------------------------------------

    # Write combined audio to stdout
    if all_audios:
        _LOGGER.debug("Writing WAV audio to stdout")
        with io.BytesIO() as wav_io:
            wav_write(wav_io, args.sample_rate, np.concatenate(all_audios))
            wav_data = wav_io.getvalue()

        sys.stdout.buffer.write(wav_data)


# -----------------------------------------------------------------------------


def get_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(prog="larynx")
    parser.add_argument(
        "--language", help="Gruut language for text input (en-us, etc.)"
    )
    parser.add_argument(
        "text", nargs="*", help="Text to convert to speech (default: stdin)"
    )
    parser.add_argument(
        "--voice", "-v", help="Name of voice (expected in <voices-dir>/<language>)"
    )
    parser.add_argument(
        "--voices-dir",
        help="Directory with voices (format is <language>/<name_model-type>)",
    )
    parser.add_argument(
        "--quality",
        "-q",
        choices=["high", "medium", "low"],
        default="high",
        help="Vocoder quality (default: high)",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available voices/vocoders"
    )
    parser.add_argument(
        "--config", help="Path to JSON configuration file with audio settings"
    )
    parser.add_argument("--output-dir", help="Directory to write WAV file(s)")
    parser.add_argument(
        "--output-naming",
        choices=[v.value for v in OutputNaming],
        default="text",
        help="Naming scheme for output WAV files (requires --output-dir)",
    )
    parser.add_argument(
        "--id-delimiter",
        default="|",
        help="Delimiter between id and text in lines (default: |). Requires --output-naming id",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Play audio after each input line (requires 'play')",
    )
    parser.add_argument("--csv", action="store_true", help="Input format is id|text")
    parser.add_argument("--sample-rate", type=int, default=22050)

    # Gruut
    parser.add_argument(
        "--word-indexes",
        action="store_true",
        help="Allow word_n form for specifying nth pronunciation of word from lexicon",
    )
    parser.add_argument(
        "--disable-currency",
        action="store_true",
        help="Disable automatic replacement of currency with words (e.g., $1 -> one dollar)",
    )
    parser.add_argument(
        "--number-converters",
        action="store_true",
        help="Allow number_conv form for specifying num2words converter (cardinal, ordinal, ordinal_num, year, currency)",
    )

    # TTS models
    parser.add_argument(
        "--tacotron2",
        help="Path to directory with encoder/decoder/postnet onnx Tacotron2 models",
    )
    parser.add_argument("--glow-tts", help="Path to onnx Glow TTS model")

    # GlowTTS setttings
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.333,
        help="Noise scale (default: 0.333, GlowTTS only)",
    )
    parser.add_argument(
        "--length-scale",
        type=float,
        default=1.0,
        help="Length scale (default: 1.0, GlowTTS only)",
    )

    # Vocoder models
    parser.add_argument("--hifi-gan", help="Path to HiFi-GAN onnx generator model")
    parser.add_argument("--waveglow", help="Path to WaveGlow onnx model")

    parser.add_argument(
        "--optimizations",
        choices=["auto", "on", "off"],
        default="auto",
        help="Enable/disable Onnx optimizations (auto=disable on armv7l)",
    )
    parser.add_argument(
        "--denoiser-strength",
        type=float,
        default=0.001,
        help="Strength of denoiser, if available (default: 0 = disabled)",
    )

    # Miscellaneous
    parser.add_argument(
        "--no-autoload-config",
        action="store_true",
        help="Don't automatically load config.json in model directory",
    )
    parser.add_argument(
        "--max-thread-workers",
        type=int,
        default=2,
        help="Maximum number of threads to concurrently run sentences through TTS/Vocoder",
    )

    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Don't automatically download voices or vocoders",
    )
    parser.add_argument(
        "--url-format",
        default=_DEFAULT_URL_FORMAT,
        help="Format string for download URLs (accepts {voice})",
    )

    parser.add_argument("--seed", type=int, help="Set random seed (default: not set)")
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # -------------------------------------------------------------------------

    if args.version:
        # Print version and exit
        from . import __version__

        print(__version__)
        sys.exit(0)

    # -------------------------------------------------------------------------

    # Directories to search for voices
    voices_dirs = get_voices_dirs(args.voices_dir)

    def list_voices_vocoders():
        """Print all vocoders and voices"""
        vocoder_model_types = set(
            v.value for v in VocoderType if v != VocoderType.GRIFFIN_LIM
        )

        # Print vocoders
        for voices_dir in voices_dirs:
            if not voices_dir.is_dir():
                continue

            for vocoder_dir in sorted(voices_dir.iterdir()):
                if not vocoder_dir.is_dir():
                    continue

                if vocoder_dir.name in sorted(vocoder_model_types):
                    for model_dir in sorted(vocoder_dir.iterdir()):
                        if not model_dir.is_dir():
                            continue

                        print(
                            "vocoder",
                            vocoder_dir.name,
                            model_dir.name,
                            model_dir,
                            sep="\t",
                        )

            # Print voices
            for lang_dir in sorted(voices_dir.iterdir()):
                if not lang_dir.is_dir():
                    continue

                if lang_dir.name not in vocoder_model_types:
                    for voice_dir in lang_dir.iterdir():
                        if not voice_dir.is_dir():
                            continue

                        print(
                            "voice", lang_dir.name, voice_dir.name, voice_dir, sep="\t"
                        )

    if args.list:
        list_voices_vocoders()
        sys.exit(0)

    # -------------------------------------------------------------------------

    # Set defaults
    setattr(args, "tts_model_type", None)
    setattr(args, "tts_model", None)
    setattr(args, "vocoder_model_type", None)
    setattr(args, "vocoder_model", None)

    if args.voice:
        # Resolve aliases
        args.voice = resolve_voice_name(args.voice)
        tts_model_dir: typing.Optional[Path] = None

        if args.language:
            # Use directory under language
            for voices_dir in voices_dirs:
                maybe_tts_model_dir = voices_dir / args.language / args.voice
                if valid_voice_dir(maybe_tts_model_dir):
                    tts_model_dir = maybe_tts_model_dir
                    break
        else:
            # Search for voice
            for voices_dir in voices_dirs:
                for model_dir in voices_dir.rglob(args.voice):
                    if valid_voice_dir(model_dir):
                        tts_model_dir = model_dir
                        break

        if (tts_model_dir is None) and (not args.no_download):
            url_voice = get_voice_download_name(args.voice)
            assert url_voice is not None, f"No download name for voice {args.voice}"

            url = args.url_format.format(voice=url_voice)
            tts_model_dir = download_voice(args.voice, voices_dirs[0], url)

        assert tts_model_dir is not None, f"Voice not found: {args.voice}"
        _LOGGER.debug("Using voice at %s", tts_model_dir)

        # Get TTS model name from end of voice name.
        # Example: harvard-glow_tts
        tts_model_type = args.voice[args.voice.rfind("-") + 1 :]
        setattr(args, "tts_model_type", tts_model_type)
        setattr(args, "tts_model", str(tts_model_dir))

    # Vocoder defaults
    vocoder_model_type = "hifi_gan"
    vocoder_model_name = "universal_large"
    setattr(args, "vocoder_model_type", vocoder_model_type)

    if args.quality == "medium":
        vocoder_model_name = "vctk_medium"
    elif args.quality == "low":
        vocoder_model_name = "vctk_small"

    vocoder_model_dir: typing.Optional[Path] = None
    for voices_dir in voices_dirs:
        maybe_vocoder_model_dir = voices_dir / vocoder_model_type / vocoder_model_name
        if valid_voice_dir(maybe_vocoder_model_dir):
            vocoder_model_dir = maybe_vocoder_model_dir
            break

    if (vocoder_model_dir is None) and (not args.no_download):
        # hifi_gan-universal_large
        vocoder_name = f"{vocoder_model_type}-{vocoder_model_name}"
        url = args.url_format.format(voice=vocoder_name)
        vocoder_model_dir = download_voice(vocoder_name, voices_dirs[0], url)

    if vocoder_model_dir is not None:
        setattr(args, "vocoder_model", str(vocoder_model_dir))
        _LOGGER.debug("Using vocoder at %s", vocoder_model_dir)

    # Ensure TTS model
    tts_model_args = [v.value for v in TextToSpeechType]

    for tts_model_arg in tts_model_args:
        tts_model_value = getattr(args, tts_model_arg)
        if tts_model_value:
            if args.tts_model is not None:
                raise ValueError("Only one TTS model can be specified")

            args.tts_model_type = tts_model_arg
            args.tts_model = tts_model_value

    if args.tts_model is None:
        list_voices_vocoders()
        _LOGGER.info("--voice required (see list above)")
        sys.exit(1)

    if not args.language:
        # Set language based on voice
        lang = Path(args.tts_model).parent.name
        setattr(args, "language", lang)
        _LOGGER.debug("Language: %s", lang)

    # Check for vocoder model
    vocoder_model_args = [v.value for v in VocoderType if v != VocoderType.GRIFFIN_LIM]

    for vocoder_model_arg in vocoder_model_args:
        vocoder_model_value = getattr(args, vocoder_model_arg)
        if vocoder_model_value:
            if args.vocoder_model is not None:
                raise ValueError("Only one vocoder model can be specified")

            args.vocoder_model_type = vocoder_model_arg
            args.vocoder_model = vocoder_model_value

    # Convert to paths
    args.tts_model = Path(args.tts_model)

    if args.vocoder_model:
        args.vocoder_model = Path(args.vocoder_model)
    else:
        # Default to griffin-lim vocoder
        args.vocoder_model = Path.cwd()
        args.vocoder_model_type = VocoderType.GRIFFIN_LIM

    if args.output_dir:
        args.output_dir = Path(args.output_dir)

    if args.config:
        args.config = Path(args.config)

    # Handle optimizations.
    # onnxruntime crashes on armv7l if optimizations are enabled.
    setattr(args, "no_optimizations", False)
    if args.optimizations == "off":
        args.no_optimizations = True
    elif args.optimizations == "auto":
        if platform.machine() == "armv7l":
            # Enabling optimizations on 32-bit ARM crashes
            args.no_optimizations = True

    _LOGGER.debug(args)

    return args


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
