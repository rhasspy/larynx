#!/usr/bin/env python3
import argparse
import io
import logging
import os
import subprocess
import sys
from pathlib import Path

import gruut

from . import load_tts_model, load_vocoder_model, text_to_speech
from .constants import VocoderType
from .wavfile import write as wav_write

_LOGGER = logging.getLogger("larynx_runtime")

# -----------------------------------------------------------------------------


def main():
    """Main entry point"""
    args = get_args()

    gruut_lang = gruut.Language.load(args.language)
    assert gruut_lang, f"Unsupported language: {args.language}"

    # Load TTS
    _LOGGER.debug(
        "Loading text to speech model (%s, %s)", args.tts_model_type, args.tts_model
    )

    tts_model = load_tts_model(
        model_type=args.tts_model_type,
        model_path=args.tts_model,
        no_optimizations=args.no_optimizations,
    )

    # Load vocoder
    _LOGGER.debug(
        "Loading vocoder model (%s, %s)", args.vocoder_model_type, args.vocoder_model
    )

    vocoder_model = load_vocoder_model(
        model_type=args.vocoder_model_type,
        model_path=args.vocoder_model,
        no_optimizations=args.no_optimizations,
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

    for line in texts:
        line = line.strip()
        if not line:
            continue

        audio = text_to_speech(
            text=line,
            gruut_lang=gruut_lang,
            tts_model=tts_model,
            vocoder_model=vocoder_model,
            number_converters=args.number_converters,
            disable_currency=args.disable_currency,
            word_indexes=args.word_indexes,
        )

        with io.BytesIO() as wav_io:
            wav_write(wav_io, 22050, audio)
            wav_data = wav_io.getvalue()

        subprocess.run(
            ["play", "-t", "wav", "-"],
            input=wav_data,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )


# -----------------------------------------------------------------------------


def get_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(prog="larynx-runtime")
    parser.add_argument("language", help="Gruut language for text input")
    parser.add_argument(
        "text", nargs="*", help="Text to convert to speech (default: stdin)"
    )

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

    # Vocoder models
    parser.add_argument("--hifi-gan", help="Path to HiFi-GAN onnx generator model")

    parser.add_argument("--csv", action="store_true", help="Input format is id|text")
    parser.add_argument(
        "--no-optimizations", action="store_true", help="Disable Onnx optimizations"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Ensure TTS model
    tts_model_args = ["tacotron2"]
    setattr(args, "tts_model_type", None)
    setattr(args, "tts_model", None)

    for tts_model_arg in tts_model_args:
        tts_model_value = getattr(args, tts_model_arg)
        if tts_model_value:
            if args.tts_model is not None:
                raise ValueError("Only one TTS model can be specified")

            args.tts_model_type = tts_model_arg
            args.tts_model = tts_model_value

    if args.tts_model is None:
        raise ValueError("A TTS model is required")

    # Check for vocoder model
    vocoder_model_args = ["hifi_gan"]
    setattr(args, "vocoder_model_type", None)
    setattr(args, "vocoder_model", None)

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

    _LOGGER.debug(args)

    return args


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
