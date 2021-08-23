#!/usr/bin/env python3
import argparse
import io
import json
import logging
import os
import platform
import shlex
import string
import subprocess
import sys
import threading
import time
import typing
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from queue import Queue

from .constants import TextToSpeechType, VocoderType
from .utils import (
    DEFAULT_VOICE_URL_FORMAT,
    VOCODER_DIR_NAMES,
    download_voice,
    get_runtime_dir,
    get_voice_download_name,
    get_voices_dirs,
    resolve_voice_name,
    valid_voice_dir,
)

_DIR = Path(__file__).parent

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

    import numpy as np

    from .audio import AudioSettings

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

    # Phoneme transformation
    phoneme_lang: typing.Optional[str] = None
    phoneme_transform: typing.Optional[typing.Callable[[str], str]] = None
    if args.phoneme_language:
        from gruut.lang import resolve_lang

        phoneme_lang = resolve_lang(args.phoneme_language)
        phoneme_map: typing.Dict[str, typing.Union[str, typing.List[str]]] = {}

        if len(args.phoneme_map) > 1:
            phoneme_map_path = args.phoneme_map[1]
            with open(phoneme_map_path, "r") as phoneme_map_file:
                phoneme_map = json.load(phoneme_map_file)
        else:
            # Guess phoneme map
            from gruut_ipa import Phonemes
            from gruut_ipa.accent import guess_phonemes

            from_phonemes, to_phonemes = (
                Phonemes.from_language(args.language),
                Phonemes.from_language(phoneme_lang),
            )

            phoneme_map = {
                from_p.text: [to_p.text for to_p in guess_phonemes(from_p, to_phonemes)]
                for from_p in from_phonemes
            }

            _LOGGER.debug(
                "Guessed phoneme map from %s to %s: %s",
                args.language,
                phoneme_lang,
                phoneme_map,
            )

        def phoneme_map_transform(p):
            return phoneme_map.get(p, p)

        phoneme_transform = phoneme_map_transform

    # -------------------------------------------------------------------------
    # Daemon
    # -------------------------------------------------------------------------

    if args.daemon:
        runtime_dir = get_runtime_dir()
        pidfile_path = runtime_dir / "daemon.pidfile"
        _LOGGER.debug("Trying to start daemon on port %s", args.daemon_port)

        daemon_cmd = [
            "python3",
            "-m",
            "larynx.server",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.daemon_port),
            "--pidfile",
            str(pidfile_path),
            "--logfile",
            str(runtime_dir / "daemon.log"),
        ]

        _LOGGER.debug(daemon_cmd)
        subprocess.Popen(
            daemon_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        _LOGGER.debug("Waiting for daemon to start...")
        while (not pidfile_path.is_file()) or (pidfile_path.stat().st_size == 0):
            time.sleep(0.1)

        daemon_pid = int(pidfile_path.read_text().strip())
        _LOGGER.info("Daemon running (pid=%s)", daemon_pid)

        if args.text:
            text = " ".join(args.text)
        else:
            text = sys.stdin.read().encode()

        vocoder = f"{args.vocoder_model_type}/{args.vocoder_model}"
        values = {"voice": args.voice, "text": text, "vocoder": vocoder}
        url = f"http://localhost:{args.daemon_port}/api/tts?" + urllib.parse.urlencode(
            values
        )

        _LOGGER.debug(url)

        start_time = time.perf_counter()
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            wav_data = response.read()
            end_time = time.perf_counter()
            _LOGGER.debug(
                "Got %s byte(s) of WAV data in %s second(s)",
                len(wav_data),
                end_time - start_time,
            )
            sys.stdout.buffer.write(wav_data)
            sys.stdout.flush()

        return

    # -------------------------------------------------------------------------
    # No Daemon
    # -------------------------------------------------------------------------

    # Read text from stdin or arguments
    if args.text:
        # Use arguments
        texts = args.text
    else:
        # Use stdin
        texts = sys.stdin

        if os.isatty(sys.stdin.fileno()):
            print("Reading text from stdin...", file=sys.stderr)

    if args.process_on_blank_line:

        # Combine text until a blank line is encountered.
        # Good for line-wrapped books where
        # sentences are broken
        # up across multiple
        # lines.
        def process_on_blank_line(lines):
            text = ""
            for line in lines:
                line = line.strip()
                if not line:
                    if text:
                        yield text

                    text = ""
                    continue

                text += " " + line

        texts = process_on_blank_line(texts)

    # -------------------------------------------------------------------------

    # Custom lexicon
    from gruut import WordPronunciation, get_phonemizer, encode_inline_pronunciations
    from gruut_ipa import IPA

    lexicon: typing.Optional[
        typing.MutableMapping[str, typing.List[WordPronunciation]]
    ] = None

    inline_prons: typing.List[typing.Tuple[str, str]] = []

    if args.lexicon:
        _LOGGER.debug("Loading lexicon from %s", args.lexicon)
        lexicon = defaultdict(list)

        with open(args.lexicon, "r") as lexicon_file:
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
                    lexicon[word].append(WordPronunciation(phonemes=phonemes.split()))

    phonemizer = get_phonemizer(
        args.language,
        word_break=IPA.BREAK_WORD.value,
        inline_pronunciations=args.inline,
        use_word_indexes=args.word_indexes,
        lexicon=lexicon,
    )

    assert phonemizer is not None, f"Unsupported language: {args.language}"

    if (lexicon is not None) and inline_prons:
        # Expand {{ inline pronunciations }}
        for word, inline_pron in inline_prons:
            encoded_pron = encode_inline_pronunciations(inline_pron, None)
            word_pron = phonemizer.get_pronunciation(
                encoded_pron, inline_pronunciations=True
            )
            if word_pron is not None:
                lexicon[word].append(word_pron)

    # -------------------------------------------------------------------------

    from . import load_tts_model, load_vocoder_model, text_to_speech
    from .wavfile import write as wav_write

    max_thread_workers: typing.Optional[int] = None

    if args.max_thread_workers is not None:
        max_thread_workers = (
            None if args.max_thread_workers < 1 else args.max_thread_workers
        )
    elif args.raw_stream:
        # Faster time to first audio
        max_thread_workers = 2

    executor = ThreadPoolExecutor(max_workers=max_thread_workers)

    # Load TTS/vocoder models
    tts_settings: typing.Optional[typing.Dict[str, typing.Any]] = None
    if args.tts_model_type == TextToSpeechType.GLOW_TTS:
        tts_settings = {
            "noise_scale": args.noise_scale,
            "length_scale": args.length_scale,
        }

    def async_load_tts():
        # Load TTS
        start_load_time = time.perf_counter()
        _LOGGER.debug(
            "Loading text to speech model (%s, %s)", args.tts_model_type, args.tts_model
        )

        tts_model = load_tts_model(
            model_type=args.tts_model_type,
            model_path=args.tts_model,
            no_optimizations=args.no_optimizations,
        )

        end_load_time = time.perf_counter()
        _LOGGER.debug(
            "Loaded text to speech model in %s second(s)",
            end_load_time - start_load_time,
        )

        return tts_model

    def async_load_vocoder():
        # Load vocoder
        start_load_time = time.perf_counter()
        _LOGGER.debug(
            "Loading vocoder model (%s, %s)",
            args.vocoder_model_type,
            args.vocoder_model,
        )

        vocoder_model = load_vocoder_model(
            model_type=args.vocoder_model_type,
            model_path=args.vocoder_model,
            executor=executor,
            no_optimizations=args.no_optimizations,
            denoiser_strength=args.denoiser_strength,
        )

        end_load_time = time.perf_counter()
        _LOGGER.debug("Loaded vocoder in %s second(s)", end_load_time - start_load_time)

        return vocoder_model

    # Load in parallel
    tts_load_future = executor.submit(async_load_tts)
    vocoder_load_future = executor.submit(async_load_vocoder)

    if os.isatty(sys.stdout.fileno()):
        if (not args.output_dir) and (not args.raw_stream):
            # No where else for the audio to go
            args.interactive = True

    # Raw stream queue
    raw_queue: typing.Optional["Queue[bytes]"] = None
    if args.raw_stream:
        # Output in a separate thread to avoid blocking audio processing
        raw_queue = Queue(maxsize=args.raw_stream_queue_size)

        def output_raw_stream():
            while True:
                audio = raw_queue.get()
                if audio is None:
                    break

                _LOGGER.debug(
                    "Writing %s byte(s) of 16-bit 22050Hz mono PCM to stdout",
                    len(audio),
                )
                sys.stdout.buffer.write(audio)
                sys.stdout.buffer.flush()

        threading.Thread(target=output_raw_stream, daemon=True).start()

    all_audios: typing.List[np.ndarray] = []
    wav_data: typing.Optional[bytes] = None
    play_command = shlex.split(args.play_command)

    # -------------------
    # Process input lines
    # -------------------
    start_time_to_first_audio = time.perf_counter()

    try:
        for line in texts:
            line_id = ""
            line = line.strip()
            if not line:
                continue

            if args.output_naming == OutputNaming.ID:
                # Line has the format id|text instead of just text
                line_id, line = line.split(args.id_delimiter, maxsplit=1)

            text_and_audios = text_to_speech(
                text=line,
                lang=args.language,
                tts_model=tts_load_future,
                vocoder_model=vocoder_load_future,
                audio_settings=audio_settings,
                number_converters=args.number_converters,
                disable_currency=args.disable_currency,
                word_indexes=args.word_indexes,
                inline_pronunciations=args.inline,
                phoneme_transform=phoneme_transform,
                phoneme_lang=phoneme_lang,
                tts_settings=tts_settings,
                max_workers=max_thread_workers,
                executor=executor,
                phonemizer=phonemizer,
            )

            text_id = ""

            for text_idx, (text, audio) in enumerate(text_and_audios):
                if text_idx == 0:
                    end_time_to_first_audio = time.perf_counter()
                    _LOGGER.debug(
                        "Seconds to first audio: %s",
                        end_time_to_first_audio - start_time_to_first_audio,
                    )

                if args.raw_stream:
                    assert raw_queue is not None
                    raw_queue.put(audio.tobytes())
                elif args.interactive or args.output_dir:
                    # Convert to WAV audio
                    with io.BytesIO() as wav_io:
                        wav_write(wav_io, args.sample_rate, audio)
                        wav_data = wav_io.getvalue()

                    assert wav_data is not None

                    if args.interactive:

                        # Play audio
                        _LOGGER.debug("Playing audio with play command")
                        subprocess.run(
                            play_command,
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
                                str.maketrans(
                                    "", "", string.punctuation.replace("_", "")
                                )
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
    except KeyboardInterrupt:
        if raw_queue is not None:
            # Draw audio playback queue
            while not raw_queue.empty():
                raw_queue.get()

            raw_queue.put(None)

    # -------------------------------------------------------------------------

    # Write combined audio to stdout
    if all_audios:
        with io.BytesIO() as wav_io:
            wav_write(wav_io, args.sample_rate, np.concatenate(all_audios))
            wav_data = wav_io.getvalue()

        _LOGGER.debug("Writing WAV audio to stdout")
        sys.stdout.buffer.write(wav_data)
        sys.stdout.buffer.flush()


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
        help="Play audio after each input line (see --play-command)",
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
    parser.add_argument(
        "--inline",
        action="store_true",
        help="Enable inline phonemes and word pronunciations",
    )
    parser.add_argument(
        "--lexicon", help="Path to custom lexicon with format WORD PHONEME PHONEME ..."
    )

    # Phonemes
    parser.add_argument("--phoneme-language", help="Target language of voice phonemes")
    parser.add_argument(
        "--phoneme-map",
        help="Path to JSON file with mapping from text phonemes to voice phonemes",
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
        default=0.005,
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
        help="Maximum number of threads to concurrently load models and run sentences through TTS/Vocoder",
    )

    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Don't automatically download voices or vocoders",
    )
    parser.add_argument(
        "--url-format",
        default=DEFAULT_VOICE_URL_FORMAT,
        help="Format string for download URLs (accepts {voice})",
    )
    parser.add_argument(
        "--play-command",
        default="play -",
        help="Shell command used to play audio in interactive model (default: play -)",
    )
    parser.add_argument(
        "--raw-stream",
        action="store_true",
        help="Stream raw 16-bit 22050Hz mono PCM audio to stdout",
    )
    parser.add_argument(
        "--raw-stream-queue-size",
        default=5,
        help="Maximum number of sentences to maintain in output queue with --raw-stream (default: 5)",
    )
    parser.add_argument(
        "--process-on-blank-line",
        action="store_true",
        help="Process text only after encountering a blank line",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Connect to or run a background HTTP server for TTS functionality",
    )
    parser.add_argument(
        "--daemon-port",
        type=int,
        default=15002,
        help="Port to run daemon HTTP server on (default: 15002)",
    )
    parser.add_argument(
        "--stop-daemon",
        action="store_true",
        help="Try to stop the currently running Larynx daemon and exit",
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

    if args.stop_daemon:
        # Try to stop daemon and exit
        import psutil

        runtime_dir = get_runtime_dir()
        pidfile = runtime_dir / "daemon.pidfile"
        if pidfile.is_file():
            # Get pid from file and block until terminated
            daemon_pid = int(pidfile.read_text().strip())
            proc = psutil.Process(daemon_pid)
            if proc.is_running():
                _LOGGER.debug("Trying to stop daemon (pid=%s)", daemon_pid)
                proc.terminate()
                psutil.wait_procs([proc])

            _LOGGER.info("Stopped daemon")
        else:
            _LOGGER.info("No daemon running")

        sys.exit(0)

    # -------------------------------------------------------------------------

    # Directories to search for voices
    voices_dirs = get_voices_dirs(args.voices_dir)

    def list_voices_vocoders():
        """Print all vocoders and voices"""
        # (type, name) -> location
        local_info = {}

        # Search for downloaded voices/vocoders
        for voices_dir in voices_dirs:
            if not voices_dir.is_dir():
                continue

            for voice_dir in voices_dir.iterdir():
                if not voice_dir.is_dir():
                    continue

                if voice_dir.name in VOCODER_DIR_NAMES:
                    # Vocoder
                    for vocoder_model_dir in voice_dir.iterdir():
                        if not vocoder_model_dir.is_dir():
                            continue

                        if vocoder_model_dir.glob("*.onnx"):
                            full_vocoder_name = (
                                f"{voice_dir.name}-{vocoder_model_dir.name}"
                            )
                            local_info[("vocoder", full_vocoder_name)] = str(
                                vocoder_model_dir
                            )
                else:
                    # Voice
                    voice_lang = voice_dir.name
                    for voice_model_dir in voice_dir.iterdir():
                        if not voice_model_dir.is_dir():
                            continue

                        if voice_model_dir.glob("*.onnx"):
                            local_info[("voice", voice_model_dir.name)] = str(
                                voice_model_dir
                            )

        # (type, lang, name, downloaded, aliases, location)
        voices_and_vocoders = []
        with open(_DIR / "VOCODERS", "r") as vocoders_file:
            for line in vocoders_file:
                line = line.strip()
                if not line:
                    continue

                *vocoder_aliases, full_vocoder_name = line.split()
                downloaded = False

                location = local_info.get(("vocoder", full_vocoder_name), "")
                if location:
                    downloaded = True

                voices_and_vocoders.append(
                    (
                        "vocoder",
                        " ",
                        "*" if downloaded else " ",
                        full_vocoder_name,
                        ",".join(vocoder_aliases),
                        location,
                    )
                )

        with open(_DIR / "VOICES", "r") as voices_file:
            for line in voices_file:
                line = line.strip()
                if not line:
                    continue

                *voice_aliases, full_voice_name, download_name = line.split()
                voice_lang = download_name.split("_", maxsplit=1)[0]

                downloaded = False

                location = local_info.get(("voice", full_voice_name), "")
                if location:
                    downloaded = True

                voices_and_vocoders.append(
                    (
                        "voice",
                        voice_lang,
                        "*" if downloaded else " ",
                        full_voice_name,
                        ",".join(voice_aliases),
                        location,
                    )
                )

        headers = ("TYPE", "LANG", "LOCAL", "NAME", "ALIASES", "LOCATION")

        # Get widths of columns
        col_widths = [0] * len(voices_and_vocoders[0])
        for item in voices_and_vocoders:
            for col in range(len(col_widths)):
                col_widths[col] = max(
                    col_widths[col], len(item[col]) + 1, len(headers[col]) + 1
                )

        # Print results
        print(*(h.ljust(col_widths[col]) for col, h in enumerate(headers)))

        for item in sorted(voices_and_vocoders):
            print(*(v.ljust(col_widths[col]) for col, v in enumerate(item)))

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
            from gruut.lang import resolve_lang

            args.language = resolve_lang(args.language)

            # Use directory under language first
            for voices_dir in voices_dirs:
                maybe_tts_model_dir = voices_dir / args.language / args.voice
                if valid_voice_dir(maybe_tts_model_dir):
                    tts_model_dir = maybe_tts_model_dir
                    break

        if tts_model_dir is None:
            # Search for voice in all directories
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
        # This is not an error. Benchmarking reveals that the misnamed "medium"
        # model is faster than "small".
        vocoder_model_name = "vctk_small"
    elif args.quality == "low":
        # This is not an error. Benchmarking reveals that the misnamed "medium"
        # model is faster than "small".
        vocoder_model_name = "vctk_medium"

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
            # Overwrite if already set
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
