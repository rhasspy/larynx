#!/usr/bin/env python3
"""Tests for text_to_speech API"""
import csv
import functools
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from larynx import text_to_speech
from larynx.utils import get_voices_dirs
from larynx import wavfile

# -----------------------------------------------------------------------------

_LOGGER = logging.getLogger("larynx.test_text_to_speech")

SAMPLE_SENTENCES = {
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


def test_voices():
    for voices_dir in get_voices_dirs():
        if not voices_dir.is_dir():
            continue

        for lang_dir in voices_dir.iterdir():
            if not lang_dir.is_dir():
                continue

            lang = lang_dir.name
            if lang not in SAMPLE_SENTENCES:
                continue

            for voice_dir in lang_dir.iterdir():
                if not voice_dir.is_dir():
                    continue

                voice_name = voice_dir.name

                wav_name = SAMPLE_SENTENCES[lang]
                samples_dir = voice_dir / "samples"

                sample_wav = samples_dir / f"{wav_name}.wav"
                sample_rate, audio = wavfile.read(sample_wav)
                expected_sec = audio.shape[-1] / sample_rate
                expected_signal = sum(audio ** 2) / len(audio)

                text = ""
                with open(
                    samples_dir / "test_sentences.txt", "r", encoding="utf-8"
                ) as test_sentences:
                    reader = csv.reader(test_sentences, delimiter="|")
                    for row in reader:
                        if row[0] == wav_name:
                            text = row[1]
                            break

                assert text, f"No text for {voice_dir}"

                yield check_voice, voice_name, text, expected_sec, expected_signal, sample_rate


def check_voice(
    voice_name: str,
    text: str,
    expected_sec: float,
    expected_signal: int,
    sample_rate: int,
    threshold_signal: float = 25.0,
    tolerance_sec: float = 1.0,
):
    """Verify audio from text to speech matches characteristics of a sample"""
    _LOGGER.debug("%s: %s", voice_name, text)

    audios = []
    for result in text_to_speech(text=text, voice_or_lang=voice_name):
        audios.append(result.audio)

    all_audio = np.concatenate(audios)

    # Ensure audio is not silence
    actual_signal = sum(all_audio ** 2) / len(all_audio)
    assert (
        actual_signal > threshold_signal
    ), f"Expected signal > {threshold_signal}, got {actual_signal}"

    # Check audio duration is within tolerance
    actual_sec = all_audio.shape[-1] / sample_rate

    assert (
        (expected_sec - tolerance_sec) <= actual_sec <= (expected_sec + tolerance_sec)
    ), f"Expected duration of {expected_sec}, got {actual_sec} (tolerance={tolerance_sec})"
