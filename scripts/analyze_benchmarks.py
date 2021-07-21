#!/usr/bin/env python3
import re
import sys

pattern_phonemes = re.compile(r".+Phonemes for '[^']+': (.+)")
pattern_mels_sec = re.compile(r".+Got mels in ([0-9.]+) .+")
pattern_vocoder_sec = re.compile(r".+Got audio in ([0-9.]+) .+")
pattern_rtf = re.compile(r".+Real-time factor: ([0-9.]+) .+")
pattern_first_sec = re.compile(r".+Real-time factor: ([0-9.]+) .+")

num_phonemes = []
mels_sec = []
vocoder_sec = []
rtf = []

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue

    match = pattern_phonemes.match(line)
    if match:
        phonemes = eval(match.group(1))
        num_phonemes.append(len(phonemes))
        continue

    match = pattern_mels_sec.match(line)
    if match:
        mels_sec.append(float(match.group(1)))
        continue

    match = pattern_vocoder_sec.match(line)
    if match:
        vocoder_sec.append(float(match.group(1)))
        continue

    match = pattern_rtf.match(line)
    if match:
        rtf.append(float(match.group(1)))
        continue

# -----------------------------------------------------------------------------

if num_phonemes and mels_sec:
    phonemes_per_second = []
    for np, ms in zip(num_phonemes, mels_sec):
        phonemes_per_second.append(np / ms)

    print(
        "Avg. mels:",
        sum(phonemes_per_second) / len(phonemes_per_second),
        " phonemes/sec",
    )

if num_phonemes and vocoder_sec:
    phonemes_per_second = []
    for np, vs in zip(num_phonemes, vocoder_sec):
        phonemes_per_second.append(np / vs)

    print(
        "Avg. vocoder:",
        sum(phonemes_per_second) / len(phonemes_per_second),
        " phonemes/sec",
    )

if rtf:
    print("Avg. RTF:", sum(rtf) / len(rtf), "infer/audio")
