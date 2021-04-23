#!/usr/bin/env python3
import sys
from collections import defaultdict
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: make_sample_html.py <LOCAL_DIR>", file=sys.stderr)
        sys.exit(1)

    print('<html lang="en">')
    print('<head><meta charset="utf-8"><title>Larynx Voice Samples</title></head>')
    print("<body>")
    print("<h1>Larynx Voice Samples</h1>")
    print('<p>Voices samples trained for <a href="https://github.com/rhasspy/larynx">Larynx.</a></p>')

    local_dir = Path(sys.argv[1])

    # language -> voice name -> samples dir
    voices = defaultdict(dict)

    # local/<LANGUAGE>/<VOICE>-<MODEL>
    for lang_dir in sorted(Path(local_dir).iterdir()):
        if not lang_dir.is_dir():
            continue

        language = lang_dir.name

        if language in ["hifi_gan", "waveglow"]:
            # Exclude vocoders
            continue

        for voice_dir in sorted(lang_dir.iterdir()):
            if not voice_dir.is_dir():
                continue

            samples_dir = voice_dir / "samples"
            if not samples_dir.is_dir():
                print("Missing", samples_dir, file=sys.stderr)
                continue

            test_sentences = samples_dir / "test_sentences.txt"
            if not test_sentences.is_file():
                print("Missing", test_sentences, file=sys.stderr)
                continue

            voices[language][voice_dir.name] = samples_dir

    # Print table of contents
    print("<ul>")
    for language, lang_voices in voices.items():
        print("<li>", f'<a href="#{language}">', language, "</a>")

        print("<ul>")
        for voice_name in lang_voices:
            print(
                "<li>",
                f'<a href="#{language}_{voice_name}">',
                voice_name,
                "</a>",
                "</li>",
            )
        print("</ul>")

        print("</li>")

    print("</ul>")
    print("<hr>")

    # Print samples
    for language, lang_voices in voices.items():
        print(f'<h2 id="{language}">', language, "</h2>")

        for voice_name, samples_dir in lang_voices.items():
            test_sentences = samples_dir / "test_sentences.txt"
            voice, model_type = voice_name.split("-", maxsplit=1)
            print(f'<h3 id="{language}_{voice_name}">', voice, f"({model_type})", "</h3>")

            with open(test_sentences, "r") as test_sentences_file:
                for line in sorted(test_sentences_file):
                    line = line.strip()
                    if not line:
                        continue

                    if "|" in line:
                        utt_id, text = line.split("|", maxsplit=1)
                    else:
                        utt_id, text = line, line

                    wav_path = samples_dir / f"{utt_id}.wav"

                    if not wav_path.is_file():
                        print("Missing", wav_path, file=sys.stderr)
                        continue

                    print("<p>", text, "</p>")
                    print(f'<audio preload="none" controls src="{wav_path}"></audio>')
                    print("<br>")

            print("<br>")

        # ---------------------------------------------------------------------

        print("<br>")
        print("<hr>")

    print("</body>")
    print("</html>")


if __name__ == "__main__":
    main()
