"""Utility methods for Larynx"""
import logging
import shutil
import tempfile
import typing
from pathlib import Path

import gruut_ipa
import requests
from tqdm.auto import tqdm

_LOGGER = logging.getLogger("larynx.utils")

# Allow ' for primary stress and , for secondary stress
# Allow : for elongation
_IPA_TRANSLATE = str.maketrans(
    "',:",
    "".join(
        [
            gruut_ipa.IPA.STRESS_PRIMARY.value,
            gruut_ipa.IPA.STRESS_SECONDARY.value,
            gruut_ipa.IPA.LONG,
        ]
    ),
)

# -----------------------------------------------------------------------------


def download_voice(
    voice_name: str, voices_dir: typing.Union[str, Path], link: str
) -> Path:
    """Download and extract a voice (or vocoder)"""
    voices_dir = Path(voices_dir)
    voices_dir.mkdir(parents=True, exist_ok=True)

    _LOGGER.debug(
        "Downloading voice/vocoder for %s to %s from %s", voice_name, voices_dir, link
    )

    response = requests.get(link, stream=True)
    assert response.ok, f"Bad response for {link}"

    with tempfile.NamedTemporaryFile(mode="wb+", suffix=".tar.gz") as temp_file:
        with tqdm(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
            desc=voice_name,
            total=int(response.headers.get("content-length", 0)),
        ) as pbar:
            for chunk in response.iter_content(chunk_size=4096):
                temp_file.write(chunk)
                pbar.update(len(chunk))

        temp_file.seek(0)

        # Extract
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            _LOGGER.debug("Extracting %s to %s", temp_file.name, temp_dir_str)
            shutil.unpack_archive(temp_file.name, temp_dir_str)

            # Expecting <language>/<voice_name>
            lang_dir = next(temp_dir.iterdir())
            assert lang_dir.is_dir()

            voice_dir = next(lang_dir.iterdir())
            assert voice_dir.is_dir()

            # Copy to destination
            dest_lang_dir = voices_dir / lang_dir.name
            dest_lang_dir.mkdir(parents=True, exist_ok=True)

            dest_voice_dir = voices_dir / lang_dir.name / voice_dir.name
            if dest_voice_dir.is_dir():
                # Delete existing files
                shutil.rmtree(str(dest_voice_dir))

            # Move files
            _LOGGER.debug("Moving %s to %s", voice_dir, dest_voice_dir)
            shutil.move(str(voice_dir), str(dest_voice_dir))

            return dest_voice_dir


# -----------------------------------------------------------------------------


def valid_voice_dir(voice_dir: typing.Union[str, Path]) -> bool:
    """Return True if directory exists and has an onnx file"""
    voice_dir = Path(voice_dir)
    return voice_dir.is_dir() and (len(list(voice_dir.glob("*.onnx"))) > 0)
