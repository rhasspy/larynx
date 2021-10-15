"""Utility methods for Larynx"""
import getpass
import logging
import os
import shutil
import tempfile
import typing
import urllib.request
from pathlib import Path
from urllib.error import HTTPError

from larynx.constants import VocoderType

_DIR = Path(__file__).parent
_LOGGER = logging.getLogger("larynx.utils")
_ENV_VOICES_DIR = "LARYNX_VOICES_DIR"

# Format string for downloading voices
DEFAULT_VOICE_URL_FORMAT = (
    "http://github.com/rhasspy/larynx/releases/download/v1.0.0/{voice}.tar.gz"
)

# Directory names that contain vocoders instead of voices
VOCODER_DIR_NAMES = set(v.value for v in VocoderType if v != VocoderType.GRIFFIN_LIM)

# Quality name to vocoder name
VOCODER_QUALITY: typing.Dict[str, str] = {
    "high": "hifi_gan/universal_large",
    "medium": "hifi_gan/vctk_medium",
    "low": "hifi_gan/vctk_small",
}

# alias -> full name
VOICE_ALIASES: typing.Dict[str, str] = {}

# voice -> <name>.tar.gz
VOICE_DOWNLOAD_NAMES: typing.Dict[str, str] = {}


def load_voices_aliases():
    """Load voice aliases from VOICES file"""
    if not VOICE_ALIASES:
        # Load voice aliases
        with open(_DIR / "VOICES", "r", encoding="utf-8") as voices_file:
            for line in voices_file:
                line = line.strip()
                if not line:
                    continue

                # alias alias ... full_name download_name
                *voice_aliases, full_voice_name, download_name = line.split()
                for voice_alias in voice_aliases:
                    VOICE_ALIASES[voice_alias] = download_name

                VOICE_ALIASES[full_voice_name] = download_name
                VOICE_DOWNLOAD_NAMES[full_voice_name] = download_name


def resolve_voice_name(voice_name: str) -> str:
    """Resolve voice name using aliases"""
    load_voices_aliases()
    return VOICE_ALIASES.get(voice_name, voice_name)


def split_voice_name(voice_name: str) -> typing.Tuple[str, str, str]:
    """Split resolved voice name (<lang>_<name>-<model_type>) into language, name, model_type"""
    lang, voice_name = voice_name.split("_", maxsplit=1)
    last_dash = voice_name.rfind("-")
    name, model_type = voice_name[:last_dash], voice_name[last_dash + 1 :]

    return lang, name, model_type


def get_voice_download_name(voice_name: str) -> str:
    """Get name of .tar.gz file name for voice (without extension)"""
    voice_name = resolve_voice_name(voice_name)
    return VOICE_DOWNLOAD_NAMES.get(voice_name, voice_name)


# -----------------------------------------------------------------------------


class VoiceDownloadError(Exception):
    """Occurs when a voice or vocoder fails to download"""


def download_voice(
    voice_name: str, voices_dir: typing.Union[str, Path], link: str
) -> Path:
    """Download and extract a voice (or vocoder)"""
    from tqdm.auto import tqdm

    voices_dir = Path(voices_dir)
    voices_dir.mkdir(parents=True, exist_ok=True)

    _LOGGER.debug(
        "Downloading voice/vocoder for %s to %s from %s", voice_name, voices_dir, link
    )

    try:
        with urllib.request.urlopen(link) as response:
            with tempfile.NamedTemporaryFile(mode="wb+", suffix=".tar.gz") as temp_file:
                with tqdm(
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    miniters=1,
                    desc=voice_name,
                    total=int(response.headers.get("content-length", 0)),
                ) as pbar:
                    chunk = response.read(4096)
                    while chunk:
                        temp_file.write(chunk)
                        pbar.update(len(chunk))
                        chunk = response.read(4096)

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
    except HTTPError as e:
        _LOGGER.exception("download_voice")
        raise VoiceDownloadError(
            f"Failed to download voice {voice_name} from {link}: {e}"
        ) from e


# -----------------------------------------------------------------------------


def get_voices_dirs(
    voices_dir: typing.Optional[typing.Union[str, Path]] = None
) -> typing.List[Path]:
    """Get directories to search for voices"""
    # Directories to search for voices
    voices_dirs: typing.List[Path] = []

    if voices_dir:
        # 1. Use --voices-dir
        voices_dirs.append(Path(voices_dir))

    # 2. Use environment variable
    env_dir = os.environ.get(_ENV_VOICES_DIR)
    if env_dir is not None:
        voices_dirs.append(Path(env_dir))

    # 3. Use ${XDG_DATA_HOME}/larynx/voices
    maybe_data_home = os.environ.get("XDG_DATA_HOME")
    if maybe_data_home:
        voices_dirs.append(Path(maybe_data_home) / "larynx" / "voices")
    else:
        # ~/.local/share/larynx/voices
        voices_dirs.append(Path.home() / ".local" / "share" / "larynx" / "voices")

    # 4. Use local directory next to module
    voices_dirs.append(_DIR.parent / "local")

    return voices_dirs


def valid_voice_dir(voice_dir: typing.Union[str, Path]) -> bool:
    """Return True if directory exists and has an Onnx model or PyTorch checkpoint"""
    voice_dir = Path(voice_dir)
    return voice_dir.is_dir() and (
        (len(list(voice_dir.glob("*.onnx"))) > 0)
        or (len(list(voice_dir.glob("*.pth"))) > 0)
    )


def get_runtime_dir() -> Path:
    """Return path to XDG_RUNTIME_DIR or fallback"""
    maybe_runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if maybe_runtime_dir:
        runtime_dir = Path(maybe_runtime_dir) / "larynx"
    else:
        # Fallback to /tmp/larynx-runtime-<user>
        user = getpass.getuser()
        runtime_dir = Path(tempfile.gettempdir()) / f"larynx-runtime-{user}"

    runtime_dir.mkdir(parents=True, exist_ok=True)

    return runtime_dir
