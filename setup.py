"""Setup file for larynx runtime"""
import os
from pathlib import Path

import setuptools

this_dir = Path(__file__).parent
module_dir = this_dir / "larynx"

# -----------------------------------------------------------------------------

# Load README in as long description
long_description: str = ""
readme_path = this_dir / "README.md"
if readme_path.is_file():
    long_description = readme_path.read_text()

requirements = []
requirements_path = this_dir / "requirements.txt"
if requirements_path.is_file():
    with open(requirements_path, "r") as requirements_file:
        requirements = requirements_file.read().splitlines()

version_path = module_dir / "VERSION"
with open(version_path, "r") as version_file:
    version = version_file.read().strip()

# -----------------------------------------------------------------------------

data_files = [
    str(f.relative_to(module_dir))
    for d in ["css", "templates", "wav"]
    for f in (module_dir / d).rglob("*")
    if f.is_file()
]

# -----------------------------------------------------------------------------

setuptools.setup(
    name="larynx",
    version=version,
    description="Neural text to speech system using the International Phonetic Alphabet",
    author="Michael Hansen",
    author_email="mike@rhasspy.org",
    url="https://github.com/rhasspy/larynx",
    packages=setuptools.find_packages(),
    package_data={"larynx": data_files + ["VERSION", "VOICES", "VOCODERS", "py.typed"]},
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "larynx = larynx.__main__:main",
            "larynx-server = larynx.server.__main__:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
)
