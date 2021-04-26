# Larynx

End-to-end text to speech system using [gruut](https://github.com/rhasspy/gruut) and [onnx](https://onnx.ai/). There are [40 voices available across 8 languages](#samples).

```sh
$ docker run -it -p 5002:5002 rhasspy/larynx:en-us
```

![Larynx screenshot](img/web_screenshot.png)

Larynx's goals are:

* "Good enough" synthesis to avoid using a cloud service
* Faster than realtime performance on a Raspberry Pi 4 (with low quality vocoder)
* Broad language support (8 languages)
* Voices trained purely from public datasets

## Samples

[Listen to voice samples](https://rhasspy.github.io/larynx/) from all of the [pre-trained models](https://github.com/rhasspy/larynx/releases).

---

## Docker Installation

Pre-built Docker images for each language are available for the following platforms:

* `linux/amd64` - desktop/laptop/server
* `linux/arm64` - Raspberry Pi 64-bit
* `linux/arm/v7` - Raspberry Pi 32-bit

Run the Larynx web server with:

```sh
$ docker run -it -p 5002:5002 rhasspy/larynx:<LANG>
```

where `<LANG>` is one of:

* `de-de` - German
* `en-us` - U.S. English
* `es-es` - Spanish
* `fr-fr` - French
* `it-it` - Italian
* `nl` - Dutch
* `ru-ru` - Russian
* `sv-se` - Swedish

Visit http://localhost:5002 for the test page. See http://localhost:5002/openapi/ for HTTP endpoint documentation.

A larger docker image with all languages is also available as `rhasspy/larynx`

## Debian Installation

Pre-built Debian packages are [available for download](https://github.com/rhasspy/larynx/releases/tag/v0.4.0).

There are three different kinds of packages, so you can install exactly what you want and no more:

* `larynx-tts_<VERSION>_<ARCH>.deb`
    * Base Larynx code and dependencies (always required)
    * `ARCH` is one of `amd64` (most desktops, laptops), `armhf` (32-bit Raspberry Pi), `arm64` (64-bit Raspberry Pi)
* `larynx-tts-lang-<LANG>_<VERSION>_all.deb`
    * Language-specific data files (at least one required)
    * See [above](#docker-installation) for a list of languages
* `larynx-tts-voice-<VOICE>_<VERSION>_all.deb`
    * Voice-specific model files (at least one required)
    * See [samples](#samples) to decide which voice(s) to choose
    
As an example, let's say you want to use the "harvard-glow_tts" voice for English on an `amd64` laptop for Larynx version 0.4.0.
You would need to download these files:

1. [`larynx-tts_0.4.0_amd64.deb`](https://github.com/rhasspy/larynx/releases/download/v0.4.0/larynx-tts_0.4.0_amd64.deb)
2. [`larynx-tts-lang-en-us_0.4.0_all.deb`](https://github.com/rhasspy/larynx/releases/download/v0.4.0/larynx-tts-lang-en-us_0.4.0_all.deb)
3. [`larynx-tts-voice-en-us-harvard-glow-tts_0.4.0_all.deb`](https://github.com/rhasspy/larynx/releases/download/v0.4.0/larynx-tts-voice-en-us-harvard-glow-tts_0.4.0_all.deb)

Once downloaded, you can install the packages all at once with:

```sh
sudo apt install \
  ./larynx-tts_0.4.0_amd64.deb \
  ./larynx-tts-lang-en-us_0.4.0_all.deb \
  ./larynx-tts-voice-en-us-harvard-glow-tts_0.4.0_all.deb
```

From there, you may run the `larynx` command or `larynx-server` to start the web server.

## Python Installation

```sh
$ pip install larynx
```

For Raspberry Pi (ARM), you will first need to [manually install phonetisaurus](https://github.com/rhasspy/phonetisaurus-pypi/releases).

For 32-bit ARM systems, a pre-built [onnxruntime wheel](https://github.com/synesthesiam/prebuilt-apps/releases/download/v1.0/) is available (official 64-bit wheels are available in [PyPI](https://pypi.org/project/onnxruntime/)).

### Language Download

Larynx uses [gruut](https://github.com/rhasspy/gruut) to transform text into phonemes. You must install the appropriate gruut language before using Larynx. U.S. English is included with gruut, but for other languages:

```sh
$ python3 -m gruut <LANGUAGE> download
```

### Voice/Vocoder Download

Voices and vocoders are available to download from the [release page](https://github.com/rhasspy/larynx/releases). They can be extracted anywhere, and the directory simply needs to be referenced in the command-line (e,g, `--voices-dir /path/to/voices`).

---

## Web Server

You can run a local web server with:

```sh
$ python3 -m larynx.server --voices-dir /path/to/voices
```

Visit http://localhost:5002 to view the site and try out voices. See http://localhost/5002/openapi for documentation on the available HTTP endpoints.

See `--help` for more options.

### MaryTTS Compatible API

To use Larynx as a drop-in replacement for a [MaryTTS](http://mary.dfki.de/) server (e.g., for use with [Home Assistant](https://www.home-assistant.io/integrations/marytts/)), run:

```sh
$ docker run -it -p 59125:5002 rhasspy/larynx:<LANG>
```

The `/process` HTTP endpoint should now work for voices formatted as `<LANG>/<VOICE>` such as `en-us/harvard-glow_tts`.

You can specify the vocoder by adding `;<VOCODER>` to the MaryTTS voice.

For example: `en-us/harvard-glow_tts;hifi_gan:vctk_small` will use the lowest quality (but fastest) vocoder. This is usually necessary to get decent performance on a Raspberry Pi.

Available vocoders are:

* `hifi_gan:universal_large` (best quality, slowest, default)
* `hifi_gan:vctk_medium` (medium quality)
* `hifi_gan:vctk_small` (lowest quality, fastest)


## Command-Line Example

The command below synthesizes multiple sentences and saves them to a directory. The `--csv` command-line flag indicates that each sentence is of the form `id|text` where `id` will be the name of the WAV file.

```sh
$ cat << EOF |
s01|The birch canoe slid on the smooth planks.
s02|Glue the sheet to the dark blue background.
s03|It's easy to tell the depth of a well.
s04|These days a chicken leg is a rare dish.
s05|Rice is often served in round bowls.
s06|The juice of lemons makes fine punch.
s07|The box was thrown beside the parked truck.
s08|The hogs were fed chopped corn and garbage.
s09|Four hours of steady work faced us.
s10|Large size in stockings is hard to sell.
EOF
  larynx \
    --debug \
    --csv \
    --voice harvard-glow_tts \
    --quality high \
    --output-dir wavs \
    --denoiser-strength 0.001
```

You can use the `--interactive` flag instead of `--output-dir` to type sentences and have the audio played immediately using the `play` command from `sox`.

### GlowTTS Settings

The GlowTTS voices support two additional parameters:

* `--noise-scale` - determines the speaker volatility during synthesis (0-1, default is  0.333)
* `--length-scale` - makes the voice speaker slower (> 1) or faster (< 1)

### Vocoder Settings

* `--denoiser-strength` - runs the denoiser if > 0; a small value like 0.005 is recommended.

### List Voices and Vocoders

```sh
$ larynx --list
```

---

## Text to Speech Models

* [GlowTTS](https://github.com/rhasspy/glow-tts-train) (40 voices)
    * English (`en-us`, 21 voices)
        * blizzard_fls (F, accent, [Blizzard](https://www.cstr.ed.ac.uk/projects/blizzard/2017/usborne_blizzard2017/license.html))
        * cmu_aew (M, [Arctic](licenses/cmuarctic.txt))
        * cmu_ahw (M, [Arctic](licenses/cmuarctic.txt))
        * cmu_aup (M, accent, [Arctic](licenses/cmuarctic.txt))
        * cmu_bdl (M, [Arctic](licenses/cmuarctic.txt))
        * cmu_clb (F, [Arctic](licenses/cmuarctic.txt))
        * cmu_eey (F, [Arctic](licenses/cmuarctic.txt))
        * cmu_fem (M, [Arctic](licenses/cmuarctic.txt))
        * cmu_jmk (M, [Arctic](licenses/cmuarctic.txt))
        * cmu_ksp (M, accent, [Arctic](licenses/cmuarctic.txt))
        * cmu_ljm (F, [Arctic](licenses/cmuarctic.txt))
        * cmu_lnh (F, [Arctic](licenses/cmuarctic.txt))
        * cmu_rms (M, [Arctic](licenses/cmuarctic.txt))
        * cmu_rxr (M, [Arctic](licenses/cmuarctic.txt))
        * cmu_slp (F, accent, [Arctic](licenses/cmuarctic.txt))
        * cmu_slt (F, [Arctic](licenses/cmuarctic.txt))
        * ek (F, accent, [M-AILabs](licenses/m-ailabs.txt))
        * harvard (F, accent, [CC/Attr/NC](https://creativecommons.org/licenses/by-nc/4.0/legalcode))
        * kathleen (F, [CC0](licenses/cc0.txt))
        * ljspeech (F, [Public Domain](https://librivox.org/pages/public-domain/))
        * mary_ann (F, [M-AILabs](licenses/m-ailabs.txt))
    * German (`de-de`, 5 voice)
        * thorsten (M, [CC0](licenses/cc0.txt))
        * eva_k (F, [M-AILabs](licenses/m-ailabs.txt))
        * karlsson (M, [M-AILabs](licenses/m-ailabs.txt))
        * rebecca\_braunert\_plunkett (F, [M-AILabs](licenses/m-ailabs.txt))
        * pavoque (M, [CC4/BY/NC/SA](https://github.com/marytts/pavoque-data))
    * French (`fr-fr`, 3 voices)
        * gilles\_le\_blanc (M, [M-AILabs](licenses/m-ailabs.txt))
        * siwis (F, [CC/Attr](licenses/cc4a.txt))
        * tom (M, [ODbL](licenses/odbl.txt))
    * Spanish (`es-es`, 2 voices)
        * carlfm (M, public domain)
        * karen_savage (F, [M-AILabs](licenses/m-ailabs.txt))
    * Dutch (`nl`, 3 voices)
        * bart\_de\_leeuw (M, [Apache2](licenses/apache2.txt))
        * flemishguy (M, [CC0](licenses/cc0.txt))
        * rdh (M, [CC0](licenses/cc0.txt))
    * Italian (`it-it`, 2 voices)
        * lisa (F, [M-AILabs](licenses/m-ailabs.txt))
        * riccardo_fasol (M, [Apache2](licenses/apache2.txt))
    * Swedish (`sv-se`, 1 voice)
        * talesyntese (M, [CC0](licenses/cc0.txt))
    * Russian (`ru-ru`, 3 voices)
        * hajdurova (F, [M-AILabs](licenses/m-ailabs.txt))
        * nikolaev (M, [M-AILabs](licenses/m-ailabs.txt))
        * minaev (M, [M-AILabs](licenses/m-ailabs.txt))
* [Tacotron2](https://github.com/rhasspy/tacotron2-train)
    * Coming soon

## Vocoders

* [Hi-Fi GAN](https://github.com/rhasspy/hifi-gan-train)
    * Universal large
    * VCTK "medium"
    * VCTK "small"
* [WaveGlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2)
    * 256 channel trained on LJ Speech
