# Larynx

End-to-end text to speech system using [gruut](https://github.com/rhasspy/gruut) and [onnx](https://onnx.ai/).

![Larynx screenshot](img/web_screenshot.png)

Larynx's goals are:

* "Good enough" synthesis to avoid using a cloud service
* Faster than realtime performance on a Raspberry Pi 4
* Broad language support
* Voices trained purely from public datasets

## Samples

[Listen to voice samples](https://rhasspy.github.io/larynx/) from all of the [pre-trained models](https://github.com/rhasspy/larynx/releases).

## Installation

```sh
$ pip install larynx
```

For Raspberry Pi (ARM), you will first need to [manually install phonetisaurus](https://github.com/rhasspy/phonetisaurus-pypi/releases).

### Language Download

Larynx uses [gruut](https://github.com/rhasspy/gruut) to transform text into phonemes. You must install the appropriate gruut language before using Larynx. U.S. English is included with gruut, but for other languages:

```sh
$ python3 -m gruut <LANGUAGE> download
```

### Voice/Vocoder Download

Voices and vocoders are available to download from the [release page](https://github.com/rhasspy/larynx/releases). They can be extracted anywhere, and the directory simply needs to be referenced in the command-line (e,g, `--glow-tts /path/to/voice`).

## Web Server

You can run a local web server with:

```sh
$ python3 -m larynx.server --voices-dir /path/to/voices
```

Visit http://localhost:5002 to view the site and try out voices. See http://localhost/5002/openapi for documentation on the available HTTP endpoints.

See `--help` for more options.

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
    --glow-tts local/en-us/harvard-glow_tts \
    --hifi-gan local/hifi_gan/universal_large \
    --output-dir wavs \
    --language en-us \
    --denoiser-strength 0.001
```

You can use the `--interactive` flag instead of `--output-dir` to type sentences and have the audio played immediately using `sox`.

### GlowTTS Settings

The GlowTTS voices support two additional parameters:

* `--noise-scale` - determines the speaker volatility during synthesis (0-1, default is  0.333)
* `--length-scale` - makes the voice speaker slower (< 1) or faster (> 1)

### Vocoder Settings

* `--denoiser-strength` - runs the denoiser if > 0; a small value like 0.005 is recommended.

## Text to Speech Models

* [GlowTTS](https://github.com/rhasspy/glow-tts-train) (35 voices)
    * English (`en-us`, 20 voices)
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
    * German (`de-de`, 1 voice)
        * thorsten (M, [CC0](licenses/cc0.txt))
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
    * VCTK medium
    * VCTK small
* [WaveGlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2)
    * 256 channel trained on LJ Speech
