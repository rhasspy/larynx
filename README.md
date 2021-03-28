# Larynx

End-to-end text to speech system using [gruut](https://github.com/rhasspy/gruut) and [onnx](https://onnx.ai/).

![Larynx logo](img/logo.png)

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

## Example

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
        * blizzard_fls (F, accent)
        * cmu_aew (M)
        * cmu_ahw (M)
        * cmu_aup (M, accent)
        * cmu_bdl (M)
        * cmu_clb (F)
        * cmu_eey (F)
        * cmu_fem (M)
        * cmu_jmk (M)
        * cmu_ksp (M, accent)
        * cmu_ljm (F)
        * cmu_lnh (F)
        * cmu_rms (M)
        * cmu_rxr (M)
        * cmu_slp (F, accent)
        * cmu_slt (F)
        * ek (F, accent)
        * harvard (F, accent)
        * kathleen (F)
        * ljspeech (F)
    * German (`de-de`, 1 voice)
        * thorsten (M)
    * French (`fr-fr`, 3 voices)
        * gilles\_le\_blanc (M)
        * siwis (F)
        * tom (M)
    * Spanish (`es-es`, 2 voices)
        * carlfm (M)
        * karen_savage (F)
    * Dutch (`nl`, 3 voices)
        * bart\_de\_leeuw (M)
        * flemishguy (M)
        * rdh (M)
    * Italian (`it-it`, 2 voices)
        * lisa (F)
        * riccardo_fasol (M)
    * Swedish (`sv-se`, 1 voice)
        * talesyntese (M)
    * Russian (`ru-ru`, 3 voices)
        * hajdurova (F)
        * nikolaev (M)
        * minaev (M)
* [Tacotron2](https://github.com/rhasspy/tacotron2-train)
    * Coming soon

## Vocoders

* [Hi-Fi GAN](https://github.com/rhasspy/hifi-gan-train)
    * Universal large
    * VCTK medium
    * VCTK small
* [WaveGlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2)
    * 256 channel trained on LJ Speech
