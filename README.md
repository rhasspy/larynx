# Larynx Runtime

End-to-end text to speech system using [gruut](https://github.com/rhasspy/gruut) and [onnx](https://onnx.ai/).

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
  larynx-runtime \
    --debug \
    --csv \
    --glow-tts local/en-us/ljspeech-glow_tts \
    --hifi-gan local/hifi_gan/universal_large \
    --output-dir wavs \
    en-us
```

## Text to Speech Models

* [GlowTTS](https://github.com/rhasspy/glow-tts-train)
    * English
    * German
    * French
    * Spanish
    * Dutch
    * Swedish
    * Russian
* [Tacotron2](https://github.com/rhasspy/tacotron2-train)

## Vocoders

* [Hi-Fi GAN](https://github.com/rhasspy/hifi-gan-train)
* [WaveGlow](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/Tacotron2)
