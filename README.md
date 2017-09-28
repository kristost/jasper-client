jasper-client
=============

**NOTE: This is a fork of the [jasper client code](https://github.com/jasperproject/jasper-client).**

>Client code for the Jasper voice computing platform. Jasper is an open source platform for developing always-on, voice-controlled applications.
>
>Learn more at [jasperproject.github.io](http://jasperproject.github.io/), where we have assembly and installation instructions, as well as extensive documentation. For the relevant disk image, please visit [SourceForge](http://sourceforge.net/projects/jasperproject/).

# Setup/Install
This repo is part of an academic project I'm involved with, requiring the use of Jasper and Snowboy on a Raspberry Pi 2 Model B.
More extensive documentation forms part of the academic project itself. I may get around to merging that documentation here in the future.

## Imaging the Raspberry Pi with Jasper v1.5
Follow the instructions here: https://github.com/mattcurrycom/Documentation/tree/master/jasper

Useful Links
* http://jasperproject.github.io/
* https://groups.google.com/forum/#!forum/jasper-support-forum

## Post Image Setup
https://github.com/mattcurrycom/Documentation/blob/master/jasper/jasper-client/Jasper-Post-Image-Setup.md

## Setting up [Snowboy](https://snowboy.kitt.ai/) on Jasper Image v1.5
~~1. Downloaded the source from git (just because, although I'm not using it -- yet...)~~
1. Downloaded the precompiled Raspberry Pi binaries from [here](https://s3-us-west-2.amazonaws.com/snowboy/snowboy-releases/rpi-arm-raspbian-8.0-1.2.0.tar.bz2) listed on the [Downloads](http://docs.kitt.ai/snowboy/#downloads) section of the docs.
1. Ran `python demo.py resources/snowboy.umdl` to test out the universal snowboy model (recognises the hotword 'snowboy', but trained on "many different speakers" whatever than implies (different accents?)).
1. Got an error about ``ImportError: libf77blas.so.3: cannot open shared object file: No such file or directory`` (similar issue raised [here](https://github.com/Kitt-AI/snowboy/issues/94))
1. So I did ```sudo apt-get install libatlas-base-dev``` which installed a bunch of dependencies as well. IAW the snowboy docs, this is supposed to be done after a ```sudo apt-get install swig3.0 python-pyaudio python3-pyaudio sox``` but I wanted to see if I could get it to work minimally without the other crud (worried about Jasper being quite brittle and sensitive to packages)
1. Re-ran `python demo.py resources/snowboy.umdl` and it worked very well.

## License

*Copyright (c) 2014-2015, Charles Marsh, Shubhro Saha & Jan Holthuis. All rights reserved.*

Jasper is covered by the MIT license, a permissive free software license that lets you do anything you want with the source code, as long as you provide back attribution and ["don't hold \[us\] liable"](http://choosealicense.com). For the full license text see the [LICENSE.md](LICENSE.md) file.

*Note that this licensing only refers to the Jasper client code (i.e.,  the code on GitHub) and not to the disk image itself (i.e., the code on SourceForge).*
