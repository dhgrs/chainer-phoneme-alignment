# chainer-phoneme-alignment

This repository is a part of VoiceStatistics WaveNet project.

# What can you do?
You can preprocess for [speech segmentation toolkit using Julius](https://github.com/julius-speech/segmentation-kit). `preprocess.py` generates `*.wav`; 16kHz resampled wav and `*.txt`; a sentence with hiragana corresponds to wav.

# Requirements
- python3.x
- tqdm
- numpy
- pandas
- librosa==**0.5**

**CAUTION**: librosa==0.6 can not save wav with `int16`. So you have to use librosa==0.5. I don't check older version but maybe you can use.

# How to use
## Step1
Download voice statistics dataset. You can download it very easily via [my repository](https://github.com/dhgrs/download_dataset).

## Step2
Set the path of dataset in `params.py`.

## Step3
Run `preprocess.py` like below.

    # In the directory of this repository
    python3 preprocess.py

## Step4
Clone [segmentation-kit](https://github.com/julius-speech/segmentation-kit) like below.

    git clone https://github.com/julius-speech/segmentation-kit

## Step5
Run segmentation-kit like below.

    # In the directory of segmentation-kit
    perl segment_julius.pl

Then `*.lab`s and `*.log`s are generated in `segmantation-kit/wav`. `*.lab` is a result of phoneme segmentation.