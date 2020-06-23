# Rhasspy ASR Kaldi

[![Continous Integration](https://github.com/rhasspy/rhasspy-asr-kaldi/workflows/Tests/badge.svg)](https://github.com/rhasspy/rhasspy-asr-kaldi/actions)
[![GitHub license](https://img.shields.io/github/license/rhasspy/rhasspy-asr-kaldi.svg)](https://github.com/rhasspy/rhasspy-asr-kaldi/blob/master/LICENSE)

Automated speech recognition in [Rhasspy](https://github.com/synesthesiam/rhasspy) voice assistant with [Kaldi](http://kaldi-asr.org).

## Requirements

* Python 3.7
* [Kaldi](https://kaldi-asr.org)
    * Expects `$KALDI_DIR` in environment
* [Opengrm](http://www.opengrm.org/twiki/bin/view/GRM/NGramLibrary)
    * Expects `ngram*` in `$PATH`
* [Phonetisaurus](https://github.com/AdolfVonKleist/Phonetisaurus)
    * Expects `phonetisaurus-apply` in `$PATH`

See [pre-built apps](https://github.com/synesthesiam/prebuilt-apps) for pre-compiled binaries.

## Installation

```bash
$ git clone https://github.com/rhasspy/rhasspy-asr-kaldi
$ cd rhasspy-asr-kaldi
$ ./configure
$ make
$ make install
```

## Transcribing

Use `python3 -m rhasspyasr_kaldi transcribe <ARGS>`

```
usage: rhasspy-asr-kaldi transcribe [-h] --model-dir MODEL_DIR
                                    [--graph-dir GRAPH_DIR]
                                    [--model-type MODEL_TYPE]
                                    [--frames-in-chunk FRAMES_IN_CHUNK]
                                    [wav_file [wav_file ...]]

positional arguments:
  wav_file              WAV file(s) to transcribe

optional arguments:
  -h, --help            show this help message and exit
  --model-dir MODEL_DIR
                        Path to Kaldi model directory (with conf, data)
  --graph-dir GRAPH_DIR
                        Path to Kaldi graph directory (with HCLG.fst)
  --model-type MODEL_TYPE
                        Either nnet3 or gmm (default: nnet3)
  --frames-in-chunk FRAMES_IN_CHUNK
                        Number of frames to process at a time
```

For nnet3 models, the `online2-tcp-nnet3-decode-faster` program is used to handle streaming audio. For gmm models, audio is buffered and packaged as a WAV file before being transcribed.

## Training

Use `python3 -m rhasspyasr_kaldi train <ARGS>`

```
usage: rhasspy-asr-kaldi train [-h] --model-dir MODEL_DIR
                               [--graph-dir GRAPH_DIR]
                               [--intent-graph INTENT_GRAPH]
                               [--dictionary DICTIONARY]
                               [--dictionary-casing {upper,lower,ignore}]
                               [--language-model LANGUAGE_MODEL]
                               --base-dictionary BASE_DICTIONARY
                               [--g2p-model G2P_MODEL]
                               [--g2p-casing {upper,lower,ignore}]

optional arguments:
  -h, --help            show this help message and exit
  --model-dir MODEL_DIR
                        Path to Kaldi model directory (with conf, data)
  --graph-dir GRAPH_DIR
                        Path to Kaldi graph directory (with HCLG.fst)
  --intent-graph INTENT_GRAPH
                        Path to intent graph JSON file (default: stdin)
  --dictionary DICTIONARY
                        Path to write custom pronunciation dictionary
  --dictionary-casing {upper,lower,ignore}
                        Case transformation for dictionary words (training,
                        default: ignore)
  --language-model LANGUAGE_MODEL
                        Path to write custom language model
  --base-dictionary BASE_DICTIONARY
                        Paths to pronunciation dictionaries
  --g2p-model G2P_MODEL
                        Path to Phonetisaurus grapheme-to-phoneme FST model
  --g2p-casing {upper,lower,ignore}
                        Case transformation for g2p words (training, default:
                        ignore)
```

This will generate a custom `HCLG.fst` from an intent graph created using [rhasspy-nlu](https://github.com/rhasspy/rhasspy-nlu). Your Kaldi model directory should be laid out like this:

* my_model/  (`--model-dir`)
    * conf/
        * mfcc_hires.conf
    * data/
        * local/
            * dict/
                * lexicon.txt (copied from `--dictionary`)
            * lang/
                * lm.arpa.gz (copied from `--language-model`)
    * graph/ (`--graph-dir`)
        * HCLG.fst (generated)
    * model/
        * final.mdl
    * phones/
        * extra_questions.txt
        * nonsilence_phones.txt
        * optional_silence.txt
        * silence_phones.txt
    * online/ (nnet3 only)
    * extractor/ (nnet3 only)

When using the `train` command, you will need to specify the following arguments:

* `--intent-graph` - path to graph json file generated using [rhasspy-nlu](https://github.com/rhasspy/rhasspy-nlu)
* `--model-type` - either nnet3 or gmm
* `--model-dir` - path to top-level model directory (my_model in example above)
* `--graph-dir` - path to directory where HCLG.fst should be written (my_model/graph in example above)
* `--base-dictionary` - pronunciation dictionary with all words from intent graph (can be used multiple times)
* `--dictionary` - path to write custom pronunciation dictionary (optional)
* `--language-model` - path to write custom ARPA language model (optional)

## Building From Source

`rhasspy-asr-kaldi` depends on the following programs that must be compiled:

* [Kaldi](http://kaldi-asr.org)
    * Speech to text engine
* [Opengrm](http://www.opengrm.org/twiki/bin/view/GRM/NGramLibrary)
    * Create ARPA language models
* [Phonetisaurus](https://github.com/AdolfVonKleist/Phonetisaurus)
    * Guesses pronunciations for unknown words

### Kaldi

Make sure you have the necessary dependencies installed:

```bash
sudo apt-get install \
    build-essential \
    libatlas-base-dev libatlas3-base gfortran \
    automake autoconf unzip sox libtool subversion \
    python3 python \
    git zlib1g-dev
```

Download Kaldi and extract it:

```bash
wget -O kaldi-master.tar.gz \
    'https://github.com/kaldi-asr/kaldi/archive/master.tar.gz'
tar -xvf kaldi-master.tar.gz
```

First, build Kaldi's tools:

```bash
cd kaldi-master/tools
make
```

Use `make -j 4` if you have multiple CPU cores. This will take a **long** time.

Next, build Kaldi itself:

```bash
cd kaldi-master
./configure --shared --mathlib=ATLAS
make depend
make
```

Use `make depend -j 4` and `make -j 4` if you have multiple CPU cores. This will take a **long** time.

There is no installation step. The `kaldi-master` directory contains all the libraries and programs that Rhasspy will need to access.

See [docker-kaldi](https://github.com/synesthesiam/docker-kaldi) for a Docker build script.

### Phonetisaurus

Make sure you have the necessary dependencies installed:

```bash
sudo apt-get install build-essential
```

First, download and build [OpenFST 1.6.2](http://www.openfst.org/)

```bash
wget http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.6.2.tar.gz
tar -xvf openfst-1.6.2.tar.gz
cd openfst-1.6.2
./configure \
    "--prefix=$(pwd)/build" \
    --enable-static --enable-shared \
    --enable-far --enable-ngram-fsts
make
make install
```

Use `make -j 4` if you have multiple CPU cores. This will take a **long** time.

Next, download and extract Phonetisaurus:

```bash
wget -O phonetisaurus-master.tar.gz \
    'https://github.com/AdolfVonKleist/Phonetisaurus/archive/master.tar.gz'
tar -xvf phonetisaurus-master.tar.gz
```

Finally, build Phonetisaurus (where `/path/to/openfst` is the `openfst-1.6.2` directory from above):

```
cd Phonetisaurus-master
./configure \
    --with-openfst-includes=/path/to/openfst/build/include \
    --with-openfst-libs=/path/to/openfst/build/lib
make
make install
```

Use `make -j 4` if you have multiple CPU cores. This will take a **long** time.

You should now be able to run the `phonetisaurus-align` program.

See [docker-phonetisaurus](https://github.com/synesthesiam/docker-phonetisaurus) for a Docker build script.
