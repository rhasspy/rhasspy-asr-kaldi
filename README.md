# Rhasspy ASR Kaldi

[![Continous Integration](https://github.com/rhasspy/rhasspy-asr-kaldi/workflows/Tests/badge.svg)](https://github.com/rhasspy/rhasspy-asr-kaldi/actions)
[![GitHub license](https://img.shields.io/github/license/rhasspy/rhasspy-asr-kaldi.svg)](https://github.com/rhasspy/rhasspy-asr-kaldi/blob/master/LICENSE)

Automated speech recognition in [Rhasspy](https://github.com/synesthesiam/rhasspy) voice assistant with [Kaldi](http://kaldi-asr.org).

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

For nnet3 models, the `online2-tcp-nnet3-decode-faster` program is used to handle streaming audio. For gmm models, audio buffered and packaged as a WAV file before being transcribed.

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
