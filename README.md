# Rhasspy ASR Kaldi

[![Continous Integration](https://github.com/rhasspy/rhasspy-asr-kaldi/workflows/Tests/badge.svg)](https://github.com/rhasspy/rhasspy-asr-kaldi/actions)
[![GitHub license](https://img.shields.io/github/license/rhasspy/rhasspy-asr-kaldi.svg)](https://github.com/rhasspy/rhasspy-asr-kaldi/blob/master/LICENSE)

Automated speech recognition in [Rhasspy](https://github.com/synesthesiam/rhasspy) voice assistant with [Kaldi](http://kaldi-asr.org).

## Transcribing

For nnet3 models, the `rhasspyasr_kaldi.KaldiExtensionTranscriber` uses a compiled Python extension and is the fastest. For gmm models, the `rhasspyasr_kaldi.KaldiCommandLineTranscriber` calls the `kaldi-decode` script in `bin`.

## Training

The `kaldi-train` script in `bin` will generate a custom `HCLG.fst` from an ARPA language model. Your Kaldi model directory should be laid out like this:

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

When calling `kaldi-train`, you will need to specify the following arguments:

* `--model-type` - either nnet3 or gmm
* `--kaldi-dir` - path to Kaldi directory
* `--model-dir` - path to top-level model directory (my_model in example above)
* `--graph-dir` - path to directory where HCLG.fst should be written (my_model/graph in example above)
* `--dictionary` - path to pronunciation dictionary
* `--language-model` - path to ARPA language model
