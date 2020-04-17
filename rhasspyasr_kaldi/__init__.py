"""Automated speech recognition in Rhasspy using Kaldi."""
from .train import get_kaldi_dir, train, train_prepare_online_decoding
from .transcribe import KaldiCommandLineTranscriber, KaldiModelType
