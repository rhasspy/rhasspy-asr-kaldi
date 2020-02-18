"""Automated speech recognition in Rhasspy using Kaldi."""
from .train import guess_pronunciations, read_dict, train
from .transcribe import KaldiCommandLineTranscriber
