"""Automated speech recognition in Rhasspy using Kaldi."""
from .train import PronunciationsType, guess_pronunciations, read_dict, train
from .transcribe import KaldiCommandLineTranscriber
