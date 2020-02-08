"""Automated speech recognition in Rhasspy using Kaldi."""
from .train import train, read_dict, guess_pronunciations
from .transcribe import KaldiCommandLineTranscriber
