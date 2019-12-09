from abc import ABC, abstractmethod
import typing

import attr


@attr.s
class Transcription:
    """Result of speech to text."""

    text: str = attr.ib()
    likelihood: float = attr.ib()
    transcribe_seconds: float = attr.ib()
    wav_seconds: float = attr.ib()


class Transcriber(ABC):
    """Base class for Kaldi transcribers."""

    @abstractmethod
    def transcribe_wav(self, wav_data: bytes) -> typing.Optional[Transcription]:
        """Speech to text from WAV data."""
        pass
