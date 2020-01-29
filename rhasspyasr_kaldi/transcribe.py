"""Automated speech recognition in Rhasspy using Kaldi."""
import io
import json
import logging
import socket
import struct
import subprocess
import tempfile
import time
import typing
import wave
from enum import Enum
from pathlib import Path

import numpy as np
from rhasspyasr import Transcriber, Transcription

# pylint: disable=E0401,E0611
# from .nnet3 import KaldiNNet3OnlineDecoder, KaldiNNet3OnlineModel

_DIR = Path(__file__).parent
_LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


class KaldiModelType(str, Enum):
    """Supported Kaldi model types."""

    NNET3 = "nnet3"
    GMM = "gmm"


# -----------------------------------------------------------------------------


# class KaldiExtensionTranscriber(Transcriber):
#     """Speech to text with Kaldi nnet3 Python extension."""

#     def __init__(
#         self, model_dir: typing.Union[str, Path], graph_dir: typing.Union[str, Path]
#     ):
#         self.model_dir = Path(model_dir)
#         self.graph_dir = Path(graph_dir)
#         self.model: typing.Optional[KaldiNNet3OnlineModel] = None
#         self.decoder: typing.Optional[KaldiNNet3OnlineDecoder] = None

#     def load_decoder(self):
#         """Load Kaldi decoder if not already loaded."""
#         if self.model is None:
#             self.model = self.get_model()

#         if self.decoder is None:
#             self.decoder = self.get_decoder(self.model)

#     def transcribe_wav(self, wav_data: bytes) -> typing.Optional[Transcription]:
#         """Speech to text from WAV data."""
#         self.load_decoder()
#         assert self.decoder

#         _LOGGER.debug("Decoding %s byte(s)", len(wav_data))
#         start_time = time.perf_counter()
#         with io.BytesIO(wav_data) as wav_buffer:
#             with wave.open(wav_buffer, "rb") as wav_file:
#                 sample_rate = wav_file.getframerate()
#                 num_frames = wav_file.getnframes()
#                 wav_duration = num_frames / float(sample_rate)

#                 frames = wav_file.readframes(num_frames)
#                 samples = struct.unpack_from("<%dh" % num_frames, frames)

#                 # Decode
#                 success = self.decoder.decode(
#                     sample_rate, np.array(samples, dtype=np.float32), True
#                 )

#                 if success:
#                     text, likelihood = self.decoder.get_decoded_string()
#                     transcribe_seconds = time.perf_counter() - start_time

#                     return Transcription(
#                         text=text.strip(),
#                         likelihood=likelihood,
#                         transcribe_seconds=transcribe_seconds,
#                         wav_seconds=wav_duration,
#                     )

#                 # Failure
#                 return None

#     def transcribe_stream(
#         self,
#         audio_stream: typing.Iterable[bytes],
#         sample_rate: int,
#         sample_width: int,
#         channels: int,
#     ) -> typing.Optional[Transcription]:
#         """Speech to text from an audio stream."""
#         assert channels == 1, "Only mono audio supported"
#         self.load_decoder()
#         assert self.decoder

#         start_time = time.perf_counter()
#         last_chunk: typing.Optional[bytes] = None
#         audio_iter = iter(audio_stream)
#         total_frames: int = 0
#         while True:
#             try:
#                 next_chunk = next(audio_iter)

#                 if last_chunk:
#                     # Don't finalize
#                     num_frames = len(last_chunk) // sample_width
#                     total_frames += num_frames
#                     samples = struct.unpack_from("<%dh" % num_frames, last_chunk)
#                     self.decoder.decode(
#                         sample_rate, np.array(samples, dtype=np.float32), False
#                     )

#                 last_chunk = next_chunk
#             except StopIteration:
#                 break

#         if not last_chunk:
#             # Add one empty frame for finalization
#             last_chunk = bytes([0] * sample_width)

#         # Finalize
#         num_frames = len(last_chunk) // sample_width
#         total_frames += num_frames
#         samples = struct.unpack_from("<%dh" % num_frames, last_chunk)
#         success = self.decoder.decode(
#             sample_rate, np.array(samples, dtype=np.float32), True
#         )

#         if success:
#             text, likelihood = self.decoder.get_decoded_string()
#             transcribe_seconds = time.perf_counter() - start_time

#             return Transcription(
#                 text=text.strip(),
#                 likelihood=likelihood,
#                 transcribe_seconds=transcribe_seconds,
#                 wav_seconds=total_frames / float(sample_rate),
#             )

#         # Failure
#         return None

#     def get_model(self) -> KaldiNNet3OnlineModel:
#         """Create nnet3 model using Python extension."""
#         _LOGGER.debug(
#             "Loading nnet3 model at %s (graph=%s)", self.model_dir, self.graph_dir
#         )

#         model = KaldiNNet3OnlineModel(str(self.model_dir), str(self.graph_dir))
#         _LOGGER.debug("Kaldi model loaded")
#         return model

#     def get_decoder(
#         self, model: typing.Optional[KaldiNNet3OnlineModel] = None
#     ) -> KaldiNNet3OnlineDecoder:
#         """Create nnet3 decoder using Python extension."""

#         _LOGGER.debug("Creating decoder")
#         decoder = KaldiNNet3OnlineDecoder(model or self.model)
#         _LOGGER.debug("Kaldi decoder loaded")

#         return decoder

#     def __repr__(self) -> str:
#         return (
#             "KaldiExtensionTranscriber("
#             f", model_dir={self.model_dir}"
#             f", graph_dir={self.graph_dir}"
#             ")"
#         )


# -----------------------------------------------------------------------------


class KaldiCommandLineTranscriber(Transcriber):
    """Speech to text with external Kaldi scripts."""

    def __init__(
        self,
        model_type: KaldiModelType,
        model_dir: typing.Union[str, Path],
        graph_dir: typing.Union[str, Path],
    ):
        self.model_type = model_type
        self.model_dir = Path(model_dir)
        self.graph_dir = Path(graph_dir)
        self.decode_proc = None

    def transcribe_wav(self, wav_data: bytes) -> typing.Optional[Transcription]:
        """Speech to text from WAV data."""
        start_time = time.perf_counter()
        words_txt = self.graph_dir / "words.txt"

        with tempfile.NamedTemporaryFile(suffix=".wav", mode="wb") as wav_file:
            wav_file.write(wav_data)
            wav_file.seek(0)

            if self.model_type == KaldiModelType.NNET3:
                online_conf = self.model_dir / "online" / "conf" / "online.conf"
                kaldi_cmd = [
                    str(_DIR / "kaldi" / "online2-wav-nnet3-latgen-faster"),
                    "--online=false",
                    "--do-endpointing=false",
                    f"--word-symbol-table={words_txt}",
                    f"--config={online_conf}",
                    str(self.model_dir / "model" / "final.mdl"),
                    str(self.graph_dir / "HCLG.fst"),
                    "ark:echo utt1 utt1|",
                    f"scp:echo utt1 {wav_file.name}|",
                    "ark:/dev/null",
                ]
            elif self.model_type == KaldiModelType.GMM:
                # TODO: online2-wav-gmm-latgen-faster
                pass
            else:
                raise ValueError(self.model_type)

            _LOGGER.debug(kaldi_cmd)

            try:
                lines = subprocess.check_output(
                    kaldi_cmd, stderr=subprocess.STDOUT, universal_newlines=True
                ).splitlines()
            except subprocess.CalledProcessError as e:
                _LOGGER.exception("transcribe_wav")
                _LOGGER.error(e.output)
                lines = []

            text = ""
            for line in lines:
                if line.startswith("utt1 "):
                    text = line.split(maxsplit=1)[1]
                    break

        if text:
            # Success
            end_time = time.perf_counter()

            return Transcription(
                text=text.strip(),
                likelihood=1,
                transcribe_seconds=(end_time - start_time),
                wav_seconds=get_wav_duration(wav_data),
            )

        # Failure
        return None

    def transcribe_stream(
        self,
        audio_stream: typing.Iterable[bytes],
        sample_rate: int,
        sample_width: int,
        channels: int,
    ) -> typing.Optional[Transcription]:
        """Speech to text from an audio stream."""
        if self.model_type == KaldiModelType.NNET3:
            # Use online2-tcp-nnet3-decode-faster
            if not self.decode_proc:
                self.start_decode()

            # Connect to decoder
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect(("localhost", 5050))
            client_file = client_socket.makefile()

            start_time = time.perf_counter()
            num_frames = 0
            for chunk in audio_stream:
                client_socket.sendall(chunk)
                num_frames += len(chunk) // sample_width

            # Partial shutdown of socket (write only).
            # This should force the Kaldi server to finalize the output.
            client_socket.shutdown(socket.SHUT_WR)

            lines = client_file.read().splitlines()
            if lines:
                text = lines[-1].strip()
            else:
                # No result
                text = ""

            if text:
                # Success
                end_time = time.perf_counter()

                return Transcription(
                    text=text,
                    likelihood=1,
                    transcribe_seconds=(end_time - start_time),
                    wav_seconds=(num_frames / sample_rate),
                )

            # Failure
            return None
        elif self.model_type == KaldiModelType.GMM:
            # No online streaming support.
            # Re-package as a WAV.
            with io.BytesIO() as wav_buffer:
                wav_file: wave.Wave_write = wave.open(wav_buffer, "wb")
                with wav_file:
                    wav_file.setframerate(sample_rate)
                    wav_file.setsampwidth(sample_width)
                    wav_file.setnchannels(channels)

                    for frame in audio_stream:
                        wav_file.writeframes(frame)

                return self.transcribe_wav(wav_buffer.getvalue())

    def stop(self):
        """Stop the transcriber."""
        if self.decode_proc:
            self.decode_proc.terminate()
            self.decode_proc.wait()
            self.decode_proc = None

    def start_decode(self):
        """Starts online2-tcp-nnet3-decode-faster process."""
        online_conf = self.model_dir / "online" / "conf" / "online.conf"
        kaldi_cmd = [
            str(_DIR / "kaldi" / "online2-tcp-nnet3-decode-faster"),
            f"--config={online_conf}",
            str(self.model_dir / "model" / "final.mdl"),
            str(self.graph_dir / "HCLG.fst"),
            str(self.graph_dir / "words.txt"),
        ]
        _LOGGER.debug(kaldi_cmd)

        self.decode_proc = subprocess.Popen(
            kaldi_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )

        # Read until started
        # TODO: Add timeout
        line = self.decode_proc.stdout.readline().lower().strip()
        if line:
            _LOGGER.debug(line)

        while "waiting for client" not in line:
            line = self.decode_proc.stdout.readline().lower().strip()
            if line:
                _LOGGER.debug(line)

        _LOGGER.debug("Decoder started")

    def __repr__(self) -> str:
        return (
            "KaldiCommandLineTranscriber("
            f"model_type={self.model_type}"
            f", model_dir={self.model_dir}"
            f", graph_dir={self.graph_dir}"
            ")"
        )


# -----------------------------------------------------------------------------


def get_wav_duration(wav_bytes: bytes) -> float:
    """Return the real-time duration of a WAV file"""
    with io.BytesIO(wav_bytes) as wav_buffer:
        wav_file: wave.Wave_read = wave.open(wav_buffer, "rb")
        with wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            return frames / float(rate)
