"""Automated speech recognition in Rhasspy using Kaldi."""
import io
import logging
import socket
import subprocess
import tempfile
import time
import typing
import wave
from enum import Enum
from pathlib import Path

from rhasspyasr import Transcriber, Transcription

_DIR = Path(__file__).parent
_LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


class KaldiModelType(str, Enum):
    """Supported Kaldi model types."""

    NNET3 = "nnet3"
    GMM = "gmm"


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

        with tempfile.NamedTemporaryFile(suffix=".wav", mode="wb") as wav_file:
            wav_file.write(wav_data)
            wav_file.seek(0)

            if self.model_type == KaldiModelType.NNET3:
                text = self._transcribe_wav_nnet3(wav_file.name)
            elif self.model_type == KaldiModelType.GMM:
                text = self._transcribe_wav_gmm(wav_file.name)
            else:
                raise ValueError(self.model_type)

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

    def _transcribe_wav_nnet3(self, wav_path: str) -> str:
        words_txt = self.graph_dir / "words.txt"
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
            f"scp:echo utt1 {wav_path}|",
            "ark:/dev/null",
        ]
        _LOGGER.debug(kaldi_cmd)

        try:
            lines = subprocess.check_output(
                kaldi_cmd, stderr=subprocess.STDOUT, universal_newlines=True
            ).splitlines()
        except subprocess.CalledProcessError as e:
            _LOGGER.exception("_transcribe_wav_nnet3")
            _LOGGER.error(e.output)
            lines = []

        text = ""
        for line in lines:
            if line.startswith("utt1 "):
                text = line.split(maxsplit=1)[1]
                break

        return text

    def _transcribe_wav_gmm(self, wav_path: str) -> str:
        # GMM decoding steps:
        # 1. compute-mfcc-feats
        # 2. compute-cmvn-stats
        # 3. apply-cmvn
        # 4. add-deltas
        # 5. gmm-latgen-faster
        with tempfile.TemporaryDirectory() as temp_dir:
            words_txt = self.graph_dir / "words.txt"
            mfcc_conf = self.model_dir / "conf" / "mfcc.conf"

            # 1. compute-mfcc-feats
            feats_cmd = [
                str(_DIR / "kaldi" / "compute-mfcc-feats"),
                f"--config={mfcc_conf}",
                f"scp:echo utt1 {wav_path}|",
                f"ark,scp:{temp_dir}/feats.ark,{temp_dir}/feats.scp",
            ]
            _LOGGER.debug(feats_cmd)
            subprocess.check_call(feats_cmd)

            # 2. compute-cmvn-stats
            stats_cmd = [
                str(_DIR / "kaldi" / "compute-cmvn-stats"),
                f"scp:{temp_dir}/feats.scp",
                f"ark,scp:{temp_dir}/cmvn.ark,{temp_dir}/cmvn.scp",
            ]
            _LOGGER.debug(stats_cmd)
            subprocess.check_call(stats_cmd)

            # 3. apply-cmvn
            norm_cmd = [
                str(_DIR / "kaldi" / "apply-cmvn"),
                f"scp:{temp_dir}/cmvn.scp",
                f"scp:{temp_dir}/feats.scp",
                f"ark,scp:{temp_dir}/feats_cmvn.ark,{temp_dir}/feats_cmvn.scp",
            ]
            _LOGGER.debug(norm_cmd)
            subprocess.check_call(norm_cmd)

            # 4. add-deltas
            delta_cmd = [
                str(_DIR / "kaldi" / "add-deltas"),
                f"scp:{temp_dir}/feats_cmvn.scp",
                f"ark,scp:{temp_dir}/deltas.ark,{temp_dir}/deltas.scp",
            ]
            _LOGGER.debug(delta_cmd)
            subprocess.check_call(delta_cmd)

            # 5. decode
            decode_cmd = [
                str(_DIR / "kaldi" / "gmm-latgen-faster"),
                f"--word-symbol-table={words_txt}",
                f"{self.model_dir}/model/final.mdl",
                f"{self.graph_dir}/HCLG.fst",
                f"scp:{temp_dir}/deltas.scp",
                f"ark,scp:{temp_dir}/lattices.ark,{temp_dir}/lattices.scp",
            ]
            _LOGGER.debug(decode_cmd)
            subprocess.check_call(decode_cmd)

            try:
                lines = subprocess.check_output(
                    decode_cmd, stderr=subprocess.STDOUT, universal_newlines=True
                ).splitlines()
            except subprocess.CalledProcessError as e:
                _LOGGER.exception("_transcribe_wav_gmm")
                _LOGGER.error(e.output)
                lines = []

            print(lines)
            text = ""
            for line in lines:
                if line.startswith("utt1 "):
                    text = line.split(maxsplit=1)[1]
                    break

            return text

    # -------------------------------------------------------------------------

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

        if self.model_type == KaldiModelType.GMM:
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
