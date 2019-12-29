"""Command-line interface to rhasspy-asr-kaldi"""
import argparse
import logging
import sys
import json
import os
import wave
from pathlib import Path

import attr

from . import KaldiExtensionTranscriber, KaldiCommandLineTranscriber

_LOGGER = logging.getLogger(__name__)


def main():
    """Main method"""
    parser = argparse.ArgumentParser("rhasspyasr_kaldi")
    parser.add_argument("wav_file", nargs="*", help="WAV file(s) to transcribe")
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to Kaldi model directory (with conf, data)",
    )
    parser.add_argument(
        "--graph-dir", help="Path to Kaldi graph directory (with HCLG.fst)"
    )
    parser.add_argument(
        "--model-type", default="nnet3", help="Either nnet3 or gmm (default: nnet3)"
    )
    parser.add_argument(
        "--no-stream", action="store_true", help="Process entire WAV file"
    )
    parser.add_argument(
        "--no-extension", action="store_true", help="Use shell scripts for Kaldi"
    )
    parser.add_argument(
        "--frames-in-chunk",
        type=int,
        default=1024,
        help="Number of frames to process at a time",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to console"
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    # Load transcriber
    args.model_dir = Path(args.model_dir)

    if args.graph_dir:
        args.graph_dir = Path(args.graph_dir)
    else:
        args.graph_dir = args.model_dir / "graph"

    if args.no_extension:
        # Use shell scripts
        transcriber = KaldiCommandLineTranscriber(
            args.model_type, args.model_dir, args.graph_dir
        )
    else:
        # Use Python extension
        transcriber = KaldiExtensionTranscriber(args.model_dir, args.graph_dir)

    if args.wav_file:
        # Transcribe WAV files
        for wav_path in args.wav_file:
            _LOGGER.debug("Processing %s", wav_path)
            wav_bytes = open(wav_path, "rb").read()
            result = transcriber.transcribe_wav(wav_bytes)
            print_json(result)
    else:
        # Read WAV data from stdin
        if os.isatty(sys.stdin.fileno()):
            print("Reading WAV data from stdin...", file=sys.stderr)

        # Stream in chunks
        with wave.open(sys.stdin.buffer, "rb") as wav_file:

            def audio_stream(wav_file, frames_in_chunk):
                num_frames = wav_file.getnframes()
                try:
                    while num_frames > frames_in_chunk:
                        yield wav_file.readframes(frames_in_chunk)
                        num_frames -= frames_in_chunk

                    if num_frames > 0:
                        # Last chunk
                        yield wav_file.readframes(num_frames)
                except KeyboardInterrupt:
                    pass

            result = transcriber.transcribe_stream(
                audio_stream(wav_file, args.frames_in_chunk),
                wav_file.getframerate(),
                wav_file.getsampwidth(),
            )

            print_json(result)


# -----------------------------------------------------------------------------


def print_json(result):
    """Print attr class as JSON"""
    json.dump(attr.asdict(result), sys.stdout)
    print("")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
