"""Command-line interface to rhasspy-asr-kaldi"""
import argparse
import dataclasses
import json
import logging
import os
import sys
import typing
import wave
from pathlib import Path

import rhasspynlu
from rhasspynlu.g2p import PronunciationsType

from . import KaldiCommandLineTranscriber
from . import train as kaldi_train

_LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


def main():
    """Main method."""
    args = get_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    # Dispatch to appropriate sub-command
    args.func(args)


# -----------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog="rhasspy-asr-kaldi")
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )

    # Create subparsers for each sub-command
    sub_parsers = parser.add_subparsers()
    sub_parsers.required = True
    sub_parsers.dest = "command"

    # -------------------------------------------------------------------------

    # Transcribe settings
    transcribe_parser = sub_parsers.add_parser(
        "transcribe", help="Do speech to text on one or more WAV files"
    )
    transcribe_parser.set_defaults(func=transcribe)
    transcribe_parser.add_argument(
        "wav_file", nargs="*", help="WAV file(s) to transcribe"
    )
    transcribe_parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to Kaldi model directory (with conf, data)",
    )
    transcribe_parser.add_argument(
        "--graph-dir", help="Path to Kaldi graph directory (with HCLG.fst)"
    )
    transcribe_parser.add_argument(
        "--model-type", default="nnet3", help="Either nnet3 or gmm (default: nnet3)"
    )
    transcribe_parser.add_argument(
        "--frames-in-chunk",
        type=int,
        default=1024,
        help="Number of frames to process at a time",
    )

    # -------------------------------------------------------------------------

    # Train settings
    train_parser = sub_parsers.add_parser(
        "train", help="Generate HCLG.fst from intent graph"
    )
    train_parser.set_defaults(func=train)
    train_parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to Kaldi model directory (with conf, data)",
    )
    train_parser.add_argument(
        "--graph-dir", help="Path to Kaldi graph directory (with HCLG.fst)"
    )
    train_parser.add_argument(
        "--intent-graph", help="Path to intent graph JSON file (default: stdin)"
    )
    train_parser.add_argument(
        "--no-intent-graph",
        action="store_true",
        help="Use language model and dictionaries to generate HCLG.fst",
    )
    train_parser.add_argument(
        "--dictionary", help="Path to write custom pronunciation dictionary"
    )
    train_parser.add_argument(
        "--dictionary-casing",
        choices=["upper", "lower", "ignore"],
        default="ignore",
        help="Case transformation for dictionary words (training, default: ignore)",
    )
    train_parser.add_argument(
        "--language-model", help="Path to write custom language model"
    )
    train_parser.add_argument(
        "--base-dictionary", action="append", help="Paths to pronunciation dictionaries"
    )
    train_parser.add_argument(
        "--g2p-model", help="Path to Phonetisaurus grapheme-to-phoneme FST model"
    )
    train_parser.add_argument(
        "--g2p-casing",
        choices=["upper", "lower", "ignore"],
        default="ignore",
        help="Case transformation for g2p words (training, default: ignore)",
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------


def transcribe(args: argparse.Namespace):
    """Do speech to text on one more WAV files."""
    # Load transcriber
    args.model_dir = Path(args.model_dir)

    if args.graph_dir:
        args.graph_dir = Path(args.graph_dir)
    else:
        args.graph_dir = args.model_dir / "graph"

    transcriber = KaldiCommandLineTranscriber(
        args.model_type, args.model_dir, args.graph_dir
    )

    # Do transcription
    try:
        if args.wav_file:
            # Transcribe WAV files
            for wav_path in args.wav_file:
                _LOGGER.debug("Processing %s", wav_path)
                wav_bytes = open(wav_path, "rb").read()
                result = transcriber.transcribe_wav(wav_bytes)

                assert result
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
                    wav_file.getnchannels(),
                )

                assert result
                print_json(result)
    except KeyboardInterrupt:
        pass
    finally:
        transcriber.stop()


# -----------------------------------------------------------------------------


def train(args: argparse.Namespace):
    """Generate HCLG.fst from intent graph."""
    # Convert to Paths
    args.model_dir = Path(args.model_dir)

    if args.graph_dir:
        args.graph_dir = Path(args.graph_dir)
    else:
        args.graph_dir = args.model_dir / "graph"

    if args.dictionary:
        args.dictionary = Path(args.dictionary)

    if args.language_model:
        args.language_model = Path(args.language_model)

    if args.g2p_model:
        args.g2p_model = Path(args.g2p_model)

    if args.base_dictionary:
        args.base_dictionary = [Path(p) for p in args.base_dictionary]
    else:
        args.base_dictionary = []

    graph_dict: typing.Optional[typing.Dict[str, typing.Any]] = None
    if args.intent_graph:
        # Load graph from file
        args.intent_graph = Path(args.intent_graph)

        _LOGGER.debug("Loading intent graph from %s", args.intent_graph)
        with open(args.intent_graph, "r") as graph_file:
            graph_dict = json.load(graph_file)
    elif not args.no_intent_graph:
        # Load graph from stdin
        if os.isatty(sys.stdin.fileno()):
            print("Reading intent graph from stdin...", file=sys.stderr)

        graph_dict = json.load(sys.stdin)

    # Load base dictionaries
    pronunciations: PronunciationsType = {}
    for dict_path in args.base_dictionary:
        if os.path.exists(dict_path):
            _LOGGER.debug("Loading dictionary %s", str(dict_path))
            rhasspynlu.g2p.read_pronunciations(dict_path, pronunciations)

    kaldi_train(
        graph_dict,
        pronunciations,
        args.model_dir,
        args.graph_dir,
        dictionary_word_transform=get_word_transform(args.dictionary_casing),
        dictionary=args.dictionary,
        language_model=args.language_model,
        g2p_model=args.g2p_model,
        g2p_word_transform=get_word_transform(args.g2p_casing),
    )


def get_word_transform(name: str) -> typing.Callable[[str], str]:
    """Gets a word transformation function by name."""
    if name == "upper":
        return str.upper

    if name == "lower":
        return str.lower

    return lambda s: s


# -----------------------------------------------------------------------------


def print_json(result):
    """Print data class as JSON"""
    json.dump(dataclasses.asdict(result), sys.stdout, ensure_ascii=False)
    print("")
    sys.stdout.flush()


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
