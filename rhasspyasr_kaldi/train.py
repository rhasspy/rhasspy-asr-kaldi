"""Methods for generating ASR artifacts."""
import logging
import os
import re
import shutil
import subprocess
import tempfile
import typing
from pathlib import Path

import networkx as nx
import rhasspynlu

PronunciationsType = typing.Dict[str, typing.List[typing.List[str]]]

_DIR = Path(__file__).parent
_LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------


class MissingWordPronunciationsException(Exception):
    """Raised when missing word pronunciations and no g2p model."""

    def __init__(self, words: typing.List[str]):
        super().__init__(self)
        self.words = words

    def __str__(self):
        return f"Missing pronunciations for: {self.words}"


# -----------------------------------------------------------------------------


def train(
    graph: nx.DiGraph,
    pronunciations: PronunciationsType,
    model_dir: typing.Union[str, Path],
    graph_dir: typing.Union[str, Path],
    dictionary: typing.Optional[typing.Union[str, Path]] = None,
    language_model: typing.Optional[typing.Union[str, Path]] = None,
    dictionary_word_transform: typing.Optional[typing.Callable[[str], str]] = None,
    g2p_model: typing.Optional[typing.Union[str, Path]] = None,
    g2p_word_transform: typing.Optional[typing.Callable[[str], str]] = None,
    missing_words_path: typing.Optional[Path] = None,
    balance_counts: bool = True,
    kaldi_dir: typing.Optional[Path] = None,
):
    """Re-generates HCLG.fst from intent graph"""
    g2p_word_transform = g2p_word_transform or (lambda s: s)

    # Convert to paths
    model_dir = Path(model_dir)
    graph_dir = Path(graph_dir)

    # Generate counts
    intent_counts = rhasspynlu.get_intent_ngram_counts(
        graph, balance_counts=balance_counts
    )

    # Use mitlm to create language model
    vocabulary: typing.Set[str] = set()

    with tempfile.NamedTemporaryFile(mode="w") as lm_file:
        # Create ngram counts
        if language_model:
            count_file = open(str(language_model) + ".counts", "w")
        else:
            count_file = typing.cast(
                typing.TextIO, tempfile.NamedTemporaryFile(mode="w")
            )

        with count_file:
            for intent_name in intent_counts:
                for ngram, count in intent_counts[intent_name].items():
                    if dictionary_word_transform:
                        ngram = [dictionary_word_transform(w) for w in ngram]

                    # word [word] ... <TAB> count
                    print(*ngram, file=count_file, end="")
                    print("\t", count, file=count_file)

            count_file.seek(0)
            with tempfile.NamedTemporaryFile(mode="w+") as vocab_file:
                estimate_ngram = shutil.which("estimate-ngram") or (
                    _DIR / "estimate-ngram"
                )
                ngram_command = [
                    str(estimate_ngram),
                    "-order",
                    "3",
                    "-counts",
                    count_file.name,
                    "-write-lm",
                    lm_file.name,
                    "-write-vocab",
                    vocab_file.name,
                ]

                _LOGGER.debug(ngram_command)
                subprocess.check_call(ngram_command)
                lm_file.seek(0)

                if language_model:
                    shutil.copy(lm_file.name, language_model)
                    _LOGGER.debug("Wrote language model to %s", str(language_model))
                else:
                    language_model = lm_file.name
                    lm_file.seek(0)

                # Extract vocabulary
                vocab_file.seek(0)
                for line in vocab_file:
                    line = line.strip()
                    if not line.startswith("<"):
                        vocabulary.add(line)

        assert vocabulary, "No words in vocabulary"

        # Write dictionary
        with tempfile.NamedTemporaryFile(mode="w") as dict_file:

            # Look up words
            missing_words: typing.Set[str] = set()

            # Look up each word
            for word in vocabulary:
                word_phonemes = pronunciations.get(word)
                if not word_phonemes:
                    # Add to missing word list
                    _LOGGER.warning("Missing word '%s'", word)
                    missing_words.add(word)
                    continue

                # Write CMU format
                for i, phonemes in enumerate(word_phonemes):
                    phoneme_str = " ".join(phonemes).strip()
                    if i == 0:
                        # word
                        print(word, phoneme_str, file=dict_file)
                    else:
                        # word(n)
                        print(f"{word}({i+1})", phoneme_str, file=dict_file)

            # Open missing words file
            missing_file: typing.Optional[typing.TextIO] = None
            if missing_words_path:
                missing_file = open(missing_words_path, "w")

            if missing_words:
                # Fail if no g2p model is available
                if not g2p_model:
                    raise MissingWordPronunciationsException(list(missing_words))

                # Guess word pronunciations
                _LOGGER.debug("Guessing pronunciations for %s", missing_words)
                guesses = guess_pronunciations(
                    missing_words,
                    g2p_model,
                    g2p_word_transform=g2p_word_transform,
                    num_guesses=1,
                )

                # Output is a pronunciation dictionary.
                # Append to existing dictionary file.
                for guess_word, guess_phonemes in guesses:
                    guess_phoneme_str = " ".join(guess_phonemes).strip()
                    print(guess_word, guess_phoneme_str, file=dict_file)

                    if missing_file:
                        print(guess_word, guess_phoneme_str, file=missing_file)

            # Close missing words file
            if missing_file:
                _LOGGER.debug("Wrote missing words to %s", str(missing_words_path))
                missing_file.close()

            # -----------------------------------------------------

            # Copy dictionary
            dict_file.seek(0)
            if dictionary:
                shutil.copy(dict_file.name, dictionary)
                _LOGGER.debug("Wrote dictionary to %s", str(dictionary))
            else:
                dictionary = dict_file.name
                dict_file.seek(0)

            # -------------------------------------------------------------------------
            # Kaldi Training
            # ---------------------------------------------------------
            # 1. prepare_lang.sh
            # 2. format_lm.sh
            # 3. mkgraph.sh
            # 4. prepare_online_decoding.sh
            # ---------------------------------------------------------

            # Determine directory with Kaldi binaries
            if kaldi_dir is None:
                # Check environment variable
                if "KALDI_DIR" in os.environ:
                    kaldi_dir = Path(os.environ["KALDI_DIR"])
                else:
                    kaldi_dir = _DIR / "kaldi"

            assert kaldi_dir is not None
            _LOGGER.debug("Using kaldi at %s", str(kaldi_dir))

            # Extend PATH
            egs_utils_dir = kaldi_dir / "egs" / "wsj" / "s5" / "utils"
            extended_env = os.environ.copy()
            extended_env["PATH"] = (
                str(kaldi_dir) + ":" + str(egs_utils_dir) + ":" + extended_env["PATH"]
            )

            # Create empty path.sh
            path_sh = model_dir / "path.sh"
            if not path_sh.is_file():
                path_sh.write_text("")

            # Delete existing data/graph
            data_dir = model_dir / "data"
            if data_dir.exists():
                shutil.rmtree(data_dir)

            if graph_dir.exists():
                shutil.rmtree(graph_dir)

            data_local_dir = model_dir / "data" / "local"

            _LOGGER.debug("Generating lexicon")
            dict_local_dir = data_local_dir / "dict"
            dict_local_dir.mkdir(parents=True, exist_ok=True)

            # Copy phones
            phones_dir = model_dir / "phones"
            for phone_file in phones_dir.glob("*.txt"):
                shutil.copy(phone_file, dict_local_dir / phone_file.name)

            # Copy dictionary
            shutil.copy(dictionary, dict_local_dir / "lexicon.txt")

            # Create utils link
            model_utils_link = model_dir / "utils"

            try:
                # Can't use missing_ok in 3.6
                model_utils_link.unlink()
            except Exception:
                pass

            model_utils_link.symlink_to(egs_utils_dir, target_is_directory=True)

            # 1. prepare_lang.sh
            lang_dir = data_dir / "lang"
            lang_local_dir = data_local_dir / "lang"
            prepare_lang = [
                "bash",
                str(egs_utils_dir / "prepare_lang.sh"),
                str(dict_local_dir),
                "",
                str(lang_local_dir),
                str(lang_dir),
            ]

            _LOGGER.debug(prepare_lang)
            subprocess.check_call(prepare_lang, cwd=model_dir, env=extended_env)

            # 2. format_lm.sh
            lm_arpa = lang_local_dir / "lm.arpa"
            lm_arpa.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(language_model, lm_arpa)

            gzip_lm = ["gzip", str(lm_arpa)]
            _LOGGER.debug(gzip_lm)
            subprocess.check_call(gzip_lm, cwd=lm_arpa.parent, env=extended_env)

            format_lm = [
                "bash",
                str(egs_utils_dir / "format_lm.sh"),
                str(lang_dir),
                str(lm_arpa.with_suffix(".arpa.gz")),
                str(dict_local_dir / "lexicon.txt"),
                str(lang_dir),
            ]

            _LOGGER.debug(format_lm)
            subprocess.check_call(format_lm, cwd=model_dir, env=extended_env)

            # 3. mkgraph.sh
            mkgraph = [
                "bash",
                str(egs_utils_dir / "mkgraph.sh"),
                str(lang_dir),
                str(model_dir / "model"),
                str(graph_dir),
            ]
            _LOGGER.debug(mkgraph)
            subprocess.check_call(mkgraph, cwd=model_dir, env=extended_env)

            # 4. prepare_online_decoding.sh
            extractor_dir = model_dir / "extractor"
            if extractor_dir.is_dir():
                # nnet3 only
                mfcc_conf = model_dir / "conf" / "mfcc_hires.conf"
                egs_steps_dir = kaldi_dir / "egs" / "wsj" / "s5" / "steps"
                prepare_online_decoding = [
                    "bash",
                    str(
                        egs_steps_dir
                        / "online"
                        / "nnet3"
                        / "prepare_online_decoding.sh"
                    ),
                    "--mfcc-config",
                    str(mfcc_conf),
                    str(lang_dir),
                    str(extractor_dir),
                    str(model_dir / "model"),
                    str(model_dir / "online"),
                ]

                _LOGGER.debug(prepare_online_decoding)
                subprocess.check_call(
                    prepare_online_decoding, cwd=model_dir, env=extended_env
                )


# -----------------------------------------------------------------------------


def guess_pronunciations(
    words: typing.Iterable[str],
    g2p_model: typing.Union[str, Path],
    g2p_word_transform: typing.Optional[typing.Callable[[str], str]] = None,
    num_guesses: int = 1,
) -> typing.Iterable[typing.Tuple[str, typing.List[str]]]:
    """Guess phonetic pronunciations for words. Yields (word, phonemes) pairs."""
    g2p_word_transform = g2p_word_transform or (lambda s: s)

    with tempfile.NamedTemporaryFile(mode="w") as wordlist_file:
        for word in words:
            word = g2p_word_transform(word)
            print(word, file=wordlist_file)

        wordlist_file.seek(0)
        phonetisaurus_apply = shutil.which("phonetisaurus-apply") or (
            _DIR / "phonetisaurus-apply"
        )
        g2p_command = [
            str(phonetisaurus_apply),
            "--model",
            str(g2p_model),
            "--word_list",
            wordlist_file.name,
            "--nbest",
            str(num_guesses),
        ]

        _LOGGER.debug(g2p_command)
        g2p_lines = subprocess.check_output(
            g2p_command, universal_newlines=True
        ).splitlines()

        # Output is a pronunciation dictionary.
        # Append to existing dictionary file.
        for line in g2p_lines:
            line = line.strip()
            if line:
                word, *phonemes = line.split()
                yield (word.strip(), phonemes)


def read_dict(
    dict_file: typing.Iterable[str],
    word_dict: typing.Optional[PronunciationsType] = None,
) -> PronunciationsType:
    """Loads a CMU pronunciation dictionary."""
    if word_dict is None:
        word_dict = {}

    for i, line in enumerate(dict_file):
        line = line.strip()
        if not line:
            continue

        try:
            # Use explicit whitespace (avoid 0xA0)
            word, *pronounce = re.split(r"[ \t]+", line)

            word = word.split("(")[0]

            if word in word_dict:
                word_dict[word].append(pronounce)
            else:
                word_dict[word] = [pronounce]
        except Exception as e:
            _LOGGER.warning("read_dict: %s (line %s)", e, i + 1)

    return word_dict
