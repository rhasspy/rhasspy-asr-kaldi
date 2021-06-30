"""Methods for generating ASR artifacts."""
import logging
import math
import os
import shlex
import shutil
import subprocess
import tempfile
import typing
from enum import Enum
from pathlib import Path

import networkx as nx
import rhasspynlu
from rhasspynlu.g2p import PronunciationsType

_DIR = Path(__file__).parent
_LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


class LanguageModelType(str, Enum):
    """Type of language model used to train Kaldi."""

    ARPA = "arpa"
    TEXT_FST = "text_fst"


def get_kaldi_dir() -> Path:
    """Get directory to Kaldi installation."""
    # Check environment variable
    if "KALDI_DIR" in os.environ:
        return Path(os.environ["KALDI_DIR"])

    return _DIR / "kaldi"


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
    vocab_path: typing.Optional[typing.Union[str, Path]] = None,
    language_model_fst: typing.Optional[typing.Union[str, Path]] = None,
    base_language_model_fst: typing.Optional[typing.Union[str, Path]] = None,
    base_language_model_weight: typing.Optional[float] = None,
    mixed_language_model_fst: typing.Optional[typing.Union[str, Path]] = None,
    balance_counts: bool = True,
    kaldi_dir: typing.Optional[Path] = None,
    language_model_type: LanguageModelType = LanguageModelType.ARPA,
    spn_phone: str = "SPN",
    eps: str = "<eps>",
    unk_nonterm: str = "#nonterm:unk",
):
    """Re-generates HCLG.fst from intent graph"""
    g2p_word_transform = g2p_word_transform or (lambda s: s)

    # Determine directory with Kaldi binaries
    if kaldi_dir is None:
        kaldi_dir = get_kaldi_dir()

    assert kaldi_dir is not None
    _LOGGER.debug("Using kaldi at %s", str(kaldi_dir))

    vocabulary: typing.Set[str] = set()
    unk_vocabulary: typing.Set[str] = set()

    if vocab_path:
        vocab_file = open(vocab_path, "w+")
    else:
        vocab_file = typing.cast(
            typing.TextIO, tempfile.NamedTemporaryFile(suffix=".txt", mode="w+")
        )
        vocab_path = vocab_file.name

    # Language model mixing
    is_mixing = False
    base_fst_weight = None
    if (
        (base_language_model_fst is not None)
        and (base_language_model_weight is not None)
        and (base_language_model_weight > 0)
    ):
        is_mixing = True
        base_fst_weight = (base_language_model_fst, base_language_model_weight)

    # Begin training
    with tempfile.NamedTemporaryFile(mode="w+") as lm_file:
        with vocab_file:
            if language_model_type == LanguageModelType.TEXT_FST:
                _LOGGER.debug("Writing G.fst directly")
                graph_to_g_fst(
                    graph, lm_file, vocab_file, eps=eps, unk_nonterm=unk_nonterm
                )
            else:
                # Create language model from ARPA
                _LOGGER.debug("Converting to ARPA language model")
                rhasspynlu.arpa_lm.graph_to_arpa(
                    graph,
                    lm_file.name,
                    vocab_path=vocab_path,
                    model_path=language_model_fst,
                    base_fst_weight=base_fst_weight,
                    merge_path=mixed_language_model_fst,
                )

            # Load vocabulary
            vocab_file.seek(0)
            vocabulary.update(line.strip() for line in vocab_file)

            if is_mixing or (language_model_type == LanguageModelType.TEXT_FST):
                # Add all known words
                # TODO: Use full vocab?
                frequent_words = (
                    line.strip()
                    for line in (model_dir.parent / "frequent_words.txt")
                    .read_text()
                    .splitlines()
                )
                all_vocabulary = set(pronunciations.keys())
                unk_vocabulary = (
                    set(frequent_words).intersection(all_vocabulary) - vocabulary
                )
                # unk_vocabulary = set(list(pronunciations.keys())[:1000]) - vocabulary
                vocabulary.update(unk_vocabulary)

        assert vocabulary, "No words in vocabulary"

        # <unk>
        vocabulary.add("<unk>")
        pronunciations["<unk>"] = [[spn_phone]]

        # Write dictionary to temporary file
        with tempfile.NamedTemporaryFile(mode="w+") as dictionary_file:
            _LOGGER.debug("Writing pronunciation dictionary")
            rhasspynlu.g2p.write_pronunciations(
                vocabulary,
                pronunciations,
                dictionary_file.name,
                g2p_model=g2p_model,
                g2p_word_transform=g2p_word_transform,
                missing_words_path=missing_words_path,
            )

            # -----------------------------------------------------------------

            dictionary_file.seek(0)
            if dictionary:
                # Copy dictionary over real file
                shutil.copy(dictionary_file.name, dictionary)
                _LOGGER.debug("Wrote dictionary to %s", str(dictionary))
            else:
                dictionary = Path(dictionary_file.name)
                dictionary_file.seek(0)

            lm_file.seek(0)
            if language_model:
                # Copy language model over real file
                shutil.copy(lm_file.name, language_model)
                _LOGGER.debug("Wrote language model to %s", str(language_model))
            else:
                language_model = Path(lm_file.name)
                lm_file.seek(0)

            # Generate HCLG.fst
            train_kaldi(
                model_dir,
                graph_dir,
                dictionary,
                language_model,
                kaldi_dir=kaldi_dir,
                language_model_type=language_model_type,
                eps=eps,
                unk_nonterm=unk_nonterm,
                unk_vocabulary=unk_vocabulary,
            )


# -----------------------------------------------------------------------------


def graph_to_g_fst(
    graph: nx.DiGraph,
    fst_file: typing.IO[str],
    vocab_file: typing.IO[str],
    eps: str = "<eps>",
    unk_nonterm: str = "#nonterm:unk",
    unk_prob: float = 1e-10,
):
    """
    Write G.fst text file using intent graph.

    Compiled later on with fstcompile.
    """
    vocabulary: typing.Set[str] = set()

    unk_log_prob: typing.Optional[float] = None

    if unk_prob <= 0:
        known_log_prob = 0.0
    else:
        unk_log_prob = -math.log(unk_prob)
        known_log_prob = -math.log(1.0 - unk_prob)

    n_data = graph.nodes(data=True)
    final_states: typing.Set[int] = set()
    state_map: typing.Dict[int, int] = {}

    # start state
    start_node: int = next(n for n, data in n_data if data.get("start"))

    # Transitions
    for _, intent_node in graph.edges(start_node):
        # Map states starting from 0
        from_state = state_map.get(start_node, len(state_map))
        state_map[start_node] = from_state

        to_state = state_map.get(intent_node, len(state_map))
        state_map[intent_node] = to_state

        print(from_state, to_state, eps, eps, 0.0, file=fst_file)

        # Add intent sub-graphs
        for edge in nx.edge_bfs(graph, intent_node):
            edge_data = graph.edges[edge]
            from_node, to_node = edge

            # Get input/output labels.
            # Empty string indicates epsilon transition (eps)
            ilabel = edge_data.get("ilabel", "") or eps

            # Check for whitespace
            assert (
                " " not in ilabel
            ), f"Input symbol cannot contain whitespace: {ilabel}"

            if ilabel != eps:
                vocabulary.add(ilabel)

            # Map states starting from 0
            from_state = state_map.get(from_node, len(state_map))
            state_map[from_node] = from_state

            to_state = state_map.get(to_node, len(state_map))
            state_map[to_node] = to_state

            print(from_state, to_state, ilabel, ilabel, known_log_prob, file=fst_file)

            # Unknown transition
            if (unk_log_prob is not None) and (ilabel != eps):
                print(
                    from_state, to_state, unk_nonterm, eps, unk_log_prob, file=fst_file
                )

            # Check if final state
            if n_data[from_node].get("final", False):
                final_states.add(from_state)

            if n_data[to_node].get("final", False):
                final_states.add(to_state)

    # Record final states
    for final_state in final_states:
        print(final_state, 0.0, file=fst_file)

    # Write vocabulary
    for word in vocabulary:
        print(word, file=vocab_file)


# -----------------------------------------------------------------------------


def train_kaldi(
    model_dir: typing.Union[str, Path],
    graph_dir: typing.Union[str, Path],
    dictionary: typing.Union[str, Path],
    language_model: typing.Union[str, Path],
    kaldi_dir: typing.Union[str, Path],
    language_model_type: LanguageModelType = LanguageModelType.ARPA,
    eps: str = "<eps>",
    unk_nonterm: str = "#nonterm:unk",
    unk_vocabulary: typing.Optional[typing.Set[str]] = None,
    unk_begin: str = "<unk_begin>",
    unk_end: str = "<unk_end>",
):
    """Generates HCLG.fst from dictionary and language model."""
    unk_vocabulary = unk_vocabulary or set()

    # Convert to paths
    model_dir = Path(model_dir)
    graph_dir = Path(graph_dir)
    kaldi_dir = Path(kaldi_dir)

    # -------------------------------------------------------------------------
    # Kaldi Training
    # ---------------------------------------------------------
    # 1. prepare_lang.sh
    # 2. format_lm.sh (or fstcompile)
    # 3. mkgraph.sh
    # 4. prepare_online_decoding.sh
    # ---------------------------------------------------------

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

    with open(dict_local_dir / "lexicon.txt", "a") as lexicon_file:
        for symbol in [unk_begin, unk_end]:
            print(symbol, "SIL", file=lexicon_file)

    # Add non-terminals
    with open(dict_local_dir / "nonterminals.txt", "w") as nonterm_file:
        print(unk_nonterm, file=nonterm_file)

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
        "<unk>",
        str(lang_local_dir),
        str(lang_dir),
    ]

    _LOGGER.debug(prepare_lang)
    subprocess.check_call(prepare_lang, cwd=model_dir, env=extended_env)

    if language_model_type == LanguageModelType.TEXT_FST:
        # 2. fstcompile > G.fst
        compile_grammar = [
            "fstcompile",
            shlex.quote(f"--isymbols={lang_dir}/words.txt"),
            shlex.quote(f"--osymbols={lang_dir}/words.txt"),
            "--keep_isymbols=false",
            "--keep_osymbols=false",
            shlex.quote(str(language_model)),
            shlex.quote(str(lang_dir / "G.fst.unsorted")),
        ]

        _LOGGER.debug(compile_grammar)
        subprocess.check_call(compile_grammar, cwd=model_dir, env=extended_env)

        arcsort = [
            "fstarcsort",
            "--sort_type=ilabel",
            shlex.quote(str(lang_dir / "G.fst.unsorted")),
            shlex.quote(str(lang_dir / "G.fst")),
        ]

        _LOGGER.debug(arcsort)
        subprocess.check_call(arcsort, cwd=model_dir, env=extended_env)

        dict_unk_dir = data_local_dir / "dict_unk"
        shutil.copytree(dict_local_dir, dict_unk_dir)

        # Create #nonterm:unk FST
        lang_unk_dir = data_dir / "lang_unk"
        lang_temp_dir = data_dir / "lang_tmp"
        prepare_lang_unk = [
            "bash",
            str(egs_utils_dir / "prepare_lang.sh"),
            str(dict_unk_dir),
            "<unk>",
            str(lang_temp_dir),
            str(lang_unk_dir),
        ]

        _LOGGER.debug(prepare_lang_unk)
        subprocess.check_call(prepare_lang_unk, cwd=model_dir, env=extended_env)

        lang_unk_fst = lang_unk_dir / "G.fst.txt"
        with open(lang_unk_fst, "w") as unk_fst_file:
            print("0", "1", "#nonterm_begin", "<unk_begin>", 0.0, file=unk_fst_file)
            print("2", "3", "#nonterm_end", "<unk_end>", 0.0, file=unk_fst_file)

            state = 4
            for word in unk_vocabulary:
                print("1", state, word, word, 0.0, file=unk_fst_file)
                print(state, "2", eps, eps, 0.0, file=unk_fst_file)

            print("3", 0.0, file=unk_fst_file)

        compile_unk_grammar = [
            "fstcompile",
            shlex.quote(f"--isymbols={lang_unk_dir}/words.txt"),
            shlex.quote(f"--osymbols={lang_unk_dir}/words.txt"),
            "--keep_isymbols=false",
            "--keep_osymbols=false",
            shlex.quote(str(lang_unk_fst)),
            shlex.quote(str(lang_unk_dir / "G.fst.unsorted")),
        ]

        _LOGGER.debug(compile_unk_grammar)
        subprocess.check_call(compile_unk_grammar, cwd=model_dir, env=extended_env)

        arcsort = [
            "fstarcsort",
            "--sort_type=ilabel",
            shlex.quote(str(lang_unk_dir / "G.fst.unsorted")),
            shlex.quote(str(lang_unk_dir / "G.fst")),
        ]

        _LOGGER.debug(arcsort)
        subprocess.check_call(arcsort, cwd=model_dir, env=extended_env)

        mkgraph_unk = [
            "bash",
            str(egs_utils_dir / "mkgraph.sh"),
            "--self-loop-scale",
            "1.0",
            str(lang_unk_dir),
            str(model_dir / "model"),
            str(graph_dir / "unk"),
        ]
        _LOGGER.debug(mkgraph_unk)
        subprocess.check_call(mkgraph_unk, cwd=model_dir, env=extended_env)
    else:
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
        "--self-loop-scale",
        "1.0",
        str(lang_dir),
        str(model_dir / "model"),
        str(graph_dir / "non_unk"),
    ]
    _LOGGER.debug(mkgraph)
    subprocess.check_call(mkgraph, cwd=model_dir, env=extended_env)

    # DEBUG
    nonterm_offset = -1
    nonterm_clist_offset = -1

    with open(lang_dir / "phones.txt", "r") as phones_file:
        for line in phones_file:
            line = line.strip()
            if not line:
                continue

            phone, phone_num = line.split(maxsplit=1)
            if phone == "#nonterm_bos":
                nonterm_offset = int(phone_num)
            elif phone == unk_nonterm:
                nonterm_clist_offset = int(phone_num)

    assert nonterm_offset >= 0
    assert nonterm_clist_offset >= 0

    make_grammar_fst = [
        "make-grammar-fst",
        f"--nonterm-phones-offset={nonterm_offset}",
        "--write-as-grammar=false",
        str(graph_dir / "non_unk" / "HCLG.fst"),
        str(nonterm_clist_offset),
        str(graph_dir / "unk" / "HCLG.fst"),
        str(graph_dir / "HCLG.fst"),
    ]
    _LOGGER.debug(make_grammar_fst)
    subprocess.check_call(make_grammar_fst, cwd=model_dir, env=extended_env)

    for graph_path in (graph_dir / "non_unk").iterdir():
        if graph_path.name == "HCLG.fst":
            continue

        if graph_path.is_dir():
            shutil.copytree(graph_path, graph_dir / graph_path.name)
        elif graph_path.is_file():
            shutil.copy(graph_path, graph_dir)

    # 4. prepare_online_decoding.sh
    train_prepare_online_decoding(model_dir, lang_dir, kaldi_dir)


def train_prepare_online_decoding(
    model_dir: typing.Union[str, Path],
    lang_dir: typing.Union[str, Path],
    kaldi_dir: typing.Union[str, Path],
):
    """Prepare model for online decoding."""
    model_dir = Path(model_dir)
    kaldi_dir = Path(kaldi_dir)

    # prepare_online_decoding.sh (nnet3 only)
    extractor_dir = model_dir / "extractor"
    if extractor_dir.is_dir():
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

        # Create utils link
        model_utils_link = model_dir / "utils"

        try:
            # Can't use missing_ok in 3.6
            model_utils_link.unlink()
        except Exception:
            pass

        model_utils_link.symlink_to(egs_utils_dir, target_is_directory=True)

        # Generate online.conf
        mfcc_conf = model_dir / "conf" / "mfcc_hires.conf"
        egs_steps_dir = kaldi_dir / "egs" / "wsj" / "s5" / "steps"
        prepare_online_decoding = [
            "bash",
            str(egs_steps_dir / "online" / "nnet3" / "prepare_online_decoding.sh"),
            "--mfcc-config",
            str(mfcc_conf),
            str(lang_dir),
            str(extractor_dir),
            str(model_dir / "model"),
            str(model_dir / "online"),
        ]

        _LOGGER.debug(prepare_online_decoding)
        subprocess.run(
            prepare_online_decoding,
            cwd=model_dir,
            env=extended_env,
            stderr=subprocess.STDOUT,
            check=True,
        )


# -----------------------------------------------------------------------------
