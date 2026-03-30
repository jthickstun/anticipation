"""Pytest common fixtures and utility functions for test suite."""

import sys
import tempfile
import importlib
from pathlib import Path
from typing import Callable, Any

import pytest

from anticipation.convert import midi_to_compound

# python itself has a top-level module named `tokenize`
from anticipation.tokenize import tokenize as anticipation_tokenize
from anticipation.v2.config import AnticipationV2Settings, Vocab, make_vocab

TestConfigPatcher = Callable[..., None]

TESTS_ROOT = Path(__file__).parent
TEST_DATA_PATH = TESTS_ROOT / "test_data"
VISUALIZATIONS_PATH = TESTS_ROOT / "visualizations"

# make this visualizations dir
VISUALIZATIONS_PATH.mkdir(exist_ok=True)


@pytest.fixture
def c_major_midi_path() -> Path:
    # deliberately simple MIDI file for diagnostic purposes
    # this is a C major scale starting at C1, with a new note every
    # 50 ticks. The C of the next octave is omitted, so it goes:
    # C1, D1, E1, F1, G1, A1, B1, C1, (rest for 50 ticks), D2, E2, ...
    # up until C5.
    # There are 29 notes in total.
    return TEST_DATA_PATH / "cmajor.mid"


@pytest.fixture
def simple_two_instrument_midi_path() -> Path:
    return TEST_DATA_PATH / "simple_2_instrument.mid"


@pytest.fixture
def dense_drums_sparse_piano_midi_path() -> Path:
    return TEST_DATA_PATH / "dense_drums_sparse_piano.mid"


@pytest.fixture
def lmd_0_example_1_midi_path() -> Path:
    # from The Lakh MIDI Dataset v0.1 (https://colinraffel.com/projects/lmd/)
    # from split 0, first file in name order
    return TEST_DATA_PATH / "0a0a2b0e4d3b7bf4c5383ba025c4683e.mid"


@pytest.fixture
def lmd_0_example_2_midi_path() -> Path:
    # from The Lakh MIDI Dataset v0.1 (https://colinraffel.com/projects/lmd/)
    # from split 0
    # this file is over 25 minutes long (??? !!!)
    # there are ~8 distinct movements in this file... several places where
    # nothing is playing
    return TEST_DATA_PATH / "08c8b965fd94c13611e26ba787e26d7f.mid"


@pytest.fixture
def lmd_0_example_3_midi_path() -> Path:
    # from The Lakh MIDI Dataset v0.1 (https://colinraffel.com/projects/lmd/)
    return TEST_DATA_PATH / "0c6b53ce52783ec7414b1fc7ce5c0286.mid"


@pytest.fixture
def lmd_0_example_4_midi_path() -> Path:
    # from The Lakh MIDI Dataset v0.1 (https://colinraffel.com/projects/lmd/)
    return TEST_DATA_PATH / "0283c50694655978acc97928705e3075.mid"


@pytest.fixture
def lmd_1_example_0_midi_path() -> Path:
    # from The Lakh MIDI Dataset v0.1 (https://colinraffel.com/projects/lmd/)
    # from split 1
    # this file is special because it can cause a segfault on MiniMidi/symusic.
    # we use it to test our patch that fixes that
    return TEST_DATA_PATH / "1a59d7118a473e7e093624f446bf3dbd.mid"


@pytest.fixture(scope="function")
def patch_config_and_reload(monkeypatch: pytest.MonkeyPatch) -> TestConfigPatcher:
    """
    Pytest fixture for changing the values of:
    - anticipation.config variables during test time

    Any code that imports those values must be reloaded, otherwise the
    changes do not take effect.
    """
    import anticipation.tokenize as anticipation_tokenize_cl
    import anticipation.vocab as anticipation_vocab
    import anticipation.ops as anticipation_ops
    import tests.util as test_utils
    from anticipation import config as anticipation_config

    def _apply(**kwargs):
        for k, v in kwargs.items():
            monkeypatch.setattr(anticipation_config, k, v)

        # reload everything that depends on config but not config itself
        # this propagates our test-time changes throughout the code
        importlib.reload(anticipation_vocab)
        importlib.reload(anticipation_tokenize_cl)
        importlib.reload(anticipation_ops)
        importlib.reload(test_utils)

    yield _apply

    # undo any changes done to the config by reloading it (and things
    # that depend on it) during teardown
    importlib.reload(anticipation_config)
    importlib.reload(anticipation_vocab)
    importlib.reload(anticipation_tokenize_cl)
    importlib.reload(anticipation_ops)
    importlib.reload(test_utils)


def _parse_midi_tokenized_text(midi_token_string: str) -> list[list[int]]:
    sequences = midi_token_string.strip().split("\n")
    return [list(map(int, x.split(" "))) for x in sequences]


def get_tokens_from_midi_file_v1(
    midi_file_paths: list[Path],
    augment_factor: int = 10,
    return_original_compound: bool = False,
    include_original: bool = True,
    do_span_augmentation: bool = True,
    do_random_augmentation: bool = True,
    do_instrument_augmentation: bool = True,
) -> dict[str, Any]:
    # Text -> Token
    with tempfile.TemporaryDirectory() as td:
        td_enclosing = Path(td)
        split = "0"

        to_iter = []
        for midi_file_path in midi_file_paths:
            assert midi_file_path.exists()
            assert midi_file_path.is_file()

            # MIDI -> Text
            midi_preprocess_token_list: list[int] = midi_to_compound(
                str(midi_file_path.absolute())
            )
            midi_preprocess_text = " ".join(
                str(tok) for tok in midi_preprocess_token_list
            )
            midi_preprocess_text_fname = td_enclosing / (
                midi_file_path.stem + ".mid.compound.txt"
            )
            midi_preprocess_text_fname.write_text(midi_preprocess_text)
            to_iter.append(midi_preprocess_text_fname)

        output_fname = td_enclosing / f"tokenized-events-{split}.txt"

        # seqcount: number of total tokens in the sequences
        # rest_count: number of total rests
        # all_truncations: number of times a sequence exceeds maximum duration:
        # `truncations = sum([1 for tok in tokens[1::3] if tok >= MAX_DUR])`
        (
            seqcount,
            rest_count,
            num_too_short,
            num_too_long,
            num_too_many_instruments,
            num_inexpressible,
            all_truncations,
        ) = anticipation_tokenize(
            to_iter,
            output_fname,
            # 1 = standard AR training
            # 10 = lowest acceptable anticipation augment factor
            augment_factor=augment_factor,
            include_original=include_original,
            do_span_augmentation=do_span_augmentation,
            do_random_augmentation=do_random_augmentation,
            do_instrument_augmentation=do_instrument_augmentation,
        )
        tokens: str = Path(output_fname).read_text()
        if not tokens:
            raise ValueError("Nothing was written.")

        parsed_tokens = _parse_midi_tokenized_text(tokens)
        parse_info = {
            "tokens": parsed_tokens,
            "seqcount": seqcount,
            "rest_count": rest_count,
            "num_too_short": num_too_short,
            "num_too_long": num_too_long,
            "num_too_many_instruments": num_too_many_instruments,
            "num_inexpressible": num_inexpressible,
            "all_truncations": all_truncations,
            "compound": None,
        }
        if return_original_compound:
            parse_info["compound"] = midi_preprocess_token_list

        return parse_info


def get_tokens_from_text_file(tokens_file: Path) -> list[list[int]]:
    tokens: str = Path(tokens_file).read_text()
    return _parse_midi_tokenized_text(tokens)


@pytest.fixture
def local_midi_vocab() -> Vocab:
    return make_vocab(
        tick_token_every_n_ticks=100,
        max_note_duration_in_seconds=10,
        time_resolution=100,
    )


@pytest.fixture
def local_midi_settings_ar_only(local_midi_vocab: Vocab) -> AnticipationV2Settings:
    return AnticipationV2Settings(
        num_autoregressive_seq_per_midi_file=1,
        num_span_anticipation_augmentations_per_midi_file=0,
        num_instrument_anticipation_augmentations_per_midi_file=0,
        vocab=local_midi_vocab,
        tick_token_every_n_ticks=100,
        num_workers_in_dataset_construction=10,
        # toggle for cap on instruments
        # max_track_instruments=129,
        # augmentation_pitch_shifts=(-5, -3, -2, -1, 1, 2, 3, 4, 5)
    )


@pytest.fixture
def local_midi_settings_anticipation(local_midi_vocab: Vocab) -> AnticipationV2Settings:
    return AnticipationV2Settings(
        num_autoregressive_seq_per_midi_file=1,
        num_span_anticipation_augmentations_per_midi_file=1,
        num_instrument_anticipation_augmentations_per_midi_file=1,
        vocab=local_midi_vocab,
        tick_token_every_n_ticks=100,
        num_workers_in_dataset_construction=10,
        do_clip_overlapping_durations_in_midi_conversion=False,
    )


@pytest.fixture
def local_midi_settings_anticipation_ctx_4096(
    local_midi_vocab: Vocab,
) -> AnticipationV2Settings:
    return AnticipationV2Settings(
        num_autoregressive_seq_per_midi_file=1,
        num_span_anticipation_augmentations_per_midi_file=1,
        num_instrument_anticipation_augmentations_per_midi_file=1,
        vocab=local_midi_vocab,
        tick_token_every_n_ticks=100,
        num_workers_in_dataset_construction=10,
        do_clip_overlapping_durations_in_midi_conversion=False,
        context_size=4096,
    )


def get_current_function_name() -> str:
    return sys._getframe(1).f_code.co_name  # noqa


def save_tokens_as_file(tokens: list[list[int]], save_to: Path) -> None:
    assert len(tokens) >= 1
    assert isinstance(tokens, list)
    assert isinstance(tokens[0], list)
    t = ""
    for token_seq in tokens:
        t += ",".join(map(str, token_seq)) + "\n"
    save_to.write_text(t.strip())


def get_tokens_from_file(from_path: Path) -> list[list[int]]:
    assert from_path.is_file()
    assert from_path.exists()
    t = from_path.read_text().split("\n")
    parsed = []
    for s in t:
        parsed.append([int(x) for x in s.split(",")])
    return parsed
