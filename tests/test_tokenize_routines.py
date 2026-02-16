from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from anticipation import config
from anticipation.tokenize import maybe_tokenize, extract_spans
import anticipation.ops as ops
import anticipation.vocab as v
from anticipation.convert import midi_to_compound
from anticipation.v2.config import AnticipationV2Settings, Vocab

from tests.conftest import (
    get_tokens_from_midi_file_v1,
    TestConfigPatcher,
    patch_config_and_reload,
    VISUALIZATIONS_PATH,
)
from tests.util.visualize_sequence import get_figure_and_open
from tests.util.entities import Note, Event


@pytest.fixture()
def v1_default_settings() -> AnticipationV2Settings:
    # I've written this so all the defaults are the same as what they were in v1
    # this object mostly does nothing in the context of these test cases
    # because the code uses what is written in `anticipation.config`. The fixture
    # `patch_config_and_reload` can edit those at test time.
    vocab = Vocab()
    return AnticipationV2Settings(vocab=vocab)


def test_note_parsing() -> None:
    # int -> str
    assert str(Note.make(0)) == "C-2"
    assert str(Note.make(12)) == "C-1"
    assert str(Note.make(24)) == "C0"
    assert str(Note.make(36)) == "C1"
    assert str(Note.make(60)) == "C3"  # middle C
    assert str(Note.make(84)) == "C5"
    assert str(Note.make(127)) == "G8"

    # str -> int
    assert int(Note.make("C-2")) == 0
    assert int(Note.make("C-1")) == 12
    assert int(Note.make("C0")) == 24
    assert int(Note.make("C1")) == 36
    assert int(Note.make("C3")) == 60
    assert int(Note.make("C5")) == 84
    assert int(Note.make("G8")) == 127


def test_parse_event_instances(v1_default_settings: AnticipationV2Settings) -> None:
    token_seq = [0, 10050, 11036, 50, 10050, 11038, 100, 10050, 11040]
    event_seq = Event.from_token_seq(token_seq, v1_default_settings)
    assert event_seq[0] == Event.from_midi_values(
        0, 50, 0, "C1", original_idx_in_token_seq=0
    )
    assert event_seq[1] == Event.from_midi_values(
        50, 50, 0, "D1", original_idx_in_token_seq=3
    )
    assert event_seq[2] == Event.from_midi_values(
        100, 50, 0, "E1", original_idx_in_token_seq=6
    )


def test_patch_config_at_test_time(patch_config_and_reload: TestConfigPatcher) -> None:
    # this is an edit that makes no sense, we wouldn't set this to 0
    patch_config_and_reload(COMPOUND_SIZE=0)
    assert config.COMPOUND_SIZE == 0


def test_config_patch_rolled_back() -> None:
    # test that changes from test patching is isolated
    assert config.COMPOUND_SIZE == 5


def test_extract_span_and_anticipate_for_small_sequence(
    c_major_midi_path: Path,
    patch_config_and_reload: TestConfigPatcher,
    v1_default_settings: AnticipationV2Settings,
) -> None:
    padding_density = 100
    anticipation_interval_seconds = 1
    patch_config_and_reload(
        MIN_TRACK_EVENTS=0,
        MIN_TRACK_TIME_IN_SECONDS=0,
        DELTA=anticipation_interval_seconds,
    )
    assert config.TIME_RESOLUTION == 100
    midi_preprocess_token_list: list[int] = midi_to_compound(
        str(c_major_midi_path.absolute())
    )
    all_events, truncations, status = maybe_tokenize(midi_preprocess_token_list)
    end_time = ops.max_time(all_events, seconds=False)

    # span augmentation
    np.random.seed(1)
    events, controls = extract_spans(all_events, 0.2)
    assert len(events) == 66
    assert len(controls) == 21
    assert len(all_events) == len(events) + len(controls) == 87

    events = ops.pad(events, end_time, density=padding_density)
    assert len(events) == 66 + (3 * 3)
    assert events == [
        *Event.from_midi_values(0, 50, 0, "C1").as_tokens(),
        *Event.from_midi_values(50, 50, 0, "D1").as_tokens(),
        *Event.from_midi_values(100, 50, 0, "E1").as_tokens(),
        *Event.from_midi_values(150, 50, 0, "F1").as_tokens(),
        *Event.from_midi_values(200, 50, 0, "G1").as_tokens(),
        *Event.from_midi_values(250, 50, 0, "A1").as_tokens(),
        # span: 269-400
        *Event.from_midi_values(250 + padding_density * 1, 0, 0, "REST").as_tokens(),
        *Event.from_midi_values(450, 50, 0, "D2").as_tokens(),
        *Event.from_midi_values(500, 50, 0, "E2").as_tokens(),
        *Event.from_midi_values(550, 50, 0, "F2").as_tokens(),
        *Event.from_midi_values(600, 50, 0, "G2").as_tokens(),
        *Event.from_midi_values(650, 50, 0, "A2").as_tokens(),
        *Event.from_midi_values(700, 50, 0, "B2").as_tokens(),
        *Event.from_midi_values(750, 50, 0, "C3").as_tokens(),
        *Event.from_midi_values(850, 50, 0, "D3").as_tokens(),
        *Event.from_midi_values(900, 50, 0, "E3").as_tokens(),
        *Event.from_midi_values(950, 50, 0, "F3").as_tokens(),
        *Event.from_midi_values(1000, 50, 0, "G3").as_tokens(),
        *Event.from_midi_values(1050, 50, 0, "A3").as_tokens(),
        # span: 1087-1200
        *Event.from_midi_values(1050 + padding_density * 1, 0, 0, "REST").as_tokens(),
        # span: 1250-1350
        *Event.from_midi_values(1050 + padding_density * 2, 0, 0, "REST").as_tokens(),
        *Event.from_midi_values(1350, 50, 0, "F4").as_tokens(),
        *Event.from_midi_values(1400, 50, 0, "G4").as_tokens(),
        *Event.from_midi_values(1450, 50, 0, "A4").as_tokens(),
        *Event.from_midi_values(1500, 50, 0, "B4").as_tokens(),
        # span: 1530-1650
    ]
    assert controls == [
        # Q: shouldn't it instead be v.ANOTE_OFFSET + get_note_instrument_token(...)?
        # A: no because, the note, duration tokens are exactly the event tokens + the control offset
        # time here is the original time the note occurred in the sequence
        *Event.from_midi_values(300, 50, 0, "B1", is_control=True).as_tokens(),
        *Event.from_midi_values(350, 50, 0, "C2", is_control=True).as_tokens(),
        # ...
        *Event.from_midi_values(1100, 50, 0, "B3", is_control=True).as_tokens(),
        *Event.from_midi_values(1150, 50, 0, "C4", is_control=True).as_tokens(),
        # ...
        *Event.from_midi_values(1250, 50, 0, "D4", is_control=True).as_tokens(),
        *Event.from_midi_values(1300, 50, 0, "E4", is_control=True).as_tokens(),
        # ...
        *Event.from_midi_values(1550, 50, 0, "C5", is_control=True).as_tokens(),
    ]

    # implicitly: delta = (anticipation_interval_seconds * config.TIME_RESOLUTION),
    # which we have set to delta = (100)
    tokens, controls = ops.anticipate(events, controls)
    # this function returns unconsumed controls - expect to use all of them
    assert len(controls) == 0
    assert len(tokens) == 96

    # anticipation interleaves controls and events s.t. controls appear after
    # stopping times
    assert tokens == [
        *Event.from_midi_values(0, 50, 0, "C1").as_tokens(),
        *Event.from_midi_values(50, 50, 0, "D1").as_tokens(),
        *Event.from_midi_values(100, 50, 0, "E1").as_tokens(),
        *Event.from_midi_values(150, 50, 0, "F1").as_tokens(),
        *Event.from_midi_values(200, 50, 0, "G1").as_tokens(),
        # B1 approx 100 ticks before its time
        *Event.from_midi_values(300, 50, 0, "B1", is_control=True).as_tokens(),
        *Event.from_midi_values(250, 50, 0, "A1").as_tokens(),
        # C2 approx 100 ticks before its time
        *Event.from_midi_values(350, 50, 0, "C2", is_control=True).as_tokens(),
        *Event.from_midi_values(250 + padding_density * 1, 0, 0, "REST").as_tokens(),
        *Event.from_midi_values(450, 50, 0, "D2").as_tokens(),
        *Event.from_midi_values(500, 50, 0, "E2").as_tokens(),
        *Event.from_midi_values(550, 50, 0, "F2").as_tokens(),
        *Event.from_midi_values(600, 50, 0, "G2").as_tokens(),
        *Event.from_midi_values(650, 50, 0, "A2").as_tokens(),
        *Event.from_midi_values(700, 50, 0, "B2").as_tokens(),
        *Event.from_midi_values(750, 50, 0, "C3").as_tokens(),
        *Event.from_midi_values(850, 50, 0, "D3").as_tokens(),
        *Event.from_midi_values(900, 50, 0, "E3").as_tokens(),
        *Event.from_midi_values(950, 50, 0, "F3").as_tokens(),
        *Event.from_midi_values(1000, 50, 0, "G3").as_tokens(),
        # B3 approx 100 ticks before its time
        *Event.from_midi_values(1100, 50, 0, "B3", is_control=True).as_tokens(),
        *Event.from_midi_values(1050, 50, 0, "A3").as_tokens(),
        # C4 approx 100 ticks before its time
        *Event.from_midi_values(1150, 50, 0, "C4", is_control=True).as_tokens(),
        *Event.from_midi_values(1050 + padding_density * 1, 0, 0, "REST").as_tokens(),
        # D4 approx 100 ticks before its time
        *Event.from_midi_values(1250, 50, 0, "D4", is_control=True).as_tokens(),
        *Event.from_midi_values(1050 + padding_density * 2, 0, 0, "REST").as_tokens(),
        *Event.from_midi_values(1300, 50, 0, "E4", is_control=True).as_tokens(),
        *Event.from_midi_values(1350, 50, 0, "F4").as_tokens(),
        *Event.from_midi_values(1400, 50, 0, "G4").as_tokens(),
        *Event.from_midi_values(1450, 50, 0, "A4").as_tokens(),
        *Event.from_midi_values(1550, 50, 0, "C5", is_control=True).as_tokens(),
        *Event.from_midi_values(1500, 50, 0, "B4").as_tokens(),
    ]

    get_figure_and_open(
        Event.from_token_seq(tokens, v1_default_settings),
        delta=config.DELTA,
        time_resolution=config.TIME_RESOLUTION,
        path=(VISUALIZATIONS_PATH / "viz_small_sequence_anticipated.html"),
        auto_open=False,
    )


def test_tokenization_small_sequence_ar(
    c_major_midi_path: Path,
    patch_config_and_reload: TestConfigPatcher,
    v1_default_settings: AnticipationV2Settings,
) -> None:
    m = 10
    event_size = 3
    augment_factor = 1
    patch_config_and_reload(MIN_TRACK_EVENTS=0, MIN_TRACK_TIME_IN_SECONDS=0, M=m)
    parse_info = get_tokens_from_midi_file_v1(
        [c_major_midi_path],
        augment_factor=augment_factor,
        return_original_compound=True,
    )
    compound = parse_info["compound"]
    sequences = parse_info["tokens"]

    assert len(compound) == 145
    assert parse_info["seqcount"] == 3
    assert len(sequences) == parse_info["seqcount"]

    # check each sequence
    seq_1 = sequences[0]
    seq_2 = sequences[1]
    seq_3 = sequences[2]
    assert len(seq_1) == (1 + (m * event_size))
    assert len(seq_2) == (1 + (m * event_size))
    assert len(seq_3) == (1 + (m * event_size))

    # 50 is quarter note in 4/4 120 BPM
    assert seq_1 == [
        v.AUTOREGRESS,
        v.SEPARATOR,
        v.SEPARATOR,
        v.SEPARATOR,
        *Event.from_midi_values(0, 50, 0, "C1").as_tokens(),
        *Event.from_midi_values(50, 50, 0, "D1").as_tokens(),
        *Event.from_midi_values(100, 50, 0, "E1").as_tokens(),
        *Event.from_midi_values(150, 50, 0, "F1").as_tokens(),
        *Event.from_midi_values(200, 50, 0, "G1").as_tokens(),
        *Event.from_midi_values(250, 50, 0, "A1").as_tokens(),
        *Event.from_midi_values(300, 50, 0, "B1").as_tokens(),
        *Event.from_midi_values(350, 50, 0, "C2").as_tokens(),
        *Event.from_midi_values(450, 50, 0, "D2").as_tokens(),
    ]
    assert seq_2 == [
        v.AUTOREGRESS,
        *Event.from_midi_values(0, 50, 0, "E2").as_tokens(),
        *Event.from_midi_values(50, 50, 0, "F2").as_tokens(),
        *Event.from_midi_values(100, 50, 0, "G2").as_tokens(),
        *Event.from_midi_values(150, 50, 0, "A2").as_tokens(),
        *Event.from_midi_values(200, 50, 0, "B2").as_tokens(),
        *Event.from_midi_values(250, 50, 0, "C3").as_tokens(),
        *Event.from_midi_values(350, 50, 0, "D3").as_tokens(),
        *Event.from_midi_values(400, 50, 0, "E3").as_tokens(),
        *Event.from_midi_values(450, 50, 0, "F3").as_tokens(),
        *Event.from_midi_values(500, 50, 0, "G3").as_tokens(),
    ]
    assert seq_3 == [
        v.AUTOREGRESS,
        *Event.from_midi_values(0, 50, 0, "A3").as_tokens(),
        *Event.from_midi_values(50, 50, 0, "B3").as_tokens(),
        *Event.from_midi_values(100, 50, 0, "C4").as_tokens(),
        *Event.from_midi_values(200, 50, 0, "D4").as_tokens(),
        *Event.from_midi_values(250, 50, 0, "E4").as_tokens(),
        *Event.from_midi_values(300, 50, 0, "F4").as_tokens(),
        *Event.from_midi_values(350, 50, 0, "G4").as_tokens(),
        *Event.from_midi_values(400, 50, 0, "A4").as_tokens(),
        *Event.from_midi_values(450, 50, 0, "B4").as_tokens(),
        *Event.from_midi_values(500, 50, 0, "C5").as_tokens(),
    ]

    # visualize the piano roll
    get_figure_and_open(
        Event.from_token_seq(seq_1, v1_default_settings),
        delta=config.DELTA,
        time_resolution=config.TIME_RESOLUTION,
        path=(VISUALIZATIONS_PATH / "viz_small_sequence_ar.html"),
        auto_open=False,
    )


def test_tokenization_lakh_anticipation_get_cold_start(
    lmd_0_example_1_midi_path: Path,
    v1_default_settings: AnticipationV2Settings,
) -> None:
    with patch("anticipation.tokenize.np.random.choice", return_value=[128]):
        # force the call to np.random.choice to always return [128] for tokenize.py
        # This means that the instrument code 128 will always be a control. This code
        # is the drum track for this sample. This makes it much easier to see the
        # cold start issue
        parse_info = get_tokens_from_midi_file_v1(
            [lmd_0_example_1_midi_path],
            augment_factor=10,
            include_original=False,
            do_span_augmentation=False,
            do_random_augmentation=False,
            do_instrument_augmentation=True,
        )

    # a list of sequences of tokens
    tokens: list[list[int]] = parse_info["tokens"]
    assert parse_info["num_too_short"] == 0
    assert parse_info["num_too_long"] == 0
    assert parse_info["num_too_many_instruments"] == 0
    assert parse_info["num_inexpressible"] == 0

    # the sequences after the 1st exhibit the cold start problem
    for i, s in enumerate(tokens[:5]):
        assert len(s) == 1024
        get_figure_and_open(
            Event.from_token_seq(s, v1_default_settings),
            delta=config.DELTA,
            time_resolution=config.TIME_RESOLUTION,
            path=(VISUALIZATIONS_PATH / f"{i}_viz_seq_anticipated_cold_start.html"),
            auto_open=False,
        )


def test_tokenization_lakh_boundary_special_tokens(
    lmd_0_example_1_midi_path: Path,
) -> None:
    np.random.seed(1)
    parse_info = get_tokens_from_midi_file_v1(
        [lmd_0_example_1_midi_path],
        augment_factor=10,
        include_original=True,
        do_span_augmentation=True,
        do_random_augmentation=False,
        do_instrument_augmentation=True,
    )

    # a list of sequences of tokens
    tokens: list[list[int]] = parse_info["tokens"]
    assert parse_info["num_too_short"] == 0
    assert parse_info["num_too_long"] == 0
    assert parse_info["num_too_many_instruments"] == 0
    assert parse_info["num_inexpressible"] == 0
    assert parse_info["seqcount"] == len(tokens) == 50

    # these are separated by a newline on disk save
    for seq in tokens:
        assert len(seq) == 1024

        # the first token is always one of these BOS-like tokens
        assert seq[0] in (v.AUTOREGRESS, v.ANTICIPATE)

        # if it is the AR/AN flag, then, the next token
        # can either be a separator...
        if seq[1] == v.SEPARATOR:
            assert all([x == v.SEPARATOR for x in seq[1:4]])
        else:
            # or it can be events
            assert seq[1] not in (v.AUTOREGRESS, v.ANTICIPATE, v.SEPARATOR)
