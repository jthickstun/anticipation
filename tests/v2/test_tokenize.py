import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from anticipation.v2.types import Token
from anticipation.v2.config import AnticipationV2Settings, Vocab
from anticipation.v2.tokenize import tokenize as v2_tokenize
from anticipation.v2.tokenize import MIDIFileIgnoredReason, TokenizationStatSummary
from anticipation.v2.io import TokenSequenceBinaryFile
from anticipation.v2.util import set_seed

from tests.util.entities import Event, EventSpecialCode, get_note_instrument_token, Note
from tests.util.visualize_sequence import get_figure_and_open

from tests.conftest import (
    VISUALIZATIONS_PATH,
)


@pytest.fixture
def lmd_0_example_1_tokens_and_parsed_events(
    lmd_0_example_1_midi_path: Path,
) -> tuple[list[list[Token]], list[Event], AnticipationV2Settings]:
    settings = AnticipationV2Settings(
        min_track_events=1,
        vocab=Vocab(),
        # AR only
        num_autoregressive_seq_per_midi_file=1,
        num_instrument_anticipation_augmentations_per_midi_file=0,
        num_span_anticipation_augmentations_per_midi_file=0,
        num_random_anticipation_augmentations_per_midi_file=0,
        tick_token_frequency_in_midi_ticks=100,
        debug=True,
        debug_flush_remaining_token_buffer=True,
    )
    midi_files = [lmd_0_example_1_midi_path]
    in_memory_tokens = []
    stats = v2_tokenize(midi_files, in_memory_tokens, settings)
    assert not stats.ignored_files

    # tokenizing this in full is 9 sequences
    assert len(in_memory_tokens) == 9
    # all but the last one has full context, we've set a debug setting to push the
    # tokens in the buffer that would typically be ignored in a production context
    assert all((len(x) == settings.context_size for x in in_memory_tokens[:-1]))

    # there are 584 remaining that do not fit exactly into a context window
    assert len(in_memory_tokens[8]) == 584

    all_tokens_flattened = [x for b in in_memory_tokens for x in b]
    assert len(all_tokens_flattened) == 584 + (1024 * 8)

    num_total_separators = 0
    for i, packed_seq in enumerate(in_memory_tokens):
        assert packed_seq[0] == settings.vocab.AUTOREGRESS
        num_total_separators += len(
            [x for x in packed_seq if x == settings.vocab.SEPARATOR]
        )

    assert num_total_separators == 1
    parsed_events = Event.from_token_seq(
        [x for b in in_memory_tokens for x in b], settings
    )
    assert len(parsed_events) == 3045
    return in_memory_tokens, parsed_events, settings


def test_tokenize_v2_lakh_ar_only_for_visualization(
    lmd_0_example_1_tokens_and_parsed_events: tuple[
        list[list[Token]], list[Event], AnticipationV2Settings
    ],
) -> None:
    _, parsed_events, settings = lmd_0_example_1_tokens_and_parsed_events
    get_figure_and_open(
        events=parsed_events,
        delta=settings.delta,
        time_resolution=settings.time_resolution,
        path=(VISUALIZATIONS_PATH / f"autoregressive_v2.html"),
        auto_open=False,
    )


def test_tokenize_v2_lakh_ar_local_midi_vocab(
    lmd_0_example_1_midi_path: Path, local_midi_settings_ar_only: AnticipationV2Settings
) -> None:
    in_memory_tokens = []
    stats = v2_tokenize(
        [lmd_0_example_1_midi_path], in_memory_tokens, local_midi_settings_ar_only
    )
    assert not stats.ignored_files
    parsed_events = Event.from_token_seq(
        [x for b in in_memory_tokens for x in b], local_midi_settings_ar_only
    )
    get_figure_and_open(
        events=parsed_events,
        delta=local_midi_settings_ar_only.delta,
        time_resolution=local_midi_settings_ar_only.time_resolution,
        path=(VISUALIZATIONS_PATH / f"autoregressive_v2_local_midi.html"),
        auto_open=False,
    )


def test_tokenize_v2_lakh_instrument_for_visualization(
    lmd_0_example_1_midi_path: Path,
) -> None:
    instrument_anticipation_sample = []
    with patch("anticipation.tokenize.np.random.choice", return_value=[128]):
        # force the call to np.random.choice to always return [128] for tokenize.py
        # This means that the instrument code 128 will always be a control. This code
        # is the drum track for this sample. This makes it much easier to see the
        # cold start issue
        settings = AnticipationV2Settings(
            vocab=Vocab(),
            num_autoregressive_seq_per_midi_file=0,
            num_instrument_anticipation_augmentations_per_midi_file=1,
            num_span_anticipation_augmentations_per_midi_file=0,
            num_random_anticipation_augmentations_per_midi_file=0,
            debug=True,
        )
        stats = v2_tokenize(
            [lmd_0_example_1_midi_path], instrument_anticipation_sample, settings
        )
        assert not stats.ignored_files

    assert len(instrument_anticipation_sample) == 8
    num_total_separators = 0
    for i, packed_seq in enumerate(instrument_anticipation_sample):
        assert len(packed_seq) == settings.context_size
        assert packed_seq[0] == settings.vocab.ANTICIPATE

        num_total_separators += len(
            [x for x in packed_seq if x == settings.vocab.SEPARATOR]
        )
    assert num_total_separators == 1
    parsed_events = Event.from_token_seq(
        [x for b in instrument_anticipation_sample for x in b], settings
    )
    get_figure_and_open(
        events=parsed_events,
        delta=settings.delta,
        time_resolution=settings.time_resolution,
        path=(VISUALIZATIONS_PATH / f"anticipated_instr_v2.html"),
        auto_open=False,
    )


def test_tokenize_with_ticks_for_small_sequence_ar(
    c_major_midi_path: Path,
) -> None:
    settings = AnticipationV2Settings(
        min_track_events=1,
        vocab=Vocab(),
        # AR only
        num_autoregressive_seq_per_midi_file=1,
        num_instrument_anticipation_augmentations_per_midi_file=0,
        num_span_anticipation_augmentations_per_midi_file=0,
        num_random_anticipation_augmentations_per_midi_file=0,
        # force small context for this specific test, otherwise tokens are left in the
        # buffer
        context_size=104,
        debug=True,
        debug_flush_remaining_token_buffer=False,
        tick_token_frequency_in_midi_ticks=100,
    )
    tokenized_seq = []
    stats = v2_tokenize([c_major_midi_path], tokenized_seq, settings)
    assert not stats.ignored_files

    assert len(tokenized_seq) == 1
    num_total_separators = 0
    for i, packed_seq in enumerate(tokenized_seq):
        assert len(packed_seq) == settings.context_size
        assert packed_seq[0] == settings.vocab.AUTOREGRESS
        num_total_separators += len(
            [x for x in packed_seq if x == settings.vocab.SEPARATOR]
        )

    parsed_events = Event.from_token_seq(
        [x for b in tokenized_seq for x in b], settings
    )

    # check that the ticks are added at specified interval
    ticks = [x for x in parsed_events if x.is_tick()]
    ticks_abs_times = [x.absolute_time for x in ticks]
    assert ticks_abs_times == list(
        range(0, 1500 + 1, settings.tick_token_frequency_in_midi_ticks)
    )
    get_figure_and_open(
        events=parsed_events,
        delta=settings.delta,
        time_resolution=settings.time_resolution,
        path=(VISUALIZATIONS_PATH / f"ar_with_ticks_small_seq.html"),
        auto_open=False,
    )


def test_tokenize_with_ticks_for_lakh_ar(lmd_0_example_1_midi_path: Path) -> None:
    settings = AnticipationV2Settings(
        min_track_events=1,
        vocab=Vocab(),
        # AR only
        num_autoregressive_seq_per_midi_file=1,
        num_instrument_anticipation_augmentations_per_midi_file=0,
        num_span_anticipation_augmentations_per_midi_file=0,
        num_random_anticipation_augmentations_per_midi_file=0,
        debug=True,
    )
    tokenized_seq = []
    stats = v2_tokenize([lmd_0_example_1_midi_path], tokenized_seq, settings)
    assert not stats.ignored_files
    assert len(tokenized_seq) == 8
    num_total_separators = 0
    for i, packed_seq in enumerate(tokenized_seq):
        assert len(packed_seq) == settings.context_size
        assert packed_seq[0] == settings.vocab.AUTOREGRESS
        num_total_separators += len(
            [x for x in packed_seq if x == settings.vocab.SEPARATOR]
        )

    assert num_total_separators == 1
    parsed_events = Event.from_token_seq(
        [x for b in tokenized_seq for x in b], settings
    )
    get_figure_and_open(
        events=parsed_events,
        delta=settings.delta,
        time_resolution=settings.time_resolution,
        path=(VISUALIZATIONS_PATH / f"ar_with_ticks_lakh_0.html"),
        auto_open=False,
    )


def test_tokenize_v2_lakh_span_anticipation(
    lmd_0_example_1_midi_path: Path,
    local_midi_vocab: Vocab,
) -> None:
    set_seed(48)

    tokens_to = []
    settings = AnticipationV2Settings(
        vocab=local_midi_vocab,
        num_autoregressive_seq_per_midi_file=0,
        num_instrument_anticipation_augmentations_per_midi_file=0,
        num_span_anticipation_augmentations_per_midi_file=1,
        num_random_anticipation_augmentations_per_midi_file=0,
        tick_token_frequency_in_midi_ticks=100,
        debug=True,
        debug_flush_remaining_token_buffer=True,
    )
    stats = v2_tokenize([lmd_0_example_1_midi_path], tokens_to, settings)
    assert not stats.ignored_files
    assert settings.vocab.TICK == 17612
    parsed_events = Event.from_token_seq(
        [x for b in tokens_to for x in b], settings
    )  # [:2000]
    get_figure_and_open(
        events=parsed_events,
        delta=settings.delta,
        time_resolution=settings.time_resolution,
        path=(VISUALIZATIONS_PATH / f"anticipated_span_v2.html"),
        auto_open=True,
    )


def test_absolute_time_is_correct_with_ticks(lmd_0_example_1_midi_path: Path) -> None:
    settings = AnticipationV2Settings(
        min_track_events=1,
        vocab=Vocab(),
        # AR only
        num_autoregressive_seq_per_midi_file=1,
        num_instrument_anticipation_augmentations_per_midi_file=0,
        num_span_anticipation_augmentations_per_midi_file=0,
        num_random_anticipation_augmentations_per_midi_file=0,
        debug=True,
    )
    # tokenize and parse without ticks added
    events_without_ticks = []
    stats = v2_tokenize([lmd_0_example_1_midi_path], events_without_ticks, settings)
    assert len(events_without_ticks) == 8
    assert not stats.ignored_files
    events_without_ticks = Event.from_token_seq(
        [x for b in events_without_ticks for x in b], settings
    )
    events_without_ticks = [
        x
        for x in events_without_ticks
        if x.special_code == EventSpecialCode.TYPICAL_EVENT
    ]

    # tokenize and parse WITH ticks added
    events_include_ticks = []
    stats = v2_tokenize([lmd_0_example_1_midi_path], events_include_ticks, settings)
    assert not stats.ignored_files
    events_include_ticks = Event.from_token_seq(
        [x for b in events_include_ticks for x in b], settings
    )
    events_include_ticks = [
        x
        for x in events_include_ticks
        if x.special_code == EventSpecialCode.TYPICAL_EVENT
    ]

    # assert no information loss and that time de-relativization works
    for a, b in zip(events_without_ticks, events_include_ticks):
        # absolute times must be the same, we should be able to recover
        # the original non-relativized time from the sequence of events that
        # has the ticks
        assert a.absolute_time == b.absolute_time
        assert a.midi_note() == b.midi_note()
        assert a.midi_duration() == b.midi_duration()
        assert a.midi_instrument() == b.midi_instrument()


def test_sequence_packing_file_correctness(
    lmd_0_example_1_midi_path: Path,
    lmd_0_example_2_midi_path: Path,
    lmd_0_example_1_tokens_and_parsed_events,
) -> None:
    settings = AnticipationV2Settings(
        min_track_events=1,
        vocab=Vocab(),
        # AR only
        num_autoregressive_seq_per_midi_file=1,
        num_instrument_anticipation_augmentations_per_midi_file=0,
        num_span_anticipation_augmentations_per_midi_file=0,
        num_random_anticipation_augmentations_per_midi_file=0,
        debug=True,
        debug_flush_remaining_token_buffer=False,
        tick_token_frequency_in_midi_ticks=100,
    )
    midi_files = [lmd_0_example_1_midi_path, lmd_0_example_2_midi_path]

    in_memory_tokens = []
    stats = v2_tokenize(midi_files, in_memory_tokens, settings)
    assert not stats.ignored_files

    # this token basically prefixes each file
    assert stats.num_separator_tokens == 2

    # 97 complete sequences
    assert len(in_memory_tokens) == 97
    # should all be context length
    assert all((len(x) == settings.context_size for x in in_memory_tokens))

    # test that we can fully recover the first file that is packed, even
    # when flushing the remaining buffer is OFF. We expect that a context window
    # contains the end of the first file and start of the second
    seq_border_frame = in_memory_tokens[8]
    sep_idx = seq_border_frame.index(settings.vocab.SEPARATOR)
    assert sep_idx == 584
    lmd_0_example_1_actual_tokens = in_memory_tokens[:8] + [
        in_memory_tokens[8][:sep_idx]
    ]
    lmd_0_example_1_actual_events = Event.from_token_seq(
        [x for b in lmd_0_example_1_actual_tokens for x in b], settings
    )
    (
        lmd_0_example_1_expected_tokens,
        lmd_0_example_1_expected_events,
        expected_settings,
    ) = lmd_0_example_1_tokens_and_parsed_events
    assert lmd_0_example_1_actual_tokens == lmd_0_example_1_expected_tokens
    assert lmd_0_example_1_actual_events[0] == lmd_0_example_1_expected_events[0]
    assert lmd_0_example_1_actual_events == lmd_0_example_1_expected_events
    assert settings.context_size == expected_settings.context_size
    assert settings.vocab == expected_settings.vocab

    # test saving data to disk is exactly same as in-memory
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        dataset_path = td_path / "dataset.bin"
        stats = v2_tokenize(midi_files, dataset_path, settings)
        assert not stats.ignored_files
        tokenized_samples = TokenSequenceBinaryFile.load_from_disk_to_numpy(
            dataset_path,
            seq_len=settings.context_size,
            vocab_size=settings.vocab.total_tokens(),
        )
        # reading from disk should yield same result as just writing to
        # in memory buffer, this tests the correctness of file io
        in_memory_tokenized = np.array(in_memory_tokens, dtype=np.uint16)
        assert np.array_equal(in_memory_tokenized, tokenized_samples)


def test_no_segfault(lmd_1_example_0_midi_path: Path) -> None:
    settings = AnticipationV2Settings(
        vocab=Vocab(),
        debug=False,
        # AR only
        num_autoregressive_seq_per_midi_file=1,
        num_instrument_anticipation_augmentations_per_midi_file=0,
        num_span_anticipation_augmentations_per_midi_file=0,
        num_random_anticipation_augmentations_per_midi_file=0,
        debug_flush_remaining_token_buffer=False,
    )
    tokens_to = []
    stats: TokenizationStatSummary = v2_tokenize(
        [lmd_1_example_0_midi_path], output=tokens_to, settings=settings
    )
    assert stats.ignored_files[MIDIFileIgnoredReason.INVALID_FILE_STRUCTURE] == [
        lmd_1_example_0_midi_path
    ]


def test_sequence_boundaries_for_truncated_end_triple(c_major_midi_path: Path) -> None:
    settings = AnticipationV2Settings(
        vocab=Vocab(),
        debug=False,
        debug_flush_remaining_token_buffer=True,
        min_track_events=1,
        min_track_time_in_seconds=1,
        # small context size for this test, we want to examine what happens at
        # the sequence boundaries
        context_size=8,
        num_autoregressive_seq_per_midi_file=1,
        num_instrument_anticipation_augmentations_per_midi_file=0,
        num_span_anticipation_augmentations_per_midi_file=0,
        num_random_anticipation_augmentations_per_midi_file=0,
        tick_token_frequency_in_midi_ticks=100,
    )
    tokens_to = []
    stats: TokenizationStatSummary = v2_tokenize(
        [c_major_midi_path], output=tokens_to, settings=settings
    )
    assert stats.num_times_end_triple_was_truncated == 2
    assert len(tokens_to) == 16
    assert stats.num_sequences == 16

    seq_0 = tokens_to[0]
    event_0 = [
        0,
        settings.vocab.DUR_OFFSET + 50,
        get_note_instrument_token(0, Note.make("C1").midi_note_int, settings),
    ]
    event_1 = [
        50,
        settings.vocab.DUR_OFFSET + 50,
        get_note_instrument_token(0, Note.make("D1").midi_note_int, settings),
    ]
    event_2 = [
        0,
        settings.vocab.DUR_OFFSET + 50,
        get_note_instrument_token(0, Note.make("E1").midi_note_int, settings),
    ]
    event_3 = [
        50,
        settings.vocab.DUR_OFFSET + 50,
        get_note_instrument_token(0, Note.make("F1").midi_note_int, settings),
    ]
    event_4 = [
        0,
        settings.vocab.DUR_OFFSET + 50,
        get_note_instrument_token(0, Note.make("G1").midi_note_int, settings),
    ]
    event_5 = [
        50,
        settings.vocab.DUR_OFFSET + 50,
        get_note_instrument_token(0, Note.make("A1").midi_note_int, settings),
    ]
    event_6 = [
        0,
        settings.vocab.DUR_OFFSET + 50,
        get_note_instrument_token(0, Note.make("B1").midi_note_int, settings),
    ]
    event_7 = [
        50,
        settings.vocab.DUR_OFFSET + 50,
        get_note_instrument_token(0, Note.make("C2").midi_note_int, settings),
    ]
    event_8 = [
        50,
        settings.vocab.DUR_OFFSET + 50,
        get_note_instrument_token(0, Note.make("D2").midi_note_int, settings),
    ]
    event_9 = [
        0,
        settings.vocab.DUR_OFFSET + 50,
        get_note_instrument_token(0, Note.make("E2").midi_note_int, settings),
    ]
    event_10 = [
        50,
        settings.vocab.DUR_OFFSET + 50,
        get_note_instrument_token(0, Note.make("F2").midi_note_int, settings),
    ]
    event_11 = [
        0,
        settings.vocab.DUR_OFFSET + 50,
        get_note_instrument_token(0, Note.make("G2").midi_note_int, settings),
    ]
    event_12 = [
        50,
        settings.vocab.DUR_OFFSET + 50,
        get_note_instrument_token(0, Note.make("A2").midi_note_int, settings),
    ]
    event_13 = [
        0,
        settings.vocab.DUR_OFFSET + 50,
        get_note_instrument_token(0, Note.make("B2").midi_note_int, settings),
    ]
    event_14 = [
        50,
        settings.vocab.DUR_OFFSET + 50,
        get_note_instrument_token(0, Note.make("C3").midi_note_int, settings),
    ]
    event_15 = [
        50,
        settings.vocab.DUR_OFFSET + 50,
        get_note_instrument_token(0, Note.make("D3").midi_note_int, settings),
    ]
    event_16 = [
        0,
        settings.vocab.DUR_OFFSET + 50,
        get_note_instrument_token(0, Note.make("E3").midi_note_int, settings),
    ]
    event_17 = [
        50,
        settings.vocab.DUR_OFFSET + 50,
        get_note_instrument_token(0, Note.make("F3").midi_note_int, settings),
    ]
    event_18 = [
        0,
        settings.vocab.DUR_OFFSET + 50,
        get_note_instrument_token(0, Note.make("G3").midi_note_int, settings),
    ]
    event_19 = [
        50,
        settings.vocab.DUR_OFFSET + 50,
        get_note_instrument_token(0, Note.make("A3").midi_note_int, settings),
    ]
    event_20 = [
        0,
        settings.vocab.DUR_OFFSET + 50,
        get_note_instrument_token(0, Note.make("B3").midi_note_int, settings),
    ]
    event_21 = [
        50,
        settings.vocab.DUR_OFFSET + 50,
        get_note_instrument_token(0, Note.make("C4").midi_note_int, settings),
    ]
    event_22 = [
        50,
        settings.vocab.DUR_OFFSET + 50,
        get_note_instrument_token(0, Note.make("D4").midi_note_int, settings),
    ]
    event_23 = [
        0,
        settings.vocab.DUR_OFFSET + 50,
        get_note_instrument_token(0, Note.make("E4").midi_note_int, settings),
    ]
    event_24 = [
        50,
        settings.vocab.DUR_OFFSET + 50,
        get_note_instrument_token(0, Note.make("F4").midi_note_int, settings),
    ]
    event_25 = [
        0,
        settings.vocab.DUR_OFFSET + 50,
        get_note_instrument_token(0, Note.make("G4").midi_note_int, settings),
    ]
    event_26 = [
        50,
        settings.vocab.DUR_OFFSET + 50,
        get_note_instrument_token(0, Note.make("A4").midi_note_int, settings),
    ]
    event_27 = [
        0,
        settings.vocab.DUR_OFFSET + 50,
        get_note_instrument_token(0, Note.make("B4").midi_note_int, settings),
    ]
    event_28 = [
        50,
        settings.vocab.DUR_OFFSET + 50,
        get_note_instrument_token(0, Note.make("C5").midi_note_int, settings),
    ]
    assert seq_0 == [
        # special
        settings.vocab.AUTOREGRESS,
        settings.vocab.SEPARATOR,
        # tick
        settings.vocab.TICK,
        # event 0
        *event_0,
        # event 1, is cut off - this is the first truncation
        *event_1[:2],
    ]
    seq_1 = tokens_to[1]
    assert seq_1 == [
        # special
        settings.vocab.AUTOREGRESS,
        # but event 1 continues the next sequence and
        # is represented in full
        *event_1,
        settings.vocab.TICK,
        # event 2
        *event_2,
    ]
    seq_2 = tokens_to[2]
    assert seq_2 == [
        # special
        settings.vocab.AUTOREGRESS,
        # event 2 appeared in full, so we start with event 3
        *event_3,
        settings.vocab.TICK,
        # event 4
        *event_4,
    ]
    seq_3 = tokens_to[3]
    assert seq_3 == [
        # special
        settings.vocab.AUTOREGRESS,
        *event_5,
        settings.vocab.TICK,
        *event_6,
    ]
    seq_4 = tokens_to[4]
    assert seq_4 == [
        # special
        settings.vocab.AUTOREGRESS,
        *event_7,
        settings.vocab.TICK,
        # event 8 comes 50 midi ticks after tick token
        # there is a 50 midi tick rest after event 7
        *event_8,
    ]
    seq_5 = tokens_to[5]
    assert seq_5 == [
        settings.vocab.AUTOREGRESS,
        # now the tick is at the front, that's just how it turned out
        # with the context length
        settings.vocab.TICK,
        *event_9,
        *event_10,
    ]
    seq_6 = tokens_to[6]
    assert seq_6 == [
        settings.vocab.AUTOREGRESS,
        settings.vocab.TICK,
        *event_11,
        *event_12,
    ]
    seq_7 = tokens_to[7]
    assert seq_7 == [
        settings.vocab.AUTOREGRESS,
        settings.vocab.TICK,
        *event_13,
        *event_14,
    ]
    seq_8 = tokens_to[8]
    assert seq_8 == [
        settings.vocab.AUTOREGRESS,
        settings.vocab.TICK,
        *event_15,
        settings.vocab.TICK,
        # second truncation
        *event_16[:2],
    ]
    seq_9 = tokens_to[9]
    assert seq_9 == [
        settings.vocab.AUTOREGRESS,
        # continue the truncated event from previous sequence
        *event_16,
        *event_17,
        # tick is now at the end
        settings.vocab.TICK,
    ]
    seq_10 = tokens_to[10]
    assert seq_10 == [
        settings.vocab.AUTOREGRESS,
        *event_18,
        *event_19,
        settings.vocab.TICK,
    ]
    seq_11 = tokens_to[11]
    assert seq_11 == [
        settings.vocab.AUTOREGRESS,
        *event_20,
        *event_21,
        settings.vocab.TICK,
    ]
    seq_12 = tokens_to[12]
    assert seq_12 == [
        settings.vocab.AUTOREGRESS,
        *event_22,
        settings.vocab.TICK,
        *event_23,
    ]
    seq_13 = tokens_to[13]
    assert seq_13 == [
        settings.vocab.AUTOREGRESS,
        *event_24,
        settings.vocab.TICK,
        *event_25,
    ]
    seq_14 = tokens_to[14]
    assert seq_14 == [
        settings.vocab.AUTOREGRESS,
        *event_26,
        settings.vocab.TICK,
        *event_27,
    ]

    # this sequence won't appear in the dataset, we have the setting
    # `debug_flush_remaining_token_buffer` set to true in this test,
    # so the tokens that are in the buffer at the end of tokenization
    # are returned rather than discarded.
    seq_15 = tokens_to[15]
    assert seq_15 == [
        settings.vocab.AUTOREGRESS,
        *event_28,
    ]

    # check reconstructing the events
    parsed_events = Event.from_token_seq([x for b in tokens_to for x in b], settings)
    note_strikes = [x for x in parsed_events if x.is_note_event()]
    tick_tokens = [x for x in parsed_events if x.is_tick()]

    # 29 original notes
    assert len(note_strikes) == 29
    # the time resolution is 100, we play a note every 50 ticks
    # so there will be a tick every sequence
    assert len(tick_tokens) == 16

    # check that there are no strange repeated events
    get_figure_and_open(
        events=parsed_events,
        delta=settings.delta,
        time_resolution=settings.time_resolution,
        path=(VISUALIZATIONS_PATH / "c_major_with_ticks.html"),
        auto_open=False,
    )


def test_apply_pitch_augmentation(c_major_midi_path: Path) -> None:
    settings = AnticipationV2Settings(
        vocab=Vocab(),
        debug=False,
        debug_flush_remaining_token_buffer=True,
        min_track_events=1,
        min_track_time_in_seconds=1,
        context_size=105,
        num_autoregressive_seq_per_midi_file=1,
        num_instrument_anticipation_augmentations_per_midi_file=0,
        num_span_anticipation_augmentations_per_midi_file=0,
        num_random_anticipation_augmentations_per_midi_file=0,
        tick_token_frequency_in_midi_ticks=100,
        # 3 tranpositions that will work, one that will fail - our implementation
        # should be able to handle the one that pushes notes out of bounds - it
        # should just ignore it
        augmentation_pitch_shifts=(-3, -2, -1, 1, 2, 3, 127),
    )
    tokens_to = []
    stats: TokenizationStatSummary = v2_tokenize(
        [c_major_midi_path], output=tokens_to, settings=settings
    )
    assert stats.num_pitch_transpose_augmentations == 6
    assert stats.num_times_end_triple_was_truncated == 0
    assert len(tokens_to) == 7  # transpositions (-3, -2, -1, 0, 1, 2, 3)

    # order here matters, it's the order in which augmentations took place
    expected_note_names = [
        [
            "C1",
            "D1",
            "E1",
            "F1",
            "G1",
            "A1",
            "B1",
            "C2",
            "D2",
            "E2",
            "F2",
            "G2",
            "A2",
            "B2",
            "C3",
            "D3",
            "E3",
            "F3",
            "G3",
            "A3",
            "B3",
            "C4",
            "D4",
            "E4",
            "F4",
            "G4",
            "A4",
            "B4",
            "C5",
        ],
        [
            "A0",
            "B0",
            "C#1",
            "D1",
            "E1",
            "F#1",
            "G#1",
            "A1",
            "B1",
            "C#2",
            "D2",
            "E2",
            "F#2",
            "G#2",
            "A2",
            "B2",
            "C#3",
            "D3",
            "E3",
            "F#3",
            "G#3",
            "A3",
            "B3",
            "C#4",
            "D4",
            "E4",
            "F#4",
            "G#4",
            "A4",
        ],
        [
            "A#0",
            "C1",
            "D1",
            "D#1",
            "F1",
            "G1",
            "A1",
            "A#1",
            "C2",
            "D2",
            "D#2",
            "F2",
            "G2",
            "A2",
            "A#2",
            "C3",
            "D3",
            "D#3",
            "F3",
            "G3",
            "A3",
            "A#3",
            "C4",
            "D4",
            "D#4",
            "F4",
            "G4",
            "A4",
            "A#4",
        ],
        [
            "B0",
            "C#1",
            "D#1",
            "E1",
            "F#1",
            "G#1",
            "A#1",
            "B1",
            "C#2",
            "D#2",
            "E2",
            "F#2",
            "G#2",
            "A#2",
            "B2",
            "C#3",
            "D#3",
            "E3",
            "F#3",
            "G#3",
            "A#3",
            "B3",
            "C#4",
            "D#4",
            "E4",
            "F#4",
            "G#4",
            "A#4",
            "B4",
        ],
        [
            "C#1",
            "D#1",
            "F1",
            "F#1",
            "G#1",
            "A#1",
            "C2",
            "C#2",
            "D#2",
            "F2",
            "F#2",
            "G#2",
            "A#2",
            "C3",
            "C#3",
            "D#3",
            "F3",
            "F#3",
            "G#3",
            "A#3",
            "C4",
            "C#4",
            "D#4",
            "F4",
            "F#4",
            "G#4",
            "A#4",
            "C5",
            "C#5",
        ],
        [
            "D1",
            "E1",
            "F#1",
            "G1",
            "A1",
            "B1",
            "C#2",
            "D2",
            "E2",
            "F#2",
            "G2",
            "A2",
            "B2",
            "C#3",
            "D3",
            "E3",
            "F#3",
            "G3",
            "A3",
            "B3",
            "C#4",
            "D4",
            "E4",
            "F#4",
            "G4",
            "A4",
            "B4",
            "C#5",
            "D5",
        ],
        [
            "D#1",
            "F1",
            "G1",
            "G#1",
            "A#1",
            "C2",
            "D2",
            "D#2",
            "F2",
            "G2",
            "G#2",
            "A#2",
            "C3",
            "D3",
            "D#3",
            "F3",
            "G3",
            "G#3",
            "A#3",
            "C4",
            "D4",
            "D#4",
            "F4",
            "G4",
            "G#4",
            "A#4",
            "C5",
            "D5",
            "D#5",
        ],
    ]
    for i, seq in enumerate(tokens_to):
        assert len(seq) == settings.context_size
        assert seq[0] == settings.vocab.AUTOREGRESS
        assert seq[1] == settings.vocab.SEPARATOR
        parsed_events = Event.from_token_seq(seq, settings)
        notes = [x.note().name for x in parsed_events if x.is_note_event()]
        assert notes == expected_note_names[i]
