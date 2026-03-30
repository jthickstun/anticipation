import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from anticipation.v2.config import (
    AnticipationV2Settings,
    Vocab,
    MIDI_DRUMS_PROGRAM_CODE,
    make_vocab,
)
from anticipation.v2.types import Token
from anticipation.v2.tokenize import (
    MIDIFileIgnoredReason,
    TokenizationStatSummary,
    _tokenize_midi_file,
    _get_augmentation_instrument,
    TokenizedMIDIFileResult,
    TokenStream,
    tokenize as v2_tokenize,
)
from anticipation.v2.io import TokenSequenceBinaryFile
from anticipation.v2.util import set_seed
from conftest import TEST_DATA_PATH

from tests.conftest import (
    get_current_function_name,
    get_tokens_from_file,
    save_tokens_as_file,
    VISUALIZATIONS_PATH,
)
from tests.util.entities import Event, EventSpecialCode, get_note_instrument_token, Note
from tests.util.visualize_sequence import get_figure_and_open


def _check_anticipation_rule_for_controls_and_token_ranges(
    token_sequences: list[list[Token]], settings: AnticipationV2Settings
) -> None:
    """This runs checks on forms and rules that should be true for all sequences."""
    assert isinstance(token_sequences, list)
    assert len(token_sequences) >= 1, "not enough sequences"
    # should be a list of lists of tokens
    assert isinstance(token_sequences[0], list)

    # basic checks for context length
    exact_size_tokens_seq = list(token_sequences)
    if settings.debug_flush_remaining_token_buffer:
        exact_size_tokens_seq = exact_size_tokens_seq[:-1]
    assert all(len(x) == settings.context_size for x in exact_size_tokens_seq)

    for seq in token_sequences:
        num_ar = seq.count(settings.vocab.AUTOREGRESS)
        num_an = seq.count(settings.vocab.ANTICIPATE)
        num_sep = seq.count(settings.vocab.SEPARATOR)
        if num_ar + num_an > 1:
            # a boundary sequence, ensure there's a separator
            assert num_sep == 1

    flattened_tokens = [x for b in token_sequences for x in b]
    max_token_val = settings.vocab.total_tokens() - 1
    assert all(0 <= x <= max_token_val for x in flattened_tokens)
    parsed_events = Event.from_token_seq(flattened_tokens, settings)

    # rule: controls must always come after a tick, or follow another control
    # ... or if the context was split, they need to follow the flag token, which
    # in this case MUST be `anticipate`
    for i, e in enumerate(parsed_events):
        if e.is_control:
            prev_event = parsed_events[i - 1]
            assert (
                prev_event.is_control
                or prev_event.is_tick()
                or prev_event.is_anticipate()
            )

    # check the ranges of all tokens
    for i, e in enumerate(parsed_events):
        if e.is_note_event():
            if e.is_control:
                t, d, ni = e.as_tokens()

                # check relativized
                if settings.tick_token_every_n_ticks > 0:
                    assert (
                        t - settings.vocab.ATIME_OFFSET
                        < settings.tick_token_every_n_ticks
                    )

                # check not too long
                assert d - settings.vocab.ADUR_OFFSET <= int(
                    settings.max_note_duration_in_seconds * settings.time_resolution
                )

                # check regions
                assert t < settings.vocab.ADUR_OFFSET
                assert d < settings.vocab.ANOTE_OFFSET
                assert ni < settings.vocab.SPECIAL_OFFSET
                assert (
                    settings.vocab.CONTROL_OFFSET
                    <= settings.vocab.ATIME_OFFSET
                    <= t
                    < d
                    < ni
                    < settings.vocab.SPECIAL_OFFSET
                )
            else:
                t, d, ni = e.as_tokens()

                # check relativized
                if settings.tick_token_every_n_ticks > 0:
                    assert (
                        t - settings.vocab.TIME_OFFSET
                        < settings.tick_token_every_n_ticks
                    )

                # check not too long
                assert d - settings.vocab.DUR_OFFSET <= int(
                    settings.max_note_duration_in_seconds * settings.time_resolution
                )

                # check regions
                assert t < settings.vocab.DUR_OFFSET
                assert d < settings.vocab.NOTE_OFFSET
                assert ni < settings.vocab.TICK
                assert (
                    0 <= settings.vocab.TIME_OFFSET <= t < d < ni < settings.vocab.TICK
                )


def _check_is_musically_same(
    token_seq_list_a: list[list[Token]],
    token_seq_list_b: list[list[Token]],
    settings_a: AnticipationV2Settings,
    settings_b: AnticipationV2Settings,
) -> None:
    events_a = Event.from_list_of_token_seq(token_seq_list_a, settings_a)
    events_b = Event.from_list_of_token_seq(token_seq_list_b, settings_b)

    # remove ticks and separators, anything that isn't a 'musical' event
    events_a_filtered = [x for x in events_a if x.is_note_event()]
    events_b_filtered = [x for x in events_b if x.is_note_event()]

    assert len(events_a_filtered) == len(events_b_filtered)

    events_a_filtered.sort(
        key=lambda x: (x.absolute_time, x.midi_instrument(), x.midi_note())
    )
    events_b_filtered.sort(
        key=lambda x: (x.absolute_time, x.midi_instrument(), x.midi_note())
    )

    for i in range(len(events_a_filtered)):
        # ensure that the musical semantics are preserved
        a = events_a_filtered[i]
        b = events_b_filtered[i]
        assert a.is_musically_equal(b)


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
        tick_token_every_n_ticks=100,
        debug=True,
        debug_flush_remaining_token_buffer=True,
        do_clip_overlapping_durations_in_midi_conversion=False,
    )
    midi_files = [lmd_0_example_1_midi_path]
    in_memory_tokens = []
    stats = v2_tokenize(midi_files, in_memory_tokens, settings)
    assert not stats.ignored_files

    _check_anticipation_rule_for_controls_and_token_ranges(in_memory_tokens, settings)

    # tokenizing this in full is 9 sequences
    assert len(in_memory_tokens) == 9
    # all but the last one has full context, we've set a debug setting to push the
    # tokens in the buffer that would typically be ignored in a production context
    assert all((len(x) == settings.context_size for x in in_memory_tokens[:-1]))
    assert len(in_memory_tokens[8]) == 584

    all_tokens_flattened = [x for b in in_memory_tokens for x in b]
    assert len(all_tokens_flattened) == 584 + (settings.context_size * 8)

    num_total_separators = 0
    for i, packed_seq in enumerate(in_memory_tokens):
        assert (packed_seq[0] == settings.vocab.AUTOREGRESS) or (
            packed_seq[0] == settings.vocab.SEPARATOR
            and packed_seq[1] == settings.vocab.AUTOREGRESS
        )
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
        path=(VISUALIZATIONS_PATH / (get_current_function_name() + ".html")),
        auto_open=False,
    )


def test_tokenize_v2_lakh_ar_local_midi_vocab(
    lmd_0_example_1_midi_path: Path, local_midi_settings_ar_only: AnticipationV2Settings
) -> None:
    in_memory_tokens = []
    stats = v2_tokenize(
        [lmd_0_example_1_midi_path], in_memory_tokens, local_midi_settings_ar_only
    )
    _check_anticipation_rule_for_controls_and_token_ranges(
        in_memory_tokens, local_midi_settings_ar_only
    )
    assert not stats.ignored_files
    parsed_events = Event.from_token_seq(
        [x for b in in_memory_tokens for x in b], local_midi_settings_ar_only
    )
    assert parsed_events[0].is_separator()
    get_figure_and_open(
        events=parsed_events,
        delta=local_midi_settings_ar_only.delta,
        time_resolution=local_midi_settings_ar_only.time_resolution,
        path=(VISUALIZATIONS_PATH / (get_current_function_name() + ".html")),
        auto_open=False,
    )


def test_tokenize_v2_lakh_ar_local_midi_vocab_4096(
    lmd_0_example_1_midi_path: Path,
    local_midi_settings_anticipation_ctx_4096: AnticipationV2Settings,
) -> None:
    in_memory_tokens = []
    stats = v2_tokenize(
        [lmd_0_example_1_midi_path],
        in_memory_tokens,
        local_midi_settings_anticipation_ctx_4096,
    )
    assert not stats.ignored_files
    _check_anticipation_rule_for_controls_and_token_ranges(
        in_memory_tokens, local_midi_settings_anticipation_ctx_4096
    )


def test_tokenize_v2_lakh_instrument_for_visualization(
    lmd_0_example_1_midi_path: Path,
) -> None:
    instrument_anticipation_sample = []
    with patch(
        "anticipation.v2.tokenize.np.random.choice",
        return_value=[MIDI_DRUMS_PROGRAM_CODE],
    ):
        # force the call to np.random.choice to always return [128] for tokenize.py
        # This means that the instrument code 128 will always be a control. This code
        # is the drum track for this sample. This makes it much easier to see the
        # cold start issue
        settings = AnticipationV2Settings(
            vocab=Vocab(),
            num_autoregressive_seq_per_midi_file=0,
            num_instrument_anticipation_augmentations_per_midi_file=1,
            num_span_anticipation_augmentations_per_midi_file=0,
            tick_token_every_n_ticks=100,
            debug=True,
            debug_flush_remaining_token_buffer=False,
        )
        stats = v2_tokenize(
            [lmd_0_example_1_midi_path], instrument_anticipation_sample, settings
        )
        assert not stats.ignored_files

    _check_anticipation_rule_for_controls_and_token_ranges(
        instrument_anticipation_sample, settings
    )

    assert len(instrument_anticipation_sample) == 8
    num_total_separators = 0
    for i, packed_seq in enumerate(instrument_anticipation_sample):
        assert len(packed_seq) == settings.context_size
        assert (packed_seq[0] == settings.vocab.ANTICIPATE) or (
            packed_seq[0] == settings.vocab.SEPARATOR
            and packed_seq[1] == settings.vocab.ANTICIPATE
        )
        num_total_separators += len(
            [x for x in packed_seq if x == settings.vocab.SEPARATOR]
        )

    # sep is a prefix
    assert num_total_separators == 1
    parsed_events = Event.from_token_seq(
        [x for b in instrument_anticipation_sample for x in b], settings
    )
    get_figure_and_open(
        events=parsed_events,
        delta=settings.delta,
        time_resolution=settings.time_resolution,
        path=(VISUALIZATIONS_PATH / (get_current_function_name() + ".html")),
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
        # force small context for this specific test, otherwise tokens are left in the
        # buffer
        context_size=104,
        debug=True,
        debug_flush_remaining_token_buffer=False,
        tick_token_every_n_ticks=100,
    )
    tokenized_seq = []
    stats = v2_tokenize([c_major_midi_path], tokenized_seq, settings)
    assert not stats.ignored_files

    _check_anticipation_rule_for_controls_and_token_ranges(tokenized_seq, settings)

    assert len(tokenized_seq) == 1
    num_total_separators = 0
    for i, packed_seq in enumerate(tokenized_seq):
        assert len(packed_seq) == settings.context_size
        assert (packed_seq[0] == settings.vocab.AUTOREGRESS) or (
            packed_seq[0] == settings.vocab.SEPARATOR
            and packed_seq[1] == settings.vocab.AUTOREGRESS
        )
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
        range(0, 1500 + 1, settings.tick_token_every_n_ticks)
    )
    get_figure_and_open(
        events=parsed_events,
        delta=settings.delta,
        time_resolution=settings.time_resolution,
        path=(VISUALIZATIONS_PATH / (get_current_function_name() + ".html")),
        auto_open=False,
    )


def test_tokenize_with_ticks_for_lakh_ar(
    lmd_0_example_1_midi_path: Path,
    local_midi_vocab: Vocab,
) -> None:
    settings = AnticipationV2Settings(
        min_track_events=1,
        vocab=local_midi_vocab,
        # AR only
        num_autoregressive_seq_per_midi_file=1,
        num_instrument_anticipation_augmentations_per_midi_file=0,
        num_span_anticipation_augmentations_per_midi_file=0,
        tick_token_every_n_ticks=100,
        do_clip_overlapping_durations_in_midi_conversion=False,
        debug=True,
        debug_flush_remaining_token_buffer=False,
    )
    tokenized_seq = []
    stats = v2_tokenize([lmd_0_example_1_midi_path], tokenized_seq, settings)
    assert not stats.ignored_files
    assert len(tokenized_seq) == 8
    num_total_separators = 0
    for i, packed_seq in enumerate(tokenized_seq):
        assert len(packed_seq) == settings.context_size
        assert (packed_seq[0] == settings.vocab.AUTOREGRESS) or (
            packed_seq[0] == settings.vocab.SEPARATOR
            and packed_seq[1] == settings.vocab.AUTOREGRESS
        )
        num_total_separators += len(
            [x for x in packed_seq if x == settings.vocab.SEPARATOR]
        )

    _check_anticipation_rule_for_controls_and_token_ranges(tokenized_seq, settings)

    assert num_total_separators == 1
    parsed_events = Event.from_token_seq(
        [x for b in tokenized_seq for x in b], settings
    )
    get_figure_and_open(
        events=parsed_events,
        delta=settings.delta,
        time_resolution=settings.time_resolution,
        path=(VISUALIZATIONS_PATH / (get_current_function_name() + ".html")),
        auto_open=False,
    )


def test_tokenize_v2_lakh_span_anticipation(
    lmd_0_example_1_midi_path: Path,
    local_midi_vocab: Vocab,
) -> None:
    set_seed(0)

    tokens_to = []
    settings = AnticipationV2Settings(
        vocab=local_midi_vocab,
        num_autoregressive_seq_per_midi_file=0,
        num_instrument_anticipation_augmentations_per_midi_file=0,
        num_span_anticipation_augmentations_per_midi_file=1,
        tick_token_every_n_ticks=100,
        debug=True,
        debug_flush_remaining_token_buffer=True,
    )
    stats = v2_tokenize([lmd_0_example_1_midi_path], tokens_to, settings)
    _check_anticipation_rule_for_controls_and_token_ranges(tokens_to, settings)

    assert not stats.ignored_files
    assert settings.vocab.TICK == 17612

    parsed_events = Event.from_token_seq([x for b in tokens_to for x in b], settings)
    get_figure_and_open(
        events=parsed_events,
        delta=settings.delta,
        time_resolution=settings.time_resolution,
        path=(VISUALIZATIONS_PATH / (get_current_function_name() + ".html")),
        auto_open=False,
    )


def test_tokenize_v2_dense_sparse_piano_span_anticipation(
    dense_drums_sparse_piano_midi_path: Path,
    local_midi_vocab: Vocab,
) -> None:
    set_seed(0)
    tokens_to = []
    settings = AnticipationV2Settings(
        vocab=local_midi_vocab,
        min_track_events=0,
        min_track_time_in_seconds=0,
        num_autoregressive_seq_per_midi_file=0,
        num_instrument_anticipation_augmentations_per_midi_file=0,
        num_span_anticipation_augmentations_per_midi_file=1,
        tick_token_every_n_ticks=100,
        # need this for this file...
        do_clip_overlapping_durations_in_midi_conversion=True,
        debug=True,
        debug_flush_remaining_token_buffer=False,
    )
    with patch(
        "anticipation.v2.tokenize.random_time_partition",
        side_effect=[
            # first call to random_time_partition's return value
            (312, 3062),
            # second call's return value
            (292, 6300),
        ],
    ):
        stats = v2_tokenize([dense_drums_sparse_piano_midi_path], tokens_to, settings)
        assert len(tokens_to) == 2
        assert len(tokens_to[0]) == settings.context_size
        assert len(tokens_to[1]) == settings.context_size

    assert not stats.ignored_files
    _check_anticipation_rule_for_controls_and_token_ranges(tokens_to, settings)

    pe = Event.from_token_seq([x for b in tokens_to for x in b], settings)
    assert len(pe) == 732

    assert pe[0].is_separator()
    assert pe[1].is_anticipate()
    assert pe[2].is_tick()

    assert pe[3].midi_instrument() == 0
    assert pe[3].midi_time() == 0
    assert pe[3].midi_duration() == 100
    assert pe[3].note().name == "C3"

    # all events here...
    for i in range(4, 254):
        assert not pe[i].is_control

    # first control we encounter
    assert pe[254].is_control
    assert pe[254].midi_time() == 50
    assert pe[254].midi_duration() == 1
    assert pe[254].midi_instrument() == 128
    assert pe[254].note().name == "E1"

    assert pe[255].is_control
    assert pe[255].midi_duration() == 6
    assert pe[255].midi_instrument() == 128
    assert pe[255].note().name == "F#1"

    assert pe[256].is_control
    assert pe[256].midi_time() == 62
    assert pe[256].midi_duration() == 38
    assert pe[256].midi_instrument() == 128
    assert pe[256].note().name == "C1"

    # not the most precise way to check / understand if something goes wrong
    # but at least it is exact
    data_path = TEST_DATA_PATH / (get_current_function_name() + ".txt")
    expected_tokens = get_tokens_from_file(
        data_path,
    )
    assert tokens_to == expected_tokens

    get_figure_and_open(
        events=pe,
        delta=settings.delta,
        time_resolution=settings.time_resolution,
        path=(VISUALIZATIONS_PATH / (get_current_function_name() + ".html")),
        auto_open=False,
    )


def test_tokenize_v2_simple_two_instrument_midi(
    simple_two_instrument_midi_path: Path,
    local_midi_vocab: Vocab,
) -> None:
    set_seed(0)

    tokens_to = []
    settings = AnticipationV2Settings(
        vocab=local_midi_vocab,
        # very small context for testing boundaries
        context_size=80,
        min_track_events=0,
        min_track_time_in_seconds=0,
        num_autoregressive_seq_per_midi_file=0,
        num_instrument_anticipation_augmentations_per_midi_file=0,
        num_span_anticipation_augmentations_per_midi_file=1,
        do_clip_overlapping_durations_in_midi_conversion=False,
        tick_token_every_n_ticks=100,
        debug=True,
        debug_flush_remaining_token_buffer=False,
    )
    with patch(
        "anticipation.v2.tokenize.random_time_partition",
        side_effect=[
            (33, 1400),
            (31, 3000),
            (23, 4300),
            (19, 5800),
        ],
    ):
        # lock in a specific random pattern for choosing span splits
        stats = v2_tokenize([simple_two_instrument_midi_path], tokens_to, settings)

    _check_anticipation_rule_for_controls_and_token_ranges(tokens_to, settings)

    assert not stats.ignored_files
    parsed_events = Event.from_token_seq(
        [x for b in [tokens_to[0]] for x in b], settings
    )
    assert len(parsed_events) == 38

    assert parsed_events[0].as_tokens() == (settings.vocab.SEPARATOR,)
    assert parsed_events[1].as_tokens() == (settings.vocab.ANTICIPATE,)
    assert parsed_events[2].as_tokens() == (settings.vocab.TICK,)

    e_2 = parsed_events[3]
    assert e_2.absolute_time == 0
    assert e_2.midi_time() == 0
    assert e_2.midi_duration() == 100
    assert e_2.note().name == "C4"
    assert e_2.midi_instrument_name() == "Acoustic Grand Piano"
    assert e_2.midi_instrument() == 0
    assert e_2.as_tokens() == (
        settings.vocab.TIME_OFFSET + 0,
        settings.vocab.DUR_OFFSET + 100,
        settings.vocab.NOTE_OFFSET + 72,
    )

    e_3 = parsed_events[4]
    assert e_3.absolute_time == 0
    assert e_3.midi_time() == 0
    assert e_3.midi_duration() == 400
    assert e_3.note().name == "E2"
    assert e_3.midi_instrument_name() == "Piccolo"
    assert e_3.midi_instrument() == 72
    assert e_3.as_tokens() == (
        settings.vocab.TIME_OFFSET + 0,
        settings.vocab.DUR_OFFSET + 400,
        settings.vocab.NOTE_OFFSET + 9_268,
    )

    e_4 = parsed_events[5]
    assert e_4.as_tokens() == (settings.vocab.TICK,)

    e_5 = parsed_events[6]
    assert e_5.absolute_time == 100
    assert e_5.midi_time() == 0  # relativized to the tick
    assert e_5.midi_duration() == 100
    assert e_5.note().name == "B3"
    assert e_5.midi_instrument() == 0
    assert e_5.as_tokens() == (
        settings.vocab.TIME_OFFSET + 0,
        settings.vocab.DUR_OFFSET + 100,
        settings.vocab.NOTE_OFFSET + 71,
    )

    e_6 = parsed_events[7]
    assert e_6.as_tokens() == (settings.vocab.TICK,)

    e_7 = parsed_events[8]
    assert e_7.absolute_time == 200
    assert e_7.midi_time() == 0  # relativized to the tick
    assert e_7.midi_duration() == 100
    assert e_7.note().name == "A#3"
    assert e_7.midi_instrument() == 0
    assert e_7.as_tokens() == (
        settings.vocab.TIME_OFFSET + 0,
        settings.vocab.DUR_OFFSET + 100,
        settings.vocab.NOTE_OFFSET + 70,
    )

    e_8 = parsed_events[9]
    assert e_8.as_tokens() == (settings.vocab.TICK,)

    e_9 = parsed_events[10]
    assert e_9.absolute_time == 300
    assert e_9.midi_time() == 0  # relativized to the tick
    assert e_9.midi_duration() == 100
    assert e_9.note().name == "A3"
    assert e_9.midi_instrument() == 0
    assert e_9.as_tokens() == (
        settings.vocab.TIME_OFFSET + 0,
        settings.vocab.DUR_OFFSET + 100,
        settings.vocab.NOTE_OFFSET + 69,
    )

    e_10 = parsed_events[11]
    assert e_10.as_tokens() == (settings.vocab.TICK,)

    e_11 = parsed_events[12]
    assert e_11.absolute_time == 400
    assert e_11.midi_time() == 0
    assert e_11.midi_duration() == 100
    assert e_11.note().name == "G#3"
    assert e_11.midi_instrument() == 0
    assert e_11.as_tokens() == (
        settings.vocab.TIME_OFFSET + 0,
        settings.vocab.DUR_OFFSET + 100,
        settings.vocab.NOTE_OFFSET + 68,
    )
    e_12 = parsed_events[13]
    assert e_12.absolute_time == 400
    assert e_12.midi_time() == 0
    assert e_12.midi_duration() == 400
    assert e_12.note().name == "F2"
    assert e_12.midi_instrument() == 72
    assert e_12.as_tokens() == (
        settings.vocab.TIME_OFFSET + 0,
        settings.vocab.DUR_OFFSET + 400,
        settings.vocab.NOTE_OFFSET + 9_269,
    )
    e_13 = parsed_events[14]
    assert e_13.as_tokens() == (settings.vocab.TICK,)

    e_14 = parsed_events[15]
    assert e_14.absolute_time == 500
    assert e_14.midi_time() == 0
    assert e_14.midi_duration() == 100
    assert e_14.note().name == "G3"
    assert e_14.midi_instrument() == 0
    assert e_14.as_tokens() == (
        settings.vocab.TIME_OFFSET + 0,
        settings.vocab.DUR_OFFSET + 100,
        settings.vocab.NOTE_OFFSET + 67,
    )

    # ... few more events...

    # we are most interested in what happens at the very
    # end when a span happens
    final_span_token_tuples = []
    for e in parsed_events[20:]:
        final_span_token_tuples.append(e.as_tokens())

    assert final_span_token_tuples == [
        (settings.vocab.TICK,),
        # -- CONTROL ---
        (
            settings.vocab.ATIME_OFFSET + 0,
            settings.vocab.ADUR_OFFSET + 100,
            settings.vocab.ANOTE_OFFSET + 71,
        ),
        (
            settings.vocab.TIME_OFFSET + 0,
            settings.vocab.DUR_OFFSET + 100,
            settings.vocab.NOTE_OFFSET + 64,
        ),
        (
            settings.vocab.TIME_OFFSET + 0,
            settings.vocab.DUR_OFFSET + 400,
            settings.vocab.NOTE_OFFSET + 9_270,
        ),
        (settings.vocab.TICK,),
        # -- CONTROL ---
        (
            settings.vocab.ATIME_OFFSET + 0,
            settings.vocab.ADUR_OFFSET + 100,
            settings.vocab.ANOTE_OFFSET + 70,
        ),
        # --------------
        (
            settings.vocab.TIME_OFFSET + 0,
            settings.vocab.DUR_OFFSET + 100,
            settings.vocab.NOTE_OFFSET + 63,
        ),
        (settings.vocab.TICK,),
        # -- CONTROL ---
        (
            settings.vocab.ATIME_OFFSET + 0,
            settings.vocab.ADUR_OFFSET + 100,
            settings.vocab.ANOTE_OFFSET + 69,
        ),
        # --------------
        (
            settings.vocab.TIME_OFFSET + 0,
            settings.vocab.DUR_OFFSET + 100,
            settings.vocab.NOTE_OFFSET + 62,
        ),
        (settings.vocab.TICK,),
        (
            settings.vocab.ATIME_OFFSET + 0,
            settings.vocab.ADUR_OFFSET + 100,
            settings.vocab.ANOTE_OFFSET + 68,
        ),
        (
            settings.vocab.TIME_OFFSET + 0,
            settings.vocab.DUR_OFFSET + 100,
            settings.vocab.NOTE_OFFSET + 61,
        ),
        (settings.vocab.TICK,),
        (
            settings.vocab.TIME_OFFSET + 0,
            settings.vocab.DUR_OFFSET + 100,
            settings.vocab.NOTE_OFFSET + 72,
        ),
        (
            settings.vocab.TIME_OFFSET + 0,
            settings.vocab.DUR_OFFSET + 400,
            settings.vocab.NOTE_OFFSET + 9_271,
        ),
        (settings.vocab.TICK,),
        (settings.vocab.TICK,),
    ]

    get_figure_and_open(
        events=parsed_events,
        delta=settings.delta,
        time_resolution=settings.time_resolution,
        path=(VISUALIZATIONS_PATH / (get_current_function_name() + ".html")),
        auto_open=False,
    )


def test_tokenize_v2_dense_sparse_piano_ar_only(
    dense_drums_sparse_piano_midi_path: Path,
    local_midi_vocab: Vocab,
) -> None:
    # testing a simple sequence with autoregressive only settings
    # also this MIDI file has strange overlaps, so ensure those are gone
    tokens_to = []
    settings = AnticipationV2Settings(
        vocab=local_midi_vocab,
        context_size=1024,
        num_autoregressive_seq_per_midi_file=1,
        num_instrument_anticipation_augmentations_per_midi_file=0,
        num_span_anticipation_augmentations_per_midi_file=0,
        do_clip_overlapping_durations_in_midi_conversion=False,
        tick_token_every_n_ticks=100,
        debug=True,
        debug_flush_remaining_token_buffer=True,
    )
    stats = v2_tokenize([dense_drums_sparse_piano_midi_path], tokens_to, settings)

    _check_anticipation_rule_for_controls_and_token_ranges(tokens_to, settings)

    assert not stats.ignored_files
    # fills context 3 times, including end buffer
    assert len(tokens_to) == 3
    assert len(tokens_to[0]) == settings.context_size
    assert len(tokens_to[1]) == settings.context_size
    assert len(tokens_to[2]) == 426

    flattened_tokens = [x for b in tokens_to for x in b]

    # 1 control token for every context length
    num_ar_tokens = flattened_tokens.count(settings.vocab.AUTOREGRESS)
    assert num_ar_tokens == 3
    num_tick_tokens = flattened_tokens.count(settings.vocab.TICK)
    assert num_tick_tokens == 85
    num_sep_tokens = flattened_tokens.count(settings.vocab.SEPARATOR)
    assert num_sep_tokens == 1

    # no duplicate events
    parsed_events = Event.from_token_seq([x for b in tokens_to for x in b], settings)
    assert len(parsed_events) == len(set(parsed_events)) == 883

    get_figure_and_open(
        events=parsed_events,
        delta=settings.delta,
        time_resolution=settings.time_resolution,
        path=(VISUALIZATIONS_PATH / (get_current_function_name() + ".html")),
        auto_open=False,
    )


def test_no_information_loss_or_added_when_anticipation_applied_dense_sparse_piano(
    dense_drums_sparse_piano_midi_path: Path,
    local_midi_vocab: Vocab,
) -> None:
    set_seed(0)

    # tokenize using AR settings
    tokens_ar = []
    settings_ar = AnticipationV2Settings(
        vocab=local_midi_vocab,
        num_autoregressive_seq_per_midi_file=1,
        num_instrument_anticipation_augmentations_per_midi_file=0,
        num_span_anticipation_augmentations_per_midi_file=0,
        do_clip_overlapping_durations_in_midi_conversion=True,
        tick_token_every_n_ticks=100,
        debug=True,
        debug_flush_remaining_token_buffer=True,
    )
    v2_tokenize([dense_drums_sparse_piano_midi_path], tokens_ar, settings_ar)
    _check_anticipation_rule_for_controls_and_token_ranges(tokens_ar, settings_ar)
    events_ar = Event.from_list_of_token_seq(tokens_ar, settings_ar)

    get_figure_and_open(
        events=events_ar,
        delta=settings_ar.delta,
        time_resolution=settings_ar.time_resolution,
        path=(VISUALIZATIONS_PATH / (get_current_function_name() + "_ar.html")),
        auto_open=False,
    )

    # tokenize using with span anticipation
    tokens_span = []
    settings_span = AnticipationV2Settings(
        vocab=local_midi_vocab,
        num_autoregressive_seq_per_midi_file=0,
        num_instrument_anticipation_augmentations_per_midi_file=0,
        num_span_anticipation_augmentations_per_midi_file=1,
        do_clip_overlapping_durations_in_midi_conversion=True,
        tick_token_every_n_ticks=100,
        debug=True,
        debug_flush_remaining_token_buffer=True,
    )
    v2_tokenize([dense_drums_sparse_piano_midi_path], tokens_span, settings_span)
    _check_anticipation_rule_for_controls_and_token_ranges(tokens_span, settings_span)
    events_span = Event.from_list_of_token_seq(tokens_span, settings_span)

    # check number of ticks is constant, note that this won't be true for
    # instrument anticipation since we need to prefix with ticks...
    num_ticks_ar = len([x for x in events_ar if x.is_tick()])
    num_ticks_span = len([x for x in events_span if x.is_tick()])
    assert num_ticks_ar == num_ticks_span

    get_figure_and_open(
        events=events_span,
        delta=settings_ar.delta,
        time_resolution=settings_ar.time_resolution,
        path=(VISUALIZATIONS_PATH / (get_current_function_name() + "_span.html")),
        auto_open=False,
    )

    _check_is_musically_same(tokens_ar, tokens_span, settings_ar, settings_span)


def test_tokenize_v2_lakh_instrument_anticipation_blockwise(
    lmd_0_example_1_midi_path: Path,
    local_midi_vocab: Vocab,
) -> None:
    set_seed(0)

    with patch(
        "anticipation.v2.tokenize.np.random.choice",
        return_value=[MIDI_DRUMS_PROGRAM_CODE],
    ):
        settings = AnticipationV2Settings(
            vocab=local_midi_vocab,
            num_autoregressive_seq_per_midi_file=0,
            num_instrument_anticipation_augmentations_per_midi_file=1,
            num_span_anticipation_augmentations_per_midi_file=0,
            tick_token_every_n_ticks=100,
            debug=True,
            debug_flush_remaining_token_buffer=True,
        )
        tokens_to = []
        stats = v2_tokenize([lmd_0_example_1_midi_path], tokens_to, settings)
        assert stats.num_tokenized_files == 1
        assert stats.num_lost_tokens_left_in_buffer == 0

        _check_anticipation_rule_for_controls_and_token_ranges(tokens_to, settings)

    assert not stats.ignored_files
    parsed_events = Event.from_list_of_token_seq(tokens_to, settings)
    get_figure_and_open(
        events=parsed_events,
        delta=settings.delta,
        time_resolution=settings.time_resolution,
        path=(VISUALIZATIONS_PATH / (get_current_function_name() + ".html")),
        auto_open=False,
    )


def test_tokenize_v2_dense_drums_sparse_piano_instrument_anticipation_blockwise(
    dense_drums_sparse_piano_midi_path: Path,
    local_midi_vocab: Vocab,
) -> None:
    with patch(
        "anticipation.v2.tokenize.np.random.choice",
        return_value=[MIDI_DRUMS_PROGRAM_CODE],
    ):
        settings = AnticipationV2Settings(
            vocab=local_midi_vocab,
            num_autoregressive_seq_per_midi_file=0,
            num_instrument_anticipation_augmentations_per_midi_file=1,
            num_span_anticipation_augmentations_per_midi_file=0,
            tick_token_every_n_ticks=100,
            debug=True,
            debug_flush_remaining_token_buffer=True,
            do_clip_overlapping_durations_in_midi_conversion=False,
        )
        result: TokenizedMIDIFileResult = _tokenize_midi_file(
            dense_drums_sparse_piano_midi_path, settings
        )

        # check that there are no repeated notes in the tokenization result, we have issues in the
        # event sequence... prevent exactly the same event from appearing if because that can happen
        # in MIDI files sometimes...
        seen = set()
        for i in range(0, len(result.events), 3):
            my_event = tuple(result.events[i : i + 3])
            assert my_event not in seen
            seen.add(my_event)

        # this test bypasses the sequence packer, so it won't have
        # separator tokens or other control tokens, also there's no truncation
        # logic to fit it into the context window... so we can assume every event
        # is complete, and the sequence won't have SEP or AR/ANTI tokens.
        token_stream: TokenStream
        token_stream = _get_augmentation_instrument(
            result.events,
            result.all_midi_program_codes,
            settings,
        )
        # ensure that the associated control prefix denotes anticipatory
        # sequence
        assert token_stream.control_prefix == (settings.vocab.ANTICIPATE,)

    token_list = list(token_stream)
    assert len(token_list) == 882

    # token list should be a list of tuples of int, e.g.
    # [(TICK,), (0, 50, 12), (50, 50, 12), ...]
    assert isinstance(token_list[0], tuple)
    assert isinstance(token_list[0][0], int)

    tick_idxs: list[int] = []
    for i, event in enumerate(token_list):
        if len(event) == 1:
            if event == (settings.vocab.TICK,):
                # this is a tick
                tick_idxs.append(i)
            else:
                # if it is length 1 and not a tick,
                # then it must be a special token
                assert event[0] >= settings.vocab.SPECIAL_OFFSET

    # expect this many ticks
    assert len(tick_idxs) == 88

    # sequence should start with a tick, but after the sep
    assert tick_idxs[0] == 0
    for i, tick_idx in enumerate(tick_idxs[:-1]):
        curr_tick_idx = tick_idx
        next_tick_idx = tick_idxs[i + 1]
        has_controls = False
        has_events = False

        # there might be a subsequence between two ticks that has only events,
        # in that case, controls need not immediately follow the tick because there
        # are none
        for k in range(curr_tick_idx, next_tick_idx):
            has_controls = (
                has_controls or token_list[k][0] >= settings.vocab.CONTROL_OFFSET
            )
            has_events = has_events or token_list[k][0] < settings.vocab.DUR_OFFSET

        if has_controls:
            # it must be the case that a control token directly follows every tick
            triple_after_tick: tuple[Token, ...] = token_list[curr_tick_idx + 1]
            assert triple_after_tick[0] >= settings.vocab.CONTROL_OFFSET
            assert triple_after_tick[1] >= settings.vocab.CONTROL_OFFSET
            assert triple_after_tick[2] >= settings.vocab.CONTROL_OFFSET

    ungrouped_tokens = [x for b in token_list for x in b]
    assert len(ungrouped_tokens) == 2470

    # parse and plot
    parsed_events = Event.from_token_seq(ungrouped_tokens, settings)
    get_figure_and_open(
        events=parsed_events,
        delta=settings.delta,
        time_resolution=settings.time_resolution,
        path=(VISUALIZATIONS_PATH / (get_current_function_name() + ".html")),
        auto_open=False,
    )


def test_absolute_time_is_correct_with_ticks(lmd_0_example_1_midi_path: Path) -> None:
    settings_no_ticks = AnticipationV2Settings(
        min_track_events=1,
        vocab=make_vocab(
            tick_token_every_n_ticks=0,
            max_note_duration_in_seconds=5,
            time_resolution=100,
        ),
        # AR only
        num_autoregressive_seq_per_midi_file=1,
        num_instrument_anticipation_augmentations_per_midi_file=0,
        num_span_anticipation_augmentations_per_midi_file=0,
        debug=True,
        debug_flush_remaining_token_buffer=False,
        tick_token_every_n_ticks=0,
    )
    # tokenize and parse without ticks added
    events_without_ticks = []
    stats = v2_tokenize(
        [lmd_0_example_1_midi_path], events_without_ticks, settings_no_ticks
    )
    assert len(events_without_ticks) == 8
    assert not stats.ignored_files
    assert stats.num_tokenized_files == 1
    assert stats.num_lost_tokens_left_in_buffer == 406

    events_without_ticks = Event.from_token_seq(
        [x for b in events_without_ticks for x in b], settings_no_ticks
    )
    events_without_ticks = [
        x
        for x in events_without_ticks
        if x.special_code == EventSpecialCode.TYPICAL_EVENT
    ]

    # tokenize and parse WITH ticks added
    events_include_ticks = []
    settings_with_ticks = AnticipationV2Settings(
        min_track_events=1,
        vocab=make_vocab(
            100,
            settings_no_ticks.max_note_duration_in_seconds,
            settings_no_ticks.time_resolution,
        ),
        # AR only
        num_autoregressive_seq_per_midi_file=1,
        num_instrument_anticipation_augmentations_per_midi_file=0,
        num_span_anticipation_augmentations_per_midi_file=0,
        debug=True,
        tick_token_every_n_ticks=100,
    )
    stats = v2_tokenize(
        [lmd_0_example_1_midi_path], events_include_ticks, settings_with_ticks
    )
    assert not stats.ignored_files
    _check_anticipation_rule_for_controls_and_token_ranges(
        events_include_ticks, settings_with_ticks
    )
    events_include_ticks = Event.from_token_seq(
        [x for b in events_include_ticks for x in b], settings_with_ticks
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
        debug=True,
        debug_flush_remaining_token_buffer=False,
        tick_token_every_n_ticks=100,
    )
    midi_files = [lmd_0_example_1_midi_path, lmd_0_example_2_midi_path]

    in_memory_tokens = []
    stats = v2_tokenize(midi_files, in_memory_tokens, settings)
    assert not stats.ignored_files
    assert stats.num_sequences == 97
    assert stats.num_given_files == 2
    assert stats.num_tokenized_files == 2
    assert stats.num_lost_tokens_left_in_buffer == 182
    assert stats.num_pitch_transpose_augmentations == 0
    assert stats.num_anticipate_tokens == 0

    # there is one sequence in here where the first song ends and the
    # other begins, when we start the next one, it must be the case
    # that the relevant control exists at that boundary
    assert stats.num_autoregress_tokens == 98
    assert stats.num_times_end_was_truncated == 57

    # should be one per tokenized file - and it appears in prefix
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
    assert settings.context_size == expected_settings.context_size
    assert settings.vocab == expected_settings.vocab
    assert lmd_0_example_1_actual_events[0] == lmd_0_example_1_expected_events[0]
    assert lmd_0_example_1_actual_events == lmd_0_example_1_expected_events
    assert lmd_0_example_1_actual_tokens == lmd_0_example_1_expected_tokens

    _check_anticipation_rule_for_controls_and_token_ranges(in_memory_tokens, settings)

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


def test_sequence_packing_file_boundaries_with_mixed_augmentations(
    lmd_0_example_1_midi_path: Path,
    lmd_0_example_2_midi_path: Path,
    dense_drums_sparse_piano_midi_path: Path,
    local_midi_vocab: Vocab,
) -> None:
    set_seed(0)
    settings = AnticipationV2Settings(
        min_track_events=1,
        vocab=local_midi_vocab,
        # do one of every kind of augmentation
        num_autoregressive_seq_per_midi_file=1,
        num_instrument_anticipation_augmentations_per_midi_file=1,
        num_span_anticipation_augmentations_per_midi_file=1,
        do_clip_overlapping_durations_in_midi_conversion=False,
        debug=True,
        debug_flush_remaining_token_buffer=False,
        tick_token_every_n_ticks=100,
    )
    midi_files = [
        lmd_0_example_1_midi_path,
        lmd_0_example_2_midi_path,
        dense_drums_sparse_piano_midi_path,
    ]
    sequences = []
    stats = v2_tokenize(midi_files, sequences, settings)
    assert not stats.ignored_files
    assert stats.num_given_files == 3
    assert stats.num_tokenized_files == 3
    assert stats.num_sequences == len(sequences) == 298
    boundaries = []
    for i in range(1, len(sequences)):
        curr_seq = sequences[i]
        if curr_seq.count(settings.vocab.SEPARATOR) > 0:
            boundaries.append(i)

    # 3 files, 3 styles of tokenizing for each of them
    # sep is in between all 9, so there are 8 boundaries.
    assert len(boundaries) == 8

    # boundary between `lmd_0_example_1_midi_path` AR and
    # `lmd_0_example_1_midi_path` Instrument Anticipation
    boundary_1: list[int] = sequences[boundaries[0]]
    split_idx = boundary_1.index(settings.vocab.SEPARATOR)

    assert boundary_1[0] == settings.vocab.AUTOREGRESS
    assert boundary_1[split_idx + 1] == settings.vocab.ANTICIPATE

    # boundary between `lmd_0_example_1_midi_path` Instrument Anticipation and
    # `lmd_0_example_1_midi_path` Span Anticipation
    boundary_2: list[int] = sequences[boundaries[1]]
    split_idx = boundary_2.index(settings.vocab.SEPARATOR)

    assert boundary_2[0] == settings.vocab.ANTICIPATE
    assert boundary_2[split_idx + 1] == settings.vocab.ANTICIPATE

    # boundary between `lmd_0_example_1_midi_path` Span Anticipation and
    # `lmd_0_example_2_midi_path` AR
    boundary_3: list[int] = sequences[boundaries[2]]
    split_idx = boundary_3.index(settings.vocab.SEPARATOR)

    assert boundary_3[0] == settings.vocab.ANTICIPATE
    assert boundary_3[split_idx + 1] == settings.vocab.AUTOREGRESS

    # boundary between `lmd_0_example_2_midi_path` AR
    # `lmd_0_example_2_midi_path` Instrument Anticipation
    boundary_4: list[int] = sequences[boundaries[3]]
    split_idx = boundary_4.index(settings.vocab.SEPARATOR)

    assert boundary_4[0] == settings.vocab.AUTOREGRESS
    assert boundary_4[split_idx + 1] == settings.vocab.ANTICIPATE

    # boundary between `lmd_0_example_2_midi_path` Instrument Anticipation
    # `lmd_0_example_2_midi_path` Span Anticipation
    boundary_5: list[int] = sequences[boundaries[4]]
    split_idx = boundary_5.index(settings.vocab.SEPARATOR)

    assert boundary_5[0] == settings.vocab.ANTICIPATE
    assert boundary_5[split_idx + 1] == settings.vocab.ANTICIPATE

    # boundary between `lmd_0_example_2_midi_path` Span Anticipation
    # `dense_drums_sparse_piano_midi_path` AR
    boundary_6: list[int] = sequences[boundaries[5]]
    split_idx = boundary_6.index(settings.vocab.SEPARATOR)

    assert boundary_6[0] == settings.vocab.ANTICIPATE
    assert boundary_6[split_idx + 1] == settings.vocab.AUTOREGRESS

    # boundary between `dense_drums_sparse_piano_midi_path` AR and
    # `dense_drums_sparse_piano_midi_path` Instrument Anticipation
    boundary_7: list[int] = sequences[boundaries[6]]
    split_idx = boundary_7.index(settings.vocab.SEPARATOR)

    assert boundary_7[0] == settings.vocab.AUTOREGRESS
    assert boundary_7[split_idx + 1] == settings.vocab.ANTICIPATE

    # boundary between `dense_drums_sparse_piano_midi_path` Instrument Anticipation
    # `dense_drums_sparse_piano_midi_path` Span Anticipation
    boundary_8: list[int] = sequences[boundaries[7]]
    split_idx = boundary_8.index(settings.vocab.SEPARATOR)

    assert boundary_8[0] == settings.vocab.ANTICIPATE
    assert boundary_8[split_idx + 1] == settings.vocab.ANTICIPATE


def test_sequence_packing_file_boundaries_with_mixed_augmentations_2(
    lmd_0_example_1_midi_path: Path,
    lmd_0_example_2_midi_path: Path,
    local_midi_vocab: Vocab,
) -> None:
    set_seed(0)
    settings = AnticipationV2Settings(
        min_track_events=1,
        vocab=local_midi_vocab,
        # AR only
        num_autoregressive_seq_per_midi_file=1,
        num_instrument_anticipation_augmentations_per_midi_file=0,
        num_span_anticipation_augmentations_per_midi_file=0,
        do_clip_overlapping_durations_in_midi_conversion=False,
        debug=True,
        debug_flush_remaining_token_buffer=False,
        tick_token_every_n_ticks=100,
    )
    midi_files = [lmd_0_example_1_midi_path, lmd_0_example_2_midi_path]

    in_memory_tokens = []
    stats = v2_tokenize([lmd_0_example_2_midi_path], in_memory_tokens, settings)
    assert not stats.ignored_files
    # assert stats.num_given_files == 2
    # assert stats.num_tokenized_files == 2
    parsed_events = Event.from_token_seq(
        [x for b in in_memory_tokens for x in b], settings
    )

    # assert len(in_memory_tokens) == 1
    get_figure_and_open(
        events=parsed_events,
        delta=settings.delta,
        time_resolution=settings.time_resolution,
        path=(VISUALIZATIONS_PATH / (get_current_function_name() + ".html")),
        auto_open=False,
    )


def test_extremely_long_span_augmentation(
    lmd_0_example_2_midi_path: Path,
    local_midi_vocab: Vocab,
) -> None:
    set_seed(0)
    settings = AnticipationV2Settings(
        min_track_events=1,
        vocab=local_midi_vocab,
        num_autoregressive_seq_per_midi_file=0,
        num_instrument_anticipation_augmentations_per_midi_file=0,
        num_span_anticipation_augmentations_per_midi_file=1,
        do_clip_overlapping_durations_in_midi_conversion=False,
        debug=True,
        debug_flush_remaining_token_buffer=False,
        tick_token_every_n_ticks=100,
    )
    sequences = []

    # this particular file is very long and also has some very note-dense parts
    # this test helped uncover a problem in sampling by the raw index of the context,
    # instead we sample in time, which prevents that issue
    stats = v2_tokenize([lmd_0_example_2_midi_path], sequences, settings)
    assert not stats.ignored_files
    _check_anticipation_rule_for_controls_and_token_ranges(sequences, settings)
    assert len(sequences) == 88

    num_good_spans = 0
    for seq in sequences:
        parsed_events = Event.from_token_seq(seq, settings)

        # unfortunately it may not always be possible to do a time-based split
        # this is because the note density might be so high that everything
        # in the context is within delta seconds
        num_controls = len([x for x in parsed_events if x.is_control])
        if num_controls > 0:
            num_good_spans += 1

    assert num_good_spans + stats.num_times_span_had_insufficient_time == len(sequences)


def test_no_segfault(lmd_1_example_0_midi_path: Path) -> None:
    settings = AnticipationV2Settings(
        vocab=Vocab(),
        debug=False,
        # AR only
        num_autoregressive_seq_per_midi_file=1,
        num_instrument_anticipation_augmentations_per_midi_file=0,
        num_span_anticipation_augmentations_per_midi_file=0,
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
        tick_token_every_n_ticks=100,
        do_clip_overlapping_durations_in_midi_conversion=False,
    )
    tokens_to = []
    stats: TokenizationStatSummary = v2_tokenize(
        [c_major_midi_path], output=tokens_to, settings=settings
    )
    _check_anticipation_rule_for_controls_and_token_ranges(tokens_to, settings)
    assert stats.num_times_end_was_truncated == 2
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
        # sequence prefix
        settings.vocab.SEPARATOR,
        # ---
        settings.vocab.AUTOREGRESS,
        settings.vocab.TICK,
        *event_0,
        # event 1 is truncated...
        *event_1[:2],
    ]
    seq_1 = tokens_to[1]
    assert seq_1 == [
        # special
        settings.vocab.AUTOREGRESS,
        # event 1 must start the next sequence then
        *event_1,
        settings.vocab.TICK,
        *event_2,
    ]
    seq_2 = tokens_to[2]
    assert seq_2 == [
        # special
        settings.vocab.AUTOREGRESS,
        *event_3,
        settings.vocab.TICK,
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
        *event_8,
    ]
    seq_5 = tokens_to[5]
    assert seq_5 == [
        settings.vocab.AUTOREGRESS,
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
        # truncated
        *event_16[:2],
    ]
    seq_9 = tokens_to[9]
    assert seq_9 == [
        settings.vocab.AUTOREGRESS,
        *event_16,
        *event_17,
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
    parsed_events = Event.from_list_of_token_seq(tokens_to, settings)
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
        path=(VISUALIZATIONS_PATH / (get_current_function_name() + ".html")),
        auto_open=False,
    )


def test_apply_pitch_augmentation(c_major_midi_path: Path) -> None:
    settings = AnticipationV2Settings(
        vocab=Vocab(),
        debug=False,
        debug_flush_remaining_token_buffer=False,
        min_track_events=1,
        min_track_time_in_seconds=1,
        # I set this so that each sequence exactly fits in the context
        # there should be no truncations
        context_size=105,
        num_autoregressive_seq_per_midi_file=1,
        num_instrument_anticipation_augmentations_per_midi_file=0,
        num_span_anticipation_augmentations_per_midi_file=0,
        tick_token_every_n_ticks=100,
        # 3 tranpositions that will work, one that will fail - our implementation
        # should be able to handle the one that pushes notes out of bounds - it
        # should just ignore it
        augmentation_pitch_shifts=(-3, -2, -1, 1, 2, 3, 127),
        do_clip_overlapping_durations_in_midi_conversion=False,
    )
    tokens_to = []
    stats: TokenizationStatSummary = v2_tokenize(
        [c_major_midi_path], output=tokens_to, settings=settings
    )

    assert stats.num_pitch_transpose_augmentations == 6
    assert stats.num_times_end_was_truncated == 0
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
        # ensure every context starts with the flag token
        assert (seq[0] == settings.vocab.AUTOREGRESS) or (
            seq[0] == settings.vocab.SEPARATOR and seq[1] == settings.vocab.AUTOREGRESS
        )

        # parse events
        parsed_events = Event.from_token_seq(seq, settings)
        notes = [x.note().name for x in parsed_events if x.is_note_event()]
        assert notes == expected_note_names[i]

    _check_anticipation_rule_for_controls_and_token_ranges(tokens_to, settings)
    all_tokens = [x for b in tokens_to for x in b]

    # should be 1 separator for every transposition
    # there are 6 alterations, 1 original
    num_sep_tokens = all_tokens.count(settings.vocab.SEPARATOR)
    assert num_sep_tokens == 7


def test_no_information_loss_dense_drums_sparse_piano_for_all_anticipation_types(
    dense_drums_sparse_piano_midi_path: Path,
    local_midi_vocab: Vocab,
) -> None:
    ar_settings = AnticipationV2Settings(
        min_track_events=1,
        context_size=512,
        vocab=local_midi_vocab,
        num_autoregressive_seq_per_midi_file=1,
        num_instrument_anticipation_augmentations_per_midi_file=0,
        num_span_anticipation_augmentations_per_midi_file=0,
        do_clip_overlapping_durations_in_midi_conversion=False,
        debug=True,
        debug_flush_remaining_token_buffer=True,
        tick_token_every_n_ticks=100,
    )
    # the reference
    ar_token_seqs = []
    stats: TokenizationStatSummary = v2_tokenize(
        [dense_drums_sparse_piano_midi_path], output=ar_token_seqs, settings=ar_settings
    )
    assert not stats.ignored_files
    assert stats.num_lost_tokens_left_in_buffer == 0
    ar_events = Event.from_list_of_token_seq(ar_token_seqs, ar_settings)
    get_figure_and_open(
        events=ar_events,
        delta=ar_settings.delta,
        time_resolution=ar_settings.time_resolution,
        path=(VISUALIZATIONS_PATH / (get_current_function_name() + "_ar.html")),
        auto_open=False,
    )
    all_instruments = list(
        sorted(set([x.midi_instrument() for x in ar_events if x.is_note_event()]))
    )
    # piano, drums
    assert all_instruments == [0, 128]

    span_settings = AnticipationV2Settings(
        min_track_events=ar_settings.min_track_events,
        context_size=ar_settings.context_size,
        vocab=local_midi_vocab,
        num_autoregressive_seq_per_midi_file=0,
        num_instrument_anticipation_augmentations_per_midi_file=0,
        num_span_anticipation_augmentations_per_midi_file=1,
        do_clip_overlapping_durations_in_midi_conversion=ar_settings.do_clip_overlapping_durations_in_midi_conversion,
        debug=True,
        debug_flush_remaining_token_buffer=True,
        tick_token_every_n_ticks=ar_settings.tick_token_every_n_ticks,
    )
    for s in range(0, 10):
        # try a few random seeds since the span region is randomly decided
        set_seed(s)
        span_token_seqs = []
        stats: TokenizationStatSummary = v2_tokenize(
            [dense_drums_sparse_piano_midi_path],
            output=span_token_seqs,
            settings=span_settings,
        )
        _check_anticipation_rule_for_controls_and_token_ranges(
            span_token_seqs, span_settings
        )
        assert not stats.ignored_files
        assert stats.num_lost_tokens_left_in_buffer == 0

        # check that span anticipation and autoregressive sequence tokenization are
        # the same (musically speaking). This is important because anticipation does
        # not add or remove information, it just restructures it. The original AR
        # sequence must be recoverable from an anticipated sequence of the same
        # piece.
        _check_is_musically_same(
            ar_token_seqs, span_token_seqs, ar_settings, span_settings
        )

    instr_settings = AnticipationV2Settings(
        min_track_events=ar_settings.min_track_events,
        context_size=ar_settings.context_size,
        vocab=local_midi_vocab,
        num_autoregressive_seq_per_midi_file=0,
        num_instrument_anticipation_augmentations_per_midi_file=1,
        num_span_anticipation_augmentations_per_midi_file=0,
        do_clip_overlapping_durations_in_midi_conversion=ar_settings.do_clip_overlapping_durations_in_midi_conversion,
        debug=True,
        debug_flush_remaining_token_buffer=True,
        tick_token_every_n_ticks=ar_settings.tick_token_every_n_ticks,
    )
    for instrument_aug_choice in all_instruments:
        with patch(
            "anticipation.v2.tokenize._sample_instrument_subset",
            return_value=[[instrument_aug_choice]],
        ):
            instr_token_seqs = []
            stats: TokenizationStatSummary = v2_tokenize(
                [dense_drums_sparse_piano_midi_path],
                output=instr_token_seqs,
                settings=instr_settings,
            )
            _check_anticipation_rule_for_controls_and_token_ranges(
                instr_token_seqs, instr_settings
            )
            assert not stats.ignored_files
            assert stats.num_lost_tokens_left_in_buffer == 0
            _check_is_musically_same(
                ar_token_seqs, instr_token_seqs, ar_settings, instr_settings
            )


def test_no_information_loss_for_multiple_packed_files(
    dense_drums_sparse_piano_midi_path: Path,
    simple_two_instrument_midi_path: Path,
    lmd_0_example_1_midi_path: Path,
    local_midi_vocab: Vocab,
) -> None:
    files_to_tokenize = [
        dense_drums_sparse_piano_midi_path,
        simple_two_instrument_midi_path,
        lmd_0_example_1_midi_path,
    ]
    ar_settings = AnticipationV2Settings(
        min_track_events=1,
        context_size=512,
        vocab=local_midi_vocab,
        num_autoregressive_seq_per_midi_file=1,
        num_instrument_anticipation_augmentations_per_midi_file=0,
        num_span_anticipation_augmentations_per_midi_file=0,
        do_clip_overlapping_durations_in_midi_conversion=False,
        debug=True,
        debug_flush_remaining_token_buffer=True,
        tick_token_every_n_ticks=100,
    )
    # the reference
    ar_token_seqs = []
    stats: TokenizationStatSummary = v2_tokenize(
        files_to_tokenize, output=ar_token_seqs, settings=ar_settings
    )
    assert not stats.ignored_files
    assert stats.num_lost_tokens_left_in_buffer == 0
    ar_events = Event.from_list_of_token_seq(ar_token_seqs, ar_settings)
    get_figure_and_open(
        events=ar_events,
        delta=ar_settings.delta,
        time_resolution=ar_settings.time_resolution,
        path=(VISUALIZATIONS_PATH / (get_current_function_name() + "_ar.html")),
        auto_open=False,
    )
    span_settings = AnticipationV2Settings(
        min_track_events=ar_settings.min_track_events,
        context_size=ar_settings.context_size,
        vocab=local_midi_vocab,
        num_autoregressive_seq_per_midi_file=0,
        num_instrument_anticipation_augmentations_per_midi_file=0,
        num_span_anticipation_augmentations_per_midi_file=1,
        do_clip_overlapping_durations_in_midi_conversion=ar_settings.do_clip_overlapping_durations_in_midi_conversion,
        debug=True,
        debug_flush_remaining_token_buffer=True,
        tick_token_every_n_ticks=ar_settings.tick_token_every_n_ticks,
    )
    for s in range(0, 10):
        # try a few random seeds since the span region is randomly decided
        set_seed(s)
        span_token_seqs = []
        stats: TokenizationStatSummary = v2_tokenize(
            files_to_tokenize,
            output=span_token_seqs,
            settings=span_settings,
        )
        _check_anticipation_rule_for_controls_and_token_ranges(
            span_token_seqs, span_settings
        )
        assert not stats.ignored_files
        assert stats.num_lost_tokens_left_in_buffer == 0

        # check that span anticipation and autoregressive sequence tokenization are
        # the same (musically speaking). This is important because anticipation does
        # not add or remove information, it just restructures it. The original AR
        # sequence must be recoverable from an anticipated sequence of the same
        # piece.
        _check_is_musically_same(
            ar_token_seqs, span_token_seqs, ar_settings, span_settings
        )

    instr_settings = AnticipationV2Settings(
        min_track_events=ar_settings.min_track_events,
        context_size=ar_settings.context_size,
        vocab=local_midi_vocab,
        num_autoregressive_seq_per_midi_file=0,
        num_instrument_anticipation_augmentations_per_midi_file=1,
        num_span_anticipation_augmentations_per_midi_file=0,
        do_clip_overlapping_durations_in_midi_conversion=ar_settings.do_clip_overlapping_durations_in_midi_conversion,
        debug=True,
        debug_flush_remaining_token_buffer=True,
        tick_token_every_n_ticks=ar_settings.tick_token_every_n_ticks,
    )

    for s in range(10, 20):
        # try a few random seeds since the instrument is randomly selected
        set_seed(s)
        instr_token_seqs = []
        stats: TokenizationStatSummary = v2_tokenize(
            files_to_tokenize,
            output=instr_token_seqs,
            settings=instr_settings,
        )
        _check_anticipation_rule_for_controls_and_token_ranges(
            instr_token_seqs, instr_settings
        )
        assert not stats.ignored_files
        assert stats.num_lost_tokens_left_in_buffer == 0
        _check_is_musically_same(
            ar_token_seqs, instr_token_seqs, ar_settings, instr_settings
        )
