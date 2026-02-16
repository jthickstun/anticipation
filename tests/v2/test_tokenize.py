import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from anticipation.v2.types import Token
from anticipation.v2.config import AnticipationV2Settings, Vocab
from anticipation.v2.tokenize import tokenize as v2_tokenize
from anticipation.v2.tokenize import MIDIFileIgnoredReason
from anticipation.v2.io import TokenSequenceBinaryFile

from tests.util.entities import Event, EventSpecialCode
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
        debug=True,
        debug_flush_remaining_token_buffer=True,
    )
    midi_files = [lmd_0_example_1_midi_path]
    in_memory_tokens = []
    any_ignored_in_memory = v2_tokenize(midi_files, in_memory_tokens, settings)
    assert not any_ignored_in_memory

    # tokenizing this in full is 9 sequences
    assert len(in_memory_tokens) == 9
    # all but the last one has full context, we've set a debug setting to push the
    # tokens in the buffer that would typically be ignored in a production context
    assert all((len(x) == settings.context_size for x in in_memory_tokens[:-1]))

    # there are 606 remaining that do not fit exactly into a context window
    assert len(in_memory_tokens[8]) == 606

    all_tokens_flattened = [x for b in in_memory_tokens for x in b]
    assert len(all_tokens_flattened) == 606 + (1024 * 8)

    num_total_separators = 0
    for i, packed_seq in enumerate(in_memory_tokens):
        if i == 0:
            # first sequence should have sample sep
            assert packed_seq[0] == settings.vocab.SEPARATOR
            assert packed_seq[1] == settings.vocab.AUTOREGRESS
        else:
            # all others should start with anticipation sample
            assert packed_seq[0] == settings.vocab.AUTOREGRESS

        num_total_separators += len(
            [x for x in packed_seq if x == settings.vocab.SEPARATOR]
        )

    assert num_total_separators == 1
    parsed_events = Event.from_token_seq(
        [x for b in in_memory_tokens for x in b], settings
    )
    assert len(parsed_events) == 3053
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
        any_ignored = v2_tokenize(
            [lmd_0_example_1_midi_path], instrument_anticipation_sample, settings
        )
        assert not any_ignored

    assert len(instrument_anticipation_sample) == 8
    num_total_separators = 0
    for i, packed_seq in enumerate(instrument_anticipation_sample):
        assert len(packed_seq) == settings.context_size
        if i == 0:
            # first sequence should have sample sep
            assert packed_seq[0] == settings.vocab.SEPARATOR
            assert packed_seq[1] == settings.vocab.ANTICIPATE
        else:
            # all others should start with anticipation sample
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
    )
    tokenized_seq = []
    any_ignored = v2_tokenize([c_major_midi_path], tokenized_seq, settings)
    assert not any_ignored

    assert len(tokenized_seq) == 1
    num_total_separators = 0
    for i, packed_seq in enumerate(tokenized_seq):
        assert len(packed_seq) == settings.context_size
        if i == 0:
            # first sequence should have sample sep
            assert packed_seq[0] == settings.vocab.SEPARATOR
            assert packed_seq[1] == settings.vocab.AUTOREGRESS
        else:
            # all others should start with anticipation sample
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
    any_ignored = v2_tokenize([lmd_0_example_1_midi_path], tokenized_seq, settings)
    assert not any_ignored
    assert len(tokenized_seq) == 8
    num_total_separators = 0
    for i, packed_seq in enumerate(tokenized_seq):
        assert len(packed_seq) == settings.context_size
        if i == 0:
            # first sequence should have sample sep
            assert packed_seq[0] == settings.vocab.SEPARATOR
            assert packed_seq[1] == settings.vocab.AUTOREGRESS
        else:
            # all others should start with anticipation sample
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
    any_ignored = v2_tokenize(
        [lmd_0_example_1_midi_path], events_without_ticks, settings
    )
    assert len(events_without_ticks) == 8
    assert not any_ignored
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
    any_ignored = v2_tokenize(
        [lmd_0_example_1_midi_path], events_include_ticks, settings
    )
    assert not any_ignored
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
    )
    midi_files = [lmd_0_example_1_midi_path, lmd_0_example_2_midi_path]

    in_memory_tokens = []
    any_ignored_in_memory = v2_tokenize(midi_files, in_memory_tokens, settings)
    assert not any_ignored_in_memory

    # 97 complete sequences
    assert len(in_memory_tokens) == 97
    # should all be context length
    assert all((len(x) == settings.context_size for x in in_memory_tokens))

    # test that we can fully recover the first file that is packed, even
    # when flushing the remaining buffer is OFF. We expect that a context window
    # contains the end of the first file and start of the second
    seq_border_frame = in_memory_tokens[8]
    sep_idx = seq_border_frame.index(settings.vocab.SEPARATOR)
    assert sep_idx == 606
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
    assert lmd_0_example_1_actual_events == lmd_0_example_1_expected_events
    assert settings.context_size == expected_settings.context_size
    assert settings.vocab == expected_settings.vocab

    # test saving data to disk is exactly same as in-memory
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        dataset_path = td_path / "dataset.bin"
        any_ignored = v2_tokenize(midi_files, dataset_path, settings)
        assert not any_ignored
        tokenized_samples = TokenSequenceBinaryFile.load_from_disk_to_numpy(
            dataset_path,
            seq_len=settings.context_size,
            vocab_size=settings.vocab.VOCAB_SIZE,
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
    ignored_files = v2_tokenize(
        [lmd_1_example_0_midi_path], output=tokens_to, settings=settings
    )
    assert ignored_files[MIDIFileIgnoredReason.INVALID_FILE_STRUCTURE] == [
        lmd_1_example_0_midi_path
    ]
