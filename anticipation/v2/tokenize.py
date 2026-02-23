from tqdm import tqdm
from typing import Optional, Iterable, Union, Iterator
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass, fields
from itertools import chain
import warnings

import numpy as np

from anticipation.v2 import ops as v2_ops
from anticipation.tokenize import (
    extract_instruments as v1_extract_instruments,
)

from anticipation.v2.types import (
    MIDIFileIgnoredReason,
    Token,
    MIDIProgramCode,
    MIDITick,
)
from anticipation.v2.config import AnticipationV2Settings
from anticipation.v2.convert import (
    midi_to_compound,
    compound_to_events,
    SymusicRuntimeError,
)
from anticipation.v2.io import TokenSequenceBinaryFile


def _maybe_tokenize(
    my_midi_file: Path,
    settings: AnticipationV2Settings,
    pitch_transpose: int = 0,
) -> tuple[list[Token], int, Optional[MIDIFileIgnoredReason]]:
    try:
        # try to convert this file to compound 5-tuple representation
        compound_tokens: list[int] = midi_to_compound(
            my_midi_file,
            settings=settings,
            pitch_transpose=pitch_transpose,
        )
    except SymusicRuntimeError as e:
        if "File header is not MThd" in str(e):
            return [], 0, MIDIFileIgnoredReason.INVALID_FILE_HEADER
        elif "Unexpected EOF" in str(e):
            return [], 0, MIDIFileIgnoredReason.UNEXPECTED_EOF_ERROR
        elif "Unexpected running status" in str(e):
            return [], 0, MIDIFileIgnoredReason.BAD_STATUS
        elif "Division type is not ticks per quarter" in str(e):
            return [], 0, MIDIFileIgnoredReason.INVALID_TICK_TYPE
        elif "Invaild midi format" in str(e):
            # NB: the misspelling of 'invalid' is required
            return [], 0, MIDIFileIgnoredReason.INVALID_MIDI_FORMAT_IN_HEADER
        elif "Invaild midi file!" in str(e):
            # NB: the misspelling of 'invalid' is required
            return [], 0, MIDIFileIgnoredReason.INVALID_FILE_STRUCTURE
        elif "Invalid chunk length" in str(e):
            return [], 0, MIDIFileIgnoredReason.INVALID_FILE_STRUCTURE
        else:
            raise e

    if len(compound_tokens) < settings.compound_size * settings.min_track_events:
        # skip sequences with very few events
        return [], 0, MIDIFileIgnoredReason.TOO_FEW_EVENTS

    try:
        events, num_note_truncations = compound_to_events(compound_tokens, settings)
    except AssertionError:
        return [], 0, MIDIFileIgnoredReason.VALUES_WERE_OUT_OF_RANGE

    end_time = v2_ops.max_time(events, settings, seconds=False)

    # don't want to deal with extremely short tracks
    if end_time < settings.time_resolution * settings.min_track_time_in_seconds:
        return [], 0, MIDIFileIgnoredReason.TOO_FEW_SECONDS

    # don't want to deal with extremely long tracks
    if end_time > settings.time_resolution * settings.max_track_time_in_seconds:
        return [], 0, MIDIFileIgnoredReason.TOO_MANY_SECONDS

    # skip sequences more instruments than MIDI channels (16)
    if len(v2_ops.get_instruments(events, settings)) > settings.max_track_instruments:
        return (
            [],
            0,
            MIDIFileIgnoredReason.TOO_MANY_INSTRUMENTS,
        )

    return events, num_note_truncations, None


@dataclass(frozen=True)
class TokenizedMIDIFileResult:
    events: list[Token]
    num_note_truncations: int
    reason_ignored: Optional[MIDIFileIgnoredReason]
    all_midi_program_codes: list[MIDIProgramCode]
    end_time_in_ticks: MIDITick


def _tokenize_midi_file(
    my_midi_file: Path,
    settings: AnticipationV2Settings,
    pitch_transpose: int = 0,
) -> TokenizedMIDIFileResult:
    assert isinstance(my_midi_file, Path)

    # now we are tokenizing the MIDI, using several tokens from our settings
    all_events, num_note_truncations, reason_ignored = _maybe_tokenize(
        my_midi_file, settings, pitch_transpose
    )

    all_midi_program_codes = []
    end_time_in_ticks = 0

    if reason_ignored is None:
        # we are actually going to use this, get some information we
        # will need later
        all_midi_program_codes = list(v2_ops.get_instruments(all_events, settings))
        end_time_in_ticks: int = v2_ops.max_time(all_events, settings, seconds=False)

    return TokenizedMIDIFileResult(
        events=all_events,
        num_note_truncations=num_note_truncations,
        reason_ignored=reason_ignored,
        all_midi_program_codes=all_midi_program_codes,
        end_time_in_ticks=end_time_in_ticks,
    )


@dataclass(frozen=True)
class TokenizationStatSummary:
    num_given_files: int
    num_tokenized_files: int
    num_sequences: int
    num_times_end_triple_was_truncated: int
    num_tick_tokens: int
    num_separator_tokens: int
    num_autoregress_tokens: int
    num_anticipate_tokens: int
    num_lost_tokens_left_in_buffer: int
    num_truncations_before_augmentation: int
    num_pitch_transpose_augmentations: int
    total_time_in_midi_ticks_before_augmentation: int
    total_time_in_midi_ticks: int
    ignored_files: dict[MIDIFileIgnoredReason, list[Path]]

    @classmethod
    def get_int_fields(cls) -> list[str]:
        return [x.name for x in fields(cls) if x.type == int]  # noqa


class TokenStream(Iterator[tuple[Token, ...]]):
    def __init__(self, stream, settings: AnticipationV2Settings):
        self._stream = stream
        self._settings = settings

    def __iter__(self):
        return self

    def __next__(self) -> tuple[Token, ...]:
        return next(self._stream)

    def transform(self, control_prefix, tokens):
        return tokens


from bisect import bisect_right
import random
from typing import Sequence, Tuple, List


def random_time_partition(xs: Sequence[int], *, rng: random.Random = None):
    """
    xs: monotonic non-decreasing times (seconds since event).
    Returns: (tau, left, right, split_index)
      left  = xs[:i]
      right = xs[i:]
      i = first index with xs[i] > tau  (so values == tau go to left)
    """
    if not xs:
        raise ValueError("xs is empty")
    rng = rng or random

    t0, t1 = xs[0], xs[-1]
    if t1 < t0:
        raise ValueError("xs must be non-decreasing")
    if t0 == t1:
        # Degenerate range: any tau is that value; choose split at the end.
        tau = t0
        i = len(xs)
        return tau, i

    tau = rng.uniform(t0, t1)
    i = bisect_right(xs, tau)  # first index with xs[i] > tau
    return tau, i


class SpanV2TokenStream(TokenStream):

    def __init__(self, stream, settings: AnticipationV2Settings):
        super().__init__(stream, settings)
        self._prefix = []

    def transform(
        self, control_prefix: tuple[Token, ...], tokens: list[tuple[Token, ...]]
    ):
        i = 0
        to_return = []
        if (
            tokens
            and len(tokens[0]) == 1
            and tokens[0][0] == self._settings.vocab.SEPARATOR
        ):
            # skip it
            to_return.append(tokens[0])
            i += 1

        max_time = 0
        min_time = 0
        for t in tokens[i:]:
            if len(t) != 1:
                time = t[0]
                max_time = max(time, max_time)
                min_time = min(time, min_time)

        earliest_allowed_time = max_time - (self._settings.time_resolution * self._settings.delta)
        earliest_allowed_time = 0
        ticks = []
        rel_time_in_window = 0
        ticks_seen = 0
        time = 0
        for j in range(i, len(tokens)):
            if len(tokens[j]) == 1 and tokens[j][0] == self._settings.vocab.TICK:
                if time > earliest_allowed_time:
                    ticks.append((j, time))
            else:
                time = tokens[j][0]

        print(ticks)
        time, split_idx = random_time_partition([x[1] for x in ticks])
        split_at_idx = ticks[split_idx][0]
        #split_at_idx = ticks[1][0]
        #
        events = []
        controls = []
        for j in range(i, len(tokens)):
            if len(tokens[j]) == 1 and tokens[j][0] == self._settings.vocab.TICK:
                events.append(tokens[j])
                continue

            if j < split_at_idx:
                events.append(tokens[j])
            else:
                time, dur, note_instr = tokens[j]
                controls.append(
                    (
                        time + self._settings.vocab.CONTROL_OFFSET,
                        dur + self._settings.vocab.CONTROL_OFFSET,
                        note_instr + self._settings.vocab.CONTROL_OFFSET,
                    )
                )

        # should not be able to add or remove logical units!
        assert len(events) + len(controls) == len(tokens) - i

        zzzz= 10
        stream = v2_ops.streaming_anticipate(
            iter(events), iter(controls), self._settings
        )
        tmp = list(stream)
        prev_len = len(self._prefix)
        self._prefix.extend(tmp)
        # print("transform", self._prefix)
        # stream = list(v2_ops.streaming_relativize_to_tick(iter(tokens), self._settings))
        # to_return.extend(list(stream)[1:])
        # #assert len(to_return) == len(tokens)
        # #return to_return
        stream = list(v2_ops.streaming_relativize_to_tick(iter(self._prefix), self._settings))
        aoi = list(stream)[max(prev_len - 1, 0):]
        return to_return + aoi


class SequencePacker:
    """Buffer-like coordinator that flushes tokens to a file or just accumulates them to a list.

    This adds punctuation-like semantic tokens to its outputs as they are streamed out. We
    also handle tracking of dataset tokenization stats and return them on close. Be careful,
    because of the stat tracking this class has a lot of state.

    NB: list given in ctor is pass by reference, will mutate in place.
    """

    def __init__(
        self,
        target: Union[list, Path],
        settings: AnticipationV2Settings,
    ) -> None:
        self._settings = settings
        self._iterator_queue: list[tuple[tuple[Token, ...], TokenStream]] = []
        self._buf = []
        if isinstance(target, list):
            self._target = target
        elif isinstance(target, Path):
            # self._buf = []
            # target may not exist yet, but the folder that
            # contains it should
            assert target.parent.exists()
            assert target.parent.is_dir()
            self._target = TokenSequenceBinaryFile(
                target,
                seq_len=settings.context_size,
                vocab_size=settings.vocab.total_tokens(),
            )
        else:
            raise TypeError("Must have path or list as target.")

        self._target: Union[list, TokenSequenceBinaryFile]
        self._most_recent_control_prefix = ()
        self._num_tokens_left_in_buffer = 0

        # stats
        self._total_seq_written = 0
        self._total_times_end_triple_was_truncated = 0
        self._total_time_in_midi_ticks_written = 0
        self._token_counter = {
            self._settings.vocab.TICK: 0,
            self._settings.vocab.SEPARATOR: 0,
            self._settings.vocab.AUTOREGRESS: 0,
            self._settings.vocab.ANTICIPATE: 0,
        }

    def add_tokenized_file(
        self,
        control_prefix: tuple[Token, ...],
        tokenized_file: TokenStream,
    ) -> None:
        # assumption: this is called once per file augmentation
        # tokenized_file is not split in the middle and does not continue some previous
        # sequence
        self._iterator_queue.append((control_prefix, tokenized_file))

    def write_sequences(self) -> None:
        i = 0
        for seq in self._pack_and_iter_seq():
            if i in (0, 1, 2):
                self._write_seq(seq)
            else:
                break
            i += 1

    def _pack_and_iter_seq(self):
        current_fragments: dict[TokenStream, list[tuple[Token, ...]]] = {}
        current_len = 0
        control_prefix = ()
        for control_prefix, doc in self._iterator_queue:
            num_non_control_tokens = self._settings.context_size - len(control_prefix)
            iterator = iter(doc)
            leftover_tuple = (self._settings.vocab.SEPARATOR,)
            while True:
                # 1. Feed from the buffer first
                if leftover_tuple:
                    tup = leftover_tuple
                    leftover_tuple = None
                else:
                    try:
                        tup = next(iterator)
                    except StopIteration:
                        break

                space_left = num_non_control_tokens - current_len

                # 2. Allocate tokens to the current document's fragment
                if len(tup) <= space_left:
                    current_fragments.setdefault(doc, []).append(tup)
                    current_len += len(tup)
                else:
                    partial_tup = tup[:space_left]
                    if partial_tup != tup:
                        self._total_times_end_triple_was_truncated += 1

                    current_fragments.setdefault(doc, []).append(tup)
                    current_len += len(partial_tup)
                    leftover_tuple = tup

                # 3. Assemble and yield the transformed window when full
                if current_len == num_non_control_tokens:
                    transformed_window = []

                    for d, tokens in current_fragments.items():
                        # Collect the mutated tokens back from the document
                        in_tok = len(tokens)
                        mutated_tokens = d.transform(control_prefix, tokens)
                        out_tok = len(mutated_tokens)
                        mutated_tokens = [x for b in mutated_tokens for x in b]
                        transformed_window.extend(mutated_tokens)

                    yield [*control_prefix, *transformed_window][
                        : self._settings.context_size
                    ]

                    # Reset state
                    current_fragments = {}
                    current_len = 0

        self._iterator_queue = []

        # 4. Flush any remaining tokens in a final, partial window
        if False and self._settings.debug_flush_remaining_token_buffer:
            if current_len > 0:
                transformed_window = []
                for d, tokens in current_fragments.items():
                    mutated_tokens = d.transform(control_prefix, tokens)
                    mutated_tokens = [x for b in mutated_tokens for x in b]
                    transformed_window.extend(mutated_tokens)
                to_return = [*control_prefix, *transformed_window][
                    : self._settings.context_size
                ]
                self._num_tokens_left_in_buffer = len(to_return)
                yield to_return

    def _write_seq(self, buf: list[Token]) -> None:
        # copy, just for safety since buf is mutated a lot
        local_copy = []
        abs_time_in_ticks = 0
        for i, token in enumerate(buf):
            if token == self._settings.vocab.TICK:
                abs_time_in_ticks += self._settings.tick_token_frequency_in_midi_ticks
            if token in self._token_counter:
                # keep a running counter of occurrences of some
                # special tokens of interest
                self._token_counter[token] += 1
            local_copy.append(token)

        self._total_time_in_midi_ticks_written += abs_time_in_ticks
        self._total_seq_written += 1

        # write it
        self._target.append(local_copy)

    def close(self) -> dict[str, int]:
        if not isinstance(self._target, list):
            self._target.close()

        # ideally keep this 1 level deep, so we can just gather them all
        # and add them up. Nested dicts might make this annoying
        return {
            "num_sequences": self._total_seq_written,
            "num_times_end_triple_was_truncated": self._total_times_end_triple_was_truncated,
            "num_tick_tokens": self._token_counter[self._settings.vocab.TICK],
            "num_separator_tokens": self._token_counter[self._settings.vocab.SEPARATOR],
            "num_autoregress_tokens": self._token_counter[
                self._settings.vocab.AUTOREGRESS
            ],
            "num_anticipate_tokens": self._token_counter[
                self._settings.vocab.ANTICIPATE
            ],
            "num_lost_tokens_left_in_buffer": self._num_tokens_left_in_buffer,
            "total_time_in_midi_ticks": self._total_time_in_midi_ticks_written,
        }


def tokenize(
    midi_files: Iterable[Path],
    output: Union[list, Path],
    settings: AnticipationV2Settings,
    shard_id: int = -1,
) -> TokenizationStatSummary:
    """Tokenizes MIDI for v2 Anticipatory Training

    Args:
      midi_files: An iterable collection of MIDI files. These are the
        source data that we will turn into tokens.
      output: This can be either a list or a Path object to a file. In
        the v1 tokenize function, this argument would be a string to
        a file location on disk. To keep parity with that pattern, we
        can pass a file here or a list. If given a file, the v1 behavior
        is performed. If given a list, the tokens are just appended to
        the list and nothing is written to disk. The list is mutated in
        place - so keep a reference to it. In python all lists are pass
        by reference.
      settings: The Anticipation v2 settings object. Settings in here
        will affect how MIDI events are tokenized.
      shard_id: Whether to show a tqdm progress bar. This is handled by
        dataset tokenization script. Defaults to not showing it.

    Returns:
      A dictionary of (reason ignored, number of times it happened) for
      the given files. This will be empty if no files were ignored.
    """
    if shard_id == 0:
        # only print progress of 1 of the shards to prevent clutter,
        # they take approximately the same amount of time
        iter_obj = tqdm(midi_files, mininterval=1.0, desc="Tokenizing MIDI Files")
    else:
        iter_obj = midi_files

    # --- all these are dataset level stats ---
    num_truncations_before_augmentation = 0
    total_time_in_midi_ticks_before_augmentation = 0
    num_tokenized_files = 0
    num_given_files = 0
    num_pitch_transpose_augmentations = 0
    ignored_files: dict[MIDIFileIgnoredReason, list[Path]] = defaultdict(list)
    # -----

    buf = SequencePacker(target=output, settings=settings)
    for file_idx, midi_file in enumerate(iter_obj):
        num_given_files += 1

        result: TokenizedMIDIFileResult = _tokenize_midi_file(midi_file, settings)
        if result.reason_ignored is not None:
            # can't use this file
            if settings.debug:
                # suppress writing to stderr/stdout unless we are debugging
                warnings.warn(
                    f"Sequence in file {midi_file} was ignored for reason: {result.reason_ignored}",
                    UserWarning,
                )
            ignored_files[result.reason_ignored].append(midi_file)
            continue

        # handle dataset stats - this is all BEFORE augmentation
        num_tokenized_files += 1
        num_truncations_before_augmentation += result.num_note_truncations
        total_time_in_midi_ticks_before_augmentation += result.end_time_in_ticks

        # actually tokenize and pack the sequences
        _make_sequences(result, buf, settings)

        # transpose the original file, now that we know it passes criteria for including
        for transposition_offset in settings.augmentation_pitch_shifts:
            try:
                result: TokenizedMIDIFileResult = _tokenize_midi_file(
                    midi_file, settings, transposition_offset
                )
            except SymusicRuntimeError as e:
                if "Overflow while adding" in str(e):
                    # transposition goes out of range
                    continue
                else:
                    raise e

            # using the transposed notes, do more augmentations
            _make_sequences(result, buf, settings)
            num_pitch_transpose_augmentations += 1

    # write all the sequences to a target
    buf.write_sequences()

    # close files if necessary
    buf_stats = buf.close()

    # return any stats, cast to regular dictionary for safety
    return TokenizationStatSummary(
        num_given_files=num_given_files,
        num_pitch_transpose_augmentations=num_pitch_transpose_augmentations,
        num_tokenized_files=num_tokenized_files,
        ignored_files=dict(ignored_files),
        num_truncations_before_augmentation=num_truncations_before_augmentation,
        total_time_in_midi_ticks_before_augmentation=total_time_in_midi_ticks_before_augmentation,
        **buf_stats,
    )


def _make_sequences(
    tokenized_midi: TokenizedMIDIFileResult,
    buf: SequencePacker,
    settings: AnticipationV2Settings,
) -> None:
    # 1. pure autoregressive sequence
    for _ in range(settings.num_autoregressive_seq_per_midi_file):
        control_prefix, token_iterator = _get_augmentation_autoregressive(
            tokenized_midi.events, settings
        )
        buf.add_tokenized_file(control_prefix, token_iterator)

    # --- augmentations ---
    # 2. instrument anticipation
    for _ in range(settings.num_instrument_anticipation_augmentations_per_midi_file):
        control_prefix, token_iterator = _get_augmentation_instrument(
            tokenized_midi.events,
            tokenized_midi.all_midi_program_codes,
            settings,
        )
        buf.add_tokenized_file(control_prefix, token_iterator)

    # 3. span anticipation
    # for _ in range(settings.num_span_anticipation_augmentations_per_midi_file):
    #     control_prefix, token_iterator = _get_span_augmentation(
    #         tokenized_midi.events, settings
    #     )
    #     buf.add_tokenized_file(control_prefix, token_iterator)

    for _ in range(settings.num_span_anticipation_augmentations_per_midi_file):
        control_prefix, token_iterator = _get_span_augmentation_v2(
            tokenized_midi.events, settings
        )
        buf.add_tokenized_file(control_prefix, token_iterator)


def _get_augmentation_autoregressive(
    tokens: list[Token], settings: AnticipationV2Settings
) -> tuple[tuple[Token, ...], TokenStream]:
    assert len(tokens) % 3 == 0, "bad length"

    if settings.tick_token_frequency_in_midi_ticks > 0:
        # step through events and ticks
        stream = v2_ops.streaming_relativize_to_tick(
            v2_ops.streaming_add_ticks(tokens, settings), settings
        )
    else:
        # when events drop below a certain density, pad them with rests
        # in the style that uses tick tokens, we do not need these
        tokens_with_rests = v2_ops.add_rests(tokens, settings)
        stream = (tokens_with_rests[i : i + 3] for i in range(0, len(tokens), 3))

    return (settings.vocab.AUTOREGRESS,), TokenStream(stream, settings)


def _sample_instrument_subset(all_midi_program_codes: list[int]) -> list[int]:
    if len(all_midi_program_codes) <= 1:
        # not really well-defined for this case...
        return []

    # instrument augmentation: at least one, but not all instruments
    # each time this is called it is like a random subset draw!
    u = 1 + np.random.randint(len(all_midi_program_codes) - 1)
    instrument_subset = np.random.choice(all_midi_program_codes, u, replace=False)
    return list(instrument_subset)


def _get_augmentation_instrument(
    tokens: list[Token],
    all_midi_program_codes: list[int],
    settings: AnticipationV2Settings,
) -> tuple[tuple[Token, ...], TokenStream]:
    assert len(tokens) % 3 == 0, "bad length"

    # in v1, REST tokens are not present in token sequence before calling
    # instrument extraction...
    events, controls = v1_extract_instruments(
        tokens, _sample_instrument_subset(all_midi_program_codes)
    )
    assert len(controls) % 3 == 0
    assert len(events) % 3 == 0

    # add ticks to the events only
    event_stream = v2_ops.streaming_add_ticks(events, settings)
    control_stream = (controls[i : i + 3] for i in range(0, len(controls), 3))
    stream = v2_ops.streaming_relativize_to_tick(
        v2_ops.streaming_anticipate(event_stream, control_stream, settings), settings
    )
    return (settings.vocab.ANTICIPATE,), TokenStream(stream, settings)


def _get_span_augmentation(
    tokens: list[Token],
    settings: AnticipationV2Settings,
) -> tuple[tuple[Token, ...], TokenStream]:
    assert len(tokens) % 3 == 0, "bad length"

    # DECIDE
    token_stream = (tokens[i:i+3] for i in range(0, len(tokens), 3))
    events, controls = v2_ops.extract_spans_v1_style(token_stream, settings)
    events = [x for b in events for x in b]
    events = v2_ops.streaming_add_ticks(events, settings)

    # ANTICIPATE
    stream = v2_ops.streaming_anticipate(iter(events), iter(controls), settings)

    # RELATIVIZE
    stream = v2_ops.streaming_relativize_to_tick(stream, settings)

    # CHUNK
    return (settings.vocab.ANTICIPATE,), TokenStream(stream, settings)


def _get_span_augmentation_v2(
    tokens: list[Token],
    settings: AnticipationV2Settings,
) -> tuple[tuple[Token, ...], TokenStream]:
    assert len(tokens) % 3 == 0, "bad length"

    events = v2_ops.streaming_add_ticks(tokens, settings)

    return (settings.vocab.ANTICIPATE,), SpanV2TokenStream(events, settings)
