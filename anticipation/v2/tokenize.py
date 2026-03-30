from typing import Optional, Iterable, Union, Iterator

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, fields
from pathlib import Path
import random
import warnings

from tqdm import tqdm
import numpy as np

from anticipation.v2 import ops as v2_ops

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
    num_times_end_was_truncated: int
    num_tick_tokens: int
    num_separator_tokens: int
    num_autoregress_tokens: int
    num_anticipate_tokens: int
    num_lost_tokens_left_in_buffer: int
    num_truncations_before_augmentation: int
    num_pitch_transpose_augmentations: int
    num_times_span_had_insufficient_time: int
    total_time_in_midi_ticks_before_augmentation: int
    total_time_in_midi_ticks: int
    ignored_files: dict[MIDIFileIgnoredReason, list[Path]]

    @classmethod
    def get_int_fields(cls) -> list[str]:
        return [x.name for x in fields(cls) if x.type == int]  # noqa


class TokenStream(Iterator[tuple[Token, ...]]):
    def __init__(
        self,
        stream: Iterator[tuple[Token, ...]],
        settings: AnticipationV2Settings,
        control_prefix: tuple[Token, ...],
    ) -> None:
        assert not isinstance(stream, list), (
            "TokenStream input must be lazy iterator, not list."
        )
        self._stream = stream
        self._settings = settings

        assert isinstance(control_prefix, tuple), (
            "Control prefix must be tuple of tokens"
        )
        self.control_prefix = control_prefix

    def __iter__(self):
        return self

    def __next__(self) -> tuple[Token, ...]:
        return next(self._stream)

    def transform(
        self, tokens: list[tuple[Token, ...]]
    ) -> tuple[list[tuple[Token, ...]], dict]:
        return tokens, {}


def random_time_partition(xs: list[tuple[int, int]], delta: float) -> tuple[int, int]:
    # xs is a list of tuples, representing
    # [(an array index, a time in ticks), ...]
    t0, t1 = xs[0][1], xs[-1][1]

    if t0 + delta == t1:
        # only one reasonable choice here, which is
        # to anticipate exactly after the final tick
        return xs[-1]

    if not (t0 + delta < t1):
        raise ValueError(
            f"Not enough time in the span. Got t0,t1 = ({t0}, {t1}), delta={delta}"
        )

    # choose tau ~ Unif[t + delta, t'], tau >= t + delta
    tau: float = random.uniform(t0 + delta, t1)
    times_only = [x[1] for x in xs]

    # i is the first index with xs[i] > tau
    i = bisect_right(times_only, tau)  # type: ignore

    return xs[i]


class SpanV2TokenStream(TokenStream):
    def __init__(
        self,
        stream: Iterator[tuple[Token, ...]],
        settings: AnticipationV2Settings,
        control_flag: tuple[Token, ...],
    ) -> None:
        super().__init__(stream, settings, control_flag)
        self._num_ticks = -1

    def _decide_events_and_controls(
        self, tokens: list[tuple[Token, ...]]
    ) -> tuple[list[tuple[Token, ...]], list[Token]]:
        all_times = []
        for i, t in enumerate(tokens):
            if t != (self._settings.vocab.TICK,):
                # skip everything that isn't a tick
                # reason being it is the most reliable notion
                # of time, and all times it represents are unique -
                # whereas several notes may play at the exact
                # same time
                continue

            # idx, time
            all_times.append(
                (i, len(all_times) * self._settings.tick_token_every_n_ticks)
            )

        # ignore the final time, it could be truncated
        all_times = all_times[:-1]
        if len(all_times) == 0:
            # not enough time in the token list to split
            events = tokens
            return events, []

        assert len(all_times) > 0
        delta = self._settings.delta * self._settings.time_resolution

        # randomly decide where the cut-off between events and controls happens
        try:
            pivot_elem = random_time_partition(all_times, delta)
        except ValueError:
            # not enough time in the token list to split
            events = tokens
            return events, []

        pivot_idx = pivot_elem[0]

        events = []
        controls = []
        control_offset = self._settings.vocab.CONTROL_OFFSET
        for i, x in enumerate(tokens):
            if not v2_ops.is_triple(x, self._settings):
                # is a tick or sep or other
                events.append(x)
                continue

            if i < pivot_idx:
                events.append(x)
            else:
                controls.append(
                    (
                        x[0] + control_offset,
                        x[1] + control_offset,
                        x[2] + control_offset,
                    )
                )

        assert len(events) + len(controls) == len(tokens)

        flattened_controls = [x for b in controls for x in b]
        return events, flattened_controls

    def transform(
        self, tokens: list[tuple[Token, ...]]
    ) -> tuple[list[tuple[Token, ...]], dict]:
        stats = {"insufficient_time": False}

        unwrapped_num_tokens_start = len([x for b in tokens for x in b])

        # NB: tokens can have ticks in it, this is necessary
        # because we need to keep the number of tokens consistent after we
        # apply anticipation on a block before using it as a sequence... if we
        # add ticks after it has filled the context, then it becomes larger
        # than the context... and we can't have that
        num_tokens_at_start = len(tokens)
        all_tokens = list(tokens)

        # get the number of ticks
        num_ticks_at_start = all_tokens.count((self._settings.vocab.TICK,))

        # designate events and controls
        events, controls = self._decide_events_and_controls(all_tokens)

        # the context did not contain enough time to split for a span
        # that is ok, just keep track of it
        stats["insufficient_time"] = len(controls) == 0

        # anticipate
        stream = v2_ops.block_anticipation(
            events,
            controls,
            self._settings,
            start_at_ticks_seen=self._num_ticks + 1,
        )

        # relativize the combined sequence
        stream = v2_ops.streaming_relativize_to_tick(
            stream, self._settings, start_from_tick=self._num_ticks
        )
        self._num_ticks += num_ticks_at_start

        realized_tokens = list(stream)
        num_ticks_at_end = realized_tokens.count((self._settings.vocab.TICK,))
        assert num_ticks_at_end == num_ticks_at_start

        # ensure that we did not add or remove anything
        assert len(realized_tokens) == num_tokens_at_start

        unwrapped_num_tokens_end = len([x for b in realized_tokens for x in b])
        assert unwrapped_num_tokens_start == unwrapped_num_tokens_end

        # return the tokens
        return realized_tokens, stats


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
        self._iterator_queue: list[TokenStream] = []

        if isinstance(target, list):
            # writing to in memory list
            self._target = target
        elif isinstance(target, Path):
            # writing to a file
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

        # stats
        self._num_tokens_left_in_buffer = 0
        self._num_times_end_was_truncated = 0
        self._total_seq_written = 0
        self._total_time_in_midi_ticks_written = 0
        self._num_times_span_had_insufficient_time = 0
        self._token_counter = {
            self._settings.vocab.TICK: 0,
            self._settings.vocab.SEPARATOR: 0,
            self._settings.vocab.AUTOREGRESS: 0,
            self._settings.vocab.ANTICIPATE: 0,
        }

        self._current_fragments = defaultdict(list)
        self._current_len = 0
        self._buf: list[Token] = []

    def add_tokenized_file(
        self,
        tokenized_file: TokenStream,
    ) -> None:
        # assumption: this is called once per file augmentation
        self._iterator_queue.append(tokenized_file)

    def write_sequences(self) -> None:
        for seq in self._pack_and_iter_seq():
            self._write_seq(seq)

    def _pack_and_iter_seq(self) -> Iterable[list[Token]]:
        mutated_tokens = []

        # go through each document
        doc: TokenStream
        for doc in self._iterator_queue:
            # new document, must be separated
            self._current_fragments[doc].append((self._settings.vocab.SEPARATOR,))
            self._current_len += 1

            # ensure the control prefix is there
            self._current_fragments[doc].append(doc.control_prefix)
            self._current_len += len(doc.control_prefix)

            # go through each logical grouping of tokens in the doc
            for tup in doc:
                # prepend the control sequence, this scenario captures when we are
                # still iterating through the same document, but a new context has
                # been created, so we must prefix it with the control - that's why
                # we put it in buf instead of current_fragments
                if self._current_len == 0:
                    self._buf = [*doc.control_prefix]
                    self._current_len += len(doc.control_prefix)

                # each group must be associated with its parent document
                # because the parent document might need to transform it
                self._current_fragments[doc].append(tup)
                self._current_len += len(tup)

                if self._current_len >= self._settings.context_size:
                    # several documents might be in the same context, apply
                    # their respective transformations to their subsequences
                    for d, tokens in self._current_fragments.items():
                        # apply transform...
                        # very unfortunate - the transform has internal state
                        # I am sorry :(
                        # if there is a better way, we should do that
                        mutated_tokens, _transform_stats = d.transform(tokens)

                        # should not add or remove token groups
                        assert len(mutated_tokens) == len(tokens)

                        # flatten and add to context
                        self._buf.extend([x for b in mutated_tokens for x in b])

                        # take note of any issues or interesting behavior
                        # that happened during the transform
                        if _transform_stats.get("insufficient_time"):
                            self._num_times_span_had_insufficient_time += 1

                    # every sequence must be prefixed with a control
                    my_seq = [*self._buf]
                    did_truncate = len(my_seq) > self._settings.context_size
                    truncated_part = my_seq[self._settings.context_size :]
                    my_seq = my_seq[: self._settings.context_size]
                    yield my_seq

                    if did_truncate:
                        # truncation happened, keep the end token(s)
                        # which was already transformed
                        self._num_times_end_was_truncated += 1

                        # find the tokens that got truncated, add them to
                        # the buffer to ensure that the next sequence starts
                        # with the non-truncated version of them
                        to_add = v2_ops.get_truncated_token_groups_from_truncated_flat_token_sequence(
                            truncated_part, mutated_tokens
                        )
                        self._buf = [*doc.control_prefix] + [
                            x for b in to_add for x in b
                        ]
                    else:
                        # nothing was cut off
                        self._buf = []

                    # reset the state
                    self._current_fragments = defaultdict(list)
                    self._current_len = len(self._buf)

        # reset, done with everything
        self._iterator_queue = []

        # any remaining tokens
        self._num_tokens_left_in_buffer = self._current_len

        # if the settings request it and there are remaining tokens, push
        # them to the sink so we can inspect them
        if self._settings.debug_flush_remaining_token_buffer and self._current_len > 0:
            # control_prefix is the same as whatever last
            # document's was
            for d, tokens in self._current_fragments.items():
                # apply transform
                mutated_tokens, _transform_stats = d.transform(tokens)

                # should not add or remove token groups
                assert len(mutated_tokens) == len(tokens)

                # flatten and add to context
                self._buf.extend([x for b in mutated_tokens for x in b])

                # take note of any issues or interesting behavior
                # that happened during the transform
                if _transform_stats.get("insufficient_time"):
                    self._num_times_span_had_insufficient_time += 1

            to_return = [*self._buf]

            # something is wrong if this is larger than the context
            assert len(to_return) <= self._settings.context_size

            # we lost nothing in the buffer
            self._num_tokens_left_in_buffer = 0

            # return it
            yield to_return[: self._settings.context_size]

            # (no truncation check because this is the last context window)

    def _write_seq(self, buf: list[Token]) -> None:
        """
        Purpose of this function is just to collect some information about the
        sequences before they are written. Do not change the sequences in here.
        """
        # copy, just for safety since buf is mutated a lot
        local_copy = []
        abs_time_in_ticks = 0

        max_token_val = self._settings.vocab.total_tokens() - 1

        token: int
        for i, token in enumerate(buf):
            if token == self._settings.vocab.TICK:
                abs_time_in_ticks += self._settings.tick_token_every_n_ticks
            if token in self._token_counter:
                # keep a running counter of occurrences of some
                # special tokens of interest
                self._token_counter[token] += 1

            assert 0 <= token <= max_token_val, (
                f"Token out of bounds: (0 <= token ({token}) <= max_token_val ({max_token_val}))"
            )

            local_copy.append(token)

        self._total_time_in_midi_ticks_written += abs_time_in_ticks
        self._total_seq_written += 1

        # every sequence must be exactly the context size
        if not self._settings.debug_flush_remaining_token_buffer:
            assert len(local_copy) == self._settings.context_size

        # every sequence must start with some control prefix (or sep)
        assert local_copy[0] >= self._settings.vocab.SPECIAL_OFFSET

        # write it
        self._target.append(local_copy)

    def close(self) -> dict[str, int]:
        if not isinstance(self._target, list):
            self._target.close()

        # ideally keep this 1 level deep, so we can just gather them all
        # and add them up. Nested dicts might make this annoying
        return {
            "num_sequences": self._total_seq_written,
            "num_times_end_was_truncated": self._num_times_end_was_truncated,
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
            "num_times_span_had_insufficient_time": self._num_times_span_had_insufficient_time,
        }


def tokenize(
    midi_files: Iterable[Path],
    output: Union[list, Path],
    settings: AnticipationV2Settings,
    shard_id: int = -1,
    is_training_split: bool = True,
    flush_seq_packer_every_k_files: int = 20,
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
      is_training_split: True if we are tokenizing files in the training split,
        false otherwise. In non-training splits, we do not add anticipation or
        pitch augmentations.
      flush_seq_packer_every_k_files: To prevent accumulation of memory,
        the buffer of collected sequences will be written to disk every k
        files tokenized. This is per file, not per augmentation of file.

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
        _make_sequences(result, buf, settings, is_training_split)

        if is_training_split:
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
                _make_sequences(result, buf, settings, is_training_split=False)
                num_pitch_transpose_augmentations += 1

        if num_given_files % flush_seq_packer_every_k_files == 0:
            # write every so often
            buf.write_sequences()

    # write all the sequences to if any remain
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
    is_training_split: bool,
) -> None:
    # 1. pure autoregressive sequence
    for _ in range(settings.num_autoregressive_seq_per_midi_file):
        token_iterator = _get_augmentation_autoregressive(
            tokenized_midi.events, settings
        )
        buf.add_tokenized_file(token_iterator)

    if not is_training_split:
        # no augmentations
        return

    # --- augmentations ---
    # 2. instrument anticipation
    for _ in range(settings.num_instrument_anticipation_augmentations_per_midi_file):
        token_iterator = _get_augmentation_instrument(
            tokenized_midi.events,
            tokenized_midi.all_midi_program_codes,
            settings,
        )
        buf.add_tokenized_file(token_iterator)

    # 3. span anticipations v2 style
    for _ in range(settings.num_span_anticipation_augmentations_per_midi_file):
        token_iterator = _get_span_augmentation(tokenized_midi.events, settings)
        buf.add_tokenized_file(token_iterator)


def _get_augmentation_autoregressive(
    tokens: list[Token], settings: AnticipationV2Settings
) -> TokenStream:
    assert len(tokens) % 3 == 0, "bad length"

    if settings.tick_token_every_n_ticks > 0:
        # add ticks
        events = v2_ops.streaming_add_ticks(tokens, settings)

        # relativize
        stream = v2_ops.streaming_relativize_to_tick(events, settings)
    else:
        # when events drop below a certain density, pad them with rests
        # in the style that uses tick tokens, we do not need these
        tokens_with_rests = v2_ops.add_rests(tokens, settings)
        stream = (tokens_with_rests[i : i + 3] for i in range(0, len(tokens), 3))

    # add prefix
    control_flag = (settings.vocab.AUTOREGRESS,)

    return TokenStream(stream, settings, control_flag)


def _sample_instrument_subset(
    all_midi_program_codes: list[MIDIProgramCode],
) -> list[MIDIProgramCode]:
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
    all_midi_program_codes: list[MIDIProgramCode],
    settings: AnticipationV2Settings,
) -> TokenStream:
    assert len(tokens) % 3 == 0, "bad length"

    # if we do not add a prefix of some ticks, then we lose
    # any anticipated tokens at the very start this issue is
    # unique to instrument anticipation because in the
    # span, the time cutoff is sampled only past t > delta
    tokens = v2_ops.translate(
        tokens, settings.time_resolution * settings.delta, settings, seconds=False
    )

    # sample a random instrument from those in the sequence
    control_instrument = _sample_instrument_subset(all_midi_program_codes)
    events, controls = v2_ops.extract_instruments(tokens, control_instrument, settings)
    assert len(controls) % 3 == 0
    assert len(events) % 3 == 0

    # add ticks to the events only
    events_and_ticks = v2_ops.streaming_add_ticks(events, settings)

    # anticipate
    stream = v2_ops.block_anticipation(events_and_ticks, controls, settings)

    # relativize
    stream = v2_ops.streaming_relativize_to_tick(stream, settings)

    # add prefix
    control_flag = (settings.vocab.ANTICIPATE,)

    return TokenStream(stream, settings, control_flag)


def _get_span_augmentation(
    tokens: list[Token],
    settings: AnticipationV2Settings,
) -> TokenStream:
    assert len(tokens) % 3 == 0, "bad length"

    # add the ticks first so that the context window does not change
    # when we designate which is a control and which is an event
    events_and_ticks = v2_ops.streaming_add_ticks(tokens, settings)

    # add prefix
    control_flag = (settings.vocab.ANTICIPATE,)

    # this one has very weird control flow, sorry
    # an operation is performed on the sequence at the moment is it packed /
    # 'flushed' into a sequence length equal to the context
    return SpanV2TokenStream(events_and_ticks, settings, control_flag)
