"""
These are utilities for testing / developing with the codebase.
"""

from typing import Union
from dataclasses import dataclass
from enum import Enum
import re

from anticipation.v2.config import AnticipationV2Settings
from anticipation.v2.types import Token, MIDITick, MIDIProgramCode, MIDINote

from tests.util.constants import MIDI_PROGRAM_NAMES


class EventSpecialCode(Enum):
    """
    Used to handle things beyond the MIDI spec but required for our tokenized
    event sequences.
    """

    TYPICAL_EVENT = 0
    AUTOREGRESSIVE_TOKEN = 1
    ANTICIPATION_TOKEN = 2
    SEQ_SEPARATION_TOKENS = 3
    TICK = 4


class Note:
    """Utility to convert notes to tokens and human-readable names.

    Spans C-2 to G8.
    We consider C3 to be middle C, MIDI note 60.
    """

    # sharps only
    NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    NAME_PC_MAP = {name: i for i, name in enumerate(NAMES)}
    _NOTE_RE = re.compile(r"^([A-G]#?)(-?\d+)$")

    def __init__(self, midi_note_int: MIDINote) -> None:
        if not (0 <= midi_note_int < 128):
            raise ValueError(
                f"MIDI note integer must be in [0, 127], got {midi_note_int}"
            )
        self.midi_note_int = midi_note_int

    @classmethod
    def make(cls, midi_note_int_or_name: Union[str, MIDINote]) -> "Note":
        if isinstance(midi_note_int_or_name, int):
            return Note(midi_note_int_or_name)
        elif isinstance(midi_note_int_or_name, str):
            return Note.from_name(midi_note_int_or_name)
        else:
            raise TypeError("midi_note_int_or_name must be int or string.")

    @classmethod
    def from_name(cls, name: str) -> "Note":
        m = cls._NOTE_RE.match(name)
        assert m

        pitch, octave_str = m.groups()
        octave = int(octave_str) + 1

        assert pitch in cls.NAME_PC_MAP

        midi_note_int = (octave + 1) * 12 + cls.NAME_PC_MAP[pitch]
        assert 0 <= midi_note_int < 128
        return cls(midi_note_int)

    def to_name(self) -> str:
        pitch_class = self.midi_note_int % 12
        # lowest is C-2 (negative 2)
        octave = (self.midi_note_int // 12) - 2
        return f"{self.NAMES[pitch_class]}{octave}"

    @property
    def name(self) -> str:
        return self.to_name()

    def __int__(self) -> MIDINote:
        return self.midi_note_int

    def __str__(self) -> str:
        return self.to_name()

    def __eq__(self, other) -> bool:
        if not isinstance(other, Note):
            return False
        return self.midi_note_int == other.midi_note_int

    def __repr__(self) -> str:
        return f"Note(midi={self.midi_note_int}, name='{self.to_name()}')"


def get_note_instrument_token(
    instrument_midi_code: MIDIProgramCode,
    note_midi_code: MIDINote,
    settings: AnticipationV2Settings,
    check_range: bool = True,
) -> Token:
    # NB: both instrument midi code and note midi code are 0-indexed
    if check_range and not (0 <= note_midi_code < 128):
        raise ValueError("MIDI note values must be within [0, 127].")

    if check_range and not (0 <= instrument_midi_code < 129):
        # I used out of range MIDI instruments for purposes of visualization...
        # sorry, turn check range off to allow this
        raise ValueError(
            "MIDI instrument values must be within [0, 128]. "
            f"Programs 0-127 as defined in the MIDI spec, then 128 for drums. Got code: {instrument_midi_code}"
        )

    return settings.vocab.NOTE_OFFSET + (
        settings.max_midi_pitch * instrument_midi_code + note_midi_code
    )


def get_midi_instrument_name_from_midi_instrument_code(
    instrument_midi_code: MIDIProgramCode,
) -> str:
    if 0 <= instrument_midi_code < len(MIDI_PROGRAM_NAMES):
        return MIDI_PROGRAM_NAMES[instrument_midi_code]
    elif instrument_midi_code == 128:
        # this happens a lot in Lakh MID, not sure if is a convention
        return "Drums (Channel 9)"
    elif instrument_midi_code == 129:
        return "REST"
    elif instrument_midi_code == 130:
        return "TICK"
    elif instrument_midi_code == 131:
        return "ATICK"
    else:
        return "?"


@dataclass
class Event:
    time: Token
    duration: Token
    note_instr: Token
    settings: AnticipationV2Settings

    # these are all 'derived', meaning they are not
    # direct properties of a token sequence
    is_control: bool
    original_idx_in_token_seq: int
    special_code: EventSpecialCode = EventSpecialCode.TYPICAL_EVENT
    absolute_time: MIDITick = 0

    def __post_init__(self):
        assert self.time >= 0
        assert self.duration >= 0
        assert self.note_instr >= 0
        assert self.absolute_time >= 0

    @classmethod
    def from_midi_values(
        cls,
        midi_time: MIDITick,
        midi_duration: MIDITick,
        midi_instrument: MIDIProgramCode,
        midi_note: Union[MIDINote, str],
        settings: AnticipationV2Settings,
        is_control: bool = False,
        special_code: EventSpecialCode = EventSpecialCode.TYPICAL_EVENT,
        original_idx_in_token_seq: int = 0,
    ) -> "Event":
        v = settings.vocab
        note_offset = v.CONTROL_OFFSET if is_control else 0
        time_offset = v.ATIME_OFFSET if is_control else v.TIME_OFFSET
        dur_offset = v.ADUR_OFFSET if is_control else v.DUR_OFFSET

        if midi_note == "REST" or midi_note == v.TICK:
            # REST appears only in v1 sequences
            note_instr = v.TICK
            note_offset = 0
            assert is_control is False
        else:
            parsed_note = Note.make(midi_note)
            note_instr = get_note_instrument_token(
                midi_instrument,
                parsed_note.midi_note_int,
                settings,
            )

        absolute_midi_time = midi_time + time_offset
        return Event(
            time=absolute_midi_time,
            duration=midi_duration + dur_offset,
            note_instr=note_instr + note_offset,
            is_control=is_control,
            special_code=special_code,
            original_idx_in_token_seq=original_idx_in_token_seq,
            absolute_time=absolute_midi_time,
            settings=settings,
        )

    @classmethod
    def from_list_of_token_seq(
        cls, seqs: list[list[Token]], settings: AnticipationV2Settings
    ) -> list["Event"]:
        return cls.from_token_seq(
            [token for seq in seqs for token in seq],
            settings,
        )

    @classmethod
    def from_token_seq(
        cls, raw_event_token_seq: list[Token], settings: AnticipationV2Settings
    ) -> list["Event"]:
        if not raw_event_token_seq:
            return []

        i = 0
        events = []
        ticks_seen = 0
        prev_tick_abs_time = 0

        while i < len(raw_event_token_seq):
            if raw_event_token_seq[i] in (
                settings.vocab.AUTOREGRESS,
                settings.vocab.ANTICIPATE,
            ):
                # handle flag
                ar = raw_event_token_seq[i] == settings.vocab.AUTOREGRESS
                special_code = (
                    EventSpecialCode.AUTOREGRESSIVE_TOKEN
                    if ar
                    else EventSpecialCode.ANTICIPATION_TOKEN
                )
                events.append(
                    Event(
                        time=settings.vocab.TIME_OFFSET,
                        duration=settings.vocab.DUR_OFFSET,
                        note_instr=get_note_instrument_token(0, 0, settings),
                        is_control=False,
                        original_idx_in_token_seq=i,
                        special_code=special_code,
                        settings=settings,
                        absolute_time=prev_tick_abs_time,
                    )
                )
                i += 1
                continue
            elif raw_event_token_seq[i] == settings.vocab.SEPARATOR:
                # handle sep
                events.append(
                    Event(
                        time=settings.vocab.TIME_OFFSET + 1,
                        duration=settings.vocab.DUR_OFFSET,
                        note_instr=get_note_instrument_token(0, 0, settings),
                        is_control=False,
                        special_code=EventSpecialCode.SEQ_SEPARATION_TOKENS,
                        original_idx_in_token_seq=i,
                        absolute_time=prev_tick_abs_time,
                        settings=settings,
                    )
                )
                i += 1
                continue
            elif raw_event_token_seq[i] == settings.vocab.TICK:
                # tick token
                tick_abs_time = ticks_seen * settings.tick_token_every_n_ticks
                events.append(
                    Event(
                        time=settings.vocab.TIME_OFFSET,
                        duration=settings.vocab.DUR_OFFSET,
                        note_instr=get_note_instrument_token(
                            130, 0, settings, check_range=False
                        ),
                        is_control=False,
                        special_code=EventSpecialCode.TICK,
                        original_idx_in_token_seq=i,
                        absolute_time=tick_abs_time,
                        settings=settings,
                    )
                )
                prev_tick_abs_time = tick_abs_time
                ticks_seen += 1
                i += 1
                continue
            else:
                # typical event
                e = raw_event_token_seq[i : i + 3]

                # ---- check that the event was split off the end ----
                if len(e) < 3:
                    # this can happen if the final event in the sequence was truncated, but
                    # the setting `debug_flush_remaining_token_buffer` is False, meaning that
                    # the subsequence of the sample remaining was less than the context length.
                    # in that case, that subsequence is dropped. In a production setting, the
                    # sequence that remains in the buffer would start the sequence as the next
                    # file comes in, unless it is the very last file in the dataset (in which
                    # case it is actually discarded).
                    i += 3
                    continue
                t, d, n = e

                if d in (settings.vocab.ANTICIPATE, settings.vocab.AUTOREGRESS):
                    # this token was cut off, it is repeated in the next few tokens
                    i += 1
                    continue
                if n in (settings.vocab.ANTICIPATE, settings.vocab.AUTOREGRESS):
                    # this token was cut off, it is repeated in the next few tokens
                    i += 2
                    continue
                # ----------------------------------------------------

                new_event = Event(
                    time=t,
                    duration=d,
                    note_instr=n,
                    is_control=(t >= settings.vocab.ATIME_OFFSET),
                    original_idx_in_token_seq=i,
                    absolute_time=0,
                    settings=settings,
                )
                if not new_event.is_control:
                    new_event.absolute_time = prev_tick_abs_time + new_event.midi_time()
                else:
                    new_event.absolute_time = (
                        prev_tick_abs_time + new_event.midi_time()
                    ) + (settings.delta * settings.time_resolution)
                events.append(new_event)
                i += 3

        return events

    def midi_time(self) -> MIDITick:
        if self.time < self.settings.vocab.TIME_OFFSET:
            return self.time
        else:
            if self.is_control:
                return self.time - self.settings.vocab.ATIME_OFFSET
            else:
                return self.time - self.settings.vocab.TIME_OFFSET

    def midi_duration(self) -> MIDITick:
        if self.is_control:
            return self.duration - self.settings.vocab.ADUR_OFFSET
        else:
            return self.duration - self.settings.vocab.DUR_OFFSET

    def midi_note(self) -> MIDINote:
        note, _ = self._separate_note_instr()
        return note

    def midi_instrument(self) -> MIDIProgramCode:
        # midi instrument aka midi 'program code'
        _, instrument = self._separate_note_instr()
        return instrument

    def midi_instrument_name(self) -> str:
        return get_midi_instrument_name_from_midi_instrument_code(
            self.midi_instrument()
        )

    def _separate_note_instr(self) -> tuple[MIDINote, MIDIProgramCode]:
        t = self.settings.vocab.CONTROL_OFFSET if self.is_control else 0
        b = self.note_instr - self.settings.vocab.NOTE_OFFSET - t
        note = b - (2**7) * (b // 2**7)
        instr = b // 2**7
        # these are midi values, not tokens
        return note, instr

    def note(self) -> Note:
        return Note(self.midi_note())

    def as_tokens(self) -> tuple[Token, ...]:
        if self.special_code == EventSpecialCode.AUTOREGRESSIVE_TOKEN:
            return (self.settings.vocab.AUTOREGRESS,)
        elif self.special_code == EventSpecialCode.ANTICIPATION_TOKEN:
            return (self.settings.vocab.ANTICIPATE,)
        elif self.special_code == EventSpecialCode.SEQ_SEPARATION_TOKENS:
            return (self.settings.vocab.SEPARATOR,)
        elif self.special_code == EventSpecialCode.TICK:
            return (self.settings.vocab.TICK,)
        else:
            # not a special token, a triple
            return self.time, self.duration, self.note_instr

    def is_rest(self) -> bool:
        if self.settings.tick_token_every_n_ticks == 0:
            return self.note_instr == self.settings.vocab.TICK
        else:
            return False

    def is_tick(self) -> bool:
        return self.special_code == EventSpecialCode.TICK

    def is_note_event(self) -> bool:
        # NB: a note just means it's a valid triple, it can
        # also be a control
        return self.special_code == EventSpecialCode.TYPICAL_EVENT

    def is_anticipate(self) -> bool:
        return self.special_code == EventSpecialCode.ANTICIPATION_TOKEN

    def is_separator(self) -> bool:
        return self.special_code == EventSpecialCode.SEQ_SEPARATION_TOKENS

    def is_musically_equal(self, other) -> bool:
        if not isinstance(other, Event):
            return False

        # equality without considering which token space this came from
        # (event vs. control does not matter)
        return (
            self.midi_time() == other.midi_time()
            and self.midi_duration() == other.midi_duration()
            and self.midi_note() == other.midi_note()
            and self.midi_instrument() == other.midi_instrument()
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, Event):
            return False

        return (
            # settings properties
            # all settings need not be equal but the ones that can
            # affect tokenization should be equal
            self.settings.vocab == other.settings.vocab
            and self.settings.time_resolution == other.settings.time_resolution
            and self.settings.max_note_duration_in_seconds
            == other.settings.max_note_duration_in_seconds
            and self.settings.tick_token_every_n_ticks
            == other.settings.tick_token_every_n_ticks
            and
            # note properties
            self.time == other.time
            and self.duration == other.duration
            and self.note_instr == other.note_instr
        )

    def __repr__(self) -> str:
        return "Event(midi_time={0}, midi_duration={1}, midi_note={2} ({6}), midi_instrument={3}, is_control={4}, special_code={5}, absolute_midi_time={7})".format(
            self.midi_time(),
            self.midi_duration(),
            self.midi_note(),
            self.midi_instrument(),
            self.is_control,
            self.special_code,
            self.note().to_name(),
            self.absolute_time,
        )

    def __hash__(self) -> int:
        return hash(repr(self))
