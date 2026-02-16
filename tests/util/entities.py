"""
These are utilities for testing / developing with the codebase.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Union
import re

import anticipation.vocab as v

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
    instrument_midi_code: MIDIProgramCode, note_midi_code: MIDINote
) -> Token:
    return v.NOTE_OFFSET + (v.MAX_PITCH * instrument_midi_code + note_midi_code)


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
    else:
        return "?"


@dataclass
class Event:
    time: Token
    duration: Token
    note_instr: Token

    # these are all 'derived', meaning they are not
    # direct properties of a token sequence
    is_control: bool
    original_idx_in_token_seq: int
    special_code: EventSpecialCode = EventSpecialCode.TYPICAL_EVENT
    absolute_time: MIDITick = 0

    @classmethod
    def from_midi_values(
        cls,
        midi_time: MIDITick,
        midi_duration: MIDITick,
        midi_instrument: MIDIProgramCode,
        midi_note: Union[MIDINote, str],
        is_control: bool = False,
        special_code: EventSpecialCode = EventSpecialCode.TYPICAL_EVENT,
        original_idx_in_token_seq: int = 0,
    ) -> "Event":
        note_offset = v.CONTROL_OFFSET if is_control else 0
        time_offset = v.ATIME_OFFSET if is_control else v.TIME_OFFSET
        dur_offset = v.ADUR_OFFSET if is_control else v.DUR_OFFSET

        if midi_note == "REST" or midi_note == v.REST:
            # TODO: consider, should REST be special code?
            note_instr = v.REST
            note_offset = 0
            assert is_control is False
        else:
            parsed_note = Note.make(midi_note)
            note_instr = get_note_instrument_token(
                midi_instrument, parsed_note.midi_note_int
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
        )

    @classmethod
    def from_token_seq(
        cls, raw_event_token_seq: list[Token], settings: AnticipationV2Settings
    ) -> list["Event"]:
        if not raw_event_token_seq:
            return []

        v = settings.vocab
        i = 0
        events = []

        # the function ops.min_time expects sequence to not contain any flag tokens
        # prev_t = ops.min_time(
        #     [
        #         x
        #         for x in raw_event_token_seq[i:]
        #         if x not in (v.ANTICIPATE, v.AUTOREGRESS, v.SEPARATOR)
        #     ],
        #     seconds=False,
        # )

        ticks_seen = 0
        prev_tick_abs_time = 0
        while i < len(raw_event_token_seq):
            if raw_event_token_seq[i] in (v.AUTOREGRESS, v.ANTICIPATE):
                # handle flag
                ar = raw_event_token_seq[i] == v.AUTOREGRESS
                special_code = (
                    EventSpecialCode.AUTOREGRESSIVE_TOKEN
                    if ar
                    else EventSpecialCode.ANTICIPATION_TOKEN
                )
                events.append(
                    Event(
                        time=v.TIME_OFFSET,
                        duration=v.DUR_OFFSET,
                        note_instr=get_note_instrument_token(0, 0),
                        is_control=False,
                        original_idx_in_token_seq=i,
                        special_code=special_code,
                    )
                )
                i += 1
                continue
            elif raw_event_token_seq[i] == v.SEPARATOR:
                # handle sep
                events.append(
                    Event(
                        time=v.TIME_OFFSET + 1,
                        duration=v.DUR_OFFSET,
                        note_instr=get_note_instrument_token(0, 0),
                        is_control=False,
                        special_code=EventSpecialCode.SEQ_SEPARATION_TOKENS,
                        original_idx_in_token_seq=i,
                        absolute_time=v.TIME_OFFSET + 1,
                    )
                )
                i += 1
                continue
            elif raw_event_token_seq[i] == settings.vocab.TICK:
                # tick token
                tick_abs_time = ticks_seen * settings.tick_token_frequency_in_midi_ticks
                events.append(
                    Event(
                        time=settings.vocab.TIME_OFFSET,
                        duration=settings.vocab.DUR_OFFSET,
                        note_instr=get_note_instrument_token(130, 0),
                        is_control=False,
                        special_code=EventSpecialCode.TICK,
                        original_idx_in_token_seq=i,
                        absolute_time=tick_abs_time,
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
                    is_control=(t >= v.ATIME_OFFSET),
                    original_idx_in_token_seq=i,
                    absolute_time=0,
                )
                new_event.absolute_time = prev_tick_abs_time + new_event.midi_time()
                events.append(new_event)

                if events[-1].is_control:
                    prev_t = t - v.ATIME_OFFSET
                else:
                    prev_t = t
                i += 3

        return events

    def midi_time(self) -> MIDITick:
        if self.time < v.TIME_OFFSET:
            return self.time
        else:
            if self.is_control:
                return self.time - v.ATIME_OFFSET
            else:
                return self.time - v.TIME_OFFSET

    def midi_duration(self) -> MIDITick:
        if self.is_control:
            return self.duration - v.ADUR_OFFSET
        else:
            return self.duration - v.DUR_OFFSET

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
        t = v.CONTROL_OFFSET if self.is_control else 0
        b = self.note_instr - v.NOTE_OFFSET - t
        note = b - (2**7) * (b // 2**7)
        instr = b // 2**7
        # these are midi values, not tokens
        return note, instr

    def note(self) -> Note:
        return Note(self.midi_note())

    def as_tokens(self) -> tuple[Token, ...]:
        if self.special_code == 1:
            return (v.AUTOREGRESS,)
        elif self.special_code == 2:
            return (v.ANTICIPATE,)
        elif self.special_code == 3:
            return v.SEPARATOR, v.SEPARATOR, v.SEPARATOR
        else:
            return self.time, self.duration, self.note_instr

    def is_rest(self, settings: AnticipationV2Settings) -> bool:
        return self.note_instr == settings.vocab.REST

    def is_tick(self) -> bool:
        return self.special_code == EventSpecialCode.TICK

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
