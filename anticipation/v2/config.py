from typing import Any
from pathlib import Path
from dataclasses import dataclass, asdict, fields
from json import dumps, loads
from functools import cache

import anticipation.vocab as v1_vocab
from anticipation.v2.util import get_md5_of_string
from anticipation import config as v1_config

# anticipation/anticipation/v2/config.py is here
_here = Path(__file__).parent

REPO_ROOT = _here.parent.parent
DATASET_ROOT = REPO_ROOT / "data"
CONFIG_ROOT = REPO_ROOT / "config"

# expect lakh midi to be here:
# <REPO_ROOT>/data/lmd_full/...
LAKH_MIDI_FULL_PATH = DATASET_ROOT / "lmd_full"
TOKENIZED_DATASETS_SAVE_TO_PATH = DATASET_ROOT / "tokenized_datasets"

NUM_MIDI_PITCHES = 128


@dataclass(frozen=True)
class Vocab:
    # events
    EVENT_OFFSET: int = v1_vocab.EVENT_OFFSET
    TIME_OFFSET: int = v1_vocab.TIME_OFFSET
    DUR_OFFSET: int = v1_vocab.DUR_OFFSET
    NOTE_OFFSET: int = v1_vocab.NOTE_OFFSET
    # Rest is now called TICK, serves a different purpose
    TICK: int = v1_vocab.REST

    # controls
    CONTROL_OFFSET: int = TICK + 1
    ATIME_OFFSET: int = v1_vocab.ATIME_OFFSET
    ADUR_OFFSET: int = v1_vocab.ADUR_OFFSET
    ANOTE_OFFSET: int = v1_vocab.ANOTE_OFFSET

    # this defines more or less where musical tokens end and control
    # or special tokens begin
    SPECIAL_OFFSET: int = ANOTE_OFFSET + v1_config.MAX_NOTE

    # special tokens for control type and sequence separation
    SEPARATOR: int = SPECIAL_OFFSET
    AUTOREGRESS: int = SEPARATOR + 1
    ANTICIPATE: int = AUTOREGRESS + 1

    def __post_init__(self) -> None:
        # check that all the token values are organized and increasing
        # event block
        assert self.EVENT_OFFSET >= 0
        assert self.TIME_OFFSET >= self.EVENT_OFFSET
        assert self.DUR_OFFSET > self.TIME_OFFSET
        assert self.NOTE_OFFSET > self.DUR_OFFSET
        assert self.TICK > self.NOTE_OFFSET

        # control block
        assert self.CONTROL_OFFSET > self.NOTE_OFFSET
        assert self.ATIME_OFFSET >= self.CONTROL_OFFSET
        assert self.ADUR_OFFSET > self.ATIME_OFFSET
        assert self.ANOTE_OFFSET > self.ADUR_OFFSET

        # this is not strong enough to guarantee correctness, we will
        # check that later in settings
        assert self.SPECIAL_OFFSET > self.ANOTE_OFFSET
        assert self.SEPARATOR >= self.SPECIAL_OFFSET
        assert self.AUTOREGRESS > self.SEPARATOR
        assert self.ANTICIPATE > self.AUTOREGRESS

    def total_tokens(self) -> int:
        field_to_val = {
            field.name: getattr(self, field.name)
            for field in fields(self)  # noqa
        }
        # zero indexed
        return max(field_to_val.values()) + 1

    def realize_as_array(self) -> list[dict]:
        v = []
        for i in range(self.TIME_OFFSET, self.DUR_OFFSET):
            v.append(
                {
                    "i": i,
                    "kind": "time",
                    "is_control": False,
                    "info": {
                        "time at": i - self.TIME_OFFSET,
                    },
                }
            )
        for i in range(self.DUR_OFFSET, self.NOTE_OFFSET):
            v.append(
                {
                    "i": i,
                    "kind": "duration",
                    "is_control": False,
                    "info": {"duration of": i - self.DUR_OFFSET},
                }
            )
        for i in range(self.NOTE_OFFSET, self.TICK):
            rel_i = i - self.NOTE_OFFSET
            v.append(
                {
                    "i": i,
                    "kind": "note",
                    "is_control": False,
                    "info": {
                        "midi_program_code": rel_i // NUM_MIDI_PITCHES,
                        "pitch": rel_i
                        - (NUM_MIDI_PITCHES * (rel_i // NUM_MIDI_PITCHES)),
                    },
                }
            )

        v.append(
            {
                "i": self.TICK,
                "kind": "tick",
                "is_control": False,
                "info": {},
            }
        )

        for i in range(self.ATIME_OFFSET, self.ADUR_OFFSET):
            v.append(
                {
                    "i": i,
                    "kind": "time",
                    "is_control": True,
                    "info": {
                        "time at": i - self.ATIME_OFFSET,
                    },
                }
            )
        for i in range(self.ADUR_OFFSET, self.ANOTE_OFFSET):
            v.append(
                {
                    "i": i,
                    "kind": "duration",
                    "is_control": True,
                    "info": {"duration of": i - self.ADUR_OFFSET},
                }
            )
        for i in range(self.ANOTE_OFFSET, self.SPECIAL_OFFSET):
            rel_i = i - self.ANOTE_OFFSET
            v.append(
                {
                    "i": i,
                    "kind": "note",
                    "is_control": True,
                    "info": {
                        "midi_program_code": rel_i // NUM_MIDI_PITCHES,
                        "pitch": rel_i
                        - (NUM_MIDI_PITCHES * (rel_i // NUM_MIDI_PITCHES)),
                    },
                }
            )

        v.append(
            {
                "i": self.SEPARATOR,
                "kind": "sep",
                "is_control": False,
                "info": {},
            }
        )
        v.append(
            {
                "i": self.AUTOREGRESS,
                "kind": "autoregress",
                "is_control": False,
                "info": {},
            }
        )
        v.append(
            {
                "i": self.ANTICIPATE,
                "kind": "anticipate",
                "is_control": False,
                "info": {},
            }
        )
        return v


def make_vocab(
    tick_token_every_n_ticks: int,
    max_note_duration_in_seconds: float,
    time_resolution: int,
) -> Vocab:
    max_note_duration_in_ticks = int(max_note_duration_in_seconds * time_resolution)
    time_offset = 0

    if tick_token_every_n_ticks == 0:
        # if no tick frequency, revert back to v1's values for these ranges
        time_stops_at = v1_vocab.DUR_OFFSET
        dur_stops_at = v1_vocab.NOTE_OFFSET
    else:
        time_stops_at = time_offset + tick_token_every_n_ticks
        dur_stops_at = time_stops_at + max_note_duration_in_ticks

    # can't really change these
    max_midi_instrument = 129
    num_notes = max_midi_instrument * NUM_MIDI_PITCHES
    note_stops_at = dur_stops_at + num_notes

    control_offset = note_stops_at + 1

    special_offset = dur_stops_at + control_offset + num_notes

    return Vocab(
        # events
        EVENT_OFFSET=0,
        # the triple of (time, dur, note x instr)
        TIME_OFFSET=time_offset,
        DUR_OFFSET=time_stops_at,
        NOTE_OFFSET=dur_stops_at,
        # the tick token
        TICK=note_stops_at,
        # controls
        CONTROL_OFFSET=control_offset,
        # the triple of (time, dur, note x instr)
        ATIME_OFFSET=time_offset + control_offset,
        ADUR_OFFSET=time_stops_at + control_offset,
        ANOTE_OFFSET=dur_stops_at + control_offset,
        # sequence-level instruction tokens
        SPECIAL_OFFSET=special_offset,
        SEPARATOR=special_offset,
        AUTOREGRESS=special_offset + 1,
        ANTICIPATE=special_offset + 2,
    )


# MIDI defines 128 specific instruments labelled 0-127, but then
# the 129th instrument, has program code 128 and is specifically
# for drum kits
MIDI_DRUMS_PROGRAM_CODE = 128


@dataclass(frozen=True)
class AnticipationV2Settings:
    """
    Object for holding 'global'-like settings. If a property is frequently referenced
    by several functions, then it is a good candidate to put here.
    """

    vocab: Vocab
    compound_size: int = 5
    time_resolution: int = 100
    min_track_events: int = 100
    min_track_time_in_seconds: int = 10
    max_track_time_in_seconds: int = 3600
    debug: bool = False
    debug_flush_remaining_token_buffer: bool = False

    # the maximum number of instruments that produce notes in a file
    max_track_instruments: int = 16

    max_midi_pitch: int = 128
    # 128 program codes + 1 for drums
    max_midi_instrument: int = 129
    max_note_duration_in_seconds: int = 10

    # if this is true, then a note cannot have overlapping sustains
    # it is possible in MIDI to have two notes from the same instrument
    # playing simultaneously and overlapping
    do_clip_overlapping_durations_in_midi_conversion: bool = False

    context_size: int = 1024
    event_size: int = 3

    # set the data mixture with these
    num_autoregressive_seq_per_midi_file: int = 1
    num_span_anticipation_augmentations_per_midi_file: int = 0
    num_instrument_anticipation_augmentations_per_midi_file: int = 4

    train_data_split_shuffle_random_seed: int = 42
    num_workers_in_dataset_construction: int = 1

    # if this is 0, will not add anything
    tick_token_every_n_ticks: int = 0

    # anticipation interval (in seconds)
    delta: int = 5

    # new data augmentation styles in v2
    augmentation_pitch_shifts: tuple[int, ...] = ()

    @property
    def max_dur(self) -> int:
        return self.time_resolution * self.max_note_duration_in_seconds

    @property
    def max_time(self) -> int:
        return self.time_resolution * self.max_track_time_in_seconds

    @property
    def max_note(self) -> int:
        return self.max_midi_pitch * self.max_midi_instrument

    def to_dict(self) -> dict[str, Any]:
        settings = asdict(self)  # noqa
        settings["augmentation_pitch_shifts"] = tuple(
            settings["augmentation_pitch_shifts"]
        )
        return settings

    def _get_as_file(self) -> tuple[str, str]:
        """
        Return serialized string of settings and the md5 hash of its string representation.
        """
        s = dumps(self.to_dict(), indent=4, sort_keys=True)
        md5 = get_md5_of_string(s)
        return s, md5

    @cache
    def md5_hash(self) -> str:
        _, md5 = self._get_as_file()
        return md5

    def save_to_disk(self, enclosing_folder: Path) -> Path:
        assert isinstance(enclosing_folder, Path)
        assert enclosing_folder.exists()
        assert enclosing_folder.is_dir()
        s, md5 = self._get_as_file()
        save_to = enclosing_folder / ("settings_" + md5 + ".json")
        save_to.write_text(s)
        return save_to

    @classmethod
    def load_from_disk(cls, load_from_file: Path) -> "AnticipationV2Settings":
        assert isinstance(load_from_file, Path)
        assert load_from_file.exists()
        assert load_from_file.is_file()
        assert load_from_file.suffix == ".json"

        # can prefix with anything as long as it is separated by _
        md5 = load_from_file.stem.split("_")[-1]
        settings_str = load_from_file.read_text()
        assert md5 == get_md5_of_string(settings_str), "file integrity compromised."

        settings_parsed = loads(settings_str)
        augmentation_pitch_shifts = tuple(
            settings_parsed.pop("augmentation_pitch_shifts")
        )
        v_str = settings_parsed.pop("vocab")
        return AnticipationV2Settings(
            vocab=Vocab(**v_str),
            augmentation_pitch_shifts=augmentation_pitch_shifts,
            **settings_parsed,
        )

    def __post_init__(self) -> None:
        # this runs after the constructor
        # check that the vocab is valid
        v = self.vocab

        # check that there are enough tokens for this time resolution in the duration
        assert v.DUR_OFFSET - v.TIME_OFFSET >= self.time_resolution

        # check that there are enough tokens for representing the max duration given res
        assert v.NOTE_OFFSET - v.DUR_OFFSET == (
            self.max_note_duration_in_seconds * self.time_resolution
        )

        # ensure that the control tokens' space does not overlap with events' space
        total_instr_note_tokens = self.max_midi_pitch * self.max_midi_instrument
        assert v.TICK == v.NOTE_OFFSET + total_instr_note_tokens  # 0-indexed
        assert v.CONTROL_OFFSET > v.TICK

        # ensure that the ranges of (time -> duration) and (duration -> note) are
        # identical for events and controls
        assert v.ADUR_OFFSET - v.ATIME_OFFSET == v.DUR_OFFSET - v.TIME_OFFSET
        assert v.ANOTE_OFFSET - v.ADUR_OFFSET == v.NOTE_OFFSET - v.DUR_OFFSET

        assert v.SPECIAL_OFFSET >= v.ANOTE_OFFSET + total_instr_note_tokens, (
            f"!({v.SPECIAL_OFFSET} >= {v.ANOTE_OFFSET + total_instr_note_tokens})"
        )
        assert v.SEPARATOR >= v.SPECIAL_OFFSET
        assert v.AUTOREGRESS > v.SEPARATOR
        assert v.ANTICIPATE > v.AUTOREGRESS

        # ensure there's no 0s in this
        assert 0 not in self.augmentation_pitch_shifts, (
            "adding zero shift results in duplication"
        )
        # causes a strange type error, but we wouldn't want to do this
        # extreme of a transposition anyway. Just check it here for faster
        # failing.
        assert all(abs(x) <= 127 for x in self.augmentation_pitch_shifts)

        # check there is sufficient spacing, probably better to
        # create vocabulary with `make_vocab` function
        assert v.DUR_OFFSET - v.TIME_OFFSET >= self.tick_token_every_n_ticks
