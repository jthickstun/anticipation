from typing import Any
from pathlib import Path
from dataclasses import dataclass, asdict, fields
from json import dumps, loads
from functools import cache

import anticipation.vocab as v1_vocab
from anticipation.v2.util import get_md5_of_string

# anticipation/anticipation/v2/config.py is here
_here = Path(__file__).parent

REPO_ROOT = _here.parent.parent
DATASET_ROOT = REPO_ROOT / "data"
CONFIG_ROOT = REPO_ROOT / "config"

# expect lakh midi to be here:
# <REPO_ROOT>/data/lmd_full/...
LAKH_MIDI_FULL_PATH = DATASET_ROOT / "lmd_full"
TOKENIZED_DATASETS_SAVE_TO_PATH = DATASET_ROOT / "tokenized_datasets"


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
    ATICK: int = ANOTE_OFFSET + 1

    # this defines more or less where musical tokens end and control
    # or special tokens begin
    SPECIAL_OFFSET: int = ATICK + 1

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


@dataclass(frozen=True)
class AnticipationV2Settings:
    """
    Object for holding 'global'-like settings. If a property is frequently referenced
    by several functions, then it is a good candidate to put here.
    """

    vocab: Vocab
    compound_size: int = 5
    time_resolution: int = 100
    debug: bool = False
    debug_flush_remaining_token_buffer: bool = False
    min_track_events: int = 100
    min_track_time_in_seconds: int = 10
    max_track_time_in_seconds: int = 3600

    # the maximum number of instruments that produce notes in a file
    max_track_instruments: int = 16

    max_midi_pitch: int = 128
    # 128 program codes + 1 for drums
    max_midi_instrument: int = 129
    max_note_duration_in_seconds: int = 10
    context_size: int = 1024
    event_size: int = 3
    m: int = 341

    # original data mixture:
    # - 10% without anticipation (standard AR)
    # - 10% span anticipation
    # - 40% instrument anticipation
    # - 40% random anticipation
    num_autoregressive_seq_per_midi_file: int = 1
    num_span_anticipation_augmentations_per_midi_file: int = 1
    num_instrument_anticipation_augmentations_per_midi_file: int = 4
    num_random_anticipation_augmentations_per_midi_file: int = 4
    span_anticipation_lambda: float = 0.05
    train_data_split_shuffle_random_seed: int = 42
    num_workers_in_dataset_construction: int = 1

    # if this is 0, will not add anything
    tick_token_frequency_in_midi_ticks: int = 0

    # anticipation interval
    delta: int = 5

    # new data augmentation styles in v2
    augmentation_pitch_shifts: tuple[int, ...] = ()

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

        assert v.SPECIAL_OFFSET == v.ANOTE_OFFSET + total_instr_note_tokens  # 0-indexed
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
