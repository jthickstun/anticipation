from typing import Any
from pathlib import Path
from dataclasses import dataclass, asdict
from json import dumps, loads
from functools import cache

import anticipation.vocab as v1_vocab
from anticipation.v2.util import get_md5_of_string

# anticipation/anticipation/v2/config.py is here
_here = Path(__file__).parent

REPO_ROOT = _here.parent.parent
DATASET_ROOT = REPO_ROOT / "data"


@dataclass(frozen=True)
class Vocab:
    SEPARATOR: int = v1_vocab.SEPARATOR
    CONTROL_OFFSET: int = v1_vocab.CONTROL_OFFSET
    NOTE_OFFSET: int = v1_vocab.NOTE_OFFSET
    DUR_OFFSET: int = v1_vocab.DUR_OFFSET
    TIME_OFFSET: int = v1_vocab.TIME_OFFSET

    SPECIAL_OFFSET: int = v1_vocab.SPECIAL_OFFSET

    ATIME_OFFSET: int = v1_vocab.ATIME_OFFSET
    ANOTE_OFFSET: int = v1_vocab.ANOTE_OFFSET

    # I am calling these two tokens the 'flag token'
    # these are basically BOS tokens
    ANTICIPATE: int = v1_vocab.ANTICIPATE
    AUTOREGRESS: int = v1_vocab.AUTOREGRESS

    REST: int = v1_vocab.REST

    _last_token_in_v1: int = v1_vocab.VOCAB_SIZE

    # add the number of new blocks in v2 here
    # added:
    # - PAD
    # - TICK
    VOCAB_SIZE: int = v1_vocab.VOCAB_SIZE + 2

    # https://github.com/jthickstun/anticipation/blob/6927699c5243fd91d1d252211c29885377d9dda5/train/tokenize-new.py#L33
    # https://github.com/jthickstun/anticipation/blob/6927699c5243fd91d1d252211c29885377d9dda5/anticipation/vocabs/localmidi.py#L65
    # aka 'tick'
    TICK: int = _last_token_in_v1 + 1


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
    max_track_instruments: int = 16
    max_midi_pitch: int = 128
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

    # set to 3 to keep parity with v1
    num_sep_tokens: int = 1

    # if this is 0, will not add anything
    tick_token_frequency_in_midi_ticks: int = 100

    # anticipation interval
    delta: int = 5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # noqa

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
        md5 = load_from_file.stem.split("settings_")[1]
        settings_str = load_from_file.read_text()
        assert md5 == get_md5_of_string(settings_str), "file integrity compromised."

        settings_parsed = loads(settings_str)
        v_str = settings_parsed.pop("vocab")
        return AnticipationV2Settings(vocab=Vocab(**v_str), **settings_parsed)
