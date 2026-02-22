import tempfile
from pathlib import Path
from json import loads

import pytest

from anticipation.v2.config import AnticipationV2Settings, Vocab


def test_serialize_anticipation_v2_settings() -> None:
    settings = AnticipationV2Settings(
        vocab=Vocab(),
    )
    assert settings.to_dict() == {
        "augmentation_pitch_shifts": (),
        "compound_size": 5,
        "context_size": 1024,
        "debug": False,
        "debug_flush_remaining_token_buffer": False,
        "delta": 5,
        "event_size": 3,
        "m": 341,
        "max_midi_instrument": 129,
        "max_midi_pitch": 128,
        "max_note_duration_in_seconds": 10,
        "max_track_instruments": 16,
        "max_track_time_in_seconds": 3600,
        "min_track_events": 100,
        "min_track_time_in_seconds": 10,
        "num_autoregressive_seq_per_midi_file": 1,
        "num_instrument_anticipation_augmentations_per_midi_file": 4,
        "num_random_anticipation_augmentations_per_midi_file": 4,
        "train_data_split_shuffle_random_seed": 42,
        "num_workers_in_dataset_construction": 1,
        "num_span_anticipation_augmentations_per_midi_file": 1,
        "span_anticipation_lambda": 0.05,
        "tick_token_frequency_in_midi_ticks": 0,
        "time_resolution": 100,
        "vocab": {
            "ADUR_OFFSET": 37513,
            "ANOTE_OFFSET": 38513,
            "ANTICIPATE": 55027,
            "ATIME_OFFSET": 27513,
            "AUTOREGRESS": 55026,
            "CONTROL_OFFSET": 27513,
            "DUR_OFFSET": 10000,
            "EVENT_OFFSET": 0,
            "NOTE_OFFSET": 11000,
            "TICK": 27512,
            "SEPARATOR": 55025,
            "SPECIAL_OFFSET": 55025,
            "TIME_OFFSET": 0,
        },
    }
    s, _ = settings._get_as_file()
    assert settings.md5_hash() == "d12e0635b62bd7a827c8455cd402d323"


def test_save_load_settings() -> None:
    with tempfile.TemporaryDirectory() as td:
        temp_enclosing_path = Path(td)
        settings = AnticipationV2Settings(
            vocab=Vocab(), augmentation_pitch_shifts=(-1, 1)
        )
        saved_to = settings.save_to_disk(temp_enclosing_path)
        assert isinstance(saved_to, Path)

        reloaded_settings = AnticipationV2Settings.load_from_disk(saved_to)

        # these might need some special reload logic
        assert isinstance(reloaded_settings.augmentation_pitch_shifts, tuple)
        assert isinstance(reloaded_settings.vocab, Vocab)

        assert settings == reloaded_settings


def test_create_invalid_pitch_augmentations_settings() -> None:
    with pytest.raises(AssertionError):
        # forbid zeros because they will lead to duplication
        AnticipationV2Settings(
            vocab=Vocab(),
            augmentation_pitch_shifts=(
                -1,
                0,
            ),
        )
    with pytest.raises(AssertionError):
        # forbid out of bounds transpositions
        AnticipationV2Settings(vocab=Vocab(), augmentation_pitch_shifts=(128,))


def test_create_invalid_vocab() -> None:
    with pytest.raises(AssertionError):
        AnticipationV2Settings(
            vocab=Vocab(
                EVENT_OFFSET=0,
                TIME_OFFSET=0,
                DUR_OFFSET=100,
                NOTE_OFFSET=1100,
                # there's a bad overlap between events and controls
                CONTROL_OFFSET=17612 - 1,
                ATIME_OFFSET=17612,
                ADUR_OFFSET=17712,
                ANOTE_OFFSET=18712,
            ),
        )


def test_get_vocab_size(local_midi_vocab: Vocab) -> None:
    assert local_midi_vocab.EVENT_OFFSET == 0
    assert local_midi_vocab.TIME_OFFSET == 0
    assert local_midi_vocab.DUR_OFFSET == 100
    assert local_midi_vocab.NOTE_OFFSET == 1100
    assert local_midi_vocab.TICK == 1100 + (129 * 128)
    assert local_midi_vocab.CONTROL_OFFSET == local_midi_vocab.TICK + 1
    assert local_midi_vocab.ATIME_OFFSET == 17613
    assert local_midi_vocab.ADUR_OFFSET == 17613 + 100
    assert local_midi_vocab.ANOTE_OFFSET == 18713
    assert local_midi_vocab.SPECIAL_OFFSET == 18713 + (129 * 128)
    assert local_midi_vocab.SPECIAL_OFFSET == 35225
    assert local_midi_vocab.SEPARATOR == 35225
    assert local_midi_vocab.AUTOREGRESS == 35226
    assert local_midi_vocab.ANTICIPATE == 35227
    assert local_midi_vocab.total_tokens() == 35228


# def test_write_settings_file(local_midi_settings_ar_only):
#     local_midi_settings_ar_only.save_to_disk(Path(__file__).parent)
