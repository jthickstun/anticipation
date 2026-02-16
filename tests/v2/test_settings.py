import tempfile
from pathlib import Path
from json import loads

from anticipation.v2.config import AnticipationV2Settings, Vocab


def test_serialize_anticipation_v2_settings() -> None:
    settings = AnticipationV2Settings(
        vocab=Vocab(),
    )
    assert settings.to_dict() == {
        "compound_size": 5,
        "context_size": 1024,
        "debug": False,
        "debug_flush_remaining_token_buffer": False,
        "delta": 5,
        "event_size": 3,
        "m": 341,
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
        "num_sep_tokens": 1,
        "num_span_anticipation_augmentations_per_midi_file": 1,
        "span_anticipation_lambda": 0.05,
        "tick_token_frequency_in_midi_ticks": 100,
        "time_resolution": 100,
        "vocab": {
            "ANOTE_OFFSET": 38513,
            "ANTICIPATE": 55027,
            "ATIME_OFFSET": 27513,
            "AUTOREGRESS": 55026,
            "CONTROL_OFFSET": 27513,
            "DUR_OFFSET": 10000,
            "NOTE_OFFSET": 11000,
            "REST": 27512,
            "SEPARATOR": 55025,
            "SPECIAL_OFFSET": 55025,
            "TICK": 55029,
            "TIME_OFFSET": 0,
            "VOCAB_SIZE": 55030,
            "_last_token_in_v1": 55028,
        },
    }
    s, _ = settings._get_as_file()
    assert settings.md5_hash() == "e58abd403796aeb4c639bf1e61d52f22"
    reloaded_settings = loads(s)
    assert settings.to_dict() == reloaded_settings


def test_save_load_settings() -> None:
    with tempfile.TemporaryDirectory() as td:
        temp_enclosing_path = Path(td)
        settings = AnticipationV2Settings(
            vocab=Vocab(),
        )
        saved_to = settings.save_to_disk(temp_enclosing_path)
        assert isinstance(saved_to, Path)

        reloaded_settings = AnticipationV2Settings.load_from_disk(saved_to)
        assert settings == reloaded_settings
