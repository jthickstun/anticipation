from anticipation.v2.config import (
    AnticipationV2Settings,
    Vocab,
    make_vocab,
    CONFIG_ROOT,
)


if __name__ == "__main__":
    tick_token_every_n_ticks = 100
    time_resolution = 100
    max_note_duration_in_seconds = 10
    my_vocab: Vocab = make_vocab(
        tick_token_every_n_ticks=tick_token_every_n_ticks,
        time_resolution=time_resolution,
        max_note_duration_in_seconds=max_note_duration_in_seconds,
    )
    to_create = AnticipationV2Settings(
        vocab=my_vocab,
        delta=5,
        context_size=4096,
        # filter settings
        max_track_instruments=16,
        max_note_duration_in_seconds=max_note_duration_in_seconds,
        # data mixture and augmentation settings
        num_autoregressive_seq_per_midi_file=1,
        num_span_anticipation_augmentations_per_midi_file=1,
        num_instrument_anticipation_augmentations_per_midi_file=1,
        # system-like settings
        num_workers_in_dataset_construction=10,
        do_clip_overlapping_durations_in_midi_conversion=False,
        # time settings
        tick_token_every_n_ticks=tick_token_every_n_ticks,
        time_resolution=time_resolution,
    )
    saved_to = to_create.save_to_disk(enclosing_folder=CONFIG_ROOT)
    print(f"Saved to: {saved_to}")
