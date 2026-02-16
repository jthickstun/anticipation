from pathlib import Path

from symusic import Score, TimeUnit


def get_trimmed_midi(
    midi_file_path: Path,
    save_trimmed_midi_to_path: Path,
    start_time_sec: float,
    end_time_sec: float,
) -> Path:
    """
    Given a midi file path, isolate the events within a window of time in seconds:

        [start_time_sec, end_time_sec)

    Saves this new midi file to the given path specified by `save_trimmed_midi_to_path`.

    The time is not shifted, so there will be a bunch of silence at the start if the
    start time is non-zero.
    """
    score = Score(midi_file_path, ttype=TimeUnit.second)
    score.clip(
        start=start_time_sec,
        end=end_time_sec,
        # clip_end=False ensures that notes that go over the end are truncated and
        # not just fully deleted.
        clip_end=False,
        inplace=True,
    )
    score.dump_midi(save_trimmed_midi_to_path)
    return save_trimmed_midi_to_path
