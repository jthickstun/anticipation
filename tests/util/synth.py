from pathlib import Path

import numpy as np
from symusic import Score, Synthesizer, BuiltInSF3, dump_wav


def get_wav_from_midi_as_array(
    midi_file_path: Path, sample_rate: int = 44_100
) -> np.ndarray:
    score = Score(midi_file_path)
    soundfont_path = BuiltInSF3.MuseScoreGeneral().path(download=True)
    synthesizer = Synthesizer(
        sf_path=soundfont_path, sample_rate=sample_rate, quality=4
    )
    return synthesizer.render(score, stereo=True)


def get_wav_from_midi_and_save_to_path(
    midi_file_path: Path, save_wav_output_to: Path, sample_rate: int = 44_100
) -> Path:
    audio_data = get_wav_from_midi_as_array(midi_file_path, sample_rate)
    dump_wav(str(save_wav_output_to.absolute()), audio_data, sample_rate=sample_rate)
    return save_wav_output_to
