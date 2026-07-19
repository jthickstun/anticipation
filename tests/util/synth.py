from typing import Optional, Any
from pathlib import Path
import subprocess
import tempfile
import wave

import numpy as np
from symusic import Score, Synthesizer, BuiltInSF3, dump_wav


def _render_midi_with_fluidsynth(
    midi_path: Path,
    soundfont_path: Path,
    output_path: Optional[Path] = None,
    sample_rate: int = 48_000,
) -> Path:
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        output_path = Path(tmp.name)

    subprocess.run(
        [
            "fluidsynth",
            "-ni",
            "-R", "no",
            "-C", "no",
            "-g", "1.0",
            "-r", str(sample_rate),
            "-O", "s16",
            "-F", str(output_path),
            str(soundfont_path),
            str(midi_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return output_path

def get_wav_from_midi_and_save_to_path(
    midi_file_path: Path,  save_wav_output_to: Path, sample_rate: int = 44_100
) -> Path:
    score = Score(midi_file_path)
    soundfont_path = BuiltInSF3.MuseScoreGeneral().path(download=True)
    try:
        synthesizer = Synthesizer(
            sf_path=soundfont_path, sample_rate=sample_rate, quality=4
        )
        audio_data = synthesizer.render(score, stereo=True)
        dump_wav(str(save_wav_output_to.absolute()), audio_data, sample_rate=sample_rate)
        return save_wav_output_to
    except TypeError:
        # there's some issue with symusic that wasn't there before...
        # TypeError: Unable to convert function return value to a Python type! The signature was
        # render(self, score: symusic.core.ScoreTick, stereo: bool = True) -> Eigen::Array<float, -1, -1, 0, -1, -1>
        my_wav_file = _render_midi_with_fluidsynth(
            midi_file_path,
            soundfont_path,
            save_wav_output_to,
            sample_rate,
        )
        return my_wav_file



def read_wav(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frame_count = wav.getnframes()
        raw_audio = wav.readframes(frame_count)

    if sample_width != 2:
        raise ValueError(
            f"Expected 16-bit PCM WAV, but got {sample_width * 8}-bit audio"
        )

    audio = np.frombuffer(raw_audio, dtype="<i2")
    audio = audio.reshape(-1, channels)
    audio = audio.astype(np.float32) / 32768.0
    return sample_rate, audio

def get_wav_from_midi_and_get_as_numpy(
    midi_file_path: Path, sample_rate: int = 44_100
) -> np.ndarray:
    # Use a context manager to ensure automatic cleanup
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td) / "my_wav.wav"
        x = get_wav_from_midi_and_save_to_path(midi_file_path, td_path, sample_rate)
        return read_wav(x)[1]

def get_relative_residual_frame_report(
    reference: np.ndarray,
    candidate: np.ndarray,
    sample_rate: int,
    frame_ms: float = 20.0,
    hop_ms: float = 10.0,
    threshold_db: float = -50.0,
    silence_dbfs: float = -80.0,
) -> dict[str, Any]:
    """
    This is supposed to emulate a kind of 'null test' that we might do in a DAW.
    If you have two nearly identical waveforms, you can invert the polarity of one
    and sum them, then you should get silence.

    We can do the same thing with numpy arrays by subtracting them, but this assumes
    sample-exact equality, which is unrealistic. We care about the perceptual
    similarity, not exact sample-wise equality. So we mimic audio workstation like
    behavior by sliding a frame over the waveforms and get the loudness of the
    residual.

    A frame is considered to have 'failed' if the residual is louder than threshold_db.

    Relative db guidelines:
        -120 db: identical
        -80 db: almost identical
        -60 db: close
        -40 db: after this point we can hear audible differences
    """
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)

    if reference.ndim == 1:
        reference = reference[:, None]
    if candidate.ndim == 1:
        candidate = candidate[:, None]

    length = min(len(reference), len(candidate))
    reference = reference[:length]
    candidate = candidate[:length]
    residual = reference - candidate

    frame_size = max(1, round(sample_rate * frame_ms / 1_000))
    hop_size = max(1, round(sample_rate * hop_ms / 1_000))
    starts = np.arange(0, length - frame_size + 1, hop_size)

    signal_rms = np.array([
        np.sqrt(np.mean(reference[start:start + frame_size] ** 2))
        for start in starts
    ])
    residual_rms = np.array([
        np.sqrt(np.mean(residual[start:start + frame_size] ** 2))
        for start in starts
    ])

    signal_dbfs = 20.0 * np.log10(np.maximum(signal_rms, 1e-15))
    relative_db = 20.0 * np.log10(
        np.maximum(residual_rms, 1e-15)
        / np.maximum(signal_rms, 1e-15)
    )

    active = signal_dbfs > silence_dbfs
    failing = active & (relative_db > threshold_db)
    failing_frames = relative_db[active]
    p99_db = np.percentile(failing_frames, 99)
    p97_db = np.percentile(failing_frames, 97)
    p95_db = np.percentile(failing_frames, 95)
    return {
        "active_frames": int(np.sum(active)),
        "failing_frames": int(np.sum(failing)),
        "failing_fraction": (
            float(np.sum(failing) / np.sum(active))
            if np.any(active)
            else 0.0
        ),
        "maximum_relative_db": (
            float(np.max(relative_db[active]))
            if np.any(active)
            else float("-inf")
        ),
        "p99_relative_db": p99_db,
        "p97_relative_db": p97_db,
        "p95_relative_db": p95_db,
    }