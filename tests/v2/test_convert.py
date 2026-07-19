from collections import defaultdict
from typing import Optional, Union
import tempfile
from pathlib import Path

import numpy as np
import pytest

import anticipation.config as v1_config
import anticipation.convert as v1_convert
from anticipation.v2.config import AnticipationV2Settings, Vocab
import anticipation.v2.convert as v2_convert

from tests.util.midi import get_trimmed_midi
from tests.util.synth import (
    get_wav_from_midi_and_get_as_numpy,
    get_wav_from_midi_and_save_to_path,
    get_relative_residual_frame_report,
)


@pytest.fixture()
def default_v2_settings() -> AnticipationV2Settings:
    return AnticipationV2Settings(
        vocab=Vocab(),
        debug=True,
    )


def get_musical_differences_in_compounds(
    compound_a: list[int],
    compound_b: list[int],
    timing_tolerance_in_ticks: int = 0,
) -> list[str]:
    """
    Multiset equality up to k ticks in timing and duration. The tolerance is useful
    because mido and symusic do a tick to second conversion and the floating point errors
    can cause drift up to a small number of ticks.
    """
    a_grouped: list[list[int]] = [
        compound_a[i : i + 5] for i in range(0, len(compound_a), 5)
    ]
    b_grouped: list[list[int]] = [
        compound_b[i : i + 5] for i in range(0, len(compound_b), 5)
    ]
    left_groups: dict[tuple[int, int, int], list[list[int]]] = defaultdict(list)
    right_groups: dict[tuple[int, int, int], list[list[int]]] = defaultdict(list)

    for note in a_grouped:
        onset, duration, pitch, instrument, velocity = note
        left_groups[pitch, instrument, velocity].append(note)
    for note in b_grouped:
        onset, duration, pitch, instrument, velocity = note
        right_groups[pitch, instrument, velocity].append(note)

    errors: list[str] = []
    for key in left_groups.keys() | right_groups.keys():
        left_notes = sorted(left_groups.get(key, ()), key=lambda _n: (_n[0], _n[1]))
        right_notes = sorted(right_groups.get(key, ()), key=lambda _n: (_n[0], _n[1]))
        for left_note, right_note in zip(left_notes, right_notes):
            onset_difference = abs(left_note[0] - right_note[0])
            duration_difference = abs(left_note[1] - right_note[1])
            if (
                onset_difference > timing_tolerance_in_ticks
                or duration_difference > timing_tolerance_in_ticks
            ):
                errors.append(
                    f"{key}: {list(left_note)} does not match {list(right_note)}; "
                    f"onset difference={onset_difference}, "
                    f"duration difference={duration_difference}"
                )
    return errors


def get_wav_from_compound(
    compound: list[int], sample_rate: int = 44_000, save_to: Optional[Path] = None
) -> Union[Path, np.ndarray]:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        midi = v1_convert.compound_to_midi(compound)

        # save the midi files
        midi_path = td_path / "compound.mid"
        midi.save(midi_path)
        if save_to is not None:
            return get_wav_from_midi_and_save_to_path(
                midi_path, save_wav_output_to=save_to, sample_rate=sample_rate
            )
        else:
            return get_wav_from_midi_and_get_as_numpy(
                midi_path, sample_rate=sample_rate
            )


def test_v2_midi_to_compound_simple(
    c_major_midi_path: Path,
    default_v2_settings: AnticipationV2Settings,
) -> None:
    v2_compound, num_times_instrument_converted, num_times_clipped = (
        v2_convert.midi_to_compound(c_major_midi_path, default_v2_settings)
    )
    assert num_times_instrument_converted == 0
    assert num_times_clipped == 0
    v1_compound = v1_convert.midi_to_compound(
        str(c_major_midi_path.absolute()),
        debug=default_v2_settings.debug,
        time_resolution=v1_config.TIME_RESOLUTION,
    )
    errors = get_musical_differences_in_compounds(
        v1_compound,
        v2_compound,
        timing_tolerance_in_ticks=1,
    )
    assert len(errors) == 0


def test_v2_midi_to_compound_lakh_0(
    lmd_0_example_1_midi_path: Path,
    default_v2_settings: AnticipationV2Settings,
) -> None:
    v2_compound, num_times_instrument_converted, num_times_clipped = (
        v2_convert.midi_to_compound(lmd_0_example_1_midi_path, default_v2_settings)
    )
    assert num_times_instrument_converted == 0
    assert num_times_clipped == 0
    v1_compound = v1_convert.midi_to_compound(
        str(lmd_0_example_1_midi_path.absolute()),
        debug=default_v2_settings.debug,
        time_resolution=v1_config.TIME_RESOLUTION,
    )
    errors = get_musical_differences_in_compounds(
        v1_compound,
        v2_compound,
        timing_tolerance_in_ticks=1,
    )
    assert len(errors) == 0


def test_v2_midi_to_compound_lakh_1(
    default_v2_settings: AnticipationV2Settings,
    lmd_0_example_3_midi_path: Path,
) -> None:
    v2_compound, num_times_instrument_converted, num_times_clipped = (
        v2_convert.midi_to_compound(lmd_0_example_3_midi_path, default_v2_settings)
    )
    assert num_times_instrument_converted == 0
    assert num_times_clipped == 0
    v1_compound = v1_convert.midi_to_compound(
        str(lmd_0_example_3_midi_path.absolute()),
        debug=default_v2_settings.debug,
        time_resolution=default_v2_settings.time_resolution,
    )
    errors = get_musical_differences_in_compounds(
        v1_compound,
        v2_compound,
        timing_tolerance_in_ticks=1,
    )
    assert len(errors) == 0


def test_v2_midi_to_compound_lakh_2_snippet(
    default_v2_settings: AnticipationV2Settings,
    lmd_0_example_2_midi_path: Path,
) -> None:
    # I manually identified this snippet to be problematic, and then used this test to
    # debug the problems - is now fixed
    with tempfile.TemporaryDirectory() as td:
        trimmed_midi_path = Path(td) / "lmd_0_example_2_trimmed.mid"
        get_trimmed_midi(lmd_0_example_2_midi_path, trimmed_midi_path, 120.425, 154.0)

        v2_compound, num_times_instrument_converted, num_times_clipped = (
            v2_convert.midi_to_compound(trimmed_midi_path, default_v2_settings)
        )
        assert num_times_instrument_converted == 0
        assert num_times_clipped == 0
        v1_compound = v1_convert.midi_to_compound(
            str(trimmed_midi_path.absolute()),
            debug=default_v2_settings.debug,
            time_resolution=v1_config.TIME_RESOLUTION,
        )
    errors = get_musical_differences_in_compounds(
        v1_compound,
        v2_compound,
        timing_tolerance_in_ticks=1,
    )
    assert len(errors) == 0


def test_v2_midi_to_compound_lakh_2(
    lmd_0_example_2_midi_path: Path,
) -> None:
    settings_different_resolution = AnticipationV2Settings.create(
        do_clip_overlapping_durations_in_midi_conversion=False,
        time_resolution=100,
        tick_token_every_n_ticks=0,
        debug=True,
    )
    v2_compound, num_times_instrument_converted, num_times_clipped = (
        v2_convert.midi_to_compound(
            lmd_0_example_2_midi_path, settings_different_resolution
        )
    )
    assert num_times_instrument_converted == 0
    assert num_times_clipped == 0
    v1_compound = v1_convert.midi_to_compound(
        str(lmd_0_example_2_midi_path.absolute()),
        debug=settings_different_resolution.debug,
        time_resolution=settings_different_resolution.time_resolution,
    )
    errors = get_musical_differences_in_compounds(
        v1_compound,
        v2_compound,
        timing_tolerance_in_ticks=1,
    )
    assert len(errors) == 0


def test_v2_midi_to_compound_lakh_with_clipped_note_events(
    lmd_0_example_6_midi_path: Path,
) -> None:
    settings_with_event_clip = AnticipationV2Settings.create(
        do_clip_overlapping_durations_in_midi_conversion=True,
        debug=True,
    )
    v2_compound, num_times_instrument_converted, num_times_clipped = (
        v2_convert.midi_to_compound(lmd_0_example_6_midi_path, settings_with_event_clip)
    )
    assert num_times_instrument_converted == 0
    assert num_times_clipped == 278
    v1_compound = v1_convert.midi_to_compound(
        str(lmd_0_example_6_midi_path.absolute()),
        debug=settings_with_event_clip.debug,
        time_resolution=settings_with_event_clip.time_resolution,
    )
    errors = get_musical_differences_in_compounds(
        v1_compound,
        v2_compound,
        timing_tolerance_in_ticks=1,
    )

    # the idea here is that if we clip overlapping but not audible MIDI,
    # it shouldn't influence the perception of the music
    assert len(errors) == 0
    v2_audio = get_wav_from_compound(v2_compound, sample_rate=22_000)
    v1_audio = get_wav_from_compound(v1_compound, sample_rate=22_000)
    null_test_result = get_relative_residual_frame_report(
        v1_audio,
        v2_audio,
        sample_rate=22_000,
    )
    assert null_test_result["p97_relative_db"] < -60
