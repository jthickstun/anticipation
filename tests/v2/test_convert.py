import tempfile
from pathlib import Path

import numpy as np
from symusic import Score

import pytest

import anticipation.config as v1_config
from anticipation.v2.config import AnticipationV2Settings, Vocab
import anticipation.v2.convert as v2_convert
import anticipation.convert as v1_convert

from tests.util.synth import (
    get_wav_from_midi_as_array,
)
from tests.util.midi import get_trimmed_midi


@pytest.fixture()
def default_v2_settings() -> AnticipationV2Settings:
    return AnticipationV2Settings(
        vocab=Vocab(),
        debug=True,
    )


def _group_compound_and_drop_unclosed(compound: list[int]) -> list[list[int]]:
    grouping = []
    for c in range(0, len(compound), 5):
        t, d, n, inst, v = compound[c : c + 5]
        if d == -1:
            # unknown duration / unclosed
            continue
        grouping.append(compound[c : c + 5])

    return grouping


def is_compound_musically_equivalent(
    compound_a: list[int], compound_b: list[int], timing_tolerance_in_ticks: int = 0
) -> bool:
    if timing_tolerance_in_ticks == 0:
        # check for exact equality

        # using v1 convert for BOTH on purpose, for consistency
        # v1 convert must be sufficient for the conversion even if the compound was
        # created by v2 converter functions
        a_midi = v1_convert.compound_to_midi(compound_a)
        b_midi = v1_convert.compound_to_midi(compound_b)
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)

            # save the midi files
            a_midi_path = td_path / "a.mid"
            b_midi_path = td_path / "b.mid"
            a_midi.save(a_midi_path)
            b_midi.save(b_midi_path)

            # use symusic to load the files and then convert to the
            # piano roll representation, which is a numpy array
            pr_a: np.ndarray = Score(a_midi_path).pianoroll()
            pr_b: np.ndarray = Score(b_midi_path).pianoroll()

            # return whether the reconstructed piano rolls are equal
            return np.array_equal(pr_a, pr_b)
    else:
        if len(compound_a) != len(compound_b):
            return False

        a_grouped = [compound_a[i : i + 5] for i in range(0, len(compound_a), 5)]
        b_grouped = [compound_b[i : i + 5] for i in range(0, len(compound_b), 5)]

        for a_comp, b_comp in zip(a_grouped, b_grouped):
            a_onset, a_duration, a_note, a_instrument, a_velocity = a_comp
            b_onset, b_duration, b_note, b_instrument, b_velocity = b_comp

            instruments_equal = a_instrument == b_instrument

            if not instruments_equal:
                return False

            notes_equal = a_note == b_note
            if not notes_equal:
                return False

            onset_approx_equal = abs(a_onset - b_onset) <= timing_tolerance_in_ticks
            if not onset_approx_equal:
                return False

            duration_approx_equal = (
                abs(a_duration - b_duration) <= timing_tolerance_in_ticks
            )
            if not duration_approx_equal:
                return False

        return True


def get_wav_from_compound(compound: list[int]) -> np.ndarray:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        midi = v1_convert.compound_to_midi(compound)

        # save the midi files
        midi_path = td_path / "compound.mid"
        midi.save(midi_path)
        return get_wav_from_midi_as_array(midi_path)


def test_v2_midi_to_compound_simple(
    c_major_midi_path: Path,
    default_v2_settings: AnticipationV2Settings,
) -> None:
    v2_compound = v2_convert.midi_to_compound(c_major_midi_path, default_v2_settings)
    v1_compound = v1_convert.midi_to_compound(
        str(c_major_midi_path.absolute()),
        debug=default_v2_settings.debug,
        time_resolution=v1_config.TIME_RESOLUTION,
    )
    assert is_compound_musically_equivalent(v1_compound, v2_compound)


def test_v2_midi_to_compound_lakh_0(
    lmd_0_example_1_midi_path: Path,
    default_v2_settings: AnticipationV2Settings,
) -> None:
    v2_compound = v2_convert.midi_to_compound(
        lmd_0_example_1_midi_path, default_v2_settings
    )
    v1_compound = v1_convert.midi_to_compound(
        str(lmd_0_example_1_midi_path.absolute()),
        debug=default_v2_settings.debug,
        time_resolution=v1_config.TIME_RESOLUTION,
    )

    assert is_compound_musically_equivalent(
        v1_compound, v2_compound, timing_tolerance_in_ticks=1
    )


def test_v2_midi_to_compound_lakh_1(
    default_v2_settings: AnticipationV2Settings,
    lmd_0_example_3_midi_path: Path,
) -> None:
    v2_compound = v2_convert.midi_to_compound(
        lmd_0_example_3_midi_path, default_v2_settings
    )
    v1_compound = v1_convert.midi_to_compound(
        str(lmd_0_example_3_midi_path.absolute()),
        debug=default_v2_settings.debug,
        time_resolution=v1_config.TIME_RESOLUTION,
    )
    assert is_compound_musically_equivalent(v1_compound, v2_compound)


def test_v2_midi_to_compound_lakh_2_snippet(
    default_v2_settings: AnticipationV2Settings,
    lmd_0_example_2_midi_path: Path,
) -> None:
    # I manually identified this snippet to be problematic, and then used this test to
    # debug the problems - is now fixed
    with tempfile.TemporaryDirectory() as td:
        trimmed_midi_path = Path(td) / "lmd_0_example_2_trimmed.mid"
        get_trimmed_midi(lmd_0_example_2_midi_path, trimmed_midi_path, 120.425, 154.0)

        v2_compound = v2_convert.midi_to_compound(
            trimmed_midi_path, default_v2_settings
        )
        v1_compound = v1_convert.midi_to_compound(
            str(trimmed_midi_path.absolute()),
            debug=default_v2_settings.debug,
            time_resolution=v1_config.TIME_RESOLUTION,
        )

    assert is_compound_musically_equivalent(
        v1_compound, v2_compound, timing_tolerance_in_ticks=1
    )


def test_v2_midi_to_compound_lakh_2(
    default_v2_settings: AnticipationV2Settings,
    lmd_0_example_2_midi_path: Path,
) -> None:
    v2_compound = v2_convert.midi_to_compound(
        lmd_0_example_2_midi_path, default_v2_settings
    )
    v1_compound = v1_convert.midi_to_compound(
        str(lmd_0_example_2_midi_path.absolute()),
        debug=default_v2_settings.debug,
        time_resolution=v1_config.TIME_RESOLUTION,
    )
    assert is_compound_musically_equivalent(
        v1_compound, v2_compound, timing_tolerance_in_ticks=1
    )
