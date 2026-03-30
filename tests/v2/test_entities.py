import pytest

from anticipation.v2.config import AnticipationV2Settings, Vocab
from tests.util.entities import get_note_instrument_token


def test_get_note_instrument_token(
    local_midi_settings_ar_only: AnticipationV2Settings,
) -> None:
    # 1st program code
    assert 1100 == get_note_instrument_token(0, 0, settings=local_midi_settings_ar_only)
    assert 1227 == get_note_instrument_token(
        0, 127, settings=local_midi_settings_ar_only
    )

    # 2nd program code
    assert 1228 == get_note_instrument_token(1, 0, settings=local_midi_settings_ar_only)
    assert 1355 == get_note_instrument_token(
        1, 127, settings=local_midi_settings_ar_only
    )

    # 129th program code, drums
    assert 17484 == get_note_instrument_token(
        128, 0, settings=local_midi_settings_ar_only
    )
    assert 17611 == get_note_instrument_token(
        128, 127, settings=local_midi_settings_ar_only
    )

    with pytest.raises(ValueError):
        get_note_instrument_token(-1, -1, local_midi_settings_ar_only)

    # check for some off-by-1-errors
    with pytest.raises(ValueError):
        get_note_instrument_token(129, 0, local_midi_settings_ar_only)

    with pytest.raises(ValueError):
        get_note_instrument_token(0, 128, local_midi_settings_ar_only)


def test_get_tick_offset() -> None:
    default_v2 = AnticipationV2Settings(vocab=Vocab())
    assert 27511 == get_note_instrument_token(128, 127, settings=default_v2)
