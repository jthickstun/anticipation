from pathlib import Path
from operator import itemgetter

from symusic import Score, TimeUnit

from anticipation.v2.config import AnticipationV2Settings
from anticipation.v2.types import Token


class SymusicRuntimeError(Exception):
    pass


def compound_to_events(
    compound: list[int], settings: AnticipationV2Settings
) -> tuple[list[Token], int]:
    assert len(compound) % 5 == 0
    tokens = compound.copy()

    # remove velocities
    del tokens[4::5]

    # combine (note, instrument)
    # in v1, a note value of -1 was used as a sentinel
    # and in this `compound_to_events` function, if -1 appeared, then it
    # was set to SEPARATOR.
    # in v2, we now enforce that this -1 never appear in the note or instrument.
    # ideally this function does not add any punctuation-like tokens (PAD, SEP,
    # AUTOREGRESS, etc.)
    assert all(0 <= tok < 2**7 for tok in tokens[2::4])
    assert all(0 <= tok < 129 for tok in tokens[3::4])
    tokens[2::4] = [
        settings.max_midi_pitch * instr + note
        for note, instr in zip(tokens[2::4], tokens[3::4])
    ]
    tokens[2::4] = [settings.vocab.NOTE_OFFSET + tok for tok in tokens[2::4]]
    del tokens[3::4]

    # max duration cutoff
    max_duration = settings.time_resolution * settings.max_note_duration_in_seconds
    num_note_truncations = sum([1 for tok in tokens[1::3] if tok >= max_duration])

    # set unknown durations to 250ms
    tokens[1::3] = [
        settings.time_resolution // 4 if tok == -1 else min(tok, max_duration - 1)
        for tok in tokens[1::3]
    ]

    tokens[1::3] = [settings.vocab.DUR_OFFSET + tok for tok in tokens[1::3]]

    assert min(tokens[0::3]) >= 0
    tokens[0::3] = [settings.vocab.TIME_OFFSET + tok for tok in tokens[0::3]]

    assert len(tokens) % 3 == 0

    # tokens are: (time, duration, note x instrument)
    return tokens, num_note_truncations


def midi_to_compound(midifile: Path, settings: AnticipationV2Settings) -> list[int]:
    time_resolution = settings.time_resolution
    try:
        # always have sanitize_data equal to true. Without it, symusic may segfault or crash
        # some MIDI files in Lakh, without any modification, can sometimes trigger reading
        # out of bounds.
        score = Score.from_midi(
            midifile.read_bytes(), ttype=TimeUnit.second, sanitize_data=True
        )
    except RuntimeError as e:
        raise SymusicRuntimeError(str(e))

    compounds = []
    for track_idx, track in enumerate(score.tracks):
        instr = 128 if track.is_drum else track.program
        for note in track.notes:
            on_set_time_in_ticks = round(time_resolution * note.time)
            duration_time_in_ticks = round(time_resolution * note.duration)
            pitch = int(note.pitch)
            velocity = int(note.velocity)
            # (abs_time, onset, duration, pitch, instrument, velocity)
            # in mido, tracks are 'merged' by converting each event to
            # absolute time in ticks, and then sorting by that absolute value
            compounds.append(
                (
                    # need to sort by unquantized time for parity with v1
                    note.time,
                    # --- these are properties we keep for tokenization ---
                    on_set_time_in_ticks,
                    duration_time_in_ticks,
                    pitch,
                    instr,
                    velocity,
                )
            )

    # mimic mido's sort behavior
    # get the 0th element from the compound, could be faster than a lambda?
    compounds.sort(key=itemgetter(0))

    # remove the absolute time, just return the quantized one
    tokens = [x for b in compounds for x in b[1:]]
    return tokens
