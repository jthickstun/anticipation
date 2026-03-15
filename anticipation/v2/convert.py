from pathlib import Path
from operator import itemgetter
from collections import defaultdict

import mido
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


def midi_to_compound(
    midifile: Path, settings: AnticipationV2Settings, pitch_transpose: int = 0
) -> list[int]:
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

        if not track.is_drum and pitch_transpose != 0:
            # only shift the pitch if the instrument is not drums
            # https://yikai-liao.github.io/symusic/api_reference/track.html#modification-methods
            try:
                track.shift_pitch(offset=pitch_transpose, inplace=True)
            except ValueError as e:
                # ValueError: Overflow while adding (x) and (offset)
                raise SymusicRuntimeError(str(e))
            except TypeError as e:
                # throws a type error if abs(offset) > 127
                raise SymusicRuntimeError(
                    f"pitch_transpose is too large, got value: {pitch_transpose}. "
                    f"There are only 128 valid MIDI note values. Original exception from symusic: "
                    + str(e)
                )

        if settings.do_clip_overlapping_durations_in_midi_conversion:
            track.notes.sort(key=lambda _note: (_note.pitch, _note.time), inplace=True)

        for note in track.notes:
            on_set_time_in_ticks = round(time_resolution * note.time)
            duration_time_in_ticks = round(time_resolution * note.duration)
            pitch = int(note.pitch)
            velocity = int(note.velocity)
            # (abs_time, onset, duration, pitch, instrument, velocity)
            # in mido, tracks are 'merged' by converting each event to
            # absolute time in ticks, and then sorting by that absolute value
            to_add = [
                # need to sort by unquantized time for parity with v1
                note.time,
                # --- these are properties we keep for tokenization ---
                on_set_time_in_ticks,
                duration_time_in_ticks,
                pitch,
                instr,
                velocity,
            ]
            if not compounds or (to_add != compounds[-1]):
                # I noticed that there are some MIDI files that, for some reason,
                # have copies of the SAME event with all the same information...
                # filter those out whenever possible
                compounds.append(to_add)

                # also ensure that the same note on the same instrument cannot sustain
                # if another note plays. This results in weird overlaps in duration
                if (
                    settings.do_clip_overlapping_durations_in_midi_conversion
                    and len(compounds) >= 2
                ):
                    prev_c = compounds[-2]
                    curr_c = compounds[-1]
                    if prev_c[1] + prev_c[2] > curr_c[1] and (
                        # ensure pitch is the same
                        prev_c[3] == curr_c[3]
                        and
                        # ensure instrument is same, it should be - but just check
                        prev_c[4] == curr_c[4]
                    ):
                        prev_c[2] = curr_c[1] - prev_c[1]

    # mimic mido's sort behavior
    # get the 0th element from the compound, could be faster than a lambda?
    compounds.sort(key=itemgetter(0))

    # remove the absolute time, just return the quantized one
    tokens = [x for b in compounds for x in b[1:]]
    return tokens


def compound_to_midi(
    tokens: list[Token], settings: AnticipationV2Settings
) -> mido.MidiFile:
    assert len(tokens) % 5 == 0

    mid = mido.MidiFile()
    mid.ticks_per_beat = settings.time_resolution // 2  # 2 beats/second at quarter=120

    it = iter(tokens)
    time_index = defaultdict(list)
    for _, (time_in_ticks, duration, note, instrument, velocity) in enumerate(
        zip(it, it, it, it, it)
    ):
        time_index[(time_in_ticks, 0)].append((note, instrument, velocity))  # 0 = onset
        time_index[(time_in_ticks + duration, 1)].append(
            (note, instrument, velocity)
        )  # 1 = offset

    track_idx = {}  # maps instrument to (track number, current time)
    num_tracks = 0
    for time_in_ticks, event_type in sorted(time_index.keys()):
        for note, instrument, velocity in time_index[(time_in_ticks, event_type)]:
            if event_type == 0:  # onset
                try:
                    track, previous_time, idx = track_idx[instrument]
                except KeyError:
                    idx = num_tracks
                    previous_time = 0
                    track = mido.MidiTrack()
                    mid.tracks.append(track)
                    if instrument == 128:  # drums always go on channel 9
                        idx = 9
                        message = mido.Message("program_change", channel=idx, program=0)
                    else:
                        message = mido.Message(
                            "program_change", channel=idx, program=instrument
                        )
                    track.append(message)
                    num_tracks += 1
                    if num_tracks == 9:
                        num_tracks += 1  # skip the drums track

                # channel idx
                assert idx <= 16
                track.append(
                    mido.Message(
                        "note_on",
                        note=note,
                        channel=idx,
                        velocity=velocity,
                        time=time_in_ticks - previous_time,
                    )
                )
                track_idx[instrument] = (track, time_in_ticks, idx)
            else:  # offset
                try:
                    track, previous_time, idx = track_idx[instrument]
                except KeyError:
                    # shouldn't happen because we should have a corresponding onset
                    if settings.debug:
                        print("IGNORING bad offset")
                    continue
                track.append(
                    mido.Message(
                        "note_off",
                        note=note,
                        channel=idx,
                        time=time_in_ticks - previous_time,
                    )
                )
                track_idx[instrument] = (track, time_in_ticks, idx)
    return mid
