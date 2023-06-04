"""
Utilities for converting to and from Midi data and encoded/tokenized data.
"""

from collections import defaultdict

import mido

from anticipation.config import *
from anticipation.vocab import *
from anticipation.ops import unpad


def midi_to_interarrival(midifile, debug=False, stats=False):
    midi = mido.MidiFile(midifile)

    tokens = []
    dt = 0

    instruments = defaultdict(int) # default to code 0 = piano
    tempo = 500000 # default tempo: 500000 microseconds per beat
    truncations = 0
    for message in midi:
        dt += message.time

        # sanity check: negative time?
        if message.time < 0:
            raise ValueError

        if message.type == 'program_change':
            instruments[message.channel] = message.program
        elif message.type in ['note_on', 'note_off']:
            delta_ticks = min(round(TIME_RESOLUTION*dt), MAX_INTERARRIVAL-1)
            if delta_ticks != round(TIME_RESOLUTION*dt):
                truncations += 1

            if delta_ticks > 0: # if time elapsed since last token
                tokens.append(MIDI_TIME_OFFSET + delta_ticks) # add a time step event

            # special case: channel 9 is drums!
            inst = 128 if message.channel == 9 else instruments[message.channel]
            offset = MIDI_START_OFFSET if message.type == 'note_on' and message.velocity > 0 else MIDI_END_OFFSET
            tokens.append(offset + (2**7)*inst + message.note)
            dt = 0
        elif message.type == 'set_tempo':
            tempo = message.tempo
        elif message.type == 'time_signature':
            pass # we use real time
        elif message.type in ['aftertouch', 'polytouch', 'pitchwheel', 'sequencer_specific']:
            pass # we don't attempt to model these
        elif message.type == 'control_change':
            pass # this includes pedal and per-track volume: ignore for now
        elif message.type in ['track_name', 'text', 'end_of_track', 'lyrics', 'key_signature',
                              'copyright', 'marker', 'instrument_name', 'cue_marker',
                              'device_name', 'sequence_number']:
            pass # possibly useful metadata but ignore for now
        elif message.type == 'channel_prefix':
            pass # relatively common, but can we ignore this?
        elif message.type in ['midi_port', 'smpte_offset', 'sysex']:
            pass # I have no idea what this is
        else:
            if debug:
                print('UNHANDLED MESSAGE', message.type, message)

    if stats:
        return tokens, truncations

    return tokens


def interarrival_to_midi(tokens, debug=False):
    mid = mido.MidiFile()
    mid.ticks_per_beat = TIME_RESOLUTION // 2 # 2 beats/second at quarter=120

    track_idx = {} # maps instrument to (track number, current time)
    time_in_ticks = 0
    num_tracks = 0
    for token in tokens:
        if token == MIDI_SEPARATOR:
            continue

        if token < MIDI_START_OFFSET:
            time_in_ticks += token - MIDI_TIME_OFFSET
        elif token < MIDI_END_OFFSET:
            token -= MIDI_START_OFFSET
            instrument = token // 2**7
            pitch = token - (2**7)*instrument

            try:
                track, previous_time, idx = track_idx[instrument]
            except KeyError:
                idx = num_tracks
                previous_time = 0
                track = mido.MidiTrack()
                mid.tracks.append(track)
                if instrument == 128: # drums always go on channel 9
                    idx = 9
                    message = mido.Message('program_change', channel=idx, program=0)
                else:
                    message = mido.Message('program_change', channel=idx, program=instrument)
                track.append(message)
                num_tracks += 1
                if num_tracks == 9:
                    num_tracks += 1 # skip the drums track

            track.append(mido.Message('note_on', note=pitch, channel=idx, velocity=96, time=time_in_ticks-previous_time))
            track_idx[instrument] = (track, time_in_ticks, idx)
        else:
            token -= MIDI_END_OFFSET
            instrument = token // 2**7
            pitch = token - (2**7)*instrument

            try:
                track, previous_time, idx = track_idx[instrument]
            except KeyError:
                # shouldn't happen because we should have a corresponding onset
                if debug:
                    print('IGNORING bad offset')

                continue

            track.append(mido.Message('note_off', note=pitch, channel=idx, time=time_in_ticks-previous_time))
            track_idx[instrument] = (track, time_in_ticks, idx)

    return mid


def midi_to_compound(midifile, debug=False):
    if type(midifile) == str:
        midi = mido.MidiFile(midifile)
    else:
        midi = midifile

    tokens = []
    note_idx = 0
    open_notes = defaultdict(list)

    time = 0
    instruments = defaultdict(int) # default to code 0 = piano
    tempo = 500000 # default tempo: 500000 microseconds per beat
    for message in midi:
        time += message.time

        # sanity check: negative time?
        if message.time < 0:
            raise ValueError

        if message.type == 'program_change':
            instruments[message.channel] = message.program
        elif message.type in ['note_on', 'note_off']:
            # special case: channel 9 is drums!
            instr = 128 if message.channel == 9 else instruments[message.channel]

            if message.type == 'note_on' and message.velocity > 0: # onset
                # time quantization
                time_in_ticks = round(TIME_RESOLUTION*time)

                # Our compound word is: (time, duration, note, instr, velocity)
                tokens.append(time_in_ticks) # 5ms resolution
                tokens.append(-1) # placeholder (we'll fill this in later)
                tokens.append(message.note)
                tokens.append(instr)
                tokens.append(message.velocity)

                open_notes[(instr,message.note,message.channel)].append((note_idx, time))
                note_idx += 1
            else: # offset
                try:
                    open_idx, onset_time = open_notes[(instr,message.note,message.channel)].pop(0)
                except IndexError:
                    if debug:
                        print('WARNING: ignoring bad offset')
                else:
                    duration_ticks = round(TIME_RESOLUTION*(time-onset_time))
                    tokens[5*open_idx + 1] = duration_ticks
                    #del open_notes[(instr,message.note,message.channel)]
        elif message.type == 'set_tempo':
            tempo = message.tempo
        elif message.type == 'time_signature':
            pass # we use real time
        elif message.type in ['aftertouch', 'polytouch', 'pitchwheel', 'sequencer_specific']:
            pass # we don't attempt to model these
        elif message.type == 'control_change':
            pass # this includes pedal and per-track volume: ignore for now
        elif message.type in ['track_name', 'text', 'end_of_track', 'lyrics', 'key_signature',
                              'copyright', 'marker', 'instrument_name', 'cue_marker',
                              'device_name', 'sequence_number']:
            pass # possibly useful metadata but ignore for now
        elif message.type == 'channel_prefix':
            pass # relatively common, but can we ignore this?
        elif message.type in ['midi_port', 'smpte_offset', 'sysex']:
            pass # I have no idea what this is
        else:
            if debug:
                print('UNHANDLED MESSAGE', message.type, message)

    unclosed_count = 0
    for _,v in open_notes.items():
        unclosed_count += len(v)

    if debug and unclosed_count > 0:
        print(f'WARNING: {unclosed_count} unclosed notes')
        print('  ', midifile)

    return tokens


def compound_to_midi(tokens, debug=False):
    mid = mido.MidiFile()
    mid.ticks_per_beat = TIME_RESOLUTION // 2 # 2 beats/second at quarter=120

    it = iter(tokens)
    time_index = defaultdict(list)
    for _, (time_in_ticks,duration,note,instrument,velocity) in enumerate(zip(it,it,it,it,it)):
        time_index[(time_in_ticks,0)].append((note, instrument, velocity)) # 0 = onset
        time_index[(time_in_ticks+duration,1)].append((note, instrument, velocity)) # 1 = offset

    track_idx = {} # maps instrument to (track number, current time)
    num_tracks = 0
    for time_in_ticks, event_type in sorted(time_index.keys()):
        for (note, instrument, velocity) in time_index[(time_in_ticks, event_type)]:
            if event_type == 0: # onset
                try:
                    track, previous_time, idx = track_idx[instrument]
                except KeyError:
                    idx = num_tracks
                    previous_time = 0
                    track = mido.MidiTrack()
                    mid.tracks.append(track)
                    if instrument == 128: # drums always go on channel 9
                        idx = 9
                        message = mido.Message('program_change', channel=idx, program=0)
                    else:
                        message = mido.Message('program_change', channel=idx, program=instrument)
                    track.append(message)
                    num_tracks += 1
                    if num_tracks == 9:
                        num_tracks += 1 # skip the drums track

                track.append(mido.Message(
                    'note_on', note=note, channel=idx, velocity=velocity,
                    time=time_in_ticks-previous_time))
                track_idx[instrument] = (track, time_in_ticks, idx)
            else: # offset
                try:
                    track, previous_time, idx = track_idx[instrument]
                except KeyError:
                    # shouldn't happen because we should have a corresponding onset
                    if debug:
                        print('IGNORING bad offset')

                    continue

                track.append(mido.Message(
                    'note_off', note=note, channel=idx,
                    time=time_in_ticks-previous_time))
                track_idx[instrument] = (track, time_in_ticks, idx)

    return mid


def compound_to_events(tokens, stats=False):
    assert len(tokens) % 5 == 0
    tokens = tokens.copy()

    # remove velocities
    del tokens[4::5]

    # combine (note, instrument)
    assert all(-1 <= tok < 2**7 for tok in tokens[2::4])
    assert all(-1 <= tok < 129 for tok in tokens[3::4])
    tokens[2::4] = [SEPARATOR if note == -1 else MAX_PITCH*instr + note
                    for note, instr in zip(tokens[2::4],tokens[3::4])]
    tokens[2::4] = [NOTE_OFFSET + tok for tok in tokens[2::4]]
    del tokens[3::4]

    # max duration cutoff and set unknown durations to 250ms
    truncations = sum([1 for tok in tokens[1::3] if tok >= MAX_DUR])
    tokens[1::3] = [TIME_RESOLUTION//4 if tok == -1 else min(tok, MAX_DUR-1)
                    for tok in tokens[1::3]]
    tokens[1::3] = [DUR_OFFSET + tok for tok in tokens[1::3]]

    assert min(tokens[0::3]) >= 0
    tokens[0::3] = [TIME_OFFSET + tok for tok in tokens[0::3]]

    assert len(tokens) % 3 == 0

    if stats:
        return tokens, truncations

    return tokens


def events_to_compound(tokens, debug=False):
    tokens = unpad(tokens)

    # move all tokens to zero-offset for synthesis
    tokens = [tok - CONTROL_OFFSET if tok >= CONTROL_OFFSET and tok != SEPARATOR else tok
              for tok in tokens]

    # remove type offsets
    tokens[0::3] = [tok - TIME_OFFSET if tok != SEPARATOR else tok for tok in tokens[0::3]]
    tokens[1::3] = [tok - DUR_OFFSET if tok != SEPARATOR else tok for tok in tokens[1::3]]
    tokens[2::3] = [tok - NOTE_OFFSET if tok != SEPARATOR else tok for tok in tokens[2::3]]

    offset = 0 # add max time from previous track for synthesis
    track_max = 0 # keep track of max time in track
    for j, (time,dur,note) in enumerate(zip(tokens[0::3],tokens[1::3],tokens[2::3])):
        if note == SEPARATOR:
            offset += track_max
            track_max = 0
            if debug:
                print('Sequence Boundary')
        else:
            track_max = max(track_max, time+dur)
            tokens[3*j] += offset

    # strip sequence separators
    assert len([tok for tok in tokens if tok == SEPARATOR]) % 3 == 0
    tokens = [tok for tok in tokens if tok != SEPARATOR]

    assert len(tokens) % 3 == 0
    out = 5*(len(tokens)//3)*[0]
    out[0::5] = tokens[0::3]
    out[1::5] = tokens[1::3]
    out[2::5] = [tok - (2**7)*(tok//2**7) for tok in tokens[2::3]]
    out[3::5] = [tok//2**7 for tok in tokens[2::3]]
    out[4::5] = (len(tokens)//3)*[72] # default velocity

    assert max(out[1::5]) < MAX_DUR
    assert max(out[2::5]) < MAX_PITCH
    assert max(out[3::5]) < MAX_INSTR
    assert all(tok >= 0 for tok in out)

    return out


def events_to_midi(tokens, debug=False):
    return compound_to_midi(events_to_compound(tokens, debug=debug), debug=debug)

def midi_to_events(midifile, debug=False):
    return compound_to_events(midi_to_compound(midifile, debug=debug))
