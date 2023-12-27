from argparse import ArgumentParser

from anticipation.mmvocab import vocab

def print_tokens(tokens):
    midi_offset = vocab['midi_offset']
    time_offset = vocab['time_offset']
    pitch_offset = vocab['pitch_offset']
    instr_offset = vocab['instrument_offset']
    dur_offset = vocab['duration_offset']

    fps = vocab['config']['audio_fps']
    res = vocab['config']['midi_quantization']

    separator = vocab['separator']
    rest = vocab['rest']
    print('---------------------')
    audio_frame = audio_time = 0
    midi_arrival = midi_time = 0
    for j, (tm, instr, pitch, dur) in enumerate(zip(tokens[0::4],tokens[1::4],tokens[2::4],tokens[3::4])):
        annotation = ''

        if tm == separator:
            assert instr == separator and pitch == separator and dur == separator 
            audio_time = 0
            midi_arrival = 0
            print(j, midi_time, audio_time, '|', 'SEPARATOR')
            continue

        if tm < midi_offset:
            audio_frame += 1
            audio_time = round(float(audio_frame)/fps,3)
            print(j, midi_time, audio_time, '|', 'AUDIO')
            continue

        if instr == rest:
            annotation += ' (REST)'

        tm = tm - time_offset
        instr = instr - instr_offset
        pitch = pitch - pitch_offset
        dur = dur - dur_offset

        midi_arrival += tm
        midi_time = round(float(midi_arrival)/res,3)

        print(j, midi_time, audio_time, '|', tm, dur, instr, pitch, annotation)

if __name__ == '__main__':
    parser = ArgumentParser(description='inspect a MIDI dataset')
    parser.add_argument('filename',
        help='file containing a tokenized MIDI dataset')
    parser.add_argument('index', type=int, default=0,
        help='the item to examine')
    args = parser.parse_args()

    with open(args.filename, 'r') as f:
        for i, line in enumerate(f):
            if i < args.index:
                continue 

            tokens = [int(token) for token in line.split()]
            control = tokens[:4]
            tokens = tokens[4:]
            print('Control tokens:', control)
            print_tokens(tokens)
            break
