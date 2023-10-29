"""
The vocabulary used for multimodal encoding.
"""

from anticipation.config import *

SEPARATOR = 0

# the audio block
AUDIO_OFFSET = 1
RESIDUAL_PAD = AUDIO_OFFSET + 0
SCALE_PAD = AUDIO_OFFSET + 1
R0_OFFSET = AUDIO_OFFSET + 2
R1_OFFSET = R0_OFFSET + 1024
R2_OFFSET = R1_OFFSET + 1024
R3_OFFSET = R2_OFFSET + 1024
SCALE_OFFSET = R3_OFFSET + 1024

# the midi block
MIDI_OFFSET = SCALE_OFFSET + 100
TIME_OFFSET = MIDI_OFFSET
PITCH_OFFSET = TIME_OFFSET + 151 # match the resolution of the audio
REST = PITCH_OFFSET + 128
INSTRUMENT_OFFSET = PITCH_OFFSET + 128 + 1
DURATION_OFFSET = INSTRUMENT_OFFSET + 129

# the control block 
CONTROL_OFFSET = DURATION_OFFSET + 1500
AUDIOGEN = CONTROL_OFFSET + 0
MIDIGEN = CONTROL_OFFSET + 1
TRANSCRIBE = CONTROL_OFFSET + 2
SYNTHESIZE = CONTROL_OFFSET + 3
CONTROL_PAD = CONTROL_OFFSET + 4
VOCAB_SIZE = CONTROL_OFFSET + 5

vocab = {
    'separator' : SEPARATOR,
    'residual_pad' : RESIDUAL_PAD,
    'scale_pad' : SCALE_PAD,
    'residuals' : 4,
    'residual_offset' : [R0_OFFSET, R1_OFFSET, R2_OFFSET, R3_OFFSET],
    'scale_offset' : SCALE_OFFSET,

    'time_offset' : TIME_OFFSET,
    'pitch_offset' : PITCH_OFFSET,
    'rest' : REST,
    'instrument_offset' : INSTRUMENT_OFFSET,
    'duration_offset' : DURATION_OFFSET,

    'audiogen' : AUDIOGEN,
    'midigen' : MIDIGEN,
    'transcribe' : TRANSCRIBE,
    'synthesize' : SYNTHESIZE,
    'control_pad' : CONTROL_PAD,

    'size' : VOCAB_SIZE
}

if __name__ == '__main__':
    print('Multimodal Training Sequence Format:')
    print('Sequence Separator :', SEPARATOR)
    print('Audio Block:', AUDIO_OFFSET)
    print('  -> residual pad :', RESIDUAL_PAD)
    print('  -> scale pad : ', SCALE_PAD)
    print('  -> r0 offset :', R0_OFFSET)
    print('  -> r1 offset :', R1_OFFSET)
    print('  -> r2 offset: ', R2_OFFSET)
    print('  -> r3 offset: ', R3_OFFSET)
    print('  -> scale offset: ', SCALE_OFFSET)
    print('Midi Block:', MIDI_OFFSET)
    print('  -> interarrival time offset :', TIME_OFFSET)
    print('  -> pitch offset :', PITCH_OFFSET)
    print('  -> rest :', REST)
    print('  -> instrument offset :', INSTRUMENT_OFFSET)
    print('  -> duration offset :', DURATION_OFFSET)
    print('Control Block:', CONTROL_OFFSET)
    print('  -> audio generation flag: ', AUDIOGEN)
    print('  -> midi generation flag: ', MIDIGEN)
    print('  -> transcription flag: ', TRANSCRIBE)
    print('  -> synthesis flag: ', SYNTHESIZE)
    print('Multimodal Vocabulary Size: ', VOCAB_SIZE)
