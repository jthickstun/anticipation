"""
The vocabulary used for multimodal encoding.
"""

#
# configuaration
#

TIME_RESOLUTION = 150                # 150 bins/second to match Encodec

# Encodec
CODEBOOK_SIZE = 1024
SCALE_QUANTIZATION = 100
RESIDUALS = 4

# MIDI
MAX_PITCH = 128                          # 128 MIDI pitches
MAX_INSTR = 128 + 1                      # 129 MIDI instruments (128 + drums)
MAX_INTERARRIVAL_IN_SECONDS = 1          # maximum interarrival time
MAX_DURATION_IN_SECONDS = 10             # maximum duration of a note

MAX_DURATION = 10*TIME_RESOLUTION        # 10 seconds maximum note duration
MAX_INTERARRIVAL = 1*TIME_RESOLUTION + 1 # 1 second maximum interarrival time

#
# vocabulary
#

SEPARATOR = 0

# the audio block
AUDIO_OFFSET = 1
RESIDUAL_PAD = AUDIO_OFFSET + 0
SCALE_PAD = AUDIO_OFFSET + 1
R0_OFFSET = AUDIO_OFFSET + 2
R1_OFFSET = R0_OFFSET + CODEBOOK_SIZE
R2_OFFSET = R1_OFFSET + CODEBOOK_SIZE
R3_OFFSET = R2_OFFSET + CODEBOOK_SIZE
SCALE_OFFSET = R3_OFFSET + CODEBOOK_SIZE

# the midi block
MIDI_OFFSET = SCALE_OFFSET + SCALE_QUANTIZATION
TIME_OFFSET = MIDI_OFFSET
PITCH_OFFSET = TIME_OFFSET + MAX_INTERARRIVAL
REST = PITCH_OFFSET + MAX_PITCH
INSTRUMENT_OFFSET = PITCH_OFFSET + MAX_PITCH + 1
DURATION_OFFSET = INSTRUMENT_OFFSET + MAX_INSTR

# the control block 
CONTROL_OFFSET = DURATION_OFFSET + MAX_DURATION
AUDIOGEN = CONTROL_OFFSET + 0
MIDIGEN = CONTROL_OFFSET + 1
TRANSCRIBE = CONTROL_OFFSET + 2
SYNTHESIZE = CONTROL_OFFSET + 3
CONTROL_PAD = CONTROL_OFFSET + 4
VOCAB_SIZE = CONTROL_OFFSET + 5

vocab = {
    'config' : {
        'time_resolution' : TIME_RESOLUTION,
        'residuals' : RESIDUALS,
        'codebook_size' : CODEBOOK_SIZE,
        'scale_quantization' : SCALE_QUANTIZATION,
        'max_interarrival' : MAX_INTERARRIVAL,
        'size' : VOCAB_SIZE
    },

    'separator' : SEPARATOR,
    'residual_pad' : RESIDUAL_PAD,
    'scale_pad' : SCALE_PAD,
    'rest' : REST,
    'audiogen' : AUDIOGEN,
    'midigen' : MIDIGEN,
    'transcribe' : TRANSCRIBE,
    'synthesize' : SYNTHESIZE,
    'control_pad' : CONTROL_PAD,

    'residual_offset' : [R0_OFFSET, R1_OFFSET, R2_OFFSET, R3_OFFSET],
    'scale_offset' : SCALE_OFFSET,
    'time_offset' : TIME_OFFSET,
    'pitch_offset' : PITCH_OFFSET,
    'instrument_offset' : INSTRUMENT_OFFSET,
    'duration_offset' : DURATION_OFFSET,
}

if __name__ == '__main__':
    print('Multimodal Vocabulary Configuration:')
    print('  -> Time Resolution:', TIME_RESOLUTION)
    print('  -> Number of Residuals:', RESIDUALS)
    print('  -> Codebook Size:', CODEBOOK_SIZE)
    print('  -> Maximum Interarrival Time:', MAX_INTERARRIVAL)
    print('  -> Vocabulary Size:', VOCAB_SIZE)
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
