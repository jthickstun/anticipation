"""
The vocabulary used for multimodal encoding.
"""

#
# configuaration
#

DELTA = 3                                # seconds of anticipation

# Encodec
FRAMES_PER_SECOND = 150 + 1              # 1 scale frame +  150 audio frames
CODEBOOK_SIZE = 1024
SCALE_QUANTIZATION = 100
RESIDUALS = 4

# MIDI
MAX_PITCH = 128                          # 128 MIDI pitches
MAX_INSTR = 128 + 1                      # 129 MIDI instruments (128 + drums)
MAX_INTERARRIVAL_IN_SECONDS = 1          # maximum interarrival time
MAX_DURATION_IN_SECONDS = 10             # maximum duration of a note

MIDI_QUANTIZATION = 100                  # time bins/second
MAX_DURATION = 10*MIDI_QUANTIZATION      # 10 seconds maximum note duration
MAX_INTERARRIVAL = 1*MIDI_QUANTIZATION   # 1 second maximum interarrival time

#
# vocabulary
#

SEPARATOR = 0
RESIDUAL_PAD = 1

# the control block 
CONTROL_OFFSET = 2
CONTROL_PAD = CONTROL_OFFSET + 0
AUDIOGEN = CONTROL_OFFSET + 1
MIDIGEN = CONTROL_OFFSET + 2
TRANSCRIBE = CONTROL_OFFSET + 3
SYNTHESIZE = CONTROL_OFFSET + 4
CLEANAUDIO = CONTROL_OFFSET + 5
CLEANMIDI = CONTROL_OFFSET + 6
SYNTHAUDIO = CONTROL_OFFSET + 7
TRANSMIDI = CONTROL_OFFSET + 8

# the audio block
AUDIO_OFFSET = CONTROL_OFFSET+9
SCALE_PAD = AUDIO_OFFSET + 0
R0_OFFSET = AUDIO_OFFSET + 1
R1_OFFSET = R0_OFFSET + CODEBOOK_SIZE
R2_OFFSET = R1_OFFSET + CODEBOOK_SIZE
R3_OFFSET = R2_OFFSET + CODEBOOK_SIZE
SCALE_OFFSET = R3_OFFSET + CODEBOOK_SIZE

# the midi block
MIDI_OFFSET = SCALE_OFFSET + SCALE_QUANTIZATION
TIME_OFFSET = MIDI_OFFSET
PITCH_OFFSET = TIME_OFFSET + MAX_INTERARRIVAL + 1
REST = PITCH_OFFSET + MAX_PITCH
INSTRUMENT_OFFSET = PITCH_OFFSET + MAX_PITCH + 1
DURATION_OFFSET = INSTRUMENT_OFFSET + MAX_INSTR

VOCAB_SIZE = DURATION_OFFSET + MAX_DURATION

vocab = {
    'config' : {
        'skew' : False,
        'anticipation' : DELTA,
        'midi_quantization' : MIDI_QUANTIZATION,
        'audio_fps' : FRAMES_PER_SECOND,
        'scale_resolution' : SCALE_QUANTIZATION,
        'residuals' : RESIDUALS,
        'codebook_size' : CODEBOOK_SIZE,
        'max_interarrival' : MAX_INTERARRIVAL,
        'max_duration' : MAX_DURATION,
        'size' : VOCAB_SIZE
    },

    'separator' : SEPARATOR,
    'residual_pad' : RESIDUAL_PAD,
    'scale_pad' : SCALE_PAD,
    'rest' : REST,
    'control_pad' : CONTROL_PAD,

    'task' : {
        'audiogen' : AUDIOGEN,
        'midigen' : MIDIGEN,
        'transcribe' : TRANSCRIBE,
        'synthesize' : SYNTHESIZE
    },

    'content_type' : {
        'clean_audio' : CLEANAUDIO,
        'clean_midi' : CLEANMIDI,
        'synthesized_audio' : SYNTHAUDIO,
        'transcribed_midi' : TRANSMIDI
    },

    'audio_offset' : AUDIO_OFFSET,
    'residual_offset' : [R0_OFFSET, R1_OFFSET, R2_OFFSET, R3_OFFSET],
    'scale_offset' : SCALE_OFFSET,
    'midi_offset' : MIDI_OFFSET,
    'time_offset' : TIME_OFFSET,
    'pitch_offset' : PITCH_OFFSET,
    'instrument_offset' : INSTRUMENT_OFFSET,
    'duration_offset' : DURATION_OFFSET,
    'control_offset' : CONTROL_OFFSET
}

if __name__ == '__main__':
    print('Multimodal Vocabulary Configuration:')
    print('  -> Audio Frames Per Second:', FRAMES_PER_SECOND)
    print('  -> Scale Quantization:', SCALE_QUANTIZATION)
    print('  -> Number of Residuals:', RESIDUALS)
    print('  -> Codebook Size:', CODEBOOK_SIZE)
    print('  -> Midi Quantization:', MIDI_QUANTIZATION)
    print('  -> Maximum Interarrival Time:', MAX_INTERARRIVAL)
    print('  -> Maximum Midi Duration:', MAX_DURATION)
    print('  -> Vocabulary Size:', VOCAB_SIZE)
    print('Multimodal Training Sequence Format')
    print(80*'-')
    print('Sequence Separator :', SEPARATOR)
    print('Residual Pad :', RESIDUAL_PAD)
    print('Control Block:', CONTROL_OFFSET)
    print('  * control pad :', CONTROL_PAD)
    print('  -> generation tasks:')
    print('    * audio generation flag :', AUDIOGEN)
    print('    * midi generation flag :', MIDIGEN)
    print('    * transcription flag :', TRANSCRIBE)
    print('    * synthesis flag :', SYNTHESIZE)
    print('  -> audio type:')
    print('    * clean audio :', CLEANAUDIO)
    print('    * clean midi :', CLEANMIDI)
    print('    * synthesized audio :', SYNTHAUDIO)
    print('    * transcribed midi :', TRANSMIDI)
    print('Audio Block:', AUDIO_OFFSET)
    print('  * scale pad :', SCALE_PAD)
    print('  -> r0 offset:', R0_OFFSET)
    print('  -> r1 offset:', R1_OFFSET)
    print('  -> r2 offset:', R2_OFFSET)
    print('  -> r3 offset:', R3_OFFSET)
    print('  -> scale offset:', SCALE_OFFSET)
    print('Midi Block:', MIDI_OFFSET)
    print('  -> interarrival time offset :', TIME_OFFSET)
    print('  -> pitch offset :', PITCH_OFFSET)
    print('    * rest :', REST)
    print('  -> instrument offset :', INSTRUMENT_OFFSET)
    print('  -> duration offset :', DURATION_OFFSET)
