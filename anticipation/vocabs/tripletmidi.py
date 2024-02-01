"""
The vocabulary used for models in the paper "Anticipatory Music Transformer".
"""

#
# configuaration
#

DELTA = 5                                # seconds of anticipation
HUMAN_DELTA = -1                         # seconds of anti-anticipation

# MIDI
MAX_PITCH = 128                          # 128 MIDI pitches
MAX_INSTR = 128 + 1                      # 129 MIDI instruments (128 + drums)
MAX_INTERARRIVAL_IN_SECONDS = 1          # maximum interarrival time
MAX_TIME_IN_SECONDS = 100                # upper bound on arrival times
MAX_DURATION_IN_SECONDS = 10             # maximum duration of a note

# quantization
MIDI_QUANTIZATION = 100                  # time bins/second
MAX_TIME = MIDI_QUANTIZATION*MAX_TIME_IN_SECONDS
MAX_DUR = MIDI_QUANTIZATION*MAX_DURATION_IN_SECONDS

# combined (pitch, instrument) vocabulary
MAX_NOTE = MAX_PITCH*MAX_INSTR     # note = pitch x instrument

#
# vocabulary
#

# the event block
EVENT_OFFSET = 0
TIME_OFFSET = EVENT_OFFSET
DUR_OFFSET = TIME_OFFSET + MAX_TIME
NOTE_OFFSET = DUR_OFFSET + MAX_DUR
REST = NOTE_OFFSET + MAX_NOTE

# the control block
CONTROL_OFFSET = NOTE_OFFSET + MAX_NOTE + 1
ATIME_OFFSET = CONTROL_OFFSET + 0
ADUR_OFFSET = ATIME_OFFSET + MAX_TIME
ANOTE_OFFSET = ADUR_OFFSET + MAX_DUR

# the special block
SPECIAL_OFFSET = ANOTE_OFFSET + MAX_NOTE
SEPARATOR = SPECIAL_OFFSET
AUTOREGRESS = SPECIAL_OFFSET + 1
ANTICIPATE = SPECIAL_OFFSET + 2
PAD = SPECIAL_OFFSET + 3
INSTR_OFFSET = SPECIAL_OFFSET + 4        # instrument-conditioning tokens
VOCAB_SIZE = INSTR_OFFSET+MAX_INSTR

# chord conditioning
CHORD_INSTR = INSTR_OFFSET + 101         # lead sheet stored in program_code=goblins

vocab = {
    'config' : {
        'skew' : False,
        'anticipation' : DELTA,
        'anti-anticipation:': HUMAN_DELTA,
        'midi_quantization' : MIDI_QUANTIZATION,
        'max_time' : MAX_TIME,
        'max_duration' : MAX_DUR,
        'size' : VOCAB_SIZE
    },

    'separator' : SEPARATOR,
    'pad' : PAD,
    'rest' : REST,

    'task' : {
        'autoregress' : AUTOREGRESS,
        'anticipate' : ANTICIPATE,
    },

    'event_offset' : EVENT_OFFSET,
    'time_offset' : TIME_OFFSET,
    'duration_offset' : DUR_OFFSET,
    'note_offset' : NOTE_OFFSET,
    'special_offset' : SPECIAL_OFFSET,
    'instrument_offset' : INSTR_OFFSET,
    'control_offset': CONTROL_OFFSET,
    'chord_instrument': CHORD_INSTR
}

if __name__ == '__main__':
    print('MIDI Vocabulary Configuration:')
    print('  -> Arrival-time Tokenization') 
    print('  -> Combined Note Vocabulary note = (pitch, instrument)') 
    print('  -> Midi Quantization:', MIDI_QUANTIZATION)
    print('  -> Maximum Arrival Time:', MAX_TIME)
    print('  -> Maximum Duration:', MAX_DUR)
    print('  -> Vocabulary Size:', VOCAB_SIZE)
    print('MIDI Training Sequence Format')
    print(80*'-')
    print('Sequence Separator :', SEPARATOR)
    print('Special Block:', SPECIAL_OFFSET)
    print('  -> generation tasks:')
    print('    * autoregress flag :', AUTOREGRESS)
    print('    * anticipate flag :', ANTICIPATE)
    print('  * pad token:', PAD)
    print('  -> instrument control offset', INSTR_OFFSET)
    print('Midi Event Block:', EVENT_OFFSET)
    print('  -> arrival time offset :', TIME_OFFSET)
    print('  -> duration offset :', DUR_OFFSET)
    print('  -> note offset :', NOTE_OFFSET)
    print('    * rest :', REST)
    print('Midi Control Block:', CONTROL_OFFSET)
    print('  -> arrival time offset :', ATIME_OFFSET)
    print('  -> duration offset :', ADUR_OFFSET)
    print('  -> note offset :', ANOTE_OFFSET)
