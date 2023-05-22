"""
The vocabularies used for arrival-time and interarrival-time encodings.
"""

# training sequence vocab

from anticipation.config import *

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
VOCAB_SIZE = ANTICIPATE+1

# interarrival-time (MIDI-like) vocab
MIDI_TIME_OFFSET = 0
MIDI_START_OFFSET = MIDI_TIME_OFFSET + MAX_INTERARRIVAL
MIDI_END_OFFSET = MIDI_START_OFFSET + MAX_NOTE
MIDI_SEPARATOR = MIDI_END_OFFSET + MAX_NOTE
MIDI_VOCAB_SIZE = MIDI_SEPARATOR + 1

if __name__ == '__main__':
    print('Arrival-Time Training Sequence Format:')
    print('Event Offset: ', EVENT_OFFSET)
    print('  -> time offset :', TIME_OFFSET)
    print('  -> duration offset :', DUR_OFFSET)
    print('  -> note offset :', NOTE_OFFSET)
    print('  -> rest token: ', REST)
    print('Anticipated Control Offset: ', CONTROL_OFFSET)
    print('  -> anticipated time offset :', ATIME_OFFSET)
    print('  -> anticipated duration offset :', ADUR_OFFSET)
    print('  -> anticipated note offset :', ANOTE_OFFSET)
    print('Special Token Offset: ', SPECIAL_OFFSET)
    print('  -> separator token: ', SEPARATOR)
    print('  -> autoregression flag: ', AUTOREGRESS)
    print('  -> anticipation flag: ', ANTICIPATE)
    print('Arrival Encoding Vocabulary Size: ', VOCAB_SIZE)
    print('')
    print('Interarrival-Time Training Sequence Format:')
    print('  -> time offset: ', MIDI_TIME_OFFSET)
    print('  -> note-on offset: ', MIDI_START_OFFSET)
    print('  -> note-off offset: ', MIDI_END_OFFSET)
    print('  -> separator token: ', MIDI_SEPARATOR)
    print('Interarrival Encoding Vocabulary Size: ', MIDI_VOCAB_SIZE)
