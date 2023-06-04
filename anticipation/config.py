"""
Global configuration for anticipatory infilling models.
"""

# model hyper-parameters

CONTEXT_SIZE = 1024                # model context
EVENT_SIZE = 3                     # each event/control is encoded as 3 tokens
M = 341                            # model context (1024 = 1 + EVENT_SIZE*M)
DELTA = 5                          # anticipation time in seconds

assert CONTEXT_SIZE == 1+EVENT_SIZE*M

# vocabulary constants

MAX_TIME_IN_SECONDS = 100          # exclude very long training sequences
MAX_DURATION_IN_SECONDS = 10       # maximum duration of a note
TIME_RESOLUTION = 100              # 10ms time resolution = 100 bins/second

MAX_PITCH = 128                    # 128 MIDI pitches
MAX_INSTR = 129                    # 129 MIDI instruments (128 + drums)
MAX_NOTE = MAX_PITCH*MAX_INSTR     # note = pitch x instrument

MAX_INTERARRIVAL_IN_SECONDS = 10   # maximum interarrival time (for MIDI-like encoding)

# preprocessing settings

PREPROC_WORKERS = 16

COMPOUND_SIZE = 5                  # event size in the intermediate compound tokenization
MAX_TRACK_INSTR = 16               # exclude tracks with large numbers of instruments
MAX_TRACK_TIME_IN_SECONDS = 3600   # exclude very long tracks (longer than 1 hour)
MIN_TRACK_TIME_IN_SECONDS = 10     # exclude very short tracks (less than 10 seconds)
MIN_TRACK_EVENTS = 100             # exclude very short tracks (less than 100 events)

# LakhMIDI dataset splits

LAKH_SPLITS = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f']
LAKH_VALID = ['e']
LAKH_TEST = ['f']

# derived quantities

MAX_TIME = TIME_RESOLUTION*MAX_TIME_IN_SECONDS
MAX_DUR = TIME_RESOLUTION*MAX_DURATION_IN_SECONDS

MAX_INTERARRIVAL = TIME_RESOLUTION*MAX_INTERARRIVAL_IN_SECONDS


if __name__ == '__main__':
    print('Model constants:')
    print(f'  -> anticipation interval: {DELTA}s')
    print('Vocabulary constants:')
    print(f'  -> maximum time of a sequence: {MAX_TIME_IN_SECONDS}s')
    print(f'  -> maximum duration of a note: {MAX_DURATION_IN_SECONDS}s')
    print(f'  -> time resolution: {TIME_RESOLUTION}bins/s ({1000//TIME_RESOLUTION}ms)')
    print(f'  -> maximum interarrival-time (MIDI-like encoding): {MAX_INTERARRIVAL_IN_SECONDS}s')
    print('Preprocessing constants:')
    print(f'  -> maximum time of a track: {MAX_TRACK_TIME_IN_SECONDS}s')
    print(f'  -> minimum events in a track: {MIN_TRACK_EVENTS}s')
