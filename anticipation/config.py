# model hyper-parameters

EVENT_SIZE = 3                     # each event/label is encoded as 3 tokens
M = 341                            # model context (1024 = 1 + EVENT_SIZE*M)
DELTA = 5                          # anticipation time in seconds

# vocabulary constants

MAX_TIME_IN_SECONDS = 100          # exclude very long training sequences
MAX_DURATION_IN_SECONDS = 10       # maximum duration of a note
TIME_RESOLUTION = 100              # 10ms time resolution = 100 bins/second

MAX_PITCH = 128                    # 128 MIDI pitches
MAX_INSTR = 129                    # 129 MIDI instruments (128 + drums)
MAX_NOTE = MAX_PITCH*MAX_INSTR     # note = pitch x instrument

# preprocessing settings

PREPROC_WORKERS = 16

MAX_TRACK_INSTR = 16               # exclude tracks with large numbers of instruments
MAX_TRACK_TIME_IN_SECONDS = 1600   # exclude very long tracks
MIN_TRACK_EVENTS = 40              # exclude tracks with very few events

AUGMENT_FACTOR = 10                # data augmentation factor (multiple of 10)

# LakhMIDI dataset splits

LAKH_SPLITS = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f']
LAKH_VALID = ['e']
LAKH_TEST = ['f']

# derived quantities

MAX_TIME = TIME_RESOLUTION*MAX_TIME_IN_SECONDS
MAX_DUR = TIME_RESOLUTION*MAX_DURATION_IN_SECONDS


if __name__ == '__main__':
    print('Model constants:')
    print(f'  -> anticipation interval: {DELTA}s')
    print('Vocabulary constants:')
    print(f'  -> maximum time of a sequence: {MAX_TIME_IN_SECONDS}s')
    print(f'  -> maximum duration of a note: {MAX_DURATION_IN_SECONDS}s')
    print(f'  -> time resolution: {TIME_RESOLUTION}bins/s ({1000//TIME_RESOLUTION}ms)')
    print('Preprocessing constants:')
    print(f'  -> maximum time of a track: {MAX_TRACK_TIME_IN_SECONDS}s')
    print(f'  -> minimum events in a track: {MIN_TRACK_EVENTS}s')
    print(f'  -> training dataset augmentation factor: {AUGMENT_FACTOR}x')
