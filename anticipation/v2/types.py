from enum import Enum

# type aliases
Token = int
MIDITick = int
MIDINote = int
MIDIProgramCode = int

Triplet = tuple[Token, Token, Token]
TickToken = tuple[Token]


class MIDIFileIgnoredReason(Enum):
    TOO_FEW_EVENTS = 0
    TOO_FEW_SECONDS = 1
    TOO_MANY_SECONDS = 2
    TOO_MANY_INSTRUMENTS = 3
    INVALID_FILE_HEADER = 4
    UNEXPECTED_EOF_ERROR = 5
    BAD_STATUS = 6
    VALUES_WERE_OUT_OF_RANGE = 7
    INVALID_TICK_TYPE = 8
    INVALID_MIDI_FORMAT_IN_HEADER = 9
    INVALID_FILE_STRUCTURE = 10
