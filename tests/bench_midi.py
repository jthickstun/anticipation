import pyperf
from pathlib import Path

from anticipation.convert import midi_to_compound as v1_midi_to_compound
from anticipation.v2.config import AnticipationV2Settings, Vocab
from anticipation.v2.convert import midi_to_compound as v2_midi_to_compound


MIDI_DIR = Path(__file__).parent / "test_data"
MIDI_PATHS = list(sorted(MIDI_DIR.glob("*.mid")))

assert MIDI_PATHS

SETTINGS = AnticipationV2Settings(vocab=Vocab())

def bench_old() -> None:
    # v1 midi file to compound
    for path in MIDI_PATHS:
        v1_midi_to_compound(str(path))


def bench_new() -> None:
    # v2 midi file to compound
    for path in MIDI_PATHS:
        v2_midi_to_compound(path, SETTINGS)


if __name__ == "__main__":
    """
    (pip install pyperf)
    
    Run with, e.g.:

        PYTHONPATH=. python tests/bench_midi.py --processes 10 --values 50
        
    Recent results:
    
        old: Mean +- std dev: 1.53 sec +- 0.02 sec
        new: Mean +- std dev: 22.5 ms +- 0.7 ms
    """
    runner = pyperf.Runner()
    runner.metadata["midi_files"] = len(MIDI_PATHS)
    runner.metadata["midi_dir"] = str(MIDI_DIR)
    runner.metadata["gc"] = "enabled"

    runner.bench_func("old", bench_old)
    runner.bench_func("new", bench_new)
