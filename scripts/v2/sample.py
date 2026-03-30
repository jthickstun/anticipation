import io
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from anticipation.v2.config import AnticipationV2Settings
from anticipation.v2.sample import generate_ar_simple


def main():
    # load an anticipatory music transformer
    checkpoint_dir = "/home/mf867/anticipation_isolated/anticipation/output/checkpoints/test_checkpoints/step-1000"
    settings_file = "data/tokenized_datasets/lmd_full/b82a7a2750e3c5836ffb9bf564720cd8/settings_b82a7a2750e3c5836ffb9bf564720cd8.json"
    settings = AnticipationV2Settings.load_from_disk(Path(settings_file))
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir).to("cuda").to(torch.float32)
    print("Starting to generate...")
    midi_file = generate_ar_simple(
        model,
        settings,
        num_events_to_generate=100,
        top_p=1.0
    )
    midi_file.save("example_midi.mid")

if __name__ == "__main__":
    main()
