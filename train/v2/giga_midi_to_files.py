from typing import Any
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from symusic import Score

from train.v2.dataset_tokenize import get_lakh_midi_splits_and_configs

DATASETS_PATH = Path(__file__).parent.parent.parent / "data"

# see: https://huggingface.co/datasets/Metacreation/GigaMIDI
LAKH_MIDI_PATH = DATASETS_PATH / "lmd_full"
GIGA_MIDI_ENCLOSING_PATH = DATASETS_PATH / "giga_midi"


def extract_files() -> None:
    # remove all the test split samples from the dataset, from any split
    # we will use the test split of lakh to eval
    exclude_lakh_midi_md5 = set()
    if LAKH_MIDI_PATH.exists() and LAKH_MIDI_PATH.is_dir():
        all_lakh_splits = get_lakh_midi_splits_and_configs(LAKH_MIDI_PATH)
        # get all the md5 files from the lakh test split
        lakh_test_split = [x for x in all_lakh_splits if x["name"] == "test"][0]
        for p in lakh_test_split["dataset_paths"]:
            all_files_mid = [x.stem for x in p.rglob("*.mid")]
            all_files_midi = [x.stem for x in p.rglob("*.midi")]
            exclude_lakh_midi_md5.update(all_files_midi + all_files_mid)
    else:
        raise RuntimeError(
            "Lakh has not yet been downloaded. Execute the script `run_get_lakh.sh`."
        )

    splits = ["train", "validation", "test"]

    # exist NOT ok... if you run this multiple times, and your intention is to exclude
    # Lakh files, then the Lakh files will still be in there unless you delete the
    # original directory. Run this clean from nothing if possible.
    GIGA_MIDI_ENCLOSING_PATH.mkdir(parents=True, exist_ok=False)

    for split in splits:
        num_skipped_in_lakh = 0
        num_ignored = 0

        # if dataset is gated, run `hf auth login`
        dataset = load_dataset("Metacreation/GigaMIDI", split=split)
        split_enclosing_path = GIGA_MIDI_ENCLOSING_PATH / split
        split_enclosing_path.mkdir(parents=True, exist_ok=True)

        iter_obj = tqdm(
            dataset, mininterval=1.0, desc=f"Saving files in split: {split}"
        )
        for i, sample in enumerate(iter_obj):
            sample: dict[str, Any]
            sample_md5 = sample["md5"]
            if sample_md5 in exclude_lakh_midi_md5:
                num_skipped_in_lakh += 1
                continue

            sample_file_name = sample_md5
            save_to = Path(split_enclosing_path / f"{sample_file_name}.midi")
            try:
                score = Score.from_midi(sample["music"], sanitize_data=True)
                score.dump_midi(save_to)
            except RuntimeError:
                # can't save this file for some reason...
                num_ignored += 1
                pass

        print(f"Unable to open: {num_ignored:,}")
        print(f"Ignored Lakh files: {num_skipped_in_lakh:,}")


def check_giga_midi_uniqueness():
    splits = ["train", "validation", "test"]
    total_files = 0
    total_md5 = set()
    for split in splits:
        # if dataset is gated, run `hf auth login`
        dataset = load_dataset("Metacreation/GigaMIDI", split=split)
        total_files += len(dataset)
        for s in dataset:
            sample_md5 = s["md5"]
            total_md5.add(sample_md5)

    print("Total Files ?= Total Unique MD5s?")
    print("Files: ", total_files)
    print("MD5s: ", len(total_md5))


if __name__ == "__main__":
    """
    From repo root, run:

        PYTHONPATH=. python train/v2/giga_midi_to_files.py

    If not yet downloaded, one might need to run: `hf auth login` because the dataset
    requires huggingface authentication.
    """
    # check_giga_midi_uniqueness()
    extract_files()
