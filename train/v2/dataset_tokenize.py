import csv
import math
from pathlib import Path
from typing import Iterable, Any
from functools import partial
from json import dumps
import multiprocessing as mp
import tempfile

import numpy as np
from tqdm.contrib.concurrent import process_map

from anticipation.v2.config import (
    AnticipationV2Settings,
    Vocab,
    DATASET_ROOT,
)
from anticipation.v2.tokenize import tokenize, MIDIFileIgnoredReason
from anticipation.v2.util import (
    iter_files,
    get_book_keeping_info,
)
from anticipation.v2.io import TokenSequenceBinaryFile, consolidate_bins


def _process_shard(
    shard_id_and_files_to_process: tuple[int, list[Path]],
    settings: AnticipationV2Settings,
    shards_container_path: Path,
) -> tuple[Path, dict[MIDIFileIgnoredReason, list[Path]]]:
    shard_id, files_to_process = shard_id_and_files_to_process
    work_dir = shards_container_path / f"./{shard_id}"
    work_dir.mkdir(exist_ok=True)
    shard_artifact_path = work_dir / f"{shard_id}_shard_dataset_processed.tmp.bin"

    # tokenize code here! this is the actual logic of what we are doing,
    # everything else is coordination
    ignored_files_summary = tokenize(
        files_to_process,
        output=shard_artifact_path,
        settings=settings,
        shard_id=shard_id,
    )
    return shard_artifact_path, ignored_files_summary


def _get_dataset_shards(
    dataset_paths: Iterable[Path],
    num_shards: int,
) -> Iterable[tuple[int, list[Path]]]:
    """
    This takes a bunch of paths of files to tokenize and splits it into n disjoint lists
    of paths - one per shard.
    """
    all_files = []
    for dataset_path in sorted(dataset_paths):
        assert dataset_path.exists()
        assert dataset_path.is_dir()

        # get all files with specific file extensions
        all_files += list(iter_files(dataset_path, file_extensions=(".mid", ".midi")))

    total_files = len(all_files)

    # number of shards is equal to number of workers
    num_files_per_shard = math.ceil(total_files / num_shards)
    shards = [
        all_files[i : i + num_files_per_shard]
        for i in range(0, total_files, num_files_per_shard)
    ]

    # give each shard an ID, just incrementing
    shards_with_id = enumerate(shards)
    return shards_with_id


def _get_dataset_file_from_paths(
    settings: AnticipationV2Settings,
    dataset_paths: Iterable[Path],
    num_workers: int,
    parent_work_dir: Path,
    shards_dir: Path,
    save_to: str,
    do_shuffle: bool,
) -> tuple[Path, list[tuple[Path, dict[MIDIFileIgnoredReason, list[Path]]]]]:
    # get division of work
    shards = _get_dataset_shards(dataset_paths, num_workers)

    # this is where the tokenization code is actually called
    process_one_with_args = partial(
        _process_shard, settings=settings, shards_container_path=shards_dir
    )

    # run tokenization, keep note of where results are saved
    # (to intermediate shards), as well as any files that are ignored
    records: list[tuple[Path, dict[MIDIFileIgnoredReason, list[Path]]]] = process_map(
        process_one_with_args,
        shards,
        max_workers=num_workers,
        total=num_workers,
        desc="Gathering Workers",
    )

    # here we take all the intermediate raw binaries created by numpy and
    # consolidate them into a single file, this time with a `.npy` extension
    # so it is clearer how to load it
    bin_out_path = shards_dir / (save_to + ".bin")
    npy_out_path = parent_work_dir / (save_to + ".npy")
    consolidate_bins(
        list(shards_dir.rglob("*.bin")),
        out_path=bin_out_path,
        dtype=TokenSequenceBinaryFile.get_dtype_for_tokens(settings.vocab.VOCAB_SIZE),
        seq_len=settings.context_size,
    )
    loaded_arr = TokenSequenceBinaryFile.load_from_disk_to_numpy(
        bin_out_path, settings.context_size, settings.vocab.VOCAB_SIZE
    )
    if do_shuffle:
        # NB: this won't work for huge datasets that don't fit in ram, might need
        # to fix later
        np.random.seed(settings.train_data_split_shuffle_random_seed)
        loaded_arr = loaded_arr[np.random.permutation(loaded_arr.shape[0])]

    # save the consolidated samples to single numpy
    np.save(npy_out_path, loaded_arr)
    return npy_out_path, records


def _get_lakh_midi_splits_and_configs(
    lahk_midi_dataset_parent_path: Path,
) -> list[dict[str, Any]]:
    """
    Follow the Lakh MIDI convention of the splits being:
    - test: "f", not shuffled
    - validation: "e", not shuffled
    - train: everything else, globally shuffled
    """
    lmd_splits = [x for x in lahk_midi_dataset_parent_path.iterdir() if x.is_dir()]
    # sort all of these for determinism
    lmd_test = sorted([x for x in lmd_splits if x.name == "f"])
    lmd_valid = sorted([x for x in lmd_splits if x.name == "e"])
    lmd_train = sorted([x for x in lmd_splits if x.name not in ("f", "e")])
    all_processing_confs = [
        {
            "name": "train",
            "dataset_paths": lmd_train,
            "do_shuffle": True,
        },
        {
            "name": "valid",
            "dataset_paths": lmd_valid,
            "do_shuffle": False,
        },
        {
            "name": "test",
            "dataset_paths": lmd_test,
            "do_shuffle": False,
        },
    ]
    return all_processing_confs


def _tokenize_dataset_in_parallel(
    settings: AnticipationV2Settings,
    lmd_dataset_path: Path,
    save_all_dataset_files_to: Path,
    put_shards_in_tmp: bool,
    split_confs: list[dict[str, Any]],
) -> None:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        if put_shards_in_tmp:
            shards_dir = td_path / "shards"
            shards_dir.mkdir(exist_ok=True)
        else:
            # in this case, we don't use the temp dir that is
            # given to us by the context manager
            shards_dir = save_all_dataset_files_to / "shards"
            shards_dir.mkdir(exist_ok=True)

        ignored_files = []
        for conf in split_confs:
            # create shard dir
            shards_dir_local = shards_dir / conf["name"]
            shards_dir_local.mkdir(exist_ok=True)

            # process shard
            npy_path, file_results = _get_dataset_file_from_paths(
                settings,
                conf["dataset_paths"],
                settings.num_workers_in_dataset_construction,
                parent_work_dir=save_all_dataset_files_to,
                shards_dir=shards_dir_local,
                save_to=conf["name"],
                do_shuffle=conf["do_shuffle"],
            )

            # gather any ignored file results
            for f in file_results:
                shard_path, files_ignored = f
                for reason, files_list in files_ignored.items():
                    for file in files_list:
                        ignored_file = {
                            "split": conf["name"],
                            "shard": shard_path.name,
                            # e.g. TOO_FEW_EVENTS
                            "reason": reason.name,
                            # e.g. f9aad86bfb384b22875d40ef15be023d.mid
                            "file": str(file.relative_to(lmd_dataset_path)),
                        }
                        ignored_files.append(ignored_file)

        # write all the ignored files to disk for awareness
        field_names = list(ignored_files[0].keys())
        with open(
            save_all_dataset_files_to / "ignored_files.csv", "w", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=field_names)  # noqa
            writer.writeheader()
            writer.writerows(ignored_files)


def _write_book_keeping_info_and_get_dataset_enclosing_path(
    settings: AnticipationV2Settings, save_tokenized_dataset_to: Path
) -> Path:
    """
    This function writes some bookkeeping information to an enclosing directory.

    Each tokenized dataset has some metadata associated with it, mainly
    - the global project settings used during generation (settings_<md5>.json)
    - process information (process_info.json)
        - the time tokenization code started to run
        - the git branch and the commit that was used to create dataset
        - a UUID associated with the dataset. We made a UUID each time this is run
          to mitigate any overwriting of existing datasets
    - any files from the dataset that were ignored and roughly why (ignored_files.csv)

    At the end of running this script, the data directory should look approximately
    like this:

    ├── lmd_full (the lakh midi dataset)
    │ ├── 0
    │ ├── 1
    ...
    │ └── f
    └── tokenized_data (where we save our tokenized data)
        └── <UUID> (e.g. 030fb457e4884ad487ef5b4edd415414)
            ├── ignored_files.csv
            ├── process_info.json
            ├── settings_<md5 hash>.json
            ├── test.npy
            ├── train.npy
            └── valid.npy

    This function handles:
    - writing the settings and process info
    - generating a UUID, creating a directory using this UUID and returns it
    """
    dataset_generation_info = get_book_keeping_info()
    uuid = dataset_generation_info["uuid"]

    # save all subsequent dataset files to `work_dir`
    work_dir = save_tokenized_dataset_to / uuid
    work_dir.mkdir()

    # save bookkeeping info
    dataset_generation_info_save_to = work_dir / "book_keeping_info.json"
    dataset_generation_info_save_to.write_text(
        dumps(dataset_generation_info, sort_keys=True, indent=4)
    )

    # save project settings
    settings.save_to_disk(work_dir)
    return work_dir


def main(put_shards_in_tmp: bool = True) -> None:
    # job settings
    dataset_path = DATASET_ROOT

    # LAKH MIDI DATASET PATH
    LAKH_MIDI_FULL_PATH = dataset_path / "lmd_full"

    # WHERE TO SAVE ALL (not just this one) TOKENIZED DATASETS
    put_tokenized_datasets_in_dir = dataset_path / "tokenized_data"
    put_tokenized_datasets_in_dir.mkdir(exist_ok=True)

    settings = AnticipationV2Settings(
        vocab=Vocab(),
        num_autoregressive_seq_per_midi_file=1,
        num_span_anticipation_augmentations_per_midi_file=0,
        num_instrument_anticipation_augmentations_per_midi_file=0,
        num_random_anticipation_augmentations_per_midi_file=0,
        debug=False,
        num_workers_in_dataset_construction=10,
        train_data_split_shuffle_random_seed=42,
    )

    # do the work now, no more config past this point
    _tokenize_dataset_in_parallel(
        settings,
        LAKH_MIDI_FULL_PATH,
        _write_book_keeping_info_and_get_dataset_enclosing_path(
            settings, put_tokenized_datasets_in_dir
        ),
        put_shards_in_tmp,
        _get_lakh_midi_splits_and_configs(LAKH_MIDI_FULL_PATH),
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
