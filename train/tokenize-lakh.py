import os
from pathlib import Path
from argparse import ArgumentParser
from multiprocessing import Pool, RLock
from glob import glob

from tqdm import tqdm

from anticipation import config as v1_config
from anticipation.tokenize import tokenize, tokenize_ia
from anticipation.v2.config import AnticipationV2Settings, Vocab


V2_AR_ONLY = AnticipationV2Settings(
    vocab=Vocab(),
    num_autoregressive_seq_per_midi_file=1,
    num_span_anticipation_augmentations_per_midi_file=0,
    num_instrument_anticipation_augmentations_per_midi_file=0,
    num_random_anticipation_augmentations_per_midi_file=0,
)


def main(args):
    if args.tokenization_style == "v1_interarrival":
        encoding = "interarrival"
    elif args.tokenization_style == "v1":
        encoding = "arrival"
    elif args.tokenization_style == "v2":
        encoding = "v2"
    else:
        raise ValueError(f"Unsupported tokenization style: {args.tokenization_style}")

    print("Tokenizing LakhMIDI")
    print(f"  encoding type: {encoding}")
    print(
        f"  train split: {[s for s in v1_config.LAKH_SPLITS if s not in v1_config.LAKH_VALID + v1_config.LAKH_TEST]}"
    )
    print(f"  validation split: {v1_config.LAKH_VALID}")
    print(f"  test split: {v1_config.LAKH_TEST}")

    print("Tokenization parameters:")
    print(f"  anticipation interval = {v1_config.DELTA}s")
    print(f"  augment = {args.augment}x")
    print(f"  max track length = {v1_config.MAX_TRACK_TIME_IN_SECONDS}s")
    print(f"  min track length = {v1_config.MIN_TRACK_TIME_IN_SECONDS}s")
    print(f"  min track events = {v1_config.MIN_TRACK_EVENTS}")

    input_data_path = Path(args.datadir)
    assert input_data_path.exists()
    assert input_data_path.is_dir()

    paths = [input_data_path / s for s in v1_config.LAKH_SPLITS]
    files = [p.glob("*.compound.txt") for p in paths]
    outputs = [
        input_data_path / f"tokenized-events-{s}.txt" for s in v1_config.LAKH_SPLITS
    ]

    # don't augment the valid/test splits
    augment = [
        1 if s in v1_config.LAKH_VALID or s in v1_config.LAKH_TEST else args.augment
        for s in v1_config.LAKH_SPLITS
    ]

    # parallel tokenization drops the last chunk of < M tokens
    # if concerned about waste: process larger groups of datafiles
    func = tokenize_ia if args.interarrival else tokenize
    with Pool(
        processes=v1_config.PREPROC_WORKERS,
        initargs=(RLock(),),
        initializer=tqdm.set_lock,
    ) as pool:
        results = pool.starmap(
            func, zip(files, outputs, augment, range(len(v1_config.LAKH_SPLITS)))
        )

    (
        seq_count,
        rest_count,
        too_short,
        too_long,
        too_manyinstr,
        discarded_seqs,
        truncations,
    ) = (sum(x) for x in zip(*results))
    rest_ratio = round(100 * float(rest_count) / (seq_count * v1_config.M), 2)

    trunc_type = "interarrival" if args.interarrival else "duration"
    trunc_ratio = round(100 * float(truncations) / (seq_count * v1_config.M), 2)

    print("Tokenization complete.")
    print(f"  => Processed {seq_count} training sequences")
    print(f"  => Inserted {rest_count} REST tokens ({rest_ratio}% of events)")
    print(f"  => Discarded {too_short + too_long} event sequences")
    print(f"      - {too_short} too short")
    print(f"      - {too_long} too long")
    print(f"      - {too_manyinstr} too many instruments")
    print(f"  => Discarded {discarded_seqs} training sequences")
    print(
        f"  => Truncated {truncations} {trunc_type} times ({trunc_ratio}% of {trunc_type}s)"
    )

    print("Remember to shuffle the training split!")


if __name__ == "__main__":
    parser = ArgumentParser(description="tokenizes a MIDI dataset")
    parser.add_argument(
        "datadir", help="directory containing preprocessed MIDI to tokenize"
    )
    parser.add_argument(
        "-k",
        "--augment",
        type=int,
        default=1,
        help="dataset augmentation factor (multiple of 10)",
    )
    parser.add_argument(
        "--tokenization_style",
        type=str,
        default="v1",
        choices=["v1_interarrival", "v1", "v2"],
    )

    main(parser.parse_args())
