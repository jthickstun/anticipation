from pathlib import Path
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset


class PreTokenizedDataset(Dataset):
    """
    Must be constructed with v2 sequence packing.
    """

    def __init__(self, path: Path) -> None:
        # O(page size) random access apparently?
        self.data = np.load(path, mmap_mode="r")
        assert self.data.flags["C_CONTIGUOUS"]

    @property
    def num_tokens(self) -> int:
        num_sequences = self.data.shape[0]
        num_tokens_per_sequence = self.data.shape[1]
        return num_sequences * num_tokens_per_sequence

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        with warnings.catch_warnings():
            # this warning is about writing to a tensor loaded in this way
            # will result in undefined behavior, but this is the dataset, we
            # are not going to mutate these samples
            warnings.filterwarnings("ignore", category=UserWarning)
            input_ids = torch.from_numpy(self.data[idx]).to(dtype=torch.long)

        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        labels = torch.roll(labels, shifts=-1, dims=0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
