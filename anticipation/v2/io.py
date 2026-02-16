from pathlib import Path

from typing import Union
import numpy as np

from anticipation.v2.types import Token


class TokenSequenceBinaryFile:
    """Buffered IO for integer sequences.

    This was original within the `SequencePacker` object, but that has its own
    buffer, and this also has its own buffer, and it got confusing.

    This handles writing binary data to disk when the buffer reaches some number of sequences
    and should be closed like a file when done.
    """

    def __init__(
        self,
        path: Path,
        seq_len: int,
        vocab_size: int,
        flush_every_n_sequences: int = 8_192,
        io_buffer_bytes: int = 8 * 1024 * 1024,
    ):
        if path.exists() and path.is_file():
            # for now, I want these to be atomic. We could change later but this is safest in the
            # short term. Might be better design for the Sequence packer to handle this check.
            raise RuntimeError(
                f"Target file to write to already exists. Appending to file would entangle things. "
                f"Got path: {str(path.absolute())}"
            )

        self.vocab_size = vocab_size
        self.dtype = self.get_dtype_for_tokens(vocab_size)
        self.path = path
        self.seq_len = seq_len
        self.flush_every_n_sequences = flush_every_n_sequences
        self.io_buffer_bytes = io_buffer_bytes
        self._f = open(path, "ab", buffering=io_buffer_bytes)
        self._buf = np.empty(
            (self.flush_every_n_sequences, self.seq_len),
            dtype=self.dtype,
        )
        self.i = 0

    def append(self, tokens: list[Token]) -> None:
        a = np.asarray(tokens, dtype=self.dtype)
        if not a.flags["C_CONTIGUOUS"]:
            a = np.ascontiguousarray(a)
        self._buf[self.i, :] = a
        self.i += 1
        if self.i == self.flush_every_n_sequences:
            self._buf.tofile(self._f)
            self.i = 0

    def close(self) -> None:
        if self.i:
            # write the remaining sequences
            self._buf[: self.i].tofile(self._f)
            self.i = 0
        self._f.flush()
        self._f.close()

    @staticmethod
    def get_dtype_for_tokens(vocab_size: int) -> Union[np.uint16, np.uint32, np.uint64]:
        if vocab_size <= (2**16):
            return np.uint16
        elif vocab_size <= (2**32):
            return np.uint32
        elif vocab_size <= (2**64):
            return np.uint64
        else:
            raise ValueError("Vocab too large.")

    @classmethod
    def load_from_disk_to_numpy(
        cls, bin_path: Path, seq_len: int, vocab_size: int
    ) -> np.ndarray:
        dtype = cls.get_dtype_for_tokens(vocab_size)
        with open(bin_path, "rb") as f:
            raw_data = np.fromfile(f, dtype=dtype)
            # sample id, seq len
            tokens_arr = raw_data.reshape(-1, seq_len)
            return tokens_arr


def consolidate_bins(
    shard_paths: list[Path],
    out_path: Path,
    dtype,
    seq_len: int,
    buffer_bytes: int = 64 * 1024 * 1024,  # 64MB
) -> None:
    dtype = np.dtype(dtype)
    dtype_nbytes = dtype.itemsize
    out_path = out_path.resolve()
    if out_path.exists():
        raise FileExistsError(out_path)

    with out_path.open("wb") as f:
        for p in shard_paths:
            p = p.resolve()
            if not p.exists():
                raise FileNotFoundError(p)

            size = p.stat().st_size
            if size % dtype_nbytes != 0:
                raise ValueError(
                    f"{p.name}: size {size} not divisible by dtype itemsize {dtype_nbytes}"
                )

            n_tokens = size // dtype_nbytes
            if n_tokens % seq_len != 0:
                raise ValueError(
                    f"{p.name}: {n_tokens} tokens not divisible by seq_len={seq_len}"
                )
            with p.open("rb") as fin:
                while True:
                    chunk = fin.read(buffer_bytes)
                    if not chunk:
                        break
                    f.write(chunk)
