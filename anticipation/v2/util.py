from typing import Iterator, Any, ContextManager

import os
import hashlib
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
import tempfile
from contextlib import contextmanager

import random

import numpy as np
import torch


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


@contextmanager
def temporary_directory(
    env_var: str = "CUSTOM_TMP_DIR",
    require_exists: bool = True,
    check_writable: bool = True,
) -> ContextManager[str]:
    root = os.environ.get(env_var)
    if root is None:
        # if the envar is not set, default to the regular behavior
        # the temporary folder is OS/environment dependent
        with tempfile.TemporaryDirectory() as td:
            yield td

    root_path = Path(root).expanduser().resolve()

    if require_exists and not root_path.exists():
        raise RuntimeError(f"{env_var}={root_path} does not exist")

    if check_writable and not os.access(root_path, os.W_OK):
        raise RuntimeError(f"{env_var}={root_path} is not writable")

    with tempfile.TemporaryDirectory(dir=root_path) as td:
        yield td


def get_book_keeping_info() -> dict[str, Any]:
    return {
        "uuid": get_uuid_string(),
        "git_info": get_git_info(),
        "started_time": get_time_info(),
    }


def get_uuid_string() -> str:
    return str(uuid.uuid4().hex)


def get_time_info() -> dict[str, str]:
    dt = datetime.now(timezone.utc)
    ts = dt.strftime("%Y_%m_%d_%H_%M_%S")
    return {
        "readable_time": ts,
        "unix_time": str(dt.timestamp()),
        "timezone": "utc",
    }


def get_git_info(path: Path = None) -> dict[str, str]:
    if path is None:
        cwd = Path(".")
    else:
        cwd = path

    def run(cmd):
        return subprocess.check_output(
            cmd, cwd=cwd, stderr=subprocess.DEVNULL, text=True
        ).strip()

    try:
        run(["git", "rev-parse", "--is-inside-work-tree"])
        commit = run(["git", "rev-parse", "HEAD"])
        branch = run(["git", "branch", "--show-current"])
        return {
            "branch": branch,
            "commit": commit,
        }
    except subprocess.CalledProcessError:
        return {
            "branch": "",
            "commit": "",
        }


def iter_files(root: Path, file_extensions: tuple[str, ...]) -> Iterator[Path]:
    """
    This ordering is NOT deterministic!
    """
    extensions_to_get = {
        e.lower() if e.startswith(".") else f".{e.lower()}" for e in file_extensions
    }
    stack = [os.fspath(root)]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(entry.path)
                        continue

                    if entry.is_file(follow_symlinks=False):
                        _, ext = os.path.splitext(entry.name)
                        if ext.lower() in extensions_to_get:
                            yield Path(entry.path)
        except (FileNotFoundError, PermissionError, NotADirectoryError):
            continue


def get_md5_of_string(input_str: str) -> str:
    # assuming the input string uses utf-8 encoding
    return hashlib.md5(bytes(input_str, encoding="utf8")).hexdigest()
