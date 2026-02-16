from typing import Iterator, Any

import os
import hashlib
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path


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
