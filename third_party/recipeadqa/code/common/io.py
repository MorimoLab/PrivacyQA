"""Canonical streaming readers shared by the numbered code pipelines."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterator


def open_text(path: str | Path):
    """Open an uncompressed or gzip-compressed UTF-8 text file."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    return opener(path, "rt", encoding="utf-8")


def iter_json_array(
    path: str | Path,
    chunk_size: int = 1024 * 1024,
) -> Iterator[Any]:
    """Stream values from a top-level JSON array without loading it wholly."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    path = Path(path)
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    eof = False
    started = False
    first_item = True

    with open_text(path) as handle:
        def fill() -> None:
            nonlocal buffer, position, eof
            if position:
                buffer = buffer[position:]
                position = 0
            chunk = handle.read(chunk_size)
            if chunk:
                buffer += chunk
            else:
                eof = True

        while True:
            if position >= len(buffer) and not eof:
                fill()
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if not started:
                if position >= len(buffer):
                    if eof:
                        raise ValueError(f"empty JSON input: {path}")
                    fill()
                    continue
                if buffer[position] != "[":
                    raise ValueError(f"expected a top-level JSON array: {path}")
                position += 1
                started = True
                continue
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position >= len(buffer):
                if eof:
                    raise ValueError(f"unterminated JSON array: {path}")
                fill()
                continue
            if buffer[position] == "]":
                position += 1
                break
            if not first_item:
                if buffer[position] != ",":
                    raise ValueError(f"expected a comma between JSON array items: {path}")
                position += 1
                while True:
                    while position < len(buffer) and buffer[position].isspace():
                        position += 1
                    if position < len(buffer):
                        break
                    if eof:
                        raise ValueError(f"unterminated JSON array: {path}")
                    fill()
                if buffer[position] == "]":
                    raise ValueError(f"trailing comma in JSON array: {path}")
            while True:
                try:
                    value, end = decoder.raw_decode(buffer, position)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise ValueError(f"invalid JSON array item in {path}")
                    fill()
            position = end
            first_item = False
            yield value

        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position < len(buffer):
                raise ValueError(f"trailing content after JSON array: {path}")
            if eof:
                break
            fill()


__all__ = ["iter_json_array", "open_text"]
