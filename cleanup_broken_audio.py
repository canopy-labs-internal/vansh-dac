#!/usr/bin/env python3
"""
cleanup_broken_audio.py
----------------------
Recursively scan a directory for audio files and delete any that are either:
  • Empty (0-byte size)
  • Corrupted / unreadable by ffprobe

Supported extensions: .wav, .flac, .mp3, .ogg, .aiff, .aif, .m4a

Usage:
    python cleanup_broken_audio.py [--root PATH] [--dry-run]

This script requires `ffprobe` (from FFmpeg) to be installed and in PATH.
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

AUDIO_EXTS: List[str] = [
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".aiff",
    ".aif",
    ".m4a",
]


def is_audio_file(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_EXTS


def ffprobe_ok(path: Path) -> bool:
    """Return True if ffprobe can read *path* (no output captured for speed)."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    # Discard stdout/stderr for performance/memory
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def iter_audio_files(root: Path) -> Iterator[Path]:
    """Recursively yield audio files beneath *root* using scandir (faster than os.walk)."""
    stack: List[Path] = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False) and is_audio_file(Path(entry.name)):
                        yield Path(entry.path)
        except PermissionError:
            # Skip directories we can't access
            continue


def _check_file(path: Path) -> Tuple[Path, Optional[str]]:
    """Return (path, reason) where reason is None if file is okay."""
    try:
        if path.stat().st_size == 0:
            return path, "empty (0 bytes)"
        if not ffprobe_ok(path):
            return path, "corrupted (ffprobe failed)"
    except Exception as exc:
        return path, f"error accessing file: {exc}"
    return path, None


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Remove empty or corrupted audio files.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Directory to scan (default: CWD)")
    parser.add_argument("--dry-run", action="store_true", help="List files that would be removed but do not delete them")
    parser.add_argument("--workers", type=int, default=os.cpu_count() * 4, help="Number of concurrent ffprobe checks (default: 4×CPU cores)")
    args = parser.parse_args(argv)

    root: Path = args.root.resolve()
    dry_run: bool = args.dry_run

    if not root.is_dir():
        print(f"Error: {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning '{root}' for empty or corrupted audio files…")

    removed: List[Path] = []
    kept = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_path = {executor.submit(_check_file, p): p for p in iter_audio_files(root)}

        for future in as_completed(future_to_path):
            path, reason = future.result()
            if reason:
                removed.append(path)
                action = "Would remove" if dry_run else "Removing"
                print(f"{action}: {path} -> {reason}")
                if not dry_run:
                    try:
                        path.unlink()
                    except Exception as exc:
                        print(f"Failed to delete {path}: {exc}", file=sys.stderr)
            else:
                kept += 1

    print(f"\nSummary: {len(removed)} files removed, {kept} kept.")
    if dry_run and removed:
        print("Run again without --dry-run to actually delete the files listed above.")


if __name__ == "__main__":
    main()
