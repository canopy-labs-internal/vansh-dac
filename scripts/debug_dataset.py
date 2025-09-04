#!/usr/bin/env python
"""Debug script to reproduce the build_dataset logic and verify that audio files
can be discovered and loaded successfully.

Usage
-----
python scripts/debug_dataset.py --folders /path/one /path/two --sample_rate 24000 --num_samples 5

The script prints:
1. Number of files discovered in each folder.
2. First *num_samples* file paths.
3. Attempts to read the first *num_samples* files via AudioSignal and prints their shapes.

This is useful when training fails with a libsndfile error; you can isolate which
file cannot be decoded.
"""


DEFAULT_FOLDERS = [
    "/workspace/dac-datasets/DAPS/daps-download/daps/clean",
]
DEFAULT_SAMPLE_RATE = 24000  # Hz
DEFAULT_NUM_SAMPLES = 5      # How many files to try loading

# --------------------------------------------------

import os

from audiotools import AudioSignal
from audiotools.data.datasets import AudioLoader, AudioDataset


def build_single_dataset(folders, sample_rate):
    loader = AudioLoader(sources=folders)
    dataset = AudioDataset(loader, sample_rate)
    return loader, dataset


def main():
    loader, dataset = build_single_dataset(DEFAULT_FOLDERS, DEFAULT_SAMPLE_RATE)
    
    print(f"Loader object: {loader.audio_lists}")
    print(f"Dataset object: {dataset.loaders}")

    # all_paths = [d["path"] for lst in loader.audio_lists for d in lst]
    # num_files = len(all_paths)
    # print(f"Total files discovered by AudioLoader: {num_files}")
    # path_str = '\n'.join(all_paths)
    # print(f"ALL PATHS: {path_str}")

    # # Print the first few file paths and try to read them.
    # print(f"\nAttempting to load first {DEFAULT_NUM_SAMPLES} examples:")
    # for idx, path in enumerate(all_paths[:DEFAULT_NUM_SAMPLES]):
    #     try:
    #         signal = AudioSignal(path)
    #         signal = signal.to_mono()  # simplify printing
    #         print(
    #             f"    Loaded OK. Shape: {signal.audio_data.shape}, SR: {signal.sample_rate}"
    #         )
    #     except Exception as e:
    #         print(f"    ERROR loading file: {e}")
    #         break


if __name__ == "__main__":
    main()
