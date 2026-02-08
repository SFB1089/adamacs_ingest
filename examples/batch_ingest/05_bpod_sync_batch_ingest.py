"""Batch template for Bpod ingestion and synchronization."""

from adamacs.ingest.bpod import Bpodfile


def ingest_bpod_files(bpod_relative_paths):
    for relative_path in bpod_relative_paths:
        Bpodfile(relative_path).ingest()


if __name__ == "__main__":
    print("Provide Bpod relative paths and call ingest_bpod_files().")
