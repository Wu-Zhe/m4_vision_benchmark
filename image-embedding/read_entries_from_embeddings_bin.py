#!/usr/bin/env python3
"""Sequentially read records from output/image_embeddings.bin."""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


MAGIC = b"EGIMG256"
HEADER = struct.Struct("<8sHHH")
RECORD_LENGTH = struct.Struct("<I")
DTYPE_FLOAT16 = 1


def read_exact(stream, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise EOFError(f"Unexpected end of file: wanted {size} bytes, got {len(data)}")
    return data


def read_record(stream, record_number: int, dimension: int) -> tuple[str, np.ndarray]:
    filename_length = RECORD_LENGTH.unpack(read_exact(stream, RECORD_LENGTH.size))[0]
    filename = read_exact(stream, filename_length).decode("utf-8")
    vector_bytes = read_exact(stream, dimension * np.dtype("<f2").itemsize)
    vector = np.frombuffer(vector_bytes, dtype="<f2").copy()
    if vector.size != dimension:
        raise ValueError(f"Record {record_number} has {vector.size} dimensions; expected {dimension}")
    return filename, vector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("output/image_embeddings.bin"),
        help="Binary embedding file (default: output/image_embeddings.bin)",
    )
    parser.add_argument("--start-record", type=int, help="1-based first record to print")
    parser.add_argument("--end-record", type=int, help="1-based last record to print")
    args = parser.parse_args()

    start_record = args.start_record if args.start_record is not None else 1
    if args.end_record is not None:
        end_record = args.end_record
    elif args.start_record is not None:
        end_record = start_record + 9
    else:
        end_record = 10

    if start_record < 1 or end_record < start_record:
        raise ValueError("Record range must satisfy 1 <= start-record <= end-record")
    if not args.path.is_file():
        raise FileNotFoundError(args.path)

    with args.path.open("rb") as stream:
        magic, version, dimension, dtype_code = HEADER.unpack(
            read_exact(stream, HEADER.size)
        )
        if magic != MAGIC:
            raise ValueError(f"Invalid magic: {magic!r}; expected {MAGIC!r}")
        if version != 2:
            raise ValueError(f"Unsupported binary format version: {version}")
        if dtype_code != DTYPE_FLOAT16:
            raise ValueError(f"Unsupported embedding dtype code: {dtype_code}")

        print(f"file={args.path}")
        print(f"dimension={dimension} dtype=float16 records=until-EOF")
        print(f"requested_range={start_record}-{end_record}")

        record_number = 1
        while True:
            length_bytes = stream.read(RECORD_LENGTH.size)
            if not length_bytes:
                break
            if len(length_bytes) != RECORD_LENGTH.size:
                raise EOFError("Truncated filename-length field")
            filename_length = RECORD_LENGTH.unpack(length_bytes)[0]
            filename = read_exact(stream, filename_length).decode("utf-8")
            vector_bytes = read_exact(stream, dimension * np.dtype("<f2").itemsize)
            embedding = np.frombuffer(vector_bytes, dtype="<f2").copy()
            if record_number >= start_record:
                print(f"record={record_number}")
                print(f"image_name={filename}")
                print(f"embedding={np.array2string(embedding, threshold=np.inf, separator=', ')}")
            if record_number >= end_record:
                break
            record_number += 1


if __name__ == "__main__":
    main()
