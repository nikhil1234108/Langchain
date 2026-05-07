from __future__ import annotations

import argparse
import mimetypes
import re
import zipfile
from pathlib import Path

import pyarrow.parquet as pq


def safe_name(value: object, fallback: str) -> str:
    text = str(value or fallback)
    text = text.replace("\\", "/").split("/")[-1]
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    return text or fallback


def extension_from_bytes(data: bytes, path: str | None) -> str:
    suffix = Path(path or "").suffix.lower()
    if suffix:
        return suffix
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return ".gif"
    guess = mimetypes.guess_extension("application/octet-stream")
    return guess or ".bin"


def iter_images(parquet_path: Path, split: str):
    parquet_file = pq.ParquetFile(parquet_path)
    row_offset = 0
    for batch in parquet_file.iter_batches(columns=["image", "id"], batch_size=128):
        for row_index, row in enumerate(batch.to_pylist()):
            image = row.get("image") or {}
            data = image.get("bytes")
            if not data:
                continue
            source_path = image.get("path")
            ext = extension_from_bytes(data, source_path)
            stem = Path(safe_name(source_path, f"image_{row_offset + row_index:06d}{ext}")).stem
            image_id = safe_name(row.get("id"), f"{row_offset + row_index:06d}")
            filename = f"{split}_{row_offset + row_index:06d}_id_{image_id}_{stem}{ext}"
            yield filename, data
        row_offset += batch.num_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--main-folder", default="FATURA2-invoice-images")
    parser.add_argument("--per-folder", default=5, type=int)
    args = parser.parse_args()

    parquet_files = [
        ("test", args.data_dir / "test-00000-of-00001.parquet"),
        ("train", args.data_dir / "train-00000-of-00001.parquet"),
    ]

    count = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_STORED) as zf:
        for split, parquet_path in parquet_files:
            for filename, data in iter_images(parquet_path, split):
                folder_number = (count // args.per_folder) + 1
                archive_name = (
                    f"{args.main_folder}/folder_{folder_number:04d}/{filename}"
                )
                zf.writestr(archive_name, data)
                count += 1
                if count % 500 == 0:
                    print(f"extracted {count} images")

    print(f"created {args.output}")
    print(f"images: {count}")
    print(f"folders: {(count + args.per_folder - 1) // args.per_folder}")


if __name__ == "__main__":
    main()
