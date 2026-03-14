"""Download LAION-Aesthetics images with captions for HED edge map fine-tuning."""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def filter_and_save_urls(output_parquet: str, num_samples: int = 200_000, min_aesthetic: float = 5.0):
    """Load LAION-Aesthetics metadata, filter by aesthetic score, sample, and save parquet."""
    print(f"Loading LAION-Aesthetics (streaming), filtering aesthetic_score >= {min_aesthetic}...")

    ds = load_dataset("laion/laion2B-en-aesthetic", split="train", streaming=True)

    rows = []
    seen = 0
    for example in ds:
        seen += 1
        if seen == 1:
            print(f"  First row keys: {list(example.keys())}")
            print(f"  First row sample: { {k: str(v)[:80] for k, v in example.items()} }")
        if seen % 100_000 == 0:
            print(f"  Scanned {seen:,} rows, collected {len(rows):,} so far...")
        # Field name is "aesthetic" in laion2B-en-aesthetic
        score = example.get("aesthetic") or example.get("AESTHETIC_SCORE") or example.get("aesthetic_score") or 0
        if float(score) >= min_aesthetic:
            url = example.get("URL") or example.get("url") or ""
            caption = example.get("TEXT") or example.get("text") or ""
            if url:
                rows.append({"url": url, "caption": caption})
            if len(rows) >= num_samples:
                break

    print(f"Collected {len(rows):,} URLs from {seen:,} scanned rows.")

    df = pd.DataFrame(rows)
    Path(output_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    print(f"Saved filtered URLs to {output_parquet}")
    return output_parquet


def download_images(url_parquet: str, output_dir: str, image_size: int = 512, num_threads: int = 64):
    """Use img2dataset to download and resize images."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        "img2dataset",
        "--url_list", url_parquet,
        "--output_folder", output_dir,
        "--output_format", "files",
        "--input_format", "parquet",
        "--url_col", "url",
        "--caption_col", "caption",
        "--image_size", str(image_size),
        "--resize_mode", "center_crop",
        "--number_sample_per_shard", "10000",
        "--thread_count", str(num_threads),
        "--encode_format", "jpg",
        "--encode_quality", "95",
    ]

    print(f"Running img2dataset: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Download complete. Images saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download LAION-Aesthetics images")
    parser.add_argument("--data_dir", type=str, default="./finetune_hed/data")
    parser.add_argument("--num_samples", type=int, default=200_000)
    parser.add_argument("--min_aesthetic", type=float, default=5.0)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_threads", type=int, default=64)
    parser.add_argument("--skip_filter", action="store_true", help="Skip URL filtering (use existing parquet)")
    parser.add_argument("--skip_download", action="store_true", help="Skip download (use existing images)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    url_parquet = str(data_dir / "laion_urls.parquet")
    images_dir = str(data_dir / "images_raw")

    if not args.skip_filter:
        filter_and_save_urls(url_parquet, args.num_samples, args.min_aesthetic)
    else:
        print(f"Skipping URL filtering, using existing {url_parquet}")

    if not args.skip_download:
        download_images(url_parquet, images_dir, args.image_size, args.num_threads)
    else:
        print(f"Skipping download, using existing {images_dir}")


if __name__ == "__main__":
    main()
