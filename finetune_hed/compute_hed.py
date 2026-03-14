"""Compute HED edge maps from downloaded LAION images."""

import argparse
import json
from pathlib import Path

import torch
from controlnet_aux import HEDdetector
from PIL import Image
from tqdm import tqdm


def find_images(images_dir: str):
    """Find all images in the download directory (handles img2dataset flat/shard layouts)."""
    images_dir = Path(images_dir)
    entries = []

    # img2dataset with output_format=files creates {shard_id}/{index}.png + {index}.txt
    for img_path in sorted(images_dir.rglob("*.png")):
        # Look for caption in companion .txt file
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists():
            caption = txt_path.read_text().strip()
        else:
            caption = ""
        entries.append({"image_path": str(img_path), "caption": caption})

    # Also check for jpg/jpeg
    for ext in ["*.jpg", "*.jpeg"]:
        for img_path in sorted(images_dir.rglob(ext)):
            txt_path = img_path.with_suffix(".txt")
            caption = txt_path.read_text().strip() if txt_path.exists() else ""
            entries.append({"image_path": str(img_path), "caption": caption})

    return entries


def main():
    parser = argparse.ArgumentParser(description="Compute HED edge maps")
    parser.add_argument("--images_dir", type=str, default="./finetune_hed/data/images_raw")
    parser.add_argument("--output_dir", type=str, default="./finetune_hed/data/hed_512")
    parser.add_argument("--resolution", type=int, default=512)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading HED detector...")
    hed = HEDdetector.from_pretrained("lllyasviel/Annotators")

    print(f"Scanning images in {args.images_dir}...")
    entries = find_images(args.images_dir)
    print(f"Found {len(entries)} images.")

    # Resume support: load existing metadata and skip already-processed images
    metadata_path = output_dir / "metadata.jsonl"
    metadata = []
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = [json.loads(line.strip()) for line in f if line.strip()]
        print(f"Resuming: {len(metadata)} images already processed.")
    idx = len(metadata)
    already_done = set(range(idx))

    for i, entry in enumerate(tqdm(entries, desc="Computing HED edges")):
        if i < idx:
            continue  # skip already processed

        try:
            img = Image.open(entry["image_path"]).convert("RGB")
            img = img.resize((args.resolution, args.resolution), Image.LANCZOS)
        except Exception as e:
            print(f"  Skipping {entry['image_path']}: {e}")
            continue

        edge_map = hed(img)

        # Save as grayscale PNG
        if edge_map.mode != "L":
            edge_map = edge_map.convert("L")
        # Convert back to RGB (3-channel grayscale) for consistency with training
        edge_map_rgb = Image.merge("RGB", [edge_map, edge_map, edge_map])

        filename = f"{idx:06d}.png"
        edge_map_rgb.save(output_dir / filename)

        metadata.append({
            "file_name": filename,
            "caption": entry["caption"],
        })
        idx += 1

        # Save metadata incrementally every 1000 images
        if idx % 1000 == 0:
            with open(metadata_path, "w") as f:
                for m in metadata:
                    f.write(json.dumps(m) + "\n")

    # Write final metadata
    with open(metadata_path, "w") as f:
        for m in metadata:
            f.write(json.dumps(m) + "\n")

    print(f"Done. Saved {len(metadata)} HED edge maps to {output_dir}")
    print(f"Metadata written to {metadata_path}")


if __name__ == "__main__":
    main()
