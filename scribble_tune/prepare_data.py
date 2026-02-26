"""Download QuickDraw stroke NDJSON and render to 512x512 PNGs."""

import argparse
import json
import random
from pathlib import Path

import yaml
from PIL import Image, ImageDraw
from quickdraw import QuickDrawData


def render_strokes(strokes, size=512):
    """Render QuickDraw strokes to a PIL image at the given resolution.

    QuickDraw raw coordinates are in [0, 255]. We scale them to fill the
    target resolution with a small margin.
    """
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)

    scale = (size - 20) / 255.0  # leave 10px margin on each side
    offset = 10

    for stroke in strokes:
        xs, ys = stroke
        points = [
            (x * scale + offset, y * scale + offset)
            for x, y in zip(xs, ys)
        ]
        if len(points) >= 2:
            draw.line(points, fill="black", width=max(2, size // 256))

    return img


def download_and_render(category, max_samples, output_dir, size=512):
    """Download drawings for a category and render them as PNGs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    qd = QuickDrawData()
    group = qd.get_drawing_group(category)

    # The quickdraw library returns a generator; collect and sample
    drawings = list(group.drawings)
    if max_samples and len(drawings) > max_samples:
        drawings = random.sample(drawings, max_samples)

    total = len(drawings)
    rendered = []
    for i, drawing in enumerate(drawings):
        if not drawing.recognized:
            continue

        img = render_strokes(drawing.strokes, size=size)
        fname = f"{category}_{i:06d}.png"
        img.save(output_dir / fname)
        rendered.append(fname)

        if (i + 1) % 1000 == 0:
            print(f"  {category}: rendered {i + 1}/{total}")

    print(f"  {category}: {len(rendered)} images saved")
    return rendered


def main():
    parser = argparse.ArgumentParser(description="Prepare QuickDraw data for LoRA training")
    parser.add_argument("--config", type=str, default="scribble_tune/config.yaml")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory (e.g. Google Drive mount path)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(args.output_dir or cfg["data_dir"])
    size = cfg["resolution"]
    categories = cfg["categories"]
    samples_per_cat = cfg["samples_per_category"]
    face_max = cfg["face_max_samples"]

    random.seed(cfg["seed"])

    all_files = []
    for cat in categories:
        max_samples = face_max if cat == "face" else samples_per_cat
        print(f"Processing category: {cat} (max {max_samples})")
        files = download_and_render(cat, max_samples, output_dir, size=size)
        all_files.extend(files)

    # Write metadata.jsonl
    caption = cfg["caption"]
    metadata_path = output_dir / "metadata.jsonl"
    with open(metadata_path, "w") as f:
        for fname in all_files:
            json.dump({"file_name": fname, "text": caption}, f)
            f.write("\n")

    print(f"\nDone. Total images: {len(all_files)}")
    print(f"Metadata written to: {metadata_path}")


if __name__ == "__main__":
    main()
