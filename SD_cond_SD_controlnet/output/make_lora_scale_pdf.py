"""Build a PDF report from the LoRA scale sweep results."""
import os
from PIL import Image, ImageDraw, ImageFont

BASE = os.path.join(os.path.dirname(__file__), "sweep_lora_scale_44228647")
SCALES = ["0_0", "0_25", "0_5", "0_75", "1_0"]
SCALE_LABELS = ["0.0", "0.25", "0.5", "0.75", "1.0"]
STEPS = list(range(1, 16))
SPRINTER_STEPS = [1, 4, 7, 10, 13]
THUMB = 140  # thumbnail size for pred_x0
STHUMB = 140  # thumbnail size for sprinter
MARGIN = 40
LABEL_W = 220
GAP = 4
FONT_SIZE = 16
TITLE_SIZE = 24

def load_font(size):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except Exception:
        return ImageFont.load_default()

def make_page():
    """Create the full composite image."""
    font = load_font(FONT_SIZE)
    title_font = load_font(TITLE_SIZE)
    small_font = load_font(12)

    # Layout: for each scale, 2 rows:
    #   Row 1: pred_x0 steps 1-15 (15 thumbs)
    #   Row 2: sprinter photos at steps 1,4,7,10,13 (5 x 5 = 25 thumbs, but show 5 groups of 5)
    # Actually simpler: pred_x0 row + sprinter row with 5 groups

    n_pred_cols = 15
    n_sprinter_groups = 5
    n_sprinter_per_group = 5

    # Width: label + 15 pred_x0 thumbs
    page_w = LABEL_W + n_pred_cols * (THUMB + GAP) + MARGIN * 2

    # For each scale: title line (30px) + pred_x0 row + gap + sprinter label (20px) + sprinter row + spacing
    scale_block_h = 30 + THUMB + 10 + 20 + STHUMB + 30

    # Source section at top
    source_h = THUMB + 60

    total_h = MARGIN + source_h + len(SCALES) * scale_block_h + MARGIN
    img = Image.new("RGB", (page_w, total_h), "white")
    draw = ImageDraw.Draw(img)

    y = MARGIN

    # Title
    draw.text((MARGIN, y), "LoRA Scale Sweep — Architect pred_x0 + Sprinter Photos",
              fill="black", font=title_font)
    y += 35

    # Source face + scribble
    source_face = Image.open(os.path.join(BASE, "source_face.png")).resize((THUMB, THUMB))
    scribble = Image.open(os.path.join(BASE, "scribble.png")).resize((THUMB, THUMB))
    draw.text((MARGIN, y), "Source face:", fill="black", font=font)
    img.paste(source_face, (MARGIN + 120, y))
    draw.text((MARGIN + 120 + THUMB + 20, y), "HED scribble:", fill="black", font=font)
    img.paste(scribble, (MARGIN + 120 + THUMB + 20 + 130, y))
    y += THUMB + 25

    # Each scale
    for si, (scale_dir, scale_label) in enumerate(zip(SCALES, SCALE_LABELS)):
        folder = os.path.join(BASE, f"scale_{scale_dir}")

        # Scale title
        draw.text((MARGIN, y), f"LoRA Scale = {scale_label}", fill="black", font=title_font)
        y += 30

        # Pred_x0 row with step labels
        draw.text((MARGIN, y + THUMB // 2 - 8), "pred x0", fill="gray", font=font)
        for step in STEPS:
            col_x = LABEL_W + (step - 1) * (THUMB + GAP) + MARGIN
            path = os.path.join(folder, f"pred_x0_step_{step:02d}.png")
            if os.path.exists(path):
                thumb = Image.open(path).resize((THUMB, THUMB))
                img.paste(thumb, (col_x, y))
            # Step number on top (only for first scale)
            if si == 0:
                draw.text((col_x + THUMB // 2 - 5, y - 15), str(step),
                          fill="gray", font=small_font)
        y += THUMB + 10

        # Sprinter row — 5 groups of 5 photos spread across the 15 columns
        draw.text((MARGIN, y + STHUMB // 2 - 8), "sprinter", fill="gray", font=font)
        for gi, sp_step in enumerate(SPRINTER_STEPS):
            for pi in range(n_sprinter_per_group):
                path = os.path.join(folder, f"sprinter_step_{sp_step:02d}_img{pi+1}.png")
                # Place: group index * 3 columns + photo index mapped within 3 cols
                col_idx = gi * 3 + pi  # 5 photos spread starting at col gi*3
                if col_idx >= n_pred_cols:
                    break
                col_x = LABEL_W + col_idx * (THUMB + GAP) + MARGIN
                if os.path.exists(path):
                    thumb = Image.open(path).resize((THUMB, THUMB))
                    img.paste(thumb, (col_x, y))
        y += STHUMB + 30

    return img


def main():
    img = make_page()
    # Save as PNG (can convert to PDF)
    png_path = os.path.join(os.path.dirname(BASE), "lora_scale_sweep_report.png")
    img.save(png_path)
    print(f"Saved {png_path} ({img.size[0]}x{img.size[1]})")

    # Also save as PDF
    pdf_path = png_path.replace(".png", ".pdf")
    img.save(pdf_path, "PDF", resolution=100.0)
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
