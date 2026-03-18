"""Generate a PDF report comparing No-LoRA vs LoRA sweep results."""
import os
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.units import inch, cm
from reportlab.platypus import SimpleDocTemplate, Image, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from PIL import Image as PILImage

base = os.path.dirname(os.path.abspath(__file__))
no_lora_dir = os.path.join(base, "sweep_no_lora_44228191")
lora_dir = os.path.join(base, "sweep_lora_44228192")
out_pdf = os.path.join(base, "sweep_guidance_report.pdf")

page_w, page_h = landscape(A4)
doc = SimpleDocTemplate(out_pdf, pagesize=landscape(A4),
                        leftMargin=1*cm, rightMargin=1*cm,
                        topMargin=1*cm, bottomMargin=1*cm)

styles = getSampleStyleSheet()
title_style = ParagraphStyle('Title2', parent=styles['Title'], fontSize=18, spaceAfter=12)
h1 = ParagraphStyle('H1', parent=styles['Heading1'], fontSize=14, spaceAfter=6, spaceBefore=12)
h2 = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=12, spaceAfter=4, spaceBefore=8,
                     alignment=TA_CENTER)
caption = ParagraphStyle('Caption', parent=styles['Normal'], fontSize=9, alignment=TA_CENTER,
                         spaceAfter=8)

zetas = ["0_0", "0_2", "1_0", "5_0"]
zeta_labels = ["0 (no guidance)", "0.2", "1.0", "5.0"]

def fit_image(path, max_w, max_h):
    """Return Image scaled to fit within max_w x max_h while preserving aspect ratio."""
    img = PILImage.open(path)
    w, h = img.size
    scale = min(max_w / w, max_h / h)
    return Image(path, width=w * scale, height=h * scale)

story = []

# Title page
story.append(Paragraph("DPS Guidance Sweep Report", title_style))
story.append(Spacer(1, 0.3*inch))
story.append(Paragraph("Comparing No-LoRA vs LoRA across DPS guidance strengths (ζ)", styles['Normal']))
story.append(Spacer(1, 0.2*inch))
story.append(Paragraph("Each row shows pred_x0 decoded at every denoising step (1→30).<br/>"
                        "Below each row: 5 sprinter-generated photos every 3 steps.<br/>"
                        "CFG=7.5, strength=1.0 (full 30 steps from noise), HED scribble, 20 targets, 20 variations.",
                        styles['Normal']))
story.append(Spacer(1, 0.3*inch))

# Source face & scribble
story.append(Paragraph("Source Face & Scribble", h1))
usable_w = page_w - 2*cm
for label, d in [("No LoRA", no_lora_dir), ("LoRA", lora_dir)]:
    story.append(Paragraph(f"{label} — Source Face & Scribble", h2))
    for fname in ["source_face.png", "scribble.png"]:
        p = os.path.join(d, fname)
        if os.path.exists(p):
            story.append(fit_image(p, usable_w * 0.3, 4*cm))
    story.append(Spacer(1, 0.2*inch))

story.append(PageBreak())

# Per-zeta comparison pages
for z, z_label in zip(zetas, zeta_labels):
    story.append(Paragraph(f"ζ = {z_label}", title_style))
    story.append(Spacer(1, 0.2*inch))

    for label, d in [("No LoRA", no_lora_dir), ("LoRA", lora_dir)]:
        fname = f"sweep_zeta_{z}.png"
        path = os.path.join(d, fname)
        if os.path.exists(path):
            story.append(Paragraph(f"{label}", h2))
            story.append(fit_image(path, usable_w, page_h * 0.38))
            story.append(Paragraph(f"{label}, ζ={z_label} — pred_x0 trajectory + sprinter photos", caption))
            story.append(Spacer(1, 0.1*inch))

    story.append(PageBreak())

doc.build(story)
print(f"PDF saved to {out_pdf}")
