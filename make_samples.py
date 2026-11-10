#!/usr/bin/env python3
"""Generate five reproducible, deliberately synthetic e-commerce test pictures.

No external image downloads or image-generation services are needed. The images
are useful for an end-to-end API smoke test, not an unbiased model benchmark.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from common import SAMPLES

W, H = 960, 720
WHITE = (248, 250, 253)
INK = (37, 50, 67)


def font(size, bold=False):
    paths = (
        ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
        if bold else
        ["/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    )
    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def label(d, xy, s, size=32, fill=INK, bold=False, anchor="mm"):
    d.text(xy, s, font=font(size, bold), fill=fill, anchor=anchor)


def canvas(title, subtitle):
    im = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((35, 28, 925, 690), radius=30, fill="white", outline=(220, 229, 238), width=3)
    label(d, (480, 85), title, 45, bold=True)
    label(d, (480, 137), subtitle, 25, fill=(89, 105, 122))
    d.line((95, 175, 865, 175), fill=(230, 235, 239), width=3)
    return im, d


def foot(d, text):
    d.rounded_rectangle((295, 610, 665, 666), 14, fill=(242, 246, 250), outline=(190, 204, 217), width=2)
    label(d, (480, 638), text, size=28, bold=True)


def red_shirt(d):
    d.polygon([(355, 225),(416, 207),(433, 234),(527, 234),(544, 207),(603, 225),
               (668, 296),(611, 356),(580, 317),(580, 560),(378, 560),(378, 317),
               (347, 356),(288, 296)], fill=(190, 38, 51))
    d.ellipse((439, 207, 520, 264), fill="white")
    d.arc((438, 206, 521, 265), 0, 180, fill=(113, 14, 25), width=6)
    d.line((381, 552, 579, 552), fill=(137, 21, 30), width=4)
    label(d, (480, 395), "NORTH", size=35, fill="white", bold=True)
    foot(d, "SIZE L  |  COTTON")


def blue_shoe(d):
    d.polygon([(195, 467),(267, 422),(336, 377),(418, 369),(472, 405),(566, 416),
               (626, 442),(730, 448),(784, 480),(785, 518),(208, 520)], fill=(40, 97, 185))
    d.rounded_rectangle((205, 501, 787, 542), radius=18, fill=(238, 241, 244), outline=(48, 64, 80), width=3)
    d.line([(295, 429),(405, 444),(500, 467)], fill=(145, 202, 242), width=13)
    for i in range(4):
        x=375+i*34
        d.line((x, 402, x+45, 442), fill="white", width=5)
    d.arc((203, 477, 778, 553), 0, 180, fill=(95, 104, 112), width=3)
    foot(d, "TRAILRUN  |  US 9")


def green_bottle(d):
    d.rounded_rectangle((405, 258, 555, 567), radius=39, fill=(46, 130, 84), outline=(24, 92, 60), width=5)
    d.rounded_rectangle((438, 225, 522, 279), radius=8, fill=(40, 99, 70))
    d.rounded_rectangle((447, 214, 513, 241), radius=4, fill=(18, 55, 42))
    d.rounded_rectangle((416, 342, 544, 470), radius=12, fill=(248, 248, 233))
    label(d, (480, 382), "PURE", 29, fill=(28, 97, 59), bold=True)
    label(d, (480, 424), "750 mL", 26, fill=INK, bold=True)
    foot(d, "STAINLESS  |  750 mL")


def headphones(d):
    d.arc((353, 242, 608, 525), 185, 355, fill=(31, 36, 46), width=30)
    d.arc((368, 262, 594, 520), 185, 355, fill=(90, 101, 115), width=5)
    d.rounded_rectangle((321, 407, 385, 552), radius=24, fill=(25, 30, 38))
    d.rounded_rectangle((574, 407, 638, 552), radius=24, fill=(25, 30, 38))
    d.rounded_rectangle((336, 428, 367, 535), radius=14, fill=(76, 85, 99))
    d.rounded_rectangle((591, 428, 621, 535), radius=14, fill=(76, 85, 99))
    label(d, (480, 524), "AURORA", 28, fill=INK, bold=True)
    foot(d, "MODEL H200  |  BLACK")


def yellow_backpack(d):
    d.arc((388, 215, 576, 445), 180, 360, fill=(117, 93, 24), width=22)
    d.rounded_rectangle((343, 267, 616, 564), radius=62, fill=(227, 176, 35), outline=(150, 115, 24), width=6)
    d.rounded_rectangle((390, 380, 570, 517), radius=24, fill=(246, 202, 67), outline=(166, 126, 23), width=4)
    d.line((404, 399, 554, 399), fill=(111, 91, 27), width=5)
    label(d, (480, 339), "TRAVEL", 31, fill=(70, 57, 16), bold=True)
    label(d, (480, 453), "20L", 42, fill=(94, 71, 12), bold=True)
    foot(d, "BACKPACK  |  20 L")


SPECS = [
    ("red_cotton_tshirt.png", "RED COTTON T-SHIRT", "Casual clothing", red_shirt,
     "red cotton T-shirt, size L, printed NORTH logo", "clothing", "red", "L", "NORTH"),
    ("blue_running_shoe.png", "BLUE RUNNING SHOE", "Athletic footwear", blue_shoe,
     "blue running shoe, US size 9, TRAILRUN", "footwear", "blue", "US 9", "TRAILRUN"),
    ("green_water_bottle.png", "GREEN WATER BOTTLE", "Reusable drinkware", green_bottle,
     "green stainless steel water bottle, 750 mL, PURE", "drinkware", "green", "750 mL", "PURE"),
    ("black_headphones.png", "BLACK HEADPHONES", "Over-ear audio accessories", headphones,
     "black over-ear headphones, AURORA model H200", "electronics", "black", "H200", "AURORA"),
    ("yellow_backpack.png", "YELLOW BACKPACK", "Travel accessory", yellow_backpack,
     "yellow travel backpack, 20 L capacity, TRAVEL", "bags", "yellow", "20 L", "TRAVEL"),
]


def main():
    SAMPLES.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, (filename, title, subtitle, draw_fn, query, category, color, size, branding) in enumerate(SPECS, 1):
        im, d = canvas(title, subtitle)
        draw_fn(d)
        im.save(SAMPLES / filename)
        manifest.append({
            "id": i,
            "file": filename,
            "text_query": query,
            "expected_attributes": {
                "product_type": {
                    "clothing": "T-shirt", "footwear": "running shoe", "drinkware": "water bottle",
                    "electronics": "headphones", "bags": "backpack"
                }[category],
                "primary_color": color,
                "size_or_capacity": None if category == "electronics" else size,
                "model_number": "H200" if category == "electronics" else None,
                "brand_or_printed_logo": branding,
            },
            "expected_printed_text": branding,
        })
        print(f"Created {SAMPLES / filename}")
    (SAMPLES / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (SAMPLES / "texts.txt").write_text("\n".join(s["text_query"] for s in manifest) + "\n", encoding="utf-8")
    print("Created manifest.json and texts.txt (5 image/text pairs)")


if __name__ == "__main__":
    main()
