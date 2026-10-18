"""Figure: the reference setup of each region, with the boxes of the application-suggested method
and the type the operator gives each feature.

argv: <test photo dir> <testset_export.json> <out png>
"""
import json
import sys

from PIL import Image, ImageDraw, ImageFont, ImageOps

PHOTOS, EXPORT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
E = json.load(open(EXPORT))["regions"]
REGIONS = [("4 screws", "a) Region 1, Hybrid v18"), ("3 thread holes", "b) Region 2, Hybrid v8"),
           ("3 holes", "c) Region 3, Hybrid v1")]
NAMES = {("4 screws", "screw"): "Threaded stud", ("4 screws", "stud"): "Stud",
         ("3 thread holes", "threaded_hole"): "Type A PEM",
         ("3 holes", "threaded_hole"): "Type C PEM", ("3 holes", "bolt_head"): "Button rivet"}
COLOURS = {"Threaded stud": (46, 125, 50), "Stud": (123, 31, 162), "Type A PEM": (21, 101, 192),
           "Type C PEM": (21, 101, 192), "Button rivet": (230, 81, 0)}
H = 900
font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 30)
cap = ImageFont.truetype("C:/Windows/Fonts/times.ttf", 40)

panels = []
for key, label in REGIONS:
    img = ImageOps.exif_transpose(Image.open(f"{PHOTOS}/{key} normal.jpg")).convert("RGB")
    img = img.resize((round(img.width * H / img.height), H), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    for f in E[key]["reference"]["features"]:
        name = NAMES[(key, f["type"])]
        x0, y0, x1, y1 = f["rect"]
        box = [x0 * img.width, y0 * img.height, x1 * img.width, y1 * img.height]
        d.rectangle(box, outline=COLOURS[name], width=6)
        tw = d.textlength(name, font=font)
        ty = box[1] - 40 if box[1] > 45 else box[3] + 6
        d.rectangle([box[0], ty, box[0] + tw + 12, ty + 38], fill=COLOURS[name])
        d.text((box[0] + 6, ty + 3), name, fill="white", font=font)
    canvas = Image.new("RGB", (img.width, H + 70), "white")
    canvas.paste(img, (0, 0))
    cd = ImageDraw.Draw(canvas)
    cd.text(((img.width - cd.textlength(label, font=cap)) / 2, H + 14), label, fill="black", font=cap)
    panels.append(canvas)

gap = 30
out = Image.new("RGB", (sum(p.width for p in panels) + gap * (len(panels) - 1), panels[0].height), "white")
x = 0
for p in panels:
    out.paste(p, (x, 0))
    x += p.width + gap
out.save(OUT)
print("saved", OUT, out.size)
