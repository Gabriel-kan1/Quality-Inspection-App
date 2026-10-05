"""
Where every installed piece of hardware is in each bundled photograph, and what type it is.

detection_ground_truth.py holds counts, and a count cannot tell a right box from a wrong one:
a method that puts four boxes on four scratches scores exact on a four-fastener plate. Every
study so far inherits that limit. A precision/recall figure for the whole pipeline needs boxes
with types: did a box land on each fastener (detection), and did its crop get the right type
(classification)? Any replacement for the edge stage will be judged against the same boxes.

    python python/label_boxes.py                  # label; resumes at the first photo not done
    python python/label_boxes.py --photo one.jpg  # start at a given photograph
    python python/label_boxes.py --check          # summary, cross-checked against the hand counts

Needs opencv-python and numpy, like the other studies; the window is tkinter, from the
standard library. Labels are saved after every change to python/detection_boxes.json, which
is meant to be committed.

WHAT TO LABEL is what detection_ground_truth.py counted: the installed hardware a competent
inspector would list - heads, studs, screws, rivets, holes. Not scratches, grime, glare or a
part lying loose on the plate. One tight box per fastener: a stud is one box, cap and shaft
together, and two fasteners touching are two boxes. Where either answer is defensible (the
loose pin in screws_flash.jpg and screws_none.jpg), draw the box and mark it optional: an
evaluation then neither requires a detector to find it nor penalises one that does.

No boxes are pre-filled from a detector. A box made by adjusting a detector's box inherits
its extents, and an evaluation would then favour the detector the labels started from.

FORMAT. Pixels in the upright photograph: EXIF rotation applied, as cv2.imread does by default
and as the app's HardwareFeatureExtractor.loadBitmap does. Six of the bundled photographs are
stored on their side, so this matters. A box is [x0, y0, x1, y1] with x1 and y1 exclusive
(width = x1 - x0), the app's RectF convention before it divides by the image size. Each
photograph also records its size and a SHA-256 prefix of the file, because the labels name
photographs by file name: a different photograph saved under the same name is reported rather
than silently scored against the old boxes. load() returns the labels for other studies.
"""
import argparse
import copy
import hashlib
import json
import os
import sys

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - dependency hint only
    sys.exit(f"missing dependency: {exc}. pip install opencv-python numpy")

try:
    import tkinter as tk
    from tkinter import font as tkfont, messagebox
except ImportError:  # a Python without Tk can still load() the labels and run --check
    tk = None

import detection_ground_truth as GT

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IMG_DIR = os.path.join(ROOT, "app", "src", "main", "assets", "chassis_images")
OUT = os.path.join(HERE, "detection_boxes.json")

FORMAT = 1
# The app's five types (ReferenceLibrary.FEATURE_TYPES), then installed hardware that is none
# of them. Keys 1-6 in the window, in this order.
TYPES = ["bolt_head", "hole", "rivet_head", "screw", "stud", "other"]
COLOURS = {"bolt_head": "#ff6b6b", "hole": "#4dabf7", "rivet_head": "#ffd43b",
           "screw": "#69db7c", "stud": "#da77f2", "other": "#ff922b"}
PHOTO_EXTS = (".jpg", ".jpeg", ".png")


# ── the labels ───────────────────────────────────────────────────

def photo_order(names):
    """Reference-distance photographs first, as they are what the detector is built for, then
    the whole-panel shots, then anything detection_ground_truth.py does not know."""
    return sorted(names, key=lambda n: (0 if n in GT.GT else 1 if n in GT.WIDE else 2,
                                        n.lower()))


def photos(img_dir):
    """Every photograph in img_dir that the hand counts do not exclude."""
    return photo_order(n for n in os.listdir(img_dir)
                       if n.lower().endswith(PHOTO_EXTS) and n not in GT.EXCLUDED)


def read_photo(path):
    """Upright BGR pixels. cv2.imread applies EXIF rotation, as the app does."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f"cannot read {path}")
    return img


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def hand_count(name):
    """(acceptable counts, approximate) from detection_ground_truth.py, or (None, False)."""
    if name in GT.GT:
        return GT.GT[name], False
    if name in GT.WIDE:
        return GT.WIDE[name], True
    return None, False


def describe(counts):
    return " or ".join(str(c) for c in sorted(counts))


def split_counts(entry):
    """(required, optional) box counts."""
    opt = sum(1 for b in entry["boxes"] if b["optional"])
    return len(entry["boxes"]) - opt, opt


def agrees(entry, counts):
    """Whether the boxes read as one of the acceptable counts: every required box plus any
    number of the optional ones."""
    req, opt = split_counts(entry)
    return any(req <= c <= req + opt for c in counts)


def load(path=OUT):
    """The labels by photograph name, {} before the first save:

        {"size": [w, h], "sha256": str, "done": bool,
         "boxes": [{"type": str, "xyxy": [x0, y0, x1, y1], "optional": bool}, ...]}

    Only photographs marked done are finished; an evaluation should skip the rest."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("format") != FORMAT:
        raise ValueError(f"{path}: format {data.get('format')!r}, expected {FORMAT}")
    images = data["images"]
    for name, e in images.items():
        w, h = e["size"]
        for b in e["boxes"]:
            b.setdefault("optional", False)
            x0, y0, x1, y1 = b["xyxy"]
            if b["type"] not in TYPES:
                raise ValueError(f"{path}: {name}: unknown type {b['type']!r}")
            if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
                raise ValueError(f"{path}: {name}: box {b['xyxy']} is not inside {w}x{h}")
    return images


def dumps(images):
    """JSON with one box per line, so a diff of the file reads as a diff of the labels."""
    lines = ["{", f'  "format": {FORMAT},', f'  "types": {json.dumps(TYPES)},', '  "images": {']
    names = photo_order(images)
    for i, name in enumerate(names):
        e = images[name]
        head = json.dumps({"size": e["size"], "sha256": e["sha256"], "done": e["done"]})
        lines.append(f'    {json.dumps(name)}: {head[:-1]}, "boxes": [')
        for j, b in enumerate(e["boxes"]):
            box = {"type": b["type"], "xyxy": b["xyxy"], "optional": b["optional"]}
            lines.append("      " + json.dumps(box) + ("," if j < len(e["boxes"]) - 1 else ""))
        lines.append("    ]}" + ("," if i < len(names) - 1 else ""))
    lines += ["  }", "}"]
    return "\n".join(lines) + "\n"


def save(images, path=OUT):
    """Writes through a temporary file, so a crash mid-write cannot cost the labels."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(dumps(images))
    os.replace(tmp, path)


def check(images, img_dir):
    """Prints each photograph's labels beside its hand count. Returns how many photographs
    marked done disagree with detection_ground_truth.py."""
    on_disk = set(photos(img_dir)) if os.path.isdir(img_dir) else set()
    names = photo_order(set(images) | on_disk | set(GT.GT) | set(GT.WIDE))
    totals = {t: [0, 0] for t in TYPES}
    disagree = 0
    print(f"{'photo':<20} {'done':<5} {'boxes':>5} {'opt':>4}  {'hand count':<10}  agrees")
    for name in names:
        e = images.get(name)
        done = bool(e and e["done"])
        req, opt = split_counts(e) if e else ("-", "-")
        counts, approx = hand_count(name)
        hand = "-" if counts is None else ("~" if approx else "") + describe(counts)
        verdict = "-"
        if done and counts is not None and not approx:
            ok = agrees(e, counts)
            verdict = "yes" if ok else "NO"
            disagree += not ok
        notes = []
        if name not in on_disk:
            notes.append("photo not on disk")
        elif e and e["sha256"] != file_digest(os.path.join(img_dir, name)):
            notes.append("file differs from the one labelled")
        if done:
            for b in e["boxes"]:
                totals[b["type"]][int(b["optional"])] += 1
        print(f"{name:<20} {'yes' if done else 'no':<5} {req:>5} {opt:>4}  {hand:<10}  {verdict}"
              + (f"  ({'; '.join(notes)})" if notes else ""))

    ndone = sum(1 for n in on_disk if images.get(n, {}).get("done"))
    print(f"\n{ndone} of {len(on_disk)} photographs on disk are done. Boxes in them, by type:")
    for t in TYPES:
        req, opt = totals[t]
        print(f"  {t:<11} {req:>4}" + (f"  (+{opt} optional)" if opt else ""))
    if disagree:
        print(f"\n{disagree} done photograph(s) disagree with detection_ground_truth.py. Look"
              " again, and if the count there is what is wrong, correct it so the studies agree.")
    return disagree


# ── the window ───────────────────────────────────────────────────

BG, PANEL_FG, MUTED = "#1e1e1e", "#e8e8e8", "#9a9a9a"
CANVAS_BG, LIST_BG = "#202020", "#2a2a2a"
GOOD, BAD = "#69db7c", "#ff6b6b"

GUIDANCE = ("Label installed hardware only: heads, studs, screws, rivets, holes. Not scratches, "
            "grime, glare or loose parts. One tight box per fastener; a stud is one box. Mark a "
            "box optional where either answer is defensible.")

KEYS = """\
drag         draw a box    Del     delete box
shift+drag   draw on a box Tab     next box
click        select        Ctrl+Z  undo
drag inside  move          wheel   zoom
drag edge    resize        f       fit photo
1-6          set type      right-drag  pan
o            optional
Enter        done, then the next photograph
n / p        next / previous photograph"""


def _dpi_aware():
    """Without this Windows bitmap-scales the window on a high-DPI screen and the photo blurs."""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass


class Labeller:
    """The window. Every change is saved as it is made, so closing it never loses work."""

    GRAB = 7       # canvas px at 100% scaling: how near a corner or edge of the selection grabs it
    CLICK = 4      # canvas px: a press that moves less than this is a click, not a drag
    MIN_BOX = 3    # photo px: a smaller drag is not kept as a box

    def __init__(self, root, img_dir, out, names, images, start):
        self.root, self.img_dir, self.out = root, img_dir, out
        self.names, self.images = names, images
        self.k = root.winfo_fpixels("1i") / 96.0      # display scaling, for sizes in pixels
        try:
            self.out_label = os.path.relpath(out, ROOT)
        except ValueError:                             # on another drive
            self.out_label = out
        self.undo = {}
        self.cur_type = TYPES[0]
        self.sel = self.drag = self.pan = self.pan_job = self.cursor = self.photo = None
        self.fitted, self.render_queued = True, False
        self.save_note, self.flash_note, self.flash_job = "Nothing changed yet", None, None
        self.warning = ""
        self._build()
        self.open(start)

    # ── layout ──

    def _build(self):
        root, k = self.root, self.k
        root.title("Box labelling")
        root.configure(bg=BG)
        try:
            root.state("zoomed")
        except tk.TclError:
            root.geometry(f"{int(1400 * k)}x{int(900 * k)}")
        self.f_base = tkfont.nametofont("TkDefaultFont").copy()
        self.f_bold = self.f_base.copy()
        self.f_bold.configure(weight="bold")
        self.f_title = self.f_base.copy()
        self.f_title.configure(size=13, weight="bold")
        self.f_small = self.f_base.copy()
        self.f_small.configure(size=8)
        self.f_head = self.f_base.copy()
        self.f_head.configure(size=8, weight="bold")
        self.f_mono = tkfont.nametofont("TkFixedFont").copy()
        self.f_mono.configure(size=8)

        side = tk.Frame(root, width=int(340 * k), bg=BG, padx=int(14 * k), pady=int(12 * k))
        side.pack(side="right", fill="y")
        side.pack_propagate(False)
        wrap = int(310 * k)

        c = self.canvas = tk.Canvas(root, bg=CANVAS_BG, highlightthickness=0, cursor="crosshair")
        c.pack(side="left", fill="both", expand=True)
        self.image_item = c.create_image(0, 0, anchor="nw")
        self.cross = [c.create_line(0, 0, 0, 0, fill="#ffffff", dash=(2, 4), state="hidden")
                      for _ in range(2)]

        def label(gap=0, **kw):
            opts = dict(bg=BG, fg=PANEL_FG, anchor="w", justify="left", wraplength=wrap,
                        font=self.f_base)
            opts.update(kw)
            w = tk.Label(side, **opts)
            w.pack(fill="x", pady=(int(gap * k), 0))
            return w

        self.l_status = tk.Label(side, bg=BG, fg=MUTED, anchor="w", justify="left",
                                 wraplength=wrap, font=self.f_small)
        self.l_status.pack(side="bottom", fill="x")

        self.l_name = label(font=self.f_title)
        self.l_pos = label(fg=MUTED)
        self.l_hand = label(gap=10)
        self.l_count = label()
        self.l_done = label(font=self.f_bold)
        self.l_warn = label(fg=BAD)

        label(gap=14, text="TYPE", fg=MUTED, font=self.f_head)
        self.legend = {}
        for i, t in enumerate(TYPES):
            row = tk.Frame(side, bg=BG, cursor="hand2")
            row.pack(fill="x", pady=1)
            key = tk.Label(row, text=str(i + 1), bg=BG, fg=MUTED, width=2, anchor="w",
                           font=self.f_base)
            swatch = tk.Label(row, bg=COLOURS[t], width=2)
            name = tk.Label(row, bg=BG, fg=PANEL_FG, anchor="w", font=self.f_base)
            count = tk.Label(row, bg=BG, fg=MUTED, anchor="e", font=self.f_base)
            key.pack(side="left")
            swatch.pack(side="left", padx=(0, int(8 * k)))
            name.pack(side="left")
            count.pack(side="right")
            for w in (row, key, swatch, name, count):
                w.bind("<Button-1>", lambda e, t=t: self.set_type(t))
            self.legend[t] = (name, count)

        label(gap=14, text="BOXES IN THIS PHOTOGRAPH", fg=MUTED, font=self.f_head)
        frame = tk.Frame(side, bg=BG)
        frame.pack(fill="x", pady=(int(4 * k), 0))
        self.listbox = tk.Listbox(frame, height=8, bg=LIST_BG, fg=PANEL_FG, font=self.f_base,
                                  selectbackground="#3d5a80", highlightthickness=0,
                                  borderwidth=0, activestyle="none", takefocus=0,
                                  exportselection=False)
        bar = tk.Scrollbar(frame, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=bar.set)
        self.listbox.pack(side="left", fill="x", expand=True)
        bar.pack(side="right", fill="y")
        self.listbox.bind("<<ListboxSelect>>", self.on_list_select)

        label(gap=14, text=GUIDANCE, fg=MUTED, font=self.f_small)
        label(gap=10, text=KEYS, fg=MUTED, font=self.f_mono)

        c.bind("<Configure>", lambda e: self.fit() if self.fitted else self.request_render())
        c.bind("<ButtonPress-1>", self.on_press)
        c.bind("<B1-Motion>", self.on_drag)
        c.bind("<ButtonRelease-1>", self.on_release)
        for b in (2, 3):
            c.bind(f"<ButtonPress-{b}>", self.on_pan_start)
            c.bind(f"<B{b}-Motion>", self.on_pan)
            c.bind(f"<ButtonRelease-{b}>", self.on_pan_end)
        c.bind("<MouseWheel>", lambda e: self.zoom_at(e.x, e.y, 1.25 if e.delta > 0 else 0.8))
        c.bind("<Button-4>", lambda e: self.zoom_at(e.x, e.y, 1.25))    # X11 wheel
        c.bind("<Button-5>", lambda e: self.zoom_at(e.x, e.y, 0.8))
        c.bind("<Motion>", self.on_motion)
        c.bind("<Enter>", lambda e: c.focus_set())    # wheel events follow keyboard focus
        c.bind("<Leave>", self.on_leave)
        root.bind("<KeyPress>", self.on_key)

    # ── the current photograph ──

    @property
    def name(self):
        return self.names[self.idx]

    @property
    def entry(self):
        return self.images[self.name]

    @property
    def boxes(self):
        return self.entry["boxes"]

    def open(self, idx):
        self.drag = None
        self.idx = idx % len(self.names)
        path = os.path.join(self.img_dir, self.name)
        try:
            img = read_photo(path)
        except OSError as exc:
            messagebox.showerror("Box labelling", str(exc), parent=self.root)
            del self.names[self.idx]
            if not self.names:
                self.root.destroy()
                return
            self.open(self.idx)
            return
        # Halved copies, so a zoomed-out view samples a level near its own size and fine
        # detail does not alias away.
        self.levels = [img]
        while max(self.levels[-1].shape[:2]) > 1000:
            self.levels.append(cv2.pyrDown(self.levels[-1]))
        h, w = img.shape[:2]
        self.size, self.digest = [w, h], file_digest(path)
        e = self.images.setdefault(self.name, {"size": self.size, "sha256": self.digest,
                                               "done": False, "boxes": []})
        if e["size"] != self.size:
            self.warning = (f"These boxes were drawn on a {e['size'][0]}x{e['size'][1]} "
                            f"photograph; this file is {w}x{h}. Check them: the next change "
                            "records them against this file.")
        elif e["sha256"] != self.digest:
            self.warning = ("This file is not the one these boxes were drawn on. Check them: "
                            "the next change records them against this file.")
        else:
            self.warning = ""
        self.sel = None
        self.fit()
        self.refresh()

    def go(self, step):
        self.open(self.idx + step)

    # ── changes ──

    def push_undo(self):
        stack = self.undo.setdefault(self.name, [])
        stack.append((copy.deepcopy(self.boxes), self.entry["done"]))
        del stack[:-200]

    def changed(self):
        """Records a change: re-anchors the labels to this file if it differs from the one
        they were drawn on, saves, and redraws."""
        e = self.entry
        if e["size"] != self.size or e["sha256"] != self.digest:
            w, h = self.size
            for b in e["boxes"]:
                x0, y0, x1, y1 = b["xyxy"]
                b["xyxy"] = [min(x0, w), min(y0, h), min(x1, w), min(y1, h)]
            e["boxes"][:] = [b for b in e["boxes"]
                             if b["xyxy"][2] > b["xyxy"][0] and b["xyxy"][3] > b["xyxy"][1]]
            e["size"], e["sha256"] = self.size, self.digest
            self.warning = ""
            if self.sel is not None and self.sel >= len(e["boxes"]):
                self.sel = None
        self.persist()
        self.draw_boxes()
        self.refresh()

    def persist(self):
        try:
            save(self.images, self.out)
            self.save_note = f"Saved to {self.out_label}"
        except OSError as exc:
            self.save_note = f"NOT SAVED: {exc}"
        self.update_status()

    def undo_last(self):
        stack = self.undo.get(self.name)
        if not stack:
            self.flash("Nothing to undo in this photograph")
            return
        self.entry["boxes"], self.entry["done"] = stack.pop()
        self.sel = None
        self.changed()

    def set_type(self, t):
        self.cur_type = t
        if self.sel is not None and self.boxes[self.sel]["type"] != t:
            self.push_undo()
            self.boxes[self.sel]["type"] = t
            self.changed()
        else:
            self.draw_boxes()
            self.refresh()

    def toggle_optional(self):
        if self.sel is None:
            self.flash("Select a box first")
            return
        self.push_undo()
        b = self.boxes[self.sel]
        b["optional"] = not b["optional"]
        self.changed()

    def delete_selected(self):
        if self.sel is None:
            self.flash("Select a box first")
            return
        self.push_undo()
        del self.boxes[self.sel]
        self.sel = None
        self.changed()

    def toggle_done(self):
        e = self.entry
        if not e["done"]:
            counts, approx = hand_count(self.name)
            if counts is not None and not approx and not agrees(e, counts):
                req, opt = split_counts(e)
                drawn = f"{req} box{'' if req == 1 else 'es'}" + (
                    f" plus {opt} optional" if opt else "")
                if not messagebox.askyesno(
                        "Mark done?",
                        f"{self.name} has {drawn}; detection_ground_truth.py counts "
                        f"{describe(counts)}.\n\nMark it done anyway? If the count there is "
                        "what is wrong, correct it so the studies agree.", parent=self.root):
                    return
        self.push_undo()
        e["done"] = not e["done"]
        self.changed()
        if e["done"]:
            self.go(1)

    def cycle(self, step):
        n = len(self.boxes)
        if n:
            self.sel = (0 if step > 0 else n - 1) if self.sel is None else (self.sel + step) % n
            self.draw_boxes()
            self.refresh_list()

    def cancel(self):
        """Escape: abandons a drag in progress, else clears the selection."""
        d, self.drag = self.drag, None
        if d and d["mode"] != "draw" and d["moved"]:
            self.boxes[d["i"]]["xyxy"] = d["orig"]
            self.undo[self.name].pop()            # the snapshot taken when the drag began
        else:
            self.sel = None
        self.draw_boxes()
        self.refresh()

    # ── view ──

    def canvas_size(self):
        return max(2, self.canvas.winfo_width()), max(2, self.canvas.winfo_height())

    def to_canvas(self, x, y):
        return (x - self.ox) * self.scale, (y - self.oy) * self.scale

    def to_photo(self, u, v):
        return self.ox + u / self.scale, self.oy + v / self.scale

    def clamp(self, x, y):
        """The nearest pixel edge inside the photograph."""
        w, h = self.size
        return min(max(round(x), 0), w), min(max(round(y), 0), h)

    def fit(self):
        cw, ch = self.canvas_size()
        w, h = self.size
        self.scale = min(cw / w, ch / h) * 0.96
        self.ox = w / 2 - cw / 2 / self.scale
        self.oy = h / 2 - ch / 2 / self.scale
        self.fitted = True
        self.request_render()

    def zoom_at(self, u, v, factor):
        cw, ch = self.canvas_size()
        w, h = self.size
        fit = min(cw / w, ch / h)
        x, y = self.to_photo(u, v)
        self.scale = min(max(self.scale * factor, fit * 0.5), max(fit * 8, 16.0))
        self.ox, self.oy = x - u / self.scale, y - v / self.scale
        self.fitted = False
        self.request_render()

    def request_render(self):
        """Coalesces a burst of view changes (a wheel spin, a pan) into one redraw."""
        if not self.render_queued:
            self.render_queued = True
            self.root.after_idle(self.render)

    def render(self):
        self.render_queued = False
        cw, ch = self.canvas_size()
        level = 0
        while level + 1 < len(self.levels) and self.scale * 2 ** (level + 1) <= 1:
            level += 1
        s = self.scale * 2 ** level
        # Pixel i of a level spans [i, i + 1) * 2^level in photo coordinates and the canvas
        # shows photo x at (x - ox) * scale; warpAffine works in pixel centres, hence the halves.
        t = 0.5 * s - 0.5
        m = np.float32([[s, 0, t - self.ox * self.scale], [0, s, t - self.oy * self.scale]])
        view = cv2.warpAffine(self.levels[level], m, (cw, ch),
                              flags=cv2.INTER_NEAREST if s >= 3 else cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(32, 32, 32))
        ppm = b"P6 %d %d 255 " % (cw, ch) + cv2.cvtColor(view, cv2.COLOR_BGR2RGB).tobytes()
        # One image, overwritten in place: building a new one per frame costs a third more.
        if self.photo is None or (self.photo.width(), self.photo.height()) != (cw, ch):
            self.photo = tk.PhotoImage(width=cw, height=ch)
            self.canvas.itemconfigure(self.image_item, image=self.photo)
        self.photo.put(ppm)
        self.canvas.coords(self.image_item, 0, 0)         # a pan may have slid it meanwhile
        self.draw_boxes()
        self.update_status()

    def draw_boxes(self):
        c = self.canvas
        c.delete("box")
        for i, b in enumerate(self.boxes):
            self._draw_box(i, b, i == self.sel)
        d = self.drag
        if d and d["mode"] == "draw" and d["moved"]:
            u0, v0 = self.to_canvas(*d["p0"])
            u1, v1 = self.to_canvas(*d["p1"])
            c.create_rectangle(u0, v0, u1, v1, outline=COLOURS[self.cur_type], width=2,
                               dash=(4, 3), tags="box")

    def _draw_box(self, i, b, selected):
        c, col = self.canvas, COLOURS[b["type"]]
        u0, v0 = self.to_canvas(*b["xyxy"][:2])
        u1, v1 = self.to_canvas(*b["xyxy"][2:])
        c.create_rectangle(u0, v0, u1, v1, outline=col, width=3 if selected else 2,
                           dash=(6, 4) if b["optional"] else None, tags="box")
        text = str(i + 1) if u1 - u0 < 40 else (
            f"{i + 1} {b['type']}" + (" (optional)" if b["optional"] else ""))
        above = v0 > 16 * self.k
        tag = c.create_text(u0 + 3, v0 - 3 if above else v0 + 3, text=text, fill="#111111",
                            anchor="sw" if above else "nw", font=self.f_small, tags="box")
        x0, y0, x1, y1 = c.bbox(tag)
        c.tag_lower(c.create_rectangle(x0 - 3, y0, x1 + 3, y1, fill=col, outline=col,
                                       tags="box"), tag)
        if selected:
            um, vm, r = (u0 + u1) / 2, (v0 + v1) / 2, 4 * self.k
            for hu, hv in ((u0, v0), (um, v0), (u1, v0), (u1, vm),
                           (u1, v1), (um, v1), (u0, v1), (u0, vm)):
                c.create_rectangle(hu - r, hv - r, hu + r, hv + r, fill="#ffffff",
                                   outline=col, tags="box")

    # ── panel ──

    def refresh(self):
        e = self.entry
        req, opt = split_counts(e)
        ndone = sum(1 for n in self.names if self.images.get(n, {}).get("done"))
        self.root.title(f"Box labelling - {self.name} ({self.idx + 1} of {len(self.names)})")
        self.l_name.configure(text=self.name)
        self.l_pos.configure(text=f"Photograph {self.idx + 1} of {len(self.names)}   "
                                  f"{ndone} done")

        counts, approx = hand_count(self.name)
        if counts is None:
            self.l_hand.configure(text="No hand count for this photograph.", fg=MUTED)
        elif approx:
            self.l_hand.configure(text=f"Hand count: about {describe(counts)} (whole panel, "
                                       "approximate).", fg=MUTED)
        else:
            self.l_hand.configure(text=f"Hand count: {describe(counts)}.", fg=PANEL_FG)

        text = f"Boxes: {req}" + (f" + {opt} optional" if opt else "")
        colour = PANEL_FG
        if counts is not None and not approx:
            if agrees(e, counts):
                text, colour = text + "   ✓ matches the hand count", GOOD
            elif e["done"]:
                text, colour = text + "   ✗ does not match the hand count", BAD
        self.l_count.configure(text=text, fg=colour)
        self.l_done.configure(text="DONE" if e["done"] else "Not done (Enter marks it done)",
                              fg=GOOD if e["done"] else MUTED)
        self.l_warn.configure(text=self.warning)

        for t, (name, count) in self.legend.items():
            n = sum(1 for b in e["boxes"] if b["type"] == t)
            current = t == self.cur_type
            name.configure(text=("▶ " if current else "   ") + t,
                           font=self.f_bold if current else self.f_base)
            count.configure(text=str(n) if n else "")
        self.refresh_list()
        self.update_status()

    def refresh_list(self):
        lb = self.listbox
        lb.delete(0, "end")
        for i, b in enumerate(self.boxes):
            x0, y0, x1, y1 = b["xyxy"]
            lb.insert("end", f"{i + 1:>2}  {b['type']}" + ("  optional" if b["optional"] else "")
                      + f"   {x1 - x0}x{y1 - y0}")
            lb.itemconfigure(i, fg=COLOURS[b["type"]])
        if self.sel is not None:
            lb.selection_set(self.sel)
            lb.see(self.sel)

    def update_status(self):
        where = f"x {self.cursor[0]}  y {self.cursor[1]}     " if self.cursor else ""
        note = self.flash_note or self.save_note
        self.l_status.configure(text=f"{where}zoom {self.scale * 100:.0f}%\n{note}",
                                fg=BAD if note.startswith("NOT SAVED") else MUTED)

    def flash(self, text):
        self.flash_note = text
        if self.flash_job:
            self.root.after_cancel(self.flash_job)
        self.flash_job = self.root.after(2500, self._unflash)
        self.update_status()

    def _unflash(self):
        self.flash_note = self.flash_job = None
        self.update_status()

    # ── input ──

    def hit(self, u, v):
        """What a press at canvas (u, v) grabs: ("resize", i, edges) on a corner or edge of the
        selected box, ("move", i, "") inside any box - the smallest, so a box inside another
        stays reachable - or None on bare photograph."""
        if self.sel is not None:
            u0, v0, u1, v1 = self._canvas_box(self.sel)
            g, edges = self.GRAB * self.k, ""
            if v0 - g <= v <= v1 + g:
                edges += "l" if abs(u - u0) <= g else "r" if abs(u - u1) <= g else ""
            if u0 - g <= u <= u1 + g:
                edges += "t" if abs(v - v0) <= g else "b" if abs(v - v1) <= g else ""
            if edges:
                return "resize", self.sel, edges
        inside = []
        for i in range(len(self.boxes)):
            u0, v0, u1, v1 = self._canvas_box(i)
            if u0 <= u <= u1 and v0 <= v <= v1:
                inside.append(((u1 - u0) * (v1 - v0), i))
        return ("move", min(inside)[1], "") if inside else None

    def _canvas_box(self, i):
        x0, y0, x1, y1 = self.boxes[i]["xyxy"]
        return (*self.to_canvas(x0, y0), *self.to_canvas(x1, y1))

    def on_press(self, e):
        self.canvas.focus_set()
        # Shift always draws, so a fastener touching an already-boxed one can still be boxed.
        hit = None if e.state & 0x1 else self.hit(e.x, e.y)
        if hit:
            mode, i, edges = hit
            self.sel = i
            self.drag = {"mode": mode, "i": i, "edges": edges, "start": (e.x, e.y),
                         "orig": list(self.boxes[i]["xyxy"]), "moved": False}
        else:
            p = self.clamp(*self.to_photo(e.x, e.y))
            self.drag = {"mode": "draw", "start": (e.x, e.y), "p0": p, "p1": p,
                         "moved": False}
        self.draw_boxes()
        self.refresh_list()

    def on_drag(self, e):
        self.on_motion(e)
        d = self.drag
        if d is None:
            return
        if not d["moved"]:
            if abs(e.x - d["start"][0]) + abs(e.y - d["start"][1]) < self.CLICK:
                return
            d["moved"] = True
            if d["mode"] != "draw":
                self.push_undo()
        x, y = self.clamp(*self.to_photo(e.x, e.y))
        if d["mode"] == "draw":
            d["p1"] = (x, y)
        else:
            x0, y0, x1, y1 = d["orig"]
            if d["mode"] == "move":
                w, h = self.size
                dx = min(max(round((e.x - d["start"][0]) / self.scale), -x0), w - x1)
                dy = min(max(round((e.y - d["start"][1]) / self.scale), -y0), h - y1)
                x0, y0, x1, y1 = x0 + dx, y0 + dy, x1 + dx, y1 + dy
            else:
                if "l" in d["edges"]:
                    x0 = min(x, x1 - 1)
                if "r" in d["edges"]:
                    x1 = max(x, x0 + 1)
                if "t" in d["edges"]:
                    y0 = min(y, y1 - 1)
                if "b" in d["edges"]:
                    y1 = max(y, y0 + 1)
            self.boxes[d["i"]]["xyxy"] = [x0, y0, x1, y1]
        self.draw_boxes()

    def on_release(self, e):
        d, self.drag = self.drag, None
        if d is None:
            return
        if d["mode"] == "draw":
            (xa, ya), (xb, yb) = d["p0"], d["p1"]
            x0, x1 = sorted((xa, xb))
            y0, y1 = sorted((ya, yb))
            if d["moved"] and x1 - x0 >= self.MIN_BOX and y1 - y0 >= self.MIN_BOX:
                self.push_undo()
                self.boxes.append({"type": self.cur_type, "xyxy": [x0, y0, x1, y1],
                                   "optional": False})
                self.sel = len(self.boxes) - 1
                self.changed()
                return
            if not d["moved"]:
                self.sel = None                   # a click on bare photograph deselects
            self.draw_boxes()
            self.refresh()
        elif d["moved"]:
            self.changed()

    def on_pan_start(self, e):
        self.pan = (e.x, e.y)

    def on_pan(self, e):
        """Slides what is already drawn and renders once the pan pauses: a full render takes
        tens of milliseconds on a large window, longer than the mouse waits between events."""
        if self.pan is None:
            return
        du, dv = e.x - self.pan[0], e.y - self.pan[1]
        self.pan = (e.x, e.y)
        self.ox -= du / self.scale
        self.oy -= dv / self.scale
        self.fitted = False
        self.canvas.move(self.image_item, du, dv)
        self.canvas.move("box", du, dv)
        if self.pan_job:
            self.root.after_cancel(self.pan_job)
        self.pan_job = self.root.after(120, self._pan_settled)
        self.on_motion(e)

    def on_pan_end(self, e):
        self.pan = None
        if self.pan_job:
            self.root.after_cancel(self.pan_job)
        self._pan_settled()

    def _pan_settled(self):
        self.pan_job = None
        self.request_render()

    def on_motion(self, e):
        cw, ch = self.canvas_size()
        self.canvas.coords(self.cross[0], 0, e.y, cw, e.y)
        self.canvas.coords(self.cross[1], e.x, 0, e.x, ch)
        for line in self.cross:
            self.canvas.itemconfigure(line, state="normal")
        x, y = self.to_photo(e.x, e.y)
        w, h = self.size
        self.cursor = (int(x), int(y)) if 0 <= x < w and 0 <= y < h else None
        self.update_status()

    def on_leave(self, e):
        for line in self.cross:
            self.canvas.itemconfigure(line, state="hidden")
        self.cursor = None
        self.update_status()

    def on_list_select(self, e):
        cur = self.listbox.curselection()
        self.sel = cur[0] if cur else None
        self.draw_boxes()
        self.canvas.focus_set()

    def on_key(self, e):
        ks = e.keysym
        if e.state & 0x4:                                 # Control
            if ks.lower() == "z":
                self.undo_last()
            elif ks.lower() == "s":
                self.persist()
                self.flash(self.save_note)
            return "break"
        if e.char and e.char in "123456"[:len(TYPES)]:     # either keyboard row or the keypad
            self.set_type(TYPES[int(e.char) - 1])
        elif ks in ("o", "O"):
            self.toggle_optional()
        elif ks in ("Delete", "BackSpace"):
            self.delete_selected()
        elif ks == "Escape":
            self.cancel()
        elif ks in ("f", "F"):
            self.fit()
        elif ks in ("plus", "equal", "KP_Add", "minus", "KP_Subtract"):
            cw, ch = self.canvas_size()
            self.zoom_at(cw / 2, ch / 2, 0.8 if "minus" in ks or "Subtract" in ks else 1.25)
        elif ks in ("Return", "KP_Enter"):
            self.toggle_done()
        elif ks in ("n", "N", "Right", "Next"):
            self.go(1)
        elif ks in ("p", "P", "Left", "Prior"):
            self.go(-1)
        elif ks == "Tab":
            self.cycle(-1 if e.state & 0x1 else 1)
        elif ks == "ISO_Left_Tab":                         # Shift+Tab on X11
            self.cycle(-1)
        else:
            return None
        return "break"


def main():
    ap = argparse.ArgumentParser(description="Label the hardware in the bundled photographs.")
    ap.add_argument("--images", default=IMG_DIR,
                    help="folder of photographs (default: the app's chassis_images)")
    ap.add_argument("--out", default=OUT,
                    help="labels file (default: python/detection_boxes.json)")
    ap.add_argument("--photo", help="open this photograph first")
    ap.add_argument("--check", action="store_true",
                    help="print each photograph's labels beside its hand count, then exit")
    args = ap.parse_args()

    if args.check:
        sys.exit(1 if check(load(args.out), args.images) else 0)

    names = photos(args.images) if os.path.isdir(args.images) else []
    if not names:
        sys.exit(f"no photographs in {args.images}")
    if tk is None:
        sys.exit("tkinter is not available in this Python; --check works without it")
    images = load(args.out)          # a damaged labels file fails here, before any window
    if args.photo:
        if args.photo not in names:
            sys.exit(f"{args.photo} is not in {args.images}")
        start = names.index(args.photo)
    else:
        start = next((i for i, n in enumerate(names) if not images.get(n, {}).get("done")), 0)

    _dpi_aware()
    root = tk.Tk()
    Labeller(root, args.images, args.out, names, images, start)
    root.mainloop()


if __name__ == "__main__":
    main()
