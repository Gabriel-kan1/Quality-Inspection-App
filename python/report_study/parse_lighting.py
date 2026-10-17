"""Parses variation_test_study.py --detail output into per-method, per-region verdicts.

argv: <study output> <json out>
"""
import json
import re
import sys

LIGHTS = ("white", "warm", "yellow")
ANGLES = ("left", "right", "bot", "front")

text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
rows = {}
summary = re.compile(r"^  (v\d+)\s+(ref OK|ref BAD.*?)\s+lighting\s+(\d+)/\s*(\d+) pass\s+defect (caught|MISSED)\s+angles (\d+)/(\d+)")
config = None
for line in text.splitlines():
    m = re.match(r"^=== (.+) ===$", line)
    if m:
        config = m.group(1)
        continue
    m = summary.match(line)
    if m and config:
        mid = m.group(1)
        rows.setdefault(mid, {})[config] = {
            "ref_ok": m.group(2).startswith("ref OK"), "ref": m.group(2).strip(),
            "light_pass": int(m.group(3)), "light_n": int(m.group(4)),
            "defect_caught": m.group(5) == "caught",
            "angle_pass": int(m.group(6)), "angle_n": int(m.group(7)),
        }

# Per-photo detail: "<config> / <mid>: reference ..." then indented photo lines.
detail = {}
cur = None
for line in text.splitlines():
    m = re.match(r"^(.+) / (v\d+): reference (\d+) boxes -> (.*)$", line)
    if m:
        cur = (m.group(1), m.group(2))
        detail[f"{cur[0]}|{cur[1]}"] = {"ref_boxes": int(m.group(3)), "ref_counts": m.group(4), "photos": {}}
        continue
    m = re.match(r"^   (\S.*?)\s+(PASS|FAIL)\s+(\d+) boxes\s+(.*)$", line)
    if m and cur:
        detail[f"{cur[0]}|{cur[1]}"]["photos"][m.group(1).strip()] = {
            "verdict": m.group(2), "boxes": int(m.group(3)), "counts": m.group(4)}

json.dump({"summary": rows, "detail": detail}, open(sys.argv[2], "w"), indent=1)
for mid, cfgs in rows.items():
    print(mid, {c: (v["light_pass"], v["light_n"], v["defect_caught"], v["ref_ok"]) for c, v in cfgs.items()})
