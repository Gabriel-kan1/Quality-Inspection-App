"""
How many hardware features are actually in each bundled photograph.

Until this existed, the detector was tuned against six calibration counts and judged
everywhere else by proxies: agreement between photographs the alignment gate treats as
interchangeable, and how far a count moved when the photograph or the parameters were
nudged. Proxies can be gamed. A stack that finds nothing on every photograph has perfect
agreement between interchangeable pairs.

These counts were made by eye from contact sheets of every asset with a detector's boxes
drawn on it, then confirmed by zooming on anything ambiguous. They are the counts of
INSTALLED hardware — heads, studs, screws, rivets, holes — that a competent inspector would
list. Where a count is a set, more than one answer is defensible and the reason is given.

Three things the counting turned up that matter beyond this file:

  * screws.jpg has FOUR screws. The calibration target supplied for it was 5, and the shipped
    stack does return 5 — but the fifth box sits on a scratch. Tuning to hit 5 there was
    tuning to reproduce a false positive.
  * The shipped stack's worst failures are not the calibration images. three.jpg has three
    rivets on a heavily textured plate under a glare hotspot; the shipped stack finds 12.
  * The detector's remaining misses are all one kind of thing: a low-contrast head that is
    barely darker than the plate (two in head3.jpg, one copper one in one.jpg). No global
    threshold found in any search recovers them without flooding textured plates elsewhere.

WIDE holds the whole-panel shots. The alignment gate keeps an operator at reference distance,
so hardware that is a few pixels across at the working resolution is outside what the
detector is asked to do; they are scored, but separately.

Run any of python/hybrid_v2_study.py's evaluations and these are what section 6 scores against.
"""

# Reference-distance photographs: what the detector is designed for.
GT = {
    "defected.jpg":      {4},
    "head1.jpg":         {4},
    "head2.jpg":         {4},
    "head3.jpg":         {4},      # the two lower heads are low-contrast; every variant misses them
    "head4.jpg":         {4},
    "hole1.jpg":         {3},      # a fourth box lands on the shadowed region at right
    "hole2.jpg":         {3},      # bench-edge artefacts along the bottom
    "one.jpg":           {4},      # the copper head at lower right is faint
    "reference.jpg":     {4},
    "screws-1.jpg":      {4},      # a fifth box sits on a scratch
    "screws.jpg":        {4},      # supplied target was 5; the fifth box is a scratch
    "screws2.jpg":       {4},      # one stud tends to split into cap + shaft
    "screws_flash.jpg":  {5, 6},   # five installed, one loose pin lying on the plate
    "screws_none.jpg":   {5, 6},   # same plate, same loose pin, no flash
    "screws_test.jpg":   {4},      # stud split, as screws2
    "screws_three.jpg":  {4},      # scratch and speck false positives
    "screws_two.jpg":    {4},      # stud split
    "stud1.jpg":         {4},
    "stud2.jpg":         {4},
    "stud3.jpg":         {4},
    "stud4.jpg":         {4},
    "three.jpg":         {3},      # textured plate under a glare hotspot; grime reads as hardware
    "whole screws1.jpg": {12},     # six down the right, four in a cluster, two along the bottom
    "whole screws2.jpg": {12},
    "whole screws3.jpg": {12},
    "zero.jpg":          {3},
}

# Whole-panel shots, well outside the working resolution's reach. Counts are approximate.
WIDE = {
    "corner.jpg":        {10},
    "whole.jpg":         {20},
    "whole_light.jpg":   {20},
}

EXCLUDED = {"calib_test.png": "the scale-calibration target, not a chassis"}

# The six counts supplied as calibration targets, kept so a config can be held to them.
# screws.jpg accepts 4 as well as the supplied 5, for the reason given above.
SUPPLIED = {"defected.jpg": {4}, "head1.jpg": {4}, "reference.jpg": {4},
            "screws.jpg": {4, 5}, "stud1.jpg": {4}, "zero.jpg": {3}}


def err(name, count):
    """L1 distance from a count to the nearest acceptable one."""
    ok = GT.get(name) or WIDE.get(name)
    return min(abs(count - c) for c in ok)


def split(name, count):
    """(misses, extras) — a fastener the detector cannot see costs more than an extra box,
    which the classifier still gets a chance to reject."""
    lo, hi = min(GT[name]), max(GT[name])
    return max(0, lo - count), max(0, count - hi)


def weighted(counts):
    """2 x misses + extras over the reference-distance set. The number the search ranks on."""
    fn = fp = 0
    for n, c in counts.items():
        if n in GT:
            a, b = split(n, c)
            fn += a
            fp += b
    return 2 * fn + fp, fn, fp
