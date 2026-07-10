OPTIONAL BUNDLED DEFAULT REFERENCE LIBRARY
==========================================

Put example crop images here to give the app a "works out of the box" default
detection library. Layout — one subfolder per feature type:

    reference_crops/
        bolt_head/   *.jpg | *.jpeg | *.png
        hole/
        rivet_head/
        screw/
        stud/

Each image should be a TIGHT crop of a single feature of that type (a handful
per type is enough; more/cleaner examples => better few-shot accuracy).

On first use, com.example.ml.ReferenceLibraryProvider embeds every crop here
with the CLIP model and averages them into per-type prototypes (mirrors
build_reference_library() in python/EdgeComputingImprove.py). The result is
cached at filesDir/clip_lib/_default.json.

This is OPTIONAL. If it is empty, the app has no default library and features
read as "unknown" until you build a per-part library in the app via
Reference Setup -> "BUILD DETECTION LIBRARY (CLIP)", which auto-detects crops
from a gold-standard image and lets you label them.

The canonical feature types are defined in ReferenceLibrary.FEATURE_TYPES:
    bolt_head, hole, rivet_head, screw, stud
