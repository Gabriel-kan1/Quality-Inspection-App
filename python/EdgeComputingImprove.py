print("Importing libraries...")
import cv2
import numpy as np
import os
import pickle
import time
import colorsys
import shutil
import matplotlib.patches as mpatches
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from PIL import Image as PILImage
print("✓ Libraries imported")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
filename = "lightring1.jpg"
TESTING_IMAGE_DIR = "testing_images"
RESULT_IMAGE_DIR = "result_images"
REFERENCE_DIR = "featureRef"
LIBRARY_FILE = "reference_library.pkl"

# --- Few-Shot Classification Config ---
SIMILARITY_THRESHOLD = 0.75

# --- VLM Text Prompt Weighting ---
TEXT_PROMPT_WEIGHT = 0.0

# --- Text prompts for each feature type ---
TEXT_PROMPTS = {
    "bolt_head":  "a circular silver metallic disc completely flush and flat with surrounding grey metal surface, smooth uniform surface with slight texture, no raised edges",
    "hole":       "a circular opening in metal with a copper or bronze coloured ring around the edge, dark centre void, metallic rim surrounding the hole",
    "rivet_head": "a circular indentation in grey metal surface, shallow concave circular mark with a dark centre point, recessed into the surface",
    "screw":      "a cylindrical metal screw shaft with visible helical thread pattern, dark textured threaded surface viewed from the side, no head visible only thread",
    "stud":       "a large protruding cylindrical silver metal stud standing above the surface, dome shaped top with shiny metallic reflection, surrounded by white sealant or washer ring",
    "unknown":    "unidentified feature of unknown type, cannot classify based on text prompts"
}

# # --- OpenCV Preprocessing Config ---
CROP_PADDING = 40               # Changed from 60
# BILATERAL_D = 6
# BILATERAL_SIGMA_COLOR = 120
# BILATERAL_SIGMA_SPACE = 40      # Changed from 100
# CANNY_THRESH1 = 50              # Hardcoded thresholds restored
# CANNY_THRESH2 = 150
# DILATION_KERNEL_SIZE = 5        # New 5x5 kernel for dilation
# MORPH_KERNEL_SIZE = 7           # Changed from 10 to 7 (for closing)

# --- OpenCV Geometry & Filtering Config ---
MAX_DIMENSION_RATIO = 0.4
MAX_ASPECT_RATIO = 5.0
MIN_ASPECT_RATIO = 0.2
EDGE_MARGIN = 10
DYNAMIC_AREA_MULTIPLIER = 0.15  # Stricter area filter (15%)
PROXIMITY_THRESHOLD = 10

# ─────────────────────────────────────────────
# LOAD CLIP MODEL
# ─────────────────────────────────────────────
print("Loading CLIP model...")
clip_model = SentenceTransformer("clip-ViT-B-32")
print("✓ CLIP model loaded")

# ─────────────────────────────────────────────
# PRE-COMPUTE TEXT EMBEDDINGS
# ─────────────────────────────────────────────
print("Pre-computing text prompt embeddings...")
text_embeddings = {}
for feature_type, prompt in TEXT_PROMPTS.items():
    text_embeddings[feature_type] = clip_model.encode(prompt)
    print(f"  ✓ Text embedding computed for '{feature_type}'")
print(f"✓ {len(text_embeddings)} text embeddings ready")

# ─────────────────────────────────────────────
# EMBEDDING FUNCTION
# ─────────────────────────────────────────────
def get_embedding(crop_bgr):
    img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(img_rgb)
    embedding = clip_model.encode(pil_img)
    return embedding

# ─────────────────────────────────────────────
# REFERENCE LIBRARY FUNCTIONS
# ─────────────────────────────────────────────
def build_reference_library(reference_dir):
    library = {}
    ref_path = Path(reference_dir)
    feature_dirs = sorted([d for d in ref_path.iterdir() if d.is_dir()])

    if not feature_dirs:
        return {}

    print(f"\nBuilding reference library from '{reference_dir}'...")

    for feature_dir in feature_dirs:
        feature_type = feature_dir.name
        image_files = (
            list(feature_dir.glob("*.jpg")) +
            list(feature_dir.glob("*.jpeg")) +
            list(feature_dir.glob("*.png"))
        )

        if not image_files:
            print(f"  ⚠ No images found in {feature_dir.name}/ — skipping")
            continue

        embeddings = []
        for img_path in sorted(image_files):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            embedding = get_embedding(img)
            embeddings.append(embedding)
            print(f"  Embedded {feature_type}/{img_path.name}")

        if embeddings:
            prototype = np.mean(embeddings, axis=0)
            library[feature_type] = {
                "prototype": prototype,
                "individual": embeddings,
                "count": len(embeddings)
            }
            print(f"  ✓ {feature_type}: {len(embeddings)} reference images")

    if library:
        print(f"\n✓ Reference library built — {len(library)} feature types: {list(library.keys())}")
    return library

def save_library(library, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(library, f)
    print(f"✓ Library saved to {filepath}")

def load_library(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, "rb") as f:
        library = pickle.load(f)
    print(f"✓ Library loaded — {len(library)} feature types: {list(library.keys())}")
    return library

# ─────────────────────────────────────────────
# CLASSIFICATION FUNCTION — HYBRID IMAGE + TEXT
# ─────────────────────────────────────────────
def classify_crop(crop, library):
    if not library:
        return {
            "type": "unknown", "closest_match": "none",
            "confidence": "low", "similarity_score": 0.0,
            "image_scores": {}, "text_scores": {}, "all_scores": {}
        }

    image_embedding = get_embedding(crop)

    image_scores = {}
    text_scores = {}
    combined_scores = {}

    for feature_type, ref_data in library.items():

        img_score = cosine_similarity(
            image_embedding.reshape(1, -1),
            ref_data["prototype"].reshape(1, -1)
        )[0][0]
        image_scores[feature_type] = round(float(img_score), 3)

        if feature_type in text_embeddings:
            txt_score = cosine_similarity(
                image_embedding.reshape(1, -1),
                text_embeddings[feature_type].reshape(1, -1)
            )[0][0]
            text_scores[feature_type] = round(float(txt_score), 3)

            combined = (img_score * (1 - TEXT_PROMPT_WEIGHT) + txt_score * TEXT_PROMPT_WEIGHT)
        else:
            text_scores[feature_type] = 0.0
            combined = img_score

        combined_scores[feature_type] = round(float(combined), 3)

    best_type = max(combined_scores, key=combined_scores.get)
    best_score = combined_scores[best_type]
    closest_match = best_type

    if best_score < SIMILARITY_THRESHOLD:
        best_type = "unknown"

    if best_score > 0.88:
        confidence = "high"
    elif best_score > SIMILARITY_THRESHOLD:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "type": best_type, "closest_match": closest_match,
        "confidence": confidence, "similarity_score": round(float(best_score), 3),
        "image_scores": image_scores, "text_scores": text_scores, "all_scores": combined_scores
    }

# ─────────────────────────────────────────────
# STEP 1 — LOAD IMAGE
# ─────────────────────────────────────────────
print(f"\nStep 1: Loading image...")

os.makedirs(TESTING_IMAGE_DIR, exist_ok=True)
os.makedirs(RESULT_IMAGE_DIR, exist_ok=True)

image_path = os.path.join(TESTING_IMAGE_DIR, filename)

if not os.path.exists(image_path):
    if os.path.exists(filename):
        print(f"  ⚠ '{filename}' not found in '{TESTING_IMAGE_DIR}/' — using file from current directory")
        image_path = filename
    else:
        print(f"ERROR: '{filename}' not found in '{TESTING_IMAGE_DIR}/' or current directory")
        exit()

img_color = cv2.imread(image_path)
if img_color is None:
    print(f"ERROR: Could not load '{image_path}'")
    exit()

img_color_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
img_height, img_width = img_color.shape[:2]
print(f"✓ Image loaded from '{image_path}' — {img_width}x{img_height} pixels")

# ─────────────────────────────────────────────
# STEP 2 — HYBRID PREPROCESS (SMART GLARE + STATIC BLURS)
# ─────────────────────────────────────────────
print("\nStep 2: Preprocessing and detecting edges...")

# -- 1. Smart Glare Detection --
glare_pixels = np.sum(img_gray > 240)
total_pixels = img_gray.shape[0] * img_gray.shape[1]
glare_ratio = glare_pixels / total_pixels
print(f"  Glare ratio detected: {glare_ratio:.3f}")

if glare_ratio > 0.05:  # 5% threshold
    print("  -> Heavy glare detected. Applying Black-Hat transform.")
    lighting_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (75, 75))
    img_preprocessed = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, lighting_kernel)
else:
    print("  -> Clean lighting detected. Bypassing Black-Hat.")
    img_preprocessed = img_gray

# -- 2. Apply CLAHE (Exactly as in image_processor.py) --
clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16, 16))
img_clahe = clahe.apply(img_preprocessed)

# -- 3. Bilateral Filter (Exactly as in image_processor.py) --
img_filtered = cv2.bilateralFilter(
    img_clahe, d=6, sigmaColor=120, sigmaSpace=40
)

# -- 4. Noise Reduction Blurs (Exactly as in image_processor.py) --
img_smoothed = cv2.medianBlur(img_filtered, 5)
img_smoothed = cv2.GaussianBlur(img_smoothed, (5, 5), 0)

# -- 5. Canny Edge Detection (Exactly as in image_processor.py) --
edges = cv2.Canny(img_smoothed, 50, 150)

# -- 6. Morphological Operations (Exactly as in image_processor.py) --
dilate_kernel = np.ones((5, 5), np.uint8)
edges = cv2.dilate(edges, dilate_kernel, iterations=1)

close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel)

print("✓ Edge detection complete")

# ─────────────────────────────────────────────
# STEP 3 & 4 — FIND CONTOURS & CROP
# ─────────────────────────────────────────────
print("\nStep 3 & 4: Finding, filtering, and cropping features...")
contours, hierarchy = cv2.findContours(
    edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

if os.path.exists("crops"):
    shutil.rmtree("crops", ignore_errors=True)
    time.sleep(0.3)
os.makedirs("crops", exist_ok=True)

candidates = []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if (w > img_width * MAX_DIMENSION_RATIO or h > img_height * MAX_DIMENSION_RATIO):
        continue
    aspect_ratio = float(w) / h
    if aspect_ratio > MAX_ASPECT_RATIO or aspect_ratio < MIN_ASPECT_RATIO:
        continue
    if (x < EDGE_MARGIN or y < EDGE_MARGIN or (x + w) > (img_width - EDGE_MARGIN) or (y + h) > (img_height - EDGE_MARGIN)):
        continue
    candidates.append(c)

boxes = []
if candidates:
    areas = [cv2.contourArea(c) for c in candidates]
    areas.sort(reverse=True)
    top_areas = areas[:3]
    reference_area = sum(top_areas) / max(1, len(top_areas))
    dynamic_min_area = max(25, reference_area * DYNAMIC_AREA_MULTIPLIER)

    for c in candidates:
        if cv2.contourArea(c) < dynamic_min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((x, y, w, h))

filtered_boxes = []
for i, (x1, y1, w1, h1) in enumerate(boxes):
    is_inside = False
    for j, (x2, y2, w2, h2) in enumerate(boxes):
        if i == j:
            continue
        if (x1 >= x2 and y1 >= y2 and (x1 + w1) <= (x2 + w2) and (y1 + h1) <= (y2 + h2)):
            is_inside = True
            break
    if not is_inside:
        filtered_boxes.append((x1, y1, w1, h1))

def merge_nearby_boxes(boxes, proximity_threshold=30):
    if not boxes:
        return []

    boxes = list(boxes)
    merged = True

    while merged:
        merged = False
        used = [False] * len(boxes)
        new_boxes = []

        for i in range(len(boxes)):
            if used[i]:
                continue

            x1, y1, w1, h1 = boxes[i]
            group = [(x1, y1, w1, h1)]
            used[i] = True

            for j in range(i + 1, len(boxes)): # Only scan forward
                if used[j]:
                    continue

                x2, y2, w2, h2 = boxes[j]

                # 1. Check Proximity
                h_gap = max(0, max(x1, x2) - min(x1 + w1, x2 + w2))
                v_gap = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))

                if h_gap <= proximity_threshold and v_gap <= proximity_threshold:
                    
                    # 2. THE MAGIC RULE: Box Fill Ratio Check
                    temp_min_x = min(x1, x2)
                    temp_min_y = min(y1, y2)
                    temp_max_x = max(x1 + w1, x2 + w2)
                    temp_max_y = max(y1 + h1, y2 + h2)
                    temp_w = temp_max_x - temp_min_x
                    temp_h = temp_max_y - temp_min_y
                    
                    combined_area = temp_w * temp_h
                    sum_of_areas = (w1 * h1) + (w2 * h2)
                    
                    # If they belong to the same broken object, they will be stacked perfectly 
                    # and fill up most of the combined box (> 70%).
                    # If they are diagonal/separate, they will fill less than 50%.
                    if sum_of_areas >= combined_area * 0.70:
                        group.append((x2, y2, w2, h2))
                        used[j] = True
                        merged = True
                        
                        # Update the base box to safely chain-merge a 3rd or 4th fragment
                        x1, y1, w1, h1 = temp_min_x, temp_min_y, temp_w, temp_h
                    else:
                        pass # They are adjacent separate features. Stop the black hole! Do not merge.

            # Finalize the group bounding box
            final_min_x = min(b[0] for b in group)
            final_min_y = min(b[1] for b in group)
            final_max_x = max(b[0] + b[2] for b in group)
            final_max_y = max(b[1] + b[3] for b in group)
            new_boxes.append((final_min_x, final_min_y, final_max_x - final_min_x, final_max_y - final_min_y))

        boxes = new_boxes

    return boxes

filtered_boxes = merge_nearby_boxes(filtered_boxes, PROXIMITY_THRESHOLD)
print(f"  After proximity merge: {len(filtered_boxes)} features")

crops = []
contour_count = 0
for x, y, w, h in filtered_boxes:
    contour_count += 1
    x1 = max(0, x - CROP_PADDING)
    y1 = max(0, y - CROP_PADDING)
    x2 = min(img_width, x + w + CROP_PADDING)
    y2 = min(img_height, y + h + CROP_PADDING)

    if (x2 - x1) < 10 or (y2 - y1) < 10:
        continue

    crop = img_color[y1:y2, x1:x2]

    min_size = 80
    if crop.shape[0] < min_size or crop.shape[1] < min_size:
        scale = min_size / min(crop.shape[0], crop.shape[1])
        crop = cv2.resize(
            crop, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

    crop_filename = f"crops/crop_{contour_count}.jpg"
    cv2.imwrite(crop_filename, crop)

    crops.append({
        "id": contour_count, "x": x, "y": y, "w": w, "h": h,
        "crop": crop, "crop_filename": crop_filename
    })

print(f"✓ OpenCV extracted {len(crops)} features")

# ─────────────────────────────────────────────
# STEP 5 — DYNAMIC REFERENCE LIBRARY
# ─────────────────────────────────────────────
print("\nStep 5: Inspecting and loading reference library...")
library = load_library(LIBRARY_FILE)

ref_path = Path(REFERENCE_DIR)
current_folders = set()

if not ref_path.exists():
    ref_path.mkdir(parents=True, exist_ok=True)
else:
    current_folders = {d.name for d in ref_path.iterdir() if d.is_dir()}

rebuild_needed = False
if library is None:
    rebuild_needed = True
elif current_folders and current_folders != set(library.keys()):
    print("  ⚠ Folder contents changed — rebuilding library...")
    rebuild_needed = True

if rebuild_needed:
    library = build_reference_library(REFERENCE_DIR)
    if not library:
        print(f"\nERROR: No feature subfolders found in '{REFERENCE_DIR}/'")
        exit()
    save_library(library, LIBRARY_FILE)

for ft in library.keys():
    if ft not in TEXT_PROMPTS:
        print(f"  ⚠ No text prompt defined for '{ft}' — add it to TEXT_PROMPTS in CONFIG for better accuracy")

# ─────────────────────────────────────────────
# STEP 6 — CLASSIFY EACH CROP WITH CLIP
# ─────────────────────────────────────────────
print(f"\nStep 6: Classifying {len(crops)} crops with CLIP...")
results = []
start_total = time.time()

for i, crop_info in enumerate(crops):
    start = time.time()
    classification = classify_crop(crop_info["crop"], library)
    elapsed = time.time() - start

    label_print = classification['type']
    if (classification['type'] == "unknown" and classification['closest_match'] != "none"):
        label_print = f"unknown (maybe: {classification['closest_match']})"

    print(f"  [{i+1}/{len(crops)}] → {label_print} "
          f"combined={classification['similarity_score']:.3f} "
          f"{classification['confidence']} ({elapsed*1000:.0f}ms)")
    print(f"    image scores: {classification['image_scores']}")
    print(f"    text scores:  {classification['text_scores']}")
    print(f"    combined:     {classification['all_scores']}")

    feature_type = classification["type"]
    type_folder = os.path.join("crops", feature_type)
    os.makedirs(type_folder, exist_ok=True)

    crop_filename = os.path.join(type_folder, f"crop_{crop_info['id']}.jpg")
    cv2.imwrite(crop_filename, crop_info["crop"])
    crop_info["crop_filename"] = crop_filename

    results.append({**crop_info, **classification})

total_elapsed = time.time() - start_total
print(f"✓ Classification complete — {total_elapsed:.1f}s total")

print("\n  Crops saved to:")
for ft in sorted(os.listdir("crops")):
    ft_path = os.path.join("crops", ft)
    if os.path.isdir(ft_path):
        count = len([f for f in os.listdir(ft_path) if f.endswith(".jpg")])
        print(f"    crops/{ft}/ — {count} images")

# ─────────────────────────────────────────────
# STEP 7 — DRAW RESULTS
# ─────────────────────────────────────────────
print("\nStep 7: Drawing results...")

feature_types = sorted(list(library.keys()))
color_map = {"unknown": (255, 0, 255)}  

for i, ft in enumerate(feature_types):
    hue = i / max(1, len(feature_types))
    rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
    color_map[ft] = (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

img_annotated = img_color_rgb.copy()

for feature in results:
    x, y, w, h = feature["x"], feature["y"], feature["w"], feature["h"]
    color = color_map.get(feature["type"], (128, 128, 128))

    cv2.rectangle(img_annotated, (x, y), (x + w, y + h), color, 2)

    if (feature["type"] == "unknown" and feature["closest_match"] != "none"):
        label = f"{feature['id']}:? ({feature['closest_match']} {feature['similarity_score']:.2f})"
    else:
        label = f"{feature['id']}:{feature['type']} {feature['similarity_score']:.2f}"

    cv2.putText(
        img_annotated, label,
        (max(0, x), max(15, y - 5)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
    )

# ─────────────────────────────────────────────
# STEP 8 — SHOW AND SAVE
# ─────────────────────────────────────────────
print("\nStep 8: Saving and displaying results...")

real_features    = [r for r in results if r["type"] != "unknown"]
unknown_features = [r for r in results if r["type"] == "unknown"]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(img_color_rgb)
axes[0].set_title("Original", fontsize=12)
axes[0].axis("off")

axes[1].imshow(edges, cmap="gray")
axes[1].set_title(f"Canny Edges & Filtering\nFound {len(crops)} features", fontsize=12)
axes[1].axis("off")

axes[2].imshow(img_annotated)
axes[2].set_title(f"Classification: {len(real_features)} classified, {len(unknown_features)} unknown", fontsize=12)
axes[2].axis("off")

legend_patches = []
for ft in feature_types:
    c = color_map[ft]
    norm_color = (c[0]/255.0, c[1]/255.0, c[2]/255.0)
    legend_patches.append(mpatches.Patch(color=norm_color, label=ft))
legend_patches.append(mpatches.Patch(color=(255/255.0, 0/255.0, 255/255.0), label="unknown"))
axes[2].legend(handles=legend_patches, loc="upper right", bbox_to_anchor=(1.25, 1), title="Features")

plt.tight_layout()
output_filename = os.path.join(RESULT_IMAGE_DIR, "results_clip_" + filename)
plt.savefig(output_filename, dpi=150, bbox_inches="tight")
plt.show()
print(f"✓ Result saved to '{output_filename}'")