package com.example.ml;

import android.content.Context;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * The expected arrangement of features on a part — a persisted set of numbered slots, each
 * with a type, a position, and the running history of its measured size.
 *
 * <h3>Why a layout rather than counts</h3>
 * Verification previously compared per-type counts only: two screws expected, two found,
 * pass. That cannot see a screw and a stud swapped between positions, a fastener present
 * but in the wrong hole, or which specific one is missing. A slot is a fixed identity that
 * survives across inspections, which is what makes both positional checking and per-slot
 * dimensional history possible.
 *
 * <h3>Coordinates</h3>
 * Slot positions are stored in the reference photo's own fractional coordinates. They are
 * only meaningful once {@link LayoutMatcher} has aligned an inspection onto them — the part
 * is photographed freehand, so nothing can be compared in raw image space.
 *
 * <h3>Reference size</h3>
 * Each slot also carries the size of the box the reference detection was found in, as a
 * fraction of the reference frame, together with the working-image dimensions that fraction
 * was measured at. That is what lets an inspection compare a fastener's size against the
 * reference <em>without a calibration marker</em>: the alignment fitted by {@link LayoutMatcher}
 * recovers how much larger or smaller the part appears in this photo, so the box the reference
 * predicts at this framing can be set against the box actually found. See
 * {@code FeatureLocalisation.SIZE_TOLERANCE}. Layouts saved before sizes were recorded carry
 * none, and such slots are simply not judged.
 *
 * <h3>Learned size</h3>
 * The reference box is one detection of the fastener, and a detection can be wrong in size:
 * a shadow or scratch beside the hardware merges into the box and it comes out larger than
 * the fastener, or an edge is lost and it comes out smaller. Each slot therefore also keeps
 * the size ratios observed at it across filed inspections ({@link Slot#sizeRatios}), and
 * {@link Slot#expectedSizeFactor()} — the median of those ratios with the reference counted
 * once at 1.0 — is the running estimate of how big the hardware there really is relative to
 * the box the reference drew. {@link SizePrior} uses it to arbitrate between preprocessing
 * strengths: the one whose boxes come closest to the learned sizes is the one that neither
 * merged a shadow in nor cropped a fastener. The size <em>check</em> itself still compares
 * against the reference box, which is what the filed reports say it does.
 *
 * <h3>Dimension statistics</h3>
 * Each slot accumulates the sizes measured for it. Median and MAD are kept rather than mean
 * and standard deviation, deliberately: an occasional genuinely out-of-spec part would drag
 * a mean and inflate a standard deviation, teaching the tolerance to accept the defect it
 * exists to catch. Median and MAD are largely unmoved by a minority of outliers.
 *
 * Stored at {@code filesDir/layouts/<partId>.json}.
 */
public final class FeatureLayout {

    private static final String TAG = "FeatureLayout";
    private static final String DIR = "layouts";

    /**
     * Samples required before a slot will flag anything. Below roughly this many, a spread
     * estimate is dominated by chance and would produce arbitrary warnings; statistical
     * process control convention wants 20-25 before fixing limits. Matches the same
     * reasoning as {@link ThresholdTuner#MIN_EVENTS}.
     */
    public static final int MIN_SAMPLES_FOR_LIMITS = 20;
    /**
     * How many MADs from the median counts as out of family. 3.0 is the conventional
     * equivalent of 3-sigma once MAD is scaled to be consistent for a normal distribution
     * (see {@link #SIGMA_PER_MAD}).
     */
    public static float OUTLIER_MADS = 3.0f;
    /** For normally distributed data, sigma ≈ 1.4826 × MAD. */
    private static final double SIGMA_PER_MAD = 1.4826;
    /**
     * Most recent observed size ratios kept per slot. Enough for the median to settle and to
     * outvote a wrong reference box many times over; bounded so a part inspected daily does
     * not grow its layout file without limit.
     */
    static final int MAX_SIZE_HISTORY = 50;

    public static final class Slot {
        public final int id;
        public String type;
        /** Position in the reference photo, fractional (0-1). */
        public final float x;
        public final float y;
        /**
         * Size of the reference detection's bounding box, as a fraction of the reference
         * frame's width and height. Zero when unknown — a layout labelled before sizes were
         * recorded — in which case the slot's size is never judged.
         */
        public final float w;
        public final float h;
        /** Every size measured for this slot, in mm, across accepted inspections. */
        public final List<Double> samples = new ArrayList<>();
        /**
         * Linear size of the detection found at this slot over the size the reference box
         * predicted, one entry per filed inspection where the slot was present and judged —
         * {@code SlotResult.sizeRatio}, accumulated. Newest last; see {@link #MAX_SIZE_HISTORY}.
         */
        public final List<Float> sizeRatios = new ArrayList<>();

        Slot(int id, String type, float x, float y, float w, float h) {
            this.id = id;
            this.type = type;
            this.x = x;
            this.y = y;
            this.w = w > 0f ? w : 0f;
            this.h = h > 0f ? h : 0f;
        }

        /** Whether the reference recorded a box size for this slot at all. */
        public boolean hasSize() {
            return w > 0f && h > 0f;
        }

        /**
         * How large the hardware at this slot has measured relative to the reference box,
         * as a linear factor: the median of every observed ratio with the reference itself
         * counted once at 1.0. Exactly 1.0 until an inspection has been filed.
         *
         * A median rather than a mean, and the reference as one vote rather than a prior it
         * takes many observations to move: if the reference box swallowed a shadow, two
         * inspections that saw the fastener at 0.55 of it are already enough to say so, and
         * one inspection that merged the same shadow again is not enough to hide it.
         */
        public float expectedSizeFactor() {
            if (sizeRatios.isEmpty()) return 1f;
            List<Float> votes = new ArrayList<>(sizeRatios);
            votes.add(1f);
            java.util.Collections.sort(votes);
            int n = votes.size();
            return n % 2 == 1 ? votes.get(n / 2)
                    : (votes.get(n / 2 - 1) + votes.get(n / 2)) / 2f;
        }

        /** How many filed inspections the size estimate rests on. */
        public int sizeEvidence() {
            return sizeRatios.size();
        }

        public boolean hasLimits() {
            return samples.size() >= MIN_SAMPLES_FOR_LIMITS;
        }

        /** Middle of the observed distribution, or NaN with no samples. */
        public double median() {
            if (samples.isEmpty()) return Double.NaN;
            List<Double> sorted = new ArrayList<>(samples);
            java.util.Collections.sort(sorted);
            int n = sorted.size();
            return n % 2 == 1 ? sorted.get(n / 2)
                    : (sorted.get(n / 2 - 1) + sorted.get(n / 2)) / 2.0;
        }

        /** Median absolute deviation — the robust analogue of standard deviation. */
        public double mad() {
            if (samples.size() < 2) return Double.NaN;
            double med = median();
            List<Double> devs = new ArrayList<>(samples.size());
            for (double s : samples) devs.add(Math.abs(s - med));
            java.util.Collections.sort(devs);
            int n = devs.size();
            return n % 2 == 1 ? devs.get(n / 2) : (devs.get(n / 2 - 1) + devs.get(n / 2)) / 2.0;
        }

        /** MAD expressed on the same scale as a standard deviation. */
        public double robustSigma() {
            return mad() * SIGMA_PER_MAD;
        }

        /**
         * How many robust sigmas {@code sizeMm} sits from this slot's median, or NaN when
         * there is not yet enough history to judge.
         *
         * A zero MAD (every sample identical, common early on with quantised pixel
         * measurements) would divide to infinity, so it is treated as no judgement rather
         * than as infinite confidence.
         */
        public double deviation(double sizeMm) {
            if (!hasLimits()) return Double.NaN;
            double sigma = robustSigma();
            if (!(sigma > 1e-9)) return Double.NaN;
            return Math.abs(sizeMm - median()) / sigma;
        }

        public boolean isOutlier(double sizeMm) {
            double d = deviation(sizeMm);
            return !Double.isNaN(d) && d > OUTLIER_MADS;
        }

        public String describeLimits() {
            if (!hasLimits()) {
                return String.format(Locale.US, "learning (%d/%d samples)",
                        samples.size(), MIN_SAMPLES_FOR_LIMITS);
            }
            return String.format(Locale.US, "%.2f mm ±%.2f (n=%d)",
                    median(), robustSigma() * OUTLIER_MADS, samples.size());
        }
    }

    private final List<Slot> slots = new ArrayList<>();

    public List<Slot> slots() {
        return slots;
    }

    public boolean isEmpty() {
        return slots.isEmpty();
    }

    public Slot addSlot(String type, float x, float y) {
        return addSlot(type, x, y, 0f, 0f);
    }

    /**
     * Adds a slot with the size of the box it was detected in, fractional. Pass zero for
     * either dimension when no box is available and the slot will not be size-checked.
     */
    public Slot addSlot(String type, float x, float y, float w, float h) {
        Slot s = new Slot(slots.size() + 1, type, x, y, w, h);
        slots.add(s);
        return s;
    }

    public Slot slotById(int id) {
        for (Slot s : slots) if (s.id == id) return s;
        return null;
    }

    /**
     * Records a size for a slot.
     *
     * Callers must only pass measurements from inspections judged good. Folding in sizes
     * from parts that failed would widen the distribution toward the defect and, given
     * enough of them, teach the slot to accept it — the same circularity that makes
     * pseudo-labelling unsafe elsewhere in this project.
     */
    public void recordSample(int slotId, double sizeMm) {
        Slot s = slotById(slotId);
        if (s != null && !Double.isNaN(sizeMm) && sizeMm > 0) {
            s.samples.add(sizeMm);
        }
    }

    /**
     * Records the size ratio an inspection measured at a slot.
     *
     * Callers pass only PRESENT findings that were size-judged: a WRONG_TYPE slot holds a
     * different fastener whose size says nothing about the expected one, and an unjudged
     * finding has no ratio. Findings from failed parts are fine — a fastener present at its
     * slot is a valid observation of that fastener whatever went wrong elsewhere — and a
     * genuinely wrong-sized one is a minority the median absorbs.
     */
    public void recordSizeRatio(int slotId, float ratio) {
        Slot s = slotById(slotId);
        if (s == null || Float.isNaN(ratio) || Float.isInfinite(ratio) || ratio <= 0f) return;
        s.sizeRatios.add(ratio);
        while (s.sizeRatios.size() > MAX_SIZE_HISTORY) s.sizeRatios.remove(0);
    }

    /**
     * A copy holding only the slots {@code keep} accepts, with their ids, sizes and histories
     * intact and every layout-level field carried over.
     *
     * Ids are preserved on purpose. A filtered copy that renumbered its slots would report
     * findings under ids the persisted layout does not have, and a size ratio recorded against
     * such a finding would land on the wrong slot or none.
     */
    public FeatureLayout retaining(java.util.function.Predicate<Slot> keep) {
        FeatureLayout kept = new FeatureLayout();
        kept.referenceUri = referenceUri;
        kept.imageWidth = imageWidth;
        kept.imageHeight = imageHeight;
        kept.edgeMethodId = edgeMethodId;
        kept.preprocessingRung = preprocessingRung;
        for (Slot s : slots) {
            if (!keep.test(s)) continue;
            Slot copy = new Slot(s.id, s.type, s.x, s.y, s.w, s.h);
            copy.samples.addAll(s.samples);
            copy.sizeRatios.addAll(s.sizeRatios);
            kept.slots.add(copy);
        }
        return kept;
    }

    // ── Persistence ──────────────────────────────────────────────────────────

    /**
     * Parts with a layout file on this device, sorted. The names are the file stems, which
     * {@link PartFiles#safeName} leaves unchanged for every id it can produce, so each loads
     * back through {@link #load}.
     */
    public static java.util.List<String> savedPartIds(Context context) {
        java.util.List<String> ids = new ArrayList<>();
        File[] files = new File(context.getFilesDir(), DIR).listFiles();
        if (files == null) return ids;
        for (File f : files) {
            String name = f.getName();
            if (name.endsWith(".json")) ids.add(name.substring(0, name.length() - 5));
        }
        java.util.Collections.sort(ids, String.CASE_INSENSITIVE_ORDER);
        return ids;
    }

    public static File fileFor(Context context, String partId) {
        File dir = new File(context.getFilesDir(), DIR);
        if (!dir.exists()) dir.mkdirs();
        // Shared with the library, threshold and crop stores: these all derive a path for the
        // same part, so a divergent sanitiser would silently point one at a different file.
        return new File(dir, PartFiles.safeName(partId) + ".json");
    }

    /**
     * The reference image these slots were labelled against, or null for a layout that was
     * derived rather than labelled.
     *
     * A layout is only meaningful for the photo it describes. Persisting the URI is what lets
     * {@link #loadFor} refuse a layout whose reference image has since been replaced, instead
     * of silently reporting correct hardware as misplaced against an arrangement that no
     * longer exists.
     */
    private String referenceUri;

    public String referenceUri() {
        return referenceUri;
    }

    public void setReferenceUri(String uri) {
        this.referenceUri = uri;
    }

    /**
     * Working-image dimensions the slot fractions were measured against, or 0 when unknown.
     *
     * Needed for one thing: turning a slot's fractional box back into pixels, so the size
     * check can refuse slots too small for the detector's boxes to be repeatable (see
     * {@code FeatureLocalisation.MIN_SIZE_JUDGED_PX}). Fractions alone cannot say how many
     * pixels a feature spanned, and the floor was measured in pixels.
     */
    private int imageWidth;
    private int imageHeight;

    public int imageWidth() {
        return imageWidth;
    }

    public int imageHeight() {
        return imageHeight;
    }

    public void setImageSize(int width, int height) {
        this.imageWidth = Math.max(0, width);
        this.imageHeight = Math.max(0, height);
    }

    /**
     * Catalogue id of the edge method the reference was detected with
     * ({@code EdgeMethodLab}), or null for a layout that predates the choice.
     *
     * Kept because a method change is a detector change: the same fastener can come out as
     * one box under one method and two under another, and an inspection run under a different
     * method from its reference would then report a count mismatch on a good part. The results
     * screen compares the two and says so, rather than filing the difference as a defect.
     */
    private String edgeMethodId;

    public String edgeMethodId() {
        return edgeMethodId;
    }

    public void setEdgeMethodId(String id) {
        this.edgeMethodId = id;
    }

    /**
     * Label of the {@link PreprocessingStrength} the reference was detected with when the edge
     * method was adaptive, or null when it was not (or the layout predates the choice).
     *
     * Informational: an inspection under the adaptive method is entitled to pick a different
     * rung — that is the point — and the results screen names both so a size disagreement can
     * be read with that in mind.
     */
    private String preprocessingRung;

    public String preprocessingRung() {
        return preprocessingRung;
    }

    public void setPreprocessingRung(String label) {
        this.preprocessingRung = label;
    }

    /**
     * Catalogue id of the edge method that actually drew this reference's boxes, or null
     * when that cannot be known.
     *
     * <p>Under the adaptive stage the {@code preprocessing} field holds what ran: a rung label
     * ({@code "light"}) for layouts written before every method was on the ballot, or a
     * catalogue id ({@code "shipped-hybrid"}) since. Under a fixed method it is null and the
     * method is {@code edgeMethod}; a layout that predates both was the shipped stack, the
     * only method that existed. Adaptive with no record of what ran is the one unknowable
     * case.
     */
    public String referenceMethodId() {
        if (preprocessingRung != null) {
            PreprocessingStrength rung = PreprocessingStrength.byLabel(preprocessingRung);
            if (rung != null) return rung.id;
            if (EdgeMethodLab.byId(preprocessingRung) != null) return preprocessingRung;
        }
        if (edgeMethodId == null) return EdgeMethodLab.SHIPPED_ID;
        if (EdgeMethodLab.ADAPTIVE_ID.equals(edgeMethodId)) return null;
        return EdgeMethodLab.byId(edgeMethodId) != null ? edgeMethodId : null;
    }


    /**
     * The shorter side of a slot's reference box in working-image pixels, or NaN when either
     * the box or the image size is unknown.
     */
    public float shortSidePx(Slot s) {
        if (s == null || !s.hasSize() || imageWidth <= 0 || imageHeight <= 0) return Float.NaN;
        return Math.min(s.w * imageWidth, s.h * imageHeight);
    }

    /** Slot counts by type — the expected configuration this reference defines. */
    public java.util.Map<String, Integer> countsByType() {
        java.util.Map<String, Integer> counts = new java.util.LinkedHashMap<>();
        for (Slot s : slots) {
            Integer prev = counts.get(s.type);
            counts.put(s.type, prev == null ? 1 : prev + 1);
        }
        return counts;
    }

    /**
     * Loads the layout only if it was labelled against {@code expectedUri}.
     *
     * Returns null on a mismatch so the caller falls back to classifying the reference image,
     * which is recoverable, rather than comparing an inspection against a stale arrangement.
     */
    public static FeatureLayout loadFor(Context context, String partId, String expectedUri) {
        FeatureLayout layout = load(context, partId);
        if (layout == null) return null;
        if (layout.referenceUri == null || expectedUri == null
                || !layout.referenceUri.equals(expectedUri)) {
            Log.d(TAG, "Layout for " + partId + " was labelled against '" + layout.referenceUri
                    + "' but the profile now points at '" + expectedUri + "' — ignoring it.");
            return null;
        }
        return layout;
    }

    public void save(Context context, String partId) {
        try {
            JSONArray arr = new JSONArray();
            for (Slot s : slots) {
                JSONObject o = new JSONObject();
                o.put("id", s.id);
                o.put("type", s.type);
                o.put("x", s.x);
                o.put("y", s.y);
                if (s.hasSize()) {
                    o.put("w", s.w);
                    o.put("h", s.h);
                }
                JSONArray samples = new JSONArray();
                for (double d : s.samples) samples.put(d);
                o.put("samples", samples);
                if (!s.sizeRatios.isEmpty()) {
                    JSONArray ratios = new JSONArray();
                    for (float r : s.sizeRatios) ratios.put((double) r);
                    o.put("sizeRatios", ratios);
                }
                arr.put(o);
            }
            JSONObject root = new JSONObject();
            root.put("slots", arr);
            if (referenceUri != null) root.put("referenceUri", referenceUri);
            if (imageWidth > 0 && imageHeight > 0) {
                root.put("imageWidth", imageWidth);
                root.put("imageHeight", imageHeight);
            }
            if (edgeMethodId != null) root.put("edgeMethod", edgeMethodId);
            if (preprocessingRung != null) root.put("preprocessing", preprocessingRung);
            try (FileOutputStream fos = new FileOutputStream(fileFor(context, partId))) {
                fos.write(root.toString().getBytes(StandardCharsets.UTF_8));
            }
        } catch (Exception e) {
            Log.e(TAG, "Failed to save layout for " + partId, e);
        }
    }

    public static FeatureLayout load(Context context, String partId) {
        File f = fileFor(context, partId);
        if (!f.exists()) return null;
        try (InputStream is = new java.io.FileInputStream(f)) {
            java.io.ByteArrayOutputStream bos = new java.io.ByteArrayOutputStream();
            byte[] buf = new byte[4096];
            int n;
            while ((n = is.read(buf)) != -1) bos.write(buf, 0, n);
            JSONObject root = new JSONObject(new String(bos.toByteArray(), StandardCharsets.UTF_8));
            JSONArray arr = root.getJSONArray("slots");
            FeatureLayout layout = new FeatureLayout();
            layout.referenceUri = root.optString("referenceUri", null);
            layout.setImageSize(root.optInt("imageWidth", 0), root.optInt("imageHeight", 0));
            layout.edgeMethodId = root.optString("edgeMethod", null);
            layout.preprocessingRung = root.optString("preprocessing", null);
            for (int i = 0; i < arr.length(); i++) {
                JSONObject o = arr.getJSONObject(i);
                // Sizes are optional: layouts written before they were recorded have none,
                // and read back as slots that are positioned but never size-judged.
                Slot s = new Slot(o.getInt("id"), o.getString("type"),
                        (float) o.getDouble("x"), (float) o.getDouble("y"),
                        (float) o.optDouble("w", 0), (float) o.optDouble("h", 0));
                JSONArray samples = o.optJSONArray("samples");
                if (samples != null) {
                    for (int j = 0; j < samples.length(); j++) s.samples.add(samples.getDouble(j));
                }
                JSONArray ratios = o.optJSONArray("sizeRatios");
                if (ratios != null) {
                    for (int j = 0; j < ratios.length(); j++) {
                        s.sizeRatios.add((float) ratios.getDouble(j));
                    }
                }
                layout.slots.add(s);
            }
            return layout.isEmpty() ? null : layout;
        } catch (Exception e) {
            Log.e(TAG, "Failed to load layout for " + partId, e);
            return null;
        }
    }
}
