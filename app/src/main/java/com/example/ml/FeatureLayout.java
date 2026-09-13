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

    // ── Persistence ──────────────────────────────────────────────────────────

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
                arr.put(o);
            }
            JSONObject root = new JSONObject();
            root.put("slots", arr);
            if (referenceUri != null) root.put("referenceUri", referenceUri);
            if (imageWidth > 0 && imageHeight > 0) {
                root.put("imageWidth", imageWidth);
                root.put("imageHeight", imageHeight);
            }
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
                layout.slots.add(s);
            }
            return layout.isEmpty() ? null : layout;
        } catch (Exception e) {
            Log.e(TAG, "Failed to load layout for " + partId, e);
            return null;
        }
    }
}
