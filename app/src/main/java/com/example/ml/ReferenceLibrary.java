package com.example.ml;

import android.graphics.Color;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Few-shot classifier library, mirroring build_reference_library() and
 * classify_crop() in python/EdgeComputingImprove.py.
 *
 * Holds one prototype embedding per feature type (the mean of that type's
 * example embeddings). Classification is cosine similarity of a query embedding
 * against each prototype, thresholded per {@link #DEFAULT_SIMILARITY_THRESHOLD}.
 *
 * The Python prototype also scored text prompts, but weighted them at 0.0
 * (TEXT_PROMPT_WEIGHT), so that path never contributed. It is not reproduced here:
 * {@link FeatureEmbedder} is a vision-only encoder with no text tower to score with.
 */
public final class ReferenceLibrary {

    private static final String TAG = "ReferenceLibrary";

    public static final String UNKNOWN = "unknown";

    /** Canonical feature types (Python REFERENCE_DIR subfolders / TEXT_PROMPTS keys). */
    public static final String[] FEATURE_TYPES =
            {"bolt_head", "hole", "rivet_head", "screw", "stud"};

    // Starting thresholds. These are only the defaults — each library instance carries
    // its own values so {@link ThresholdTuner} can fit them per-part from that part's
    // confirmed classification history.
    //
    // Calibrated for {@link FeatureEmbedder#ENCODER_ID} by sweeping every
    // (similarity, margin) pair over leave-one-out prototype classification of the
    // bundled reference crops (see python/compare_encoders.py). These numbers are
    // NOT portable across encoders — the previous CLIP encoder scored every crop in
    // a 0.86-0.97 band, which made a 0.85 threshold unreachable and therefore dead:
    // it never once rejected anything, and the margin check was left doing all the
    // work at a setting that discarded half of the *correct* answers as near-ties.
    //
    // At 0.74/0.02 the current encoder accepts 94.1% of crops correctly with zero
    // false accepts over the 34 available crops. Note the inversion: with DINOv2 the
    // similarity threshold does the rejecting and the margin is a light safety net,
    // exactly the reverse of the CLIP configuration.
    public static final float DEFAULT_SIMILARITY_THRESHOLD = 0.74f;
    /**
     * Minimum gap between the best and second-best prototype score. Below this the
     * match is too close to call and we report "unknown" with the best guess as
     * closestMatch, rather than silently picking a winner.
     *
     * Kept deliberately low rather than removed. The sweep found that this encoder's
     * mistakes are confidently wrong (the one leave-one-out error scored 0.734 with a
     * 0.091 margin) rather than near-ties, so raising the margin buys nothing here and
     * only costs correct answers. It stays non-zero because near-ties remain the
     * failure mode this check exists for, and they get more likely as inspectors add
     * feature types that genuinely do sit close together.
     */
    public static final float DEFAULT_MARGIN_THRESHOLD = 0.02f;
    /**
     * Above this a classification is reported as "high" confidence, which is also what
     * makes it eligible for {@link ActiveLearningPolicy} spot checks. Set near the upper
     * quartile of correct scores (measured 0.881) so "high" means genuinely top-tier
     * rather than merely accepted.
     */
    public static final float DEFAULT_HIGH_CONFIDENCE = 0.88f;

    private float similarityThreshold = DEFAULT_SIMILARITY_THRESHOLD;
    private float highConfidence = DEFAULT_HIGH_CONFIDENCE;
    private float marginThreshold = DEFAULT_MARGIN_THRESHOLD;

    /** Why a classification landed where it did — drives active-learning review selection. */
    public enum Reason {
        /** Accepted: cleared both the absolute threshold and the runner-up margin. */
        ACCEPTED,
        /** Best score was below the absolute similarity threshold. */
        BELOW_THRESHOLD,
        /** Score cleared the threshold but two types were too close to call. */
        AMBIGUOUS,
        /** No prototypes at all (no encoder model, or nothing labelled yet). */
        NO_LIBRARY
    }

    public float getSimilarityThreshold() { return similarityThreshold; }
    public float getMarginThreshold() { return marginThreshold; }
    public float getHighConfidence() { return highConfidence; }

    public void setThresholds(float similarityThreshold, float marginThreshold) {
        this.similarityThreshold = similarityThreshold;
        this.marginThreshold = marginThreshold;
    }

    // Running accumulation so prototypes can be built incrementally (in-app builder
    // or offline bootstrap). prototype(type) = sums[type] / counts[type].
    private final Map<String, float[]> sums = new LinkedHashMap<>();
    private final Map<String, Integer> counts = new LinkedHashMap<>();

    public static final class Classification {
        public final String type;         // best type, or "unknown" if below threshold
        public final String closestMatch; // nearest type regardless of threshold
        public final float score;         // best cosine similarity
        public final String confidence;   // "high" | "medium" | "low"
        public final String runnerUp;     // second-best type, or null when only one prototype exists
        public final float runnerUpScore; // second-best cosine similarity, or -1
        public final Reason reason;
        /** Cosine similarity against every prototype, so the UI can show the full picture. */
        public final Map<String, Float> scores;

        Classification(String type, String closestMatch, float score, String confidence,
                       String runnerUp, float runnerUpScore, Reason reason,
                       Map<String, Float> scores) {
            this.type = type;
            this.closestMatch = closestMatch;
            this.score = score;
            this.confidence = confidence;
            this.runnerUp = runnerUp;
            this.runnerUpScore = runnerUpScore;
            this.reason = reason;
            this.scores = scores;
        }

        /** Gap to the runner-up; Float.MAX_VALUE when there is no runner-up to compete with. */
        public float margin() {
            return runnerUp == null ? Float.MAX_VALUE : score - runnerUpScore;
        }
    }

    public boolean isEmpty() {
        return counts.isEmpty();
    }

    public List<String> types() {
        return new ArrayList<>(counts.keySet());
    }

    /**
     * Accumulates one example embedding for a feature type into the running mean.
     * The embedding is L2-normalized first so every example contributes an equal
     * "vote" on direction regardless of its raw magnitude — otherwise a few
     * larger-magnitude crops can dominate the mean and drag the prototype toward
     * a neighboring class (this is what was causing stud crops to skew into the
     * bolt_head prototype despite having enough examples).
     */
    public void addExample(String type, float[] embedding) {
        if (type == null || embedding == null) return;
        float[] normalized = normalize(embedding);
        if (normalized == null) return;
        float[] sum = sums.get(type);
        if (sum == null) {
            sums.put(type, normalized);
            counts.put(type, 1);
        } else {
            int n = Math.min(sum.length, normalized.length);
            for (int i = 0; i < n; i++) sum[i] += normalized[i];
            counts.put(type, counts.get(type) + 1);
        }
    }

    /** Returns a unit-length copy of v, or null if v has zero magnitude. */
    private static float[] normalize(float[] v) {
        double normSq = 0;
        for (float x : v) normSq += (double) x * x;
        if (normSq == 0) return null;
        float norm = (float) Math.sqrt(normSq);
        float[] out = new float[v.length];
        for (int i = 0; i < v.length; i++) out[i] = v[i] / norm;
        return out;
    }

    /** Averaged prototype for a type, or null if none. */
    public float[] prototype(String type) {
        float[] sum = sums.get(type);
        Integer c = counts.get(type);
        if (sum == null || c == null || c == 0) return null;
        float[] proto = new float[sum.length];
        for (int i = 0; i < sum.length; i++) proto[i] = sum[i] / c;
        return proto;
    }

    /**
     * Mirrors classify_crop(): cosine similarity vs each prototype + thresholding.
     * Also requires the winner to beat the runner-up by MARGIN_THRESHOLD — a high
     * absolute score isn't enough if a different type scored almost as high, since
     * that's exactly the case where two classes (e.g. stud vs bolt_head) sit close
     * together in embedding space and a small centroid shift picks the wrong one.
     */
    public Classification classify(float[] embedding) {
        if (embedding == null || counts.isEmpty()) {
            return new Classification(UNKNOWN, "none", 0f, "low", null, -1f, Reason.NO_LIBRARY,
                    new LinkedHashMap<>());
        }

        String bestType = null;
        float bestScore = -1f;
        String secondType = null;
        float secondScore = -1f;
        Map<String, Float> allScores = new LinkedHashMap<>();
        for (String type : counts.keySet()) {
            float score = cosine(embedding, prototype(type));
            allScores.put(type, score);
            if (score > bestScore) {
                secondType = bestType;
                secondScore = bestScore;
                bestScore = score;
                bestType = type;
            } else if (score > secondScore) {
                secondScore = score;
                secondType = type;
            }
        }

        String closest = bestType;
        String type = bestType;
        boolean belowThreshold = bestScore < similarityThreshold;
        // Inclusive: a gap sitting exactly on the threshold is a tie, not a pass. With a strict
        // "<" a 0.93-vs-0.92 pair cleared a 0.01 margin and was committed silently.
        boolean tooClose = secondType != null && (bestScore - secondScore) <= marginThreshold;

        Reason reason;
        if (belowThreshold) {
            reason = Reason.BELOW_THRESHOLD;
            type = UNKNOWN;
        } else if (tooClose) {
            reason = Reason.AMBIGUOUS;
            type = UNKNOWN;
        } else {
            reason = Reason.ACCEPTED;
        }

        String confidence;
        if (tooClose) confidence = "low";
        else if (bestScore > highConfidence) confidence = "high";
        else if (bestScore > similarityThreshold) confidence = "medium";
        else confidence = "low";

        return new Classification(type, closest, bestScore, confidence, secondType, secondScore,
                reason, allScores);
    }

    private static float cosine(float[] a, float[] b) {
        if (a == null || b == null) return -1f;
        int n = Math.min(a.length, b.length);
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < n; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        if (na == 0 || nb == 0) return -1f;
        return (float) (dot / (Math.sqrt(na) * Math.sqrt(nb)));
    }

    // ── Persistence ──────────────────────────────────────────────────────────
    // Stored as { "_encoder": "<id>", "_counts": {"type": n}, "type": [floats...] },
    // where each type's array is its averaged prototype and _counts records how many
    // examples that average was taken over.

    /**
     * Which embedding space this library's prototypes live in. Prototypes are only
     * meaningful against queries from the same encoder, and {@link #cosine} compares
     * only the overlapping prefix of two vectors — so a library built by a different
     * encoder does not fail, it returns confident nonsense. Persisting the id is what
     * turns that silent corruption into a detectable mismatch.
     */
    private String encoderId = FeatureEmbedder.ENCODER_ID;

    /** Key holding {@link #encoderId}; underscore-prefixed so it can't collide with a type name. */
    private static final String ENCODER_KEY = "_encoder";

    /**
     * Key holding the example count behind each prototype, same underscore convention.
     *
     * Without it a loaded prototype comes back as a single example, so the next
     * {@link #addExample} averages the entire accumulated history against one new crop
     * fifty-fifty — and the one after that leaves the original contributing a quarter. A
     * library built from thirty labelled crops could be dragged onto a different feature
     * by three corrections, including three mistaken ones. Persisting the count keeps the
     * running mean weighted the way it was when the examples were actually added.
     *
     * Absent in files written before this existed; those load as one example each, which is
     * exactly the behaviour they were saved under.
     */
    private static final String COUNTS_KEY = "_counts";

    public String encoderId() {
        return encoderId;
    }

    public void save(File file) {
        try {
            File parent = file.getParentFile();
            if (parent != null && !parent.exists()) parent.mkdirs();
            JSONObject root = new JSONObject();
            root.put(ENCODER_KEY, encoderId);
            // Prototypes are stored as the mean, not the running sum, so the file stays
            // readable on its own; the counts alongside are what make the mean resumable.
            JSONObject countsJson = new JSONObject();
            for (String type : counts.keySet()) {
                float[] proto = prototype(type);
                JSONArray arr = new JSONArray();
                for (float v : proto) arr.put((double) v);
                root.put(type, arr);
                countsJson.put(type, counts.get(type));
            }
            root.put(COUNTS_KEY, countsJson);
            try (FileOutputStream fos = new FileOutputStream(file)) {
                fos.write(root.toString().getBytes(StandardCharsets.UTF_8));
            }
        } catch (Exception e) {
            Log.e(TAG, "save() failed for " + file, e);
        }
    }

    public static ReferenceLibrary load(File file) {
        if (file == null || !file.exists()) return null;
        try (InputStream is = new java.io.FileInputStream(file)) {
            ReferenceLibrary lib = fromJson(PartFiles.readAllText(is));
            if (lib == null) {
                // fromJson already logged why. Named here so the stale file is identifiable.
                Log.w(TAG, "Ignoring unusable library at " + file
                        + " — it will be rebuilt from reference crops.");
            }
            return lib;
        } catch (Exception e) {
            Log.e(TAG, "load() failed for " + file, e);
            return null;
        }
    }

    /**
     * Parses a saved library, or returns null if it was written by a different encoder.
     *
     * Rejecting is the safe direction: a caller that gets null rebuilds from reference
     * crops, which is recoverable, whereas accepting a mismatched library means every
     * subsequent classification is wrong in a way no threshold or log would reveal.
     * Libraries written before encoder tagging existed have no id and are also rejected,
     * since they came from the CLIP encoder and their 512-d prototypes cannot be
     * compared against this encoder's 384-d embeddings.
     */
    public static ReferenceLibrary fromJson(String json) {
        try {
            JSONObject root = new JSONObject(json);
            String storedEncoder = root.optString(ENCODER_KEY, null);
            if (storedEncoder == null || !storedEncoder.equals(FeatureEmbedder.ENCODER_ID)) {
                Log.w(TAG, "Library was built by encoder '"
                        + (storedEncoder == null ? "<untagged/legacy>" : storedEncoder)
                        + "' but this build uses '" + FeatureEmbedder.ENCODER_ID
                        + "'. Discarding it rather than comparing incompatible embeddings.");
                return null;
            }

            ReferenceLibrary lib = new ReferenceLibrary();
            lib.encoderId = storedEncoder;
            JSONObject storedCounts = root.optJSONObject(COUNTS_KEY);
            java.util.Iterator<String> keys = root.keys();
            while (keys.hasNext()) {
                String type = keys.next();
                if (ENCODER_KEY.equals(type) || COUNTS_KEY.equals(type)) continue;
                JSONArray arr = root.getJSONArray(type);
                float[] proto = new float[arr.length()];
                for (int i = 0; i < arr.length(); i++) proto[i] = (float) arr.getDouble(i);

                // The file stores the mean; restore the running sum it came from so further
                // examples are averaged in at their true weight rather than at one half.
                // Files predating COUNTS_KEY have no entry and load as one example, which is
                // the weighting they were written under.
                int count = storedCounts != null ? Math.max(1, storedCounts.optInt(type, 1)) : 1;
                for (int i = 0; i < proto.length; i++) proto[i] *= count;
                lib.sums.put(type, proto);
                lib.counts.put(type, count);
            }
            return lib.isEmpty() ? null : lib;
        } catch (Exception e) {
            Log.e(TAG, "fromJson() failed", e);
            return null;
        }
    }

    // ── Display helpers (used by the overlay + results screens) ───────────────

    private static final Map<String, Integer> COLORS = new LinkedHashMap<>();
    private static final Map<String, String> DISPLAY = new LinkedHashMap<>();
    static {
        COLORS.put("bolt_head", Color.rgb(56, 189, 248));   // sky blue
        COLORS.put("hole", Color.rgb(251, 146, 60));        // orange
        COLORS.put("rivet_head", Color.rgb(250, 204, 21));  // yellow
        COLORS.put("screw", Color.rgb(74, 222, 128));       // green
        COLORS.put("stud", Color.rgb(167, 139, 250));       // violet
        COLORS.put(UNKNOWN, Color.rgb(255, 0, 255));        // magenta (Python)

        DISPLAY.put("bolt_head", "Bolt Heads");
        DISPLAY.put("hole", "Holes");
        DISPLAY.put("rivet_head", "Rivet Heads");
        DISPLAY.put("screw", "Screws");
        DISPLAY.put("stud", "Studs");
        DISPLAY.put(UNKNOWN, "Unknown");
    }

    public static int colorFor(String type) {
        Integer c = COLORS.get(type);
        return c != null ? c : Color.rgb(128, 128, 128);
    }

    public static String displayName(String type) {
        String d = DISPLAY.get(type);
        return d != null ? d : type;
    }
}
