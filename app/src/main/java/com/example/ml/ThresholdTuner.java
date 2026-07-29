package com.example.ml;

import android.content.Context;
import android.util.Log;

import com.example.data.entity.ClassificationEvent;

import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.List;

/**
 * Fits {@code similarityThreshold} / {@code marginThreshold} per part from that part's
 * confirmed classification history, replacing hand-tuned global constants.
 *
 * Each reviewed {@link ClassificationEvent} records what the classifier scored and what
 * the human said it actually was. Replaying candidate threshold pairs over that history
 * tells us which pair would have produced the most correct answers, which is a far better
 * basis than guessing 0.85 and hoping. The search is a plain grid — a few hundred events
 * times ~600 combinations is trivial, and it avoids gradient/convexity assumptions that
 * don't hold for a step-function accuracy metric.
 *
 * Guards that keep this from making things worse:
 *  - Nothing is fitted until {@link #MIN_EVENTS} confirmations exist for the part.
 *  - A fitted pair must beat the current thresholds by {@link #MIN_IMPROVEMENT} to be adopted.
 *  - Ties break toward the defaults, so noise doesn't cause aimless drift between runs.
 */
public final class ThresholdTuner {

    private static final String TAG = "ThresholdTuner";
    private static final String DIR = "thresholds";

    /** Below this many confirmed labels, a fit would be overfitting noise. */
    public static final int MIN_EVENTS = 20;
    /** Accuracy gain required before replacing the incumbent thresholds. */
    public static final float MIN_IMPROVEMENT = 0.02f;

    private static final float SIM_MIN = 0.60f, SIM_MAX = 0.98f, SIM_STEP = 0.01f;
    private static final float MARGIN_MIN = 0.00f, MARGIN_MAX = 0.12f, MARGIN_STEP = 0.005f;

    public static final class Tuned {
        public final float similarityThreshold;
        public final float marginThreshold;
        public final int sampleSize;
        public final float accuracy;         // accuracy of the fitted pair over the history
        public final float baselineAccuracy; // accuracy the defaults would have scored

        public Tuned(float similarityThreshold, float marginThreshold, int sampleSize,
                     float accuracy, float baselineAccuracy) {
            this.similarityThreshold = similarityThreshold;
            this.marginThreshold = marginThreshold;
            this.sampleSize = sampleSize;
            this.accuracy = accuracy;
            this.baselineAccuracy = baselineAccuracy;
        }
    }

    private ThresholdTuner() {}

    /**
     * Fits thresholds over a part's confirmed history. Returns null when there isn't
     * enough evidence, or when nothing beats the defaults by a meaningful margin.
     */
    public static Tuned fit(List<ClassificationEvent> reviewed) {
        if (reviewed == null || reviewed.size() < MIN_EVENTS) return null;

        float baseline = accuracyAt(reviewed,
                ReferenceLibrary.DEFAULT_SIMILARITY_THRESHOLD,
                ReferenceLibrary.DEFAULT_MARGIN_THRESHOLD);

        float bestSim = ReferenceLibrary.DEFAULT_SIMILARITY_THRESHOLD;
        float bestMargin = ReferenceLibrary.DEFAULT_MARGIN_THRESHOLD;
        float bestAccuracy = baseline;
        float bestDistanceToDefault = 0f;

        for (float sim = SIM_MIN; sim <= SIM_MAX + 1e-6f; sim += SIM_STEP) {
            for (float margin = MARGIN_MIN; margin <= MARGIN_MAX + 1e-6f; margin += MARGIN_STEP) {
                float acc = accuracyAt(reviewed, sim, margin);
                float distance = Math.abs(sim - ReferenceLibrary.DEFAULT_SIMILARITY_THRESHOLD)
                        + Math.abs(margin - ReferenceLibrary.DEFAULT_MARGIN_THRESHOLD);
                // Strictly better wins; equal accuracy only wins if it's closer to the
                // defaults, which keeps repeat runs from wandering across a flat optimum.
                if (acc > bestAccuracy + 1e-6f
                        || (Math.abs(acc - bestAccuracy) <= 1e-6f && distance < bestDistanceToDefault)) {
                    bestAccuracy = acc;
                    bestSim = sim;
                    bestMargin = margin;
                    bestDistanceToDefault = distance;
                }
            }
        }

        if (bestAccuracy < baseline + MIN_IMPROVEMENT) {
            Log.d(TAG, "No threshold pair beat the baseline by " + MIN_IMPROVEMENT
                    + " (best=" + bestAccuracy + ", baseline=" + baseline + "); keeping current values.");
            return null;
        }

        return new Tuned(round(bestSim), round(bestMargin), reviewed.size(), bestAccuracy, baseline);
    }

    /** Fraction of the history a given threshold pair would have gotten right. */
    private static float accuracyAt(List<ClassificationEvent> events, float sim, float margin) {
        int correct = 0;
        int total = 0;
        for (ClassificationEvent e : events) {
            String truth = e.getConfirmedType();
            if (truth == null) continue;
            total++;
            if (replay(e, sim, margin).equals(truth)) correct++;
        }
        return total == 0 ? 0f : (float) correct / total;
    }

    /**
     * What the classifier would have output for this event under the given thresholds.
     *
     * Events store the *winning prototype* in predictedType — not the reported label —
     * precisely so a replay can ask "what if the threshold had been lower". A crop
     * reported as "unknown" at 0.85 still knows it was nearest to "stud" at 0.83.
     */
    private static String replay(ClassificationEvent e, float sim, float margin) {
        String winner = e.getPredictedType();
        if (winner == null || ReferenceLibrary.UNKNOWN.equals(winner) || "none".equals(winner)) {
            // No prototype competed at all (empty library) — thresholds can't rescue it.
            return ReferenceLibrary.UNKNOWN;
        }
        if (e.getPredictedScore() < sim) return ReferenceLibrary.UNKNOWN;
        if (e.getRunnerUpType() != null
                && (e.getPredictedScore() - e.getRunnerUpScore()) < margin) {
            return ReferenceLibrary.UNKNOWN;
        }
        return winner;
    }

    // ── Persistence (filesDir/thresholds/<partId>.json) ──────────────────────

    public static void save(Context context, String partId, Tuned tuned) {
        try {
            File dir = new File(context.getFilesDir(), DIR);
            if (!dir.exists()) dir.mkdirs();
            JSONObject root = new JSONObject();
            root.put("similarityThreshold", tuned.similarityThreshold);
            root.put("marginThreshold", tuned.marginThreshold);
            root.put("sampleSize", tuned.sampleSize);
            root.put("accuracy", tuned.accuracy);
            root.put("baselineAccuracy", tuned.baselineAccuracy);
            root.put("fittedAt", System.currentTimeMillis());
            try (FileOutputStream fos = new FileOutputStream(fileFor(context, partId))) {
                fos.write(root.toString().getBytes(StandardCharsets.UTF_8));
            }
        } catch (Exception e) {
            Log.e(TAG, "Failed to save tuned thresholds for " + partId, e);
        }
    }

    public static Tuned load(Context context, String partId) {
        File f = fileFor(context, partId);
        if (!f.exists()) return null;
        try (InputStream is = new java.io.FileInputStream(f)) {
            java.io.ByteArrayOutputStream bos = new java.io.ByteArrayOutputStream();
            byte[] buf = new byte[2048];
            int n;
            while ((n = is.read(buf)) != -1) bos.write(buf, 0, n);
            JSONObject root = new JSONObject(new String(bos.toByteArray(), StandardCharsets.UTF_8));
            return new Tuned(
                    (float) root.getDouble("similarityThreshold"),
                    (float) root.getDouble("marginThreshold"),
                    root.optInt("sampleSize", 0),
                    (float) root.optDouble("accuracy", 0),
                    (float) root.optDouble("baselineAccuracy", 0));
        } catch (Exception e) {
            Log.e(TAG, "Failed to load tuned thresholds for " + partId, e);
            return null;
        }
    }

    /** Applies any stored per-part thresholds to a library before it classifies anything. */
    public static Tuned applyTo(Context context, String partId, ReferenceLibrary library) {
        Tuned tuned = load(context, partId);
        if (tuned != null && library != null) {
            library.setThresholds(tuned.similarityThreshold, tuned.marginThreshold);
        }
        return tuned;
    }

    private static File fileFor(Context context, String partId) {
        String safe = partId == null || partId.isEmpty()
                ? "unknown_part" : partId.replaceAll("[^A-Za-z0-9_.-]", "_");
        return new File(new File(context.getFilesDir(), DIR), safe + ".json");
    }

    private static float round(float v) {
        return Math.round(v * 1000f) / 1000f;
    }
}
