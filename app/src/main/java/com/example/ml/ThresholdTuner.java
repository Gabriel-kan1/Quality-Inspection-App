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
 * basis than shipping one global guess. The search is a plain grid — a few hundred events
 * times ~2,000 combinations is trivial, and it avoids gradient/convexity assumptions that
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
    /**
     * A fitted pair must also be right this often in absolute terms, not merely better than the
     * defaults. Beating a bad baseline by two points is not evidence of a good threshold: a fit
     * once scored 34% against a 21% baseline and was adopted, so the classifier committed on
     * near-ties that the operator then had to correct two times in three. When no pair clears
     * this bar the confirmed history is too inconsistent to tune from and the defaults stand.
     */
    public static final float MIN_ACCURACY = 0.60f;

    /**
     * Search bounds, scaled to the score distribution {@link FeatureEmbedder} actually
     * produces: correct matches land between roughly 0.65 and 0.93 on the bundled crops,
     * so the grid brackets that with a little room on each side.
     *
     * These were 0.60-0.98 when the encoder was CLIP, whose scores all sat above 0.86 —
     * two thirds of that range addressed scores that could never occur. Bounds that don't
     * overlap the encoder's real output cost accuracy silently: the grid spends its search
     * on dead values and the fit is drawn from whatever narrow slice happens to be live.
     */
    private static final float SIM_MIN = 0.50f, SIM_MAX = 0.95f, SIM_STEP = 0.01f;
    /**
     * The margin floor is deliberately above zero. Tuning may loosen the ambiguity check but
     * must never switch it off: two prototypes scoring within a hair of each other means the
     * embedding genuinely cannot separate them, no matter what the confirmed history implies.
     *
     * Without a floor the search will happily pick 0 whenever the history is one-sided — a run
     * of reviews that all agreed with the top guess is enough — and the classifier goes back to
     * silently committing on coin-flips, which is the failure this check exists to prevent.
     *
     * The ceiling was 0.12, which predates the encoder swap and is now below the median
     * runner-up gap of a correct match (0.185). The tuner could not have reached a useful
     * margin even when the history called for one; it now spans the observed range.
     */
    private static final float MARGIN_MIN = 0.02f, MARGIN_MAX = 0.25f, MARGIN_STEP = 0.005f;

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
     *
     * Events from a different encoder are dropped before anything is counted, including
     * the {@link #MIN_EVENTS} check — a mixed history is worse than a short one. Scores
     * from two encoders don't share a numeric range, so replaying one threshold across
     * both fits to an average of two distributions and lands somewhere useful for
     * neither. The filter is here rather than in the query so every caller inherits it.
     */
    public static List<ClassificationEvent> forCurrentEncoder(List<ClassificationEvent> events) {
        List<ClassificationEvent> kept = new java.util.ArrayList<>();
        if (events == null) return kept;
        for (ClassificationEvent e : events) {
            if (FeatureEmbedder.ENCODER_ID.equals(e.getEncoderId())) kept.add(e);
        }
        int dropped = events.size() - kept.size();
        if (dropped > 0) {
            Log.d(TAG, "Ignoring " + dropped + " of " + events.size()
                    + " confirmed events from a previous encoder; fitting on " + kept.size()
                    + " scored by '" + FeatureEmbedder.ENCODER_ID + "'.");
        }
        return kept;
    }

    public static Tuned fit(List<ClassificationEvent> allReviewed) {
        List<ClassificationEvent> reviewed = forCurrentEncoder(allReviewed);
        if (reviewed.size() < MIN_EVENTS) return null;

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
        if (bestAccuracy < MIN_ACCURACY) {
            Log.d(TAG, "Best fit is only " + Math.round(bestAccuracy * 100) + "% accurate over "
                    + reviewed.size() + " labels, under the " + Math.round(MIN_ACCURACY * 100)
                    + "% floor; history is too inconsistent to tune from. Keeping defaults.");
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
        // Inclusive, matching ReferenceLibrary.classify(): a gap sitting exactly on the
        // threshold is a tie, not a pass. With a strict "<" here the replay credited a
        // threshold pair for boundary cases the live classifier would have rejected, so
        // the tuner optimised against behaviour the app does not have.
        if (e.getRunnerUpType() != null
                && (e.getPredictedScore() - e.getRunnerUpScore()) <= margin) {
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
            // Thresholds are only meaningful for the score distribution they were fitted
            // against, so they are tagged like libraries are. See load().
            root.put("encoder", FeatureEmbedder.ENCODER_ID);
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

    /**
     * Loads a part's fitted thresholds, or null to fall back to the defaults.
     *
     * Thresholds fitted against a different encoder are discarded. A value is only
     * meaningful relative to the score distribution it was fitted on: a threshold fitted
     * for an encoder whose scores all sat above 0.86, applied to one that tops out at
     * 0.93, rejects nearly every detection and reports the part as entirely unknown.
     * Falling back to the calibrated defaults is always recoverable; the tuner refits from
     * this part's history once enough same-encoder confirmations exist.
     */
    public static Tuned load(Context context, String partId) {
        File f = fileFor(context, partId);
        if (!f.exists()) return null;
        try (InputStream is = new java.io.FileInputStream(f)) {
            java.io.ByteArrayOutputStream bos = new java.io.ByteArrayOutputStream();
            byte[] buf = new byte[2048];
            int n;
            while ((n = is.read(buf)) != -1) bos.write(buf, 0, n);
            JSONObject root = new JSONObject(new String(bos.toByteArray(), StandardCharsets.UTF_8));
            String storedEncoder = root.optString("encoder", null);
            if (storedEncoder == null || !storedEncoder.equals(FeatureEmbedder.ENCODER_ID)) {
                Log.w(TAG, "Discarding thresholds for " + partId + " fitted by encoder '"
                        + (storedEncoder == null ? "<untagged/legacy>" : storedEncoder)
                        + "'; this build uses '" + FeatureEmbedder.ENCODER_ID
                        + "'. Falling back to defaults until they can be refitted.");
                return null;
            }
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
            // Clamp on read as well as on fit, so a file written before the floor existed
            // (or hand-edited) can't reintroduce a disabled ambiguity check.
            library.setThresholds(tuned.similarityThreshold,
                    Math.max(MARGIN_MIN, tuned.marginThreshold));
        }
        return tuned;
    }

    private static File fileFor(Context context, String partId) {
        return new File(new File(context.getFilesDir(), DIR), PartFiles.safeName(partId) + ".json");
    }

    private static float round(float v) {
        return Math.round(v * 1000f) / 1000f;
    }
}
