package com.example.ml;

import android.util.Log;

import com.example.data.entity.ClassificationEvent;

import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Learns, per part, that detections which resemble an ignored type are reliably noise — and
 * once that is established, stops asking the operator about them.
 *
 * <h3>The problem</h3>
 * The detector is deliberately generous: it finds every contour that could be hardware, and
 * on a reflective or scratched surface that is a lot of contours that are not. Each one is a
 * crop the classifier has never seen, so each lands as "unknown", and every unknown competes
 * for a review slot. The operator labels it {@code surface}, the surface prototype grows, and
 * the next similar crop scores a little higher — but it still lands under the similarity
 * threshold, so it is still an unknown, and it is still asked about. The loop learns without
 * ever concluding. On a noisy photo that is a dozen dialogs about nothing.
 *
 * <h3>What it learns</h3>
 * Not a new classifier. The evidence is the operator's own confirmed history: among reviewed
 * detections that were rejected below threshold, that most resembled an <em>ignored</em> type,
 * and that scored above {@link ActiveLearningPolicy#NOISE_FLOOR} (so they resemble that type
 * rather than nothing), how often did the operator say it was actually critical hardware?
 * When that population is large enough and the critical rate is close enough to zero, a live
 * detection in it is treated as its ignored type without asking.
 *
 * <h3>What it is not</h3>
 * Suppression is a routing decision, not a label. A suppressed crop is never added to any
 * prototype — the project's no-pseudo-labelling rule holds, and for the same reason: a crop
 * that already resembles the surface prototype would teach it nothing, while a mislabelled one
 * would drag it. It is still drawn on the overlay as its ignored type, still tappable, still
 * logged as a {@link ReferenceLibrary.Reason#AUTO_IGNORED} event, and one per inspection is put
 * back in front of the operator as a {@link ActiveLearningPolicy.Kind#NOISE_AUDIT} so the gate
 * can find out when it is wrong.
 *
 * <h3>Why the guard is against critical, not against being wrong</h3>
 * A scratch absorbed into {@code surface} costs an advisory trend count. A stud absorbed into
 * {@code surface} hides a fastener and passes a part that should have failed — the worst
 * failure this app has. The evidence therefore counts only confirmed <em>critical</em> types
 * as the gate having been wrong, and stays closed on any meaningful rate of them. Advisory
 * losses are the accepted price, and the noise audit is what bounds them.
 *
 * <h3>Why the evidence population is wider than the live rule</h3>
 * Live, a detection is only suppressed if no critical type scores within the margin of the
 * ignored one — a near-tie with a stud is exactly the case to ask about. Stored events carry
 * only the runner-up, not every score, so that condition cannot be replayed exactly. The
 * evidence is counted over the population <em>without</em> it. That is a superset of what gets
 * suppressed, so if the superset is safe, the subset is at least as safe. Conservative in the
 * direction that matters.
 */
public final class NoiseGate {

    private static final String TAG = "NoiseGate";

    /**
     * Reviewed detections in the noise population before the gate will act. Matches
     * {@link ThresholdTuner#MIN_EVENTS}: below roughly this many, a rate is dominated by chance.
     */
    public static final int MIN_EVIDENCE = 20;
    /**
     * Share of that population the operator may have confirmed as critical before the gate
     * stays shut. At the minimum evidence this is zero — one critical fastener in twenty
     * noise-looking crops is not a population that can be ignored — and it relaxes to one in
     * fifty only as the sample grows enough for one to plausibly be an operator slip.
     */
    public static final float MAX_CRITICAL_RATE = 0.02f;

    /** How a type key maps to its severity — supplied by the caller so this stays testable. */
    public interface SeverityLookup {
        FeatureTypeRegistry.Severity of(String type);
    }

    /** What the history said, kept so the results screen can explain the gate's state. */
    public static final class Evidence {
        /** Reviewed detections in the noise population. */
        public final int sampleSize;
        /** Of those, how many the operator said were critical hardware. */
        public final int critical;
        public final boolean mature;

        Evidence(int sampleSize, int critical) {
            this.sampleSize = sampleSize;
            this.critical = critical;
            this.mature = sampleSize >= MIN_EVIDENCE
                    && critical <= (int) Math.floor(sampleSize * MAX_CRITICAL_RATE);
        }

        public String describe() {
            if (mature) {
                return String.format(Locale.US,
                        "Noise filter active: %d of %d noise-looking crops you reviewed were"
                                + " actually critical.", critical, sampleSize);
            }
            if (sampleSize < MIN_EVIDENCE) {
                return String.format(Locale.US,
                        "Noise filter learning: %d of %d noise-looking crops reviewed.",
                        sampleSize, MIN_EVIDENCE);
            }
            return String.format(Locale.US,
                    "Noise filter held: %d of %d noise-looking crops you reviewed were critical"
                            + " hardware, so nothing is being filtered.", critical, sampleSize);
        }
    }

    public final Evidence evidence;
    private final SeverityLookup severity;
    private final float marginThreshold;

    private NoiseGate(Evidence evidence, SeverityLookup severity, float marginThreshold) {
        this.evidence = evidence;
        this.severity = severity;
        this.marginThreshold = marginThreshold;
    }

    /** A gate that never fires — for callers with no history to fit from. */
    public static NoiseGate inactive(SeverityLookup severity) {
        return new NoiseGate(new Evidence(0, 0), severity, 0f);
    }

    /**
     * Fits the gate from a part's confirmed history.
     *
     * Events from a different encoder are dropped first, as {@link ThresholdTuner} does: the
     * noise floor is a property of the embedding space, so a score from a retired encoder says
     * nothing about whether a crop resembles anything under this one.
     *
     * @param allReviewed every reviewed event for the part, in any order.
     * @param library     supplies the live margin, so the near-tie check matches the classifier.
     */
    public static NoiseGate fit(List<ClassificationEvent> allReviewed, SeverityLookup severity,
                                ReferenceLibrary library) {
        int sample = 0, critical = 0;
        for (ClassificationEvent e : ThresholdTuner.forCurrentEncoder(allReviewed)) {
            String truth = e.getConfirmedType();
            // A skipped review is not evidence either way.
            if (truth == null || ReferenceLibrary.UNKNOWN.equals(truth)) continue;
            if (!inNoisePopulation(e.getPredictedType(), e.getPredictedScore(), e.getReason(),
                    severity)) continue;
            sample++;
            if (severity.of(truth) == FeatureTypeRegistry.Severity.CRITICAL) critical++;
        }
        Evidence evidence = new Evidence(sample, critical);
        Log.d(TAG, evidence.describe());
        return new NoiseGate(evidence, severity, library.getMarginThreshold());
    }

    /**
     * The population the evidence is counted over: rejected below threshold, nearest to an
     * ignored type, and resembling it rather than nothing.
     *
     * Below-threshold only — never AMBIGUOUS. An ambiguous crop cleared the threshold and two
     * types are fighting over it; if one of them is critical that is precisely what the
     * operator needs to see, and if neither is, the classifier already accepts it once the
     * margin is resolved.
     */
    static boolean inNoisePopulation(String closest, float score, String reason,
                                     SeverityLookup severity) {
        if (closest == null || !ReferenceLibrary.Reason.BELOW_THRESHOLD.name().equals(reason)) {
            return false;
        }
        if (score < ActiveLearningPolicy.NOISE_FLOOR) return false;
        return severity.of(closest) == FeatureTypeRegistry.Severity.IGNORED;
    }

    /**
     * Whether a live classification should be treated as its ignored type without asking.
     *
     * Adds the one condition the evidence cannot replay: no critical type may score within the
     * margin of the ignored one. A crop that is nearly as much "stud" as "surface" is a
     * question, not noise, however mature the gate is.
     */
    public boolean suppresses(ReferenceLibrary.Classification c) {
        if (!evidence.mature || c == null || c.reason == null) return false;
        if (!inNoisePopulation(c.closestMatch, c.score, c.reason.name(), severity)) return false;
        // Without the full score map the near-tie check cannot be made, and an unverifiable
        // suppression is a suppression the gate has no business making.
        if (c.scores == null) return false;
        float bestCritical = -1f;
        for (Map.Entry<String, Float> entry : c.scores.entrySet()) {
            if (severity.of(entry.getKey()) == FeatureTypeRegistry.Severity.CRITICAL) {
                bestCritical = Math.max(bestCritical, entry.getValue());
            }
        }
        return bestCritical < c.score - marginThreshold;
    }

    /**
     * The classification a suppressed detection is carried as: its ignored type, at its own
     * score, flagged {@link ReferenceLibrary.Reason#AUTO_IGNORED} so the event log and the
     * review policy can tell it apart from a prediction the classifier itself accepted.
     */
    public static ReferenceLibrary.Classification asIgnored(ReferenceLibrary.Classification c) {
        return new ReferenceLibrary.Classification(c.closestMatch, c.closestMatch, c.score,
                "low", c.runnerUp, c.runnerUpScore, ReferenceLibrary.Reason.AUTO_IGNORED,
                c.scores);
    }

    /**
     * Retypes every suppressible detection in place and reports how many.
     *
     * Mirrors what a manual correction does to the feature — type, colour, label — so the
     * overlay and the counts treat it exactly as a confidently-classified ignored type, with
     * the label marked so the operator can tell the two apart at a glance.
     */
    public int apply(List<HardwareFeatureExtractor.ExtractedFeature> features) {
        if (features == null) return 0;
        int suppressed = 0;
        for (HardwareFeatureExtractor.ExtractedFeature f : features) {
            if (f == null || !suppresses(f.classification)) continue;
            ReferenceLibrary.Classification ignored = asIgnored(f.classification);
            f.classification = ignored;
            f.type = ignored.type;
            f.color = ReferenceLibrary.colorFor(ignored.type);
            String prefix = f.label != null && f.label.contains(":")
                    ? f.label.substring(0, f.label.indexOf(':')) : "?";
            f.label = String.format(Locale.US, "%s:%s %.2f (auto)",
                    prefix, ignored.type, ignored.score);
            suppressed++;
        }
        if (suppressed > 0) {
            Log.d(TAG, suppressed + " detection(s) treated as noise without review");
        }
        return suppressed;
    }
}
