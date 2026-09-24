package com.example.ml;

import android.graphics.RectF;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * What size each critical feature on a part is expected to be, and how far a set of
 * detections sits from it — the evidence {@link AdaptivePreprocessing} uses to choose a
 * preprocessing strength.
 *
 * <h3>Why size, and why learned</h3>
 * The two ways the edge stage gets a fastener's box wrong are both size errors before they
 * are anything else. A shadow or scratch lying beside the hardware fragments into contours
 * that the proximity merge joins onto the fastener's, and the box comes out larger than the
 * hardware; a faint edge is smoothed away and the box stops short of it, or the fastener
 * splits into two boxes each smaller than it. Counts cannot see either — the first is still
 * one box, the second may still be one box — but a size comparison against a known expected
 * size sees both, and a strength that neither merged nor cropped is the strength whose boxes
 * agree with the expected sizes.
 *
 * <p>The expected size starts as the box the reference recorded at each slot and is refined
 * by every filed inspection, as a median ({@link FeatureLayout.Slot#expectedSizeFactor()}).
 * The reference is one detection and can carry the same errors, so it is one vote among the
 * observations rather than the truth they are compared to. This is the "learn from previous
 * inspections" half of the mechanism: the estimate of how big each labelled feature should
 * be moves toward the consensus of what has been seen there.
 *
 * <h3>Scoring</h3>
 * Detections are aligned onto the layout by {@link LayoutMatcher} — positional, as always —
 * and the score is the <em>mean</em> over everything the strength ought to have accounted
 * for. There are three ways to be wrong and they are charged on one scale:
 *
 * <ul>
 *   <li>a box the wrong size — {@code |ln(found / expected)|}, capped at {@link #DEVIATION_CAP};
 *   <li>a judgeable slot nothing matched — the cap: the strength lost a feature;
 *   <li>a detection that explains no slot and lies on the part — the cap: it invented one.
 * </ul>
 *
 * <p>A mean rather than a median on purpose: the failure this exists to catch is one box that
 * swallowed a shadow, and a median over four slots would not see it. Charging absent slots is
 * what stops a strength that erased a fastener from scoring well on the ones it kept.
 *
 * <p><b>The extras term is what lets every strength be put on the ballot.</b> Without it the
 * score judges boxes <em>at</em> the layout's slots and is blind to what a strength scatters
 * <em>between</em> them, so one that over-detects could win on size while flooding the frame.
 * That blindness is why {@link PreprocessingStrength#LIGHT} used to be withheld unless the
 * photograph's edges were weak — a blunt rule that also withheld it on the photographs where
 * it was the only strength that found a feature at all. Measured: on the bundled photographs
 * light is charged 0.14–0.57 where it over-detects and 0.067 where it is right, against
 * 0.00–0.22 for the other strengths, so the score separates the two cases on its own.
 *
 * <p>Slots are judgeable on the same terms as the size check ({@code
 * FeatureLocalisation.MIN_SIZE_JUDGED_PX}): the reference must have recorded a box, and the
 * box must be big enough that the detector's boxes repeat between shots at all.
 *
 * <h3>Known limit</h3>
 * Agreement with the reference is the criterion, so on the very first inspections a strength
 * that reproduces the reference's own merge artefact scores well. The median moves off the
 * artefact as inspections under other lighting accumulate — one observation is enough to
 * pull the factor halfway, two to outvote the reference — but until then "closest to the
 * reference" and "closest to the hardware" can differ. Measured over the same-scene pairs in
 * {@code python/adaptive_preprocessing_study.py} the rule never chose worse than the
 * noise-only rung and improved count accuracy in every degraded condition.
 */
public final class SizePrior {

    /**
     * Deviation at which a box is as wrong as an absent one: twice or half the expected
     * linear size. Beyond it the box is not the fastener with a shadow, it is something else.
     */
    public static final double DEVIATION_CAP = Math.log(2.0);
    /**
     * How far past the slots' own bounding box a detection still counts as "on the part", as
     * a fraction of that box's extent on each side.
     *
     * <p>Zero is too tight to see over-detection on a small layout: with four slots the box
     * spans only the hardware itself, and on {@code head1.jpg} the light strength's eight
     * detections scored 0.102 against medium's 0.000 — inside the decision margin. At 0.5 the
     * same case separates at 0.355, and every measured case lands 0.14–0.57 for a strength
     * that over-detects against 0.00–0.15 for one that does not. Further out would start
     * charging a strength for hardware that genuinely sits outside the layout.
     */
    public static final double EXTRA_ENVELOPE_PAD = 0.5;

    private final FeatureLayout layout;
    /** Expected fractional area in the reference frame, per judgeable slot id. */
    private final Map<Integer, Double> expectedArea = new LinkedHashMap<>();
    private final int evidence;

    private SizePrior(FeatureLayout layout) {
        this.layout = layout;
        int votes = 0;
        for (FeatureLayout.Slot s : layout.slots()) {
            if (!s.hasSize()) continue;
            float shortSide = layout.shortSidePx(s);
            if (Float.isNaN(shortSide) || shortSide < FeatureLocalisation.MIN_SIZE_JUDGED_PX) {
                continue;
            }
            double f = s.expectedSizeFactor();
            expectedArea.put(s.id, (double) s.w * s.h * f * f);
            votes += s.sizeEvidence();
        }
        this.evidence = votes;
    }

    /**
     * Builds the prior for a layout, or returns null when it could not judge anything: no
     * layout, too few slots to ever align, or no slot with a usable reference box. A null
     * prior means the edge stage falls back to the image-quality choice alone.
     */
    public static SizePrior from(FeatureLayout layout) {
        if (layout == null || layout.isEmpty()) return null;
        if (LayoutMatcher.tooSmallToRegister(layout.slots().size())) return null;
        SizePrior prior = new SizePrior(layout);
        return prior.expectedArea.isEmpty() ? null : prior;
    }

    /** Slots with an expected size to compare against. */
    public int judgedSlots() {
        return expectedArea.size();
    }

    /** Filed size observations behind the expected sizes, summed over judged slots. */
    public int evidence() {
        return evidence;
    }

    /**
     * Mean capped size deviation of these detections from the expected sizes, or NaN when
     * they could not be aligned onto the layout at all — which is itself information: a
     * strength whose boxes cannot be registered found something other than the part.
     *
     * @param rects detections in the inspection photo's fractional coordinates
     */
    public double agreement(List<RectF> rects) {
        if (rects == null || rects.isEmpty()) return Double.NaN;
        List<String> types = new ArrayList<>(rects.size());
        // Unclassified at this point — the edge stage runs before the encoder — and the
        // matcher only ever uses types to order seeds, never to score.
        for (int i = 0; i < rects.size(); i++) types.add(ReferenceLibrary.UNKNOWN);
        return agreement(rects, LayoutMatcher.match(layout, rects, types));
    }

    /** As above, against an alignment already computed. Pure; the tests use this entry. */
    double agreement(List<RectF> rects, LayoutMatcher.Result match) {
        if (match == null || !match.aligned) return Double.NaN;
        double areaScale = match.areaScale();
        if (!(areaScale > 1e-12)) return Double.NaN;

        Map<Integer, Double> deviation = new LinkedHashMap<>();
        for (Integer slotId : expectedArea.keySet()) deviation.put(slotId, DEVIATION_CAP);
        for (LayoutMatcher.Match m : match.matches) {
            Double expected = expectedArea.get(m.slotId);
            if (expected == null) continue;
            if (m.featureIndex < 0 || m.featureIndex >= rects.size()) continue;
            RectF found = rects.get(m.featureIndex);
            double actual = (double) found.width() * found.height();
            if (!(actual > 0)) continue;
            double ratio = Math.sqrt(actual / (expected * areaScale));
            deviation.put(m.slotId, Math.min(Math.abs(Math.log(ratio)), DEVIATION_CAP));
        }
        double sum = 0;
        for (double d : deviation.values()) sum += d;
        int terms = deviation.size();

        // Detections that explain no slot but sit on the part: hardware the strength invented.
        float[] envelope = projectedEnvelope(match);
        if (envelope != null) {
            for (Integer index : match.unmatchedFeatureIndices) {
                if (index == null || index < 0 || index >= rects.size()) continue;
                RectF r = rects.get(index);
                float cx = (r.left + r.right) / 2f, cy = (r.top + r.bottom) / 2f;
                if (cx >= envelope[0] && cx <= envelope[2]
                        && cy >= envelope[1] && cy <= envelope[3]) {
                    sum += DEVIATION_CAP;
                    terms++;
                }
            }
        }
        return terms == 0 ? Double.NaN : sum / terms;
    }

    /**
     * The part's extent in this photo's coordinates — every slot projected through the
     * alignment, bounded, then padded by {@link #EXTRA_ENVELOPE_PAD} of its own extent.
     * Null when nothing could be projected. Uses every slot, not only the judgeable ones:
     * this is asking where the part is, not which of it can be size-checked.
     */
    private float[] projectedEnvelope(LayoutMatcher.Result match) {
        float minX = Float.MAX_VALUE, minY = Float.MAX_VALUE;
        float maxX = -Float.MAX_VALUE, maxY = -Float.MAX_VALUE;
        for (FeatureLayout.Slot s : layout.slots()) {
            float[] p = match.project(s.x, s.y);
            if (p == null) continue;
            minX = Math.min(minX, p[0]);
            minY = Math.min(minY, p[1]);
            maxX = Math.max(maxX, p[0]);
            maxY = Math.max(maxY, p[1]);
        }
        if (minX > maxX) return null;
        float padX = (float) Math.max(maxX - minX, 1e-6) * (float) EXTRA_ENVELOPE_PAD;
        float padY = (float) Math.max(maxY - minY, 1e-6) * (float) EXTRA_ENVELOPE_PAD;
        return new float[]{minX - padX, minY - padY, maxX + padX, maxY + padY};
    }

    public String describe() {
        return String.format(Locale.US, "%d slot%s sized, %d filed observation%s",
                judgedSlots(), judgedSlots() == 1 ? "" : "s",
                evidence, evidence == 1 ? "" : "s");
    }
}
