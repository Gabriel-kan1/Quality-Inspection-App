package com.example.ml;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import android.graphics.RectF;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;

import java.util.ArrayList;
import java.util.List;

/**
 * Covers the size-agreement score the adaptive edge stage ranks its rungs by: that boxes of
 * the expected size score zero whatever the framing, that a merged or cropped box is seen,
 * that a lost fastener is charged rather than ignored, and that the expected size learns.
 *
 * Runs under Robolectric because {@link RectF} is an Android type and the matcher logs.
 */
@RunWith(RobolectricTestRunner.class)
public class SizePriorTest {

    /** Reference box side, fractional. At 1280x960 that is 64x48 px: comfortably judgeable. */
    private static final float SIDE = 0.05f;

    /**
     * Five critical slots in the asymmetric arrangement the localisation tests use, each
     * with a reference box of {@link #SIDE}, plus a sixth whose box is too small to judge.
     */
    private static FeatureLayout layout() {
        FeatureLayout layout = new FeatureLayout();
        layout.setImageSize(1280, 960);
        float[][] at = {{0.15f, 0.15f}, {0.85f, 0.20f}, {0.30f, 0.85f}, {0.85f, 0.80f},
                {0.45f, 0.50f}};
        for (float[] p : at) layout.addSlot("screw", p[0], p[1], SIDE, SIDE);
        layout.addSlot("screw", 0.60f, 0.30f, 0.015f, 0.015f);
        return layout;
    }

    /** The detections a perfect inspection would produce: one box per slot, at its size. */
    private static List<RectF> perfect(FeatureLayout layout, float scale) {
        List<RectF> rects = new ArrayList<>();
        for (FeatureLayout.Slot s : layout.slots()) {
            float w = (s.hasSize() ? s.w : 0.015f) * scale, h = (s.hasSize() ? s.h : 0.015f) * scale;
            float cx = s.x * scale, cy = s.y * scale;
            rects.add(new RectF(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2));
        }
        return rects;
    }

    private static RectF grown(RectF r, float factor) {
        float cx = r.centerX(), cy = r.centerY();
        float w = r.width() * factor / 2, h = r.height() * factor / 2;
        return new RectF(cx - w, cy - h, cx + w, cy + h);
    }

    @Test
    public void slotsTooSmallToJudgeAreNotCounted() {
        SizePrior prior = SizePrior.from(layout());
        assertNotNull(prior);
        assertEquals(5, prior.judgedSlots());
        assertEquals(0, prior.evidence());
    }

    @Test
    public void boxesOfTheExpectedSizeScoreZero() {
        FeatureLayout layout = layout();
        SizePrior prior = SizePrior.from(layout);
        assertEquals(0.0, prior.agreement(perfect(layout, 1f)), 1e-6);
    }

    /** A part shot closer has every box larger; that is framing, and the alignment removes it. */
    @Test
    public void framingScaleIsNotASizeError() {
        FeatureLayout layout = layout();
        SizePrior prior = SizePrior.from(layout);
        assertEquals(0.0, prior.agreement(perfect(layout, 0.8f)), 0.02);
    }

    /**
     * The failure the score exists to catch: one box that swallowed a shadow and came out
     * twice the linear size. A median over five slots would not see it; the mean does.
     */
    @Test
    public void oneMergedBoxIsSeen() {
        FeatureLayout layout = layout();
        SizePrior prior = SizePrior.from(layout);
        List<RectF> rects = perfect(layout, 1f);
        rects.set(1, grown(rects.get(1), 2f));
        double expected = Math.log(2) / 5;
        assertEquals(expected, prior.agreement(rects), 0.02);
    }

    @Test
    public void aCroppedBoxIsSeenTheSameWay() {
        FeatureLayout layout = layout();
        SizePrior prior = SizePrior.from(layout);
        List<RectF> rects = perfect(layout, 1f);
        rects.set(2, grown(rects.get(2), 0.5f));
        assertEquals(Math.log(2) / 5, prior.agreement(rects), 0.02);
    }

    /**
     * A rung that erased a fastener must not score well on the ones it kept: the empty slot
     * is charged the cap, the same as a box twice the size.
     */
    @Test
    public void aLostFastenerIsChargedNotIgnored() {
        FeatureLayout layout = layout();
        SizePrior prior = SizePrior.from(layout);
        List<RectF> rects = perfect(layout, 1f);
        rects.remove(3);
        assertEquals(SizePrior.DEVIATION_CAP / 5, prior.agreement(rects), 0.02);
    }

    /** Something worse than a shadow — a box five times the size — is capped, not unbounded. */
    @Test
    public void deviationIsCapped() {
        FeatureLayout layout = layout();
        SizePrior prior = SizePrior.from(layout);
        List<RectF> rects = perfect(layout, 1f);
        rects.set(0, grown(rects.get(0), 5f));
        assertEquals(SizePrior.DEVIATION_CAP / 5, prior.agreement(rects), 0.02);
    }

    @Test
    public void detectionsThatCannotBeAlignedScoreNaN() {
        SizePrior prior = SizePrior.from(layout());
        List<RectF> junk = new ArrayList<>();
        junk.add(new RectF(0.1f, 0.1f, 0.12f, 0.12f));
        junk.add(new RectF(0.11f, 0.1f, 0.13f, 0.12f));
        assertTrue(Double.isNaN(prior.agreement(junk)));
        assertTrue(Double.isNaN(prior.agreement(new ArrayList<>())));
        assertTrue(Double.isNaN(prior.agreement(null)));
    }

    // ── Learning ─────────────────────────────────────────────────────────────

    /**
     * The reference box swallowed a shadow: the hardware at slot 2 has measured half its size
     * in every inspection since. The expected size follows the observations, so a rung whose
     * box is the true size now scores zero there and one that reproduces the artefact is
     * charged for it.
     */
    @Test
    public void expectedSizeFollowsWhatInspectionsHaveMeasured() {
        FeatureLayout layout = layout();
        layout.recordSizeRatio(2, 0.5f);
        layout.recordSizeRatio(2, 0.5f);
        assertEquals(0.5f, layout.slotById(2).expectedSizeFactor(), 1e-6);
        SizePrior prior = SizePrior.from(layout);
        assertEquals(2, prior.evidence());

        List<RectF> trueSize = perfect(layout, 1f);
        trueSize.set(1, grown(trueSize.get(1), 0.5f));
        assertEquals(0.0, prior.agreement(trueSize), 0.02);

        List<RectF> artefact = perfect(layout, 1f);
        assertEquals(Math.log(2) / 5, prior.agreement(artefact), 0.02);
    }

    /** The reference is one vote: a single observation pulls halfway, two outvote it. */
    @Test
    public void theReferenceCountsOnce() {
        FeatureLayout layout = layout();
        FeatureLayout.Slot slot = layout.slotById(1);
        assertEquals(1f, slot.expectedSizeFactor(), 0);
        layout.recordSizeRatio(1, 0.5f);
        assertEquals(0.75f, slot.expectedSizeFactor(), 1e-6);
        layout.recordSizeRatio(1, 0.5f);
        assertEquals(0.5f, slot.expectedSizeFactor(), 1e-6);
        // One inspection that merged the same shadow again is not enough to hide it.
        layout.recordSizeRatio(1, 1.0f);
        assertEquals(0.75f, slot.expectedSizeFactor(), 1e-6);
        layout.recordSizeRatio(1, 0.5f);
        assertEquals(0.5f, slot.expectedSizeFactor(), 1e-6);
    }

    @Test
    public void junkRatiosAreRefusedAndHistoryIsBounded() {
        FeatureLayout layout = layout();
        layout.recordSizeRatio(1, Float.NaN);
        layout.recordSizeRatio(1, 0f);
        layout.recordSizeRatio(1, -1f);
        layout.recordSizeRatio(99, 1f);
        assertEquals(0, layout.slotById(1).sizeEvidence());
        for (int i = 0; i < FeatureLayout.MAX_SIZE_HISTORY + 10; i++) layout.recordSizeRatio(1, 1f);
        assertEquals(FeatureLayout.MAX_SIZE_HISTORY, layout.slotById(1).sizeEvidence());
    }

    // ── When there is nothing to judge ───────────────────────────────────────

    @Test
    public void noPriorWithoutALayoutWorthAligning() {
        assertNull(SizePrior.from(null));
        assertNull(SizePrior.from(new FeatureLayout()));

        FeatureLayout three = new FeatureLayout();
        three.setImageSize(1280, 960);
        three.addSlot("screw", 0.2f, 0.2f, SIDE, SIDE);
        three.addSlot("screw", 0.8f, 0.2f, SIDE, SIDE);
        three.addSlot("screw", 0.5f, 0.8f, SIDE, SIDE);
        assertNull("too few slots to ever register", SizePrior.from(three));

        FeatureLayout unsized = new FeatureLayout();
        unsized.setImageSize(1280, 960);
        for (int i = 0; i < 5; i++) unsized.addSlot("screw", 0.1f + 0.2f * i, 0.5f);
        assertNull("no reference boxes to size against", SizePrior.from(unsized));

        FeatureLayout noFrame = layout();
        noFrame.setImageSize(0, 0);
        assertNull("no frame size, so no pixel floor can be applied", SizePrior.from(noFrame));
    }
}
