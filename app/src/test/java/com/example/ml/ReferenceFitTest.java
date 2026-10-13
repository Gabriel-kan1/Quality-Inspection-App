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
 * Covers the position score the adaptive edge stage chooses its method on, and the label
 * check that decides between close fits.
 *
 * Runs under Robolectric because {@link RectF} is an Android type.
 */
@RunWith(RobolectricTestRunner.class)
public class ReferenceFitTest {

    private static final float SIDE = 0.05f;

    /** The 3-hole plate: one threaded hole above two bolt heads, in a column. */
    private static FeatureLayout threeHoles() {
        FeatureLayout layout = new FeatureLayout();
        layout.setImageSize(960, 1280);
        layout.addSlot("threaded_hole", 0.19f, 0.18f, SIDE, SIDE);
        layout.addSlot("bolt_head", 0.22f, 0.41f, SIDE, SIDE);
        layout.addSlot("bolt_head", 0.22f, 0.91f, SIDE, SIDE);
        return layout;
    }

    private static RectF box(float cx, float cy) {
        return new RectF(cx - SIDE / 2, cy - SIDE / 2, cx + SIDE / 2, cy + SIDE / 2);
    }

    private static List<RectF> onEveryFeature(FeatureLayout layout, float dx, float dy) {
        List<RectF> out = new ArrayList<>();
        for (FeatureLayout.Slot s : layout.slots()) out.add(box(s.x + dx, s.y + dy));
        return out;
    }

    /** Three features cannot be aligned by the layout matcher, and must still be scored. */
    @Test
    public void aThreeFeaturePartIsScored() {
        ReferenceFit fit = ReferenceFit.from(threeHoles());
        assertNotNull(fit);
        assertEquals(3, fit.featureCount());
        assertNull(ReferenceFit.from(null));
        assertNull(ReferenceFit.from(new FeatureLayout()));
    }

    @Test
    public void aBoxOnEveryFeatureAndNothingElseScoresZero() {
        FeatureLayout layout = threeHoles();
        ReferenceFit.Fit f = ReferenceFit.from(layout).fit(onEveryFeature(layout, 0, 0));
        assertEquals(0.0, f.score, 1e-6);
        assertEquals(3, f.matched);
        assertEquals(0, f.extras);
    }

    /** The part framed a little to one side is framing, not a wrong method. */
    @Test
    public void aSmallFramingShiftIsAllowedFor() {
        FeatureLayout layout = threeHoles();
        ReferenceFit.Fit f = ReferenceFit.from(layout).fit(onEveryFeature(layout, 0.08f, -0.05f));
        assertEquals(0.0, f.score, 1e-4);
        assertEquals(3, f.matched);
    }

    @Test
    public void aLostFeatureCostsAsMuchAsAWholeFeature() {
        FeatureLayout layout = threeHoles();
        List<RectF> rects = onEveryFeature(layout, 0, 0);
        rects.remove(0);
        ReferenceFit.Fit f = ReferenceFit.from(layout).fit(rects);
        assertEquals(1.0 / 3, f.score, 1e-6);
        assertEquals(-1, f.boxForFeature[0]);
    }

    @Test
    public void anInventedBoxOnThePartCountsAgainstTheMethod() {
        FeatureLayout layout = threeHoles();
        List<RectF> rects = onEveryFeature(layout, 0, 0);
        rects.add(box(0.21f, 0.65f));      // between the bolt heads: on the part
        rects.add(box(0.90f, 0.10f));      // far off the part: not charged
        ReferenceFit.Fit f = ReferenceFit.from(layout).fit(rects);
        assertEquals(1, f.extras);
        assertEquals(1.0 / 4, f.score, 1e-6);
    }

    /** One box swallowing the column of features can stand in for one of them at most. */
    @Test
    public void oneBigBoxCannotStandInForEveryFeature() {
        FeatureLayout layout = threeHoles();
        List<RectF> rects = new ArrayList<>();
        rects.add(new RectF(0.10f, 0.10f, 0.30f, 0.95f));
        ReferenceFit.Fit f = ReferenceFit.from(layout).fit(rects);
        assertTrue("got " + f.matched, f.matched <= 1);
        assertTrue("got " + f.score, f.score > 0.6);
    }

    @Test
    public void aBoxBesideAFeatureIsNotThatFeature() {
        FeatureLayout layout = threeHoles();
        List<RectF> rects = onEveryFeature(layout, 0, 0);
        rects.set(0, box(0.19f + 2.5f * SIDE, 0.18f));
        ReferenceFit.Fit f = ReferenceFit.from(layout).fit(rects);
        assertEquals(2, f.matched);
    }

    @Test
    public void nothingFoundScoresOne() {
        ReferenceFit.Fit f = ReferenceFit.from(threeHoles()).fit(new ArrayList<>());
        assertEquals(1.0, f.score, 1e-9);
    }

    @Test
    public void labelsAreCountedOnTheMatchedBoxesOnly() {
        FeatureLayout layout = threeHoles();
        List<RectF> rects = onEveryFeature(layout, 0, 0);
        // A labeller that calls everything above y=0.5 a threaded hole.
        ReferenceFit fit = ReferenceFit.from(layout).withLabeller(
                b -> b.centerY() < 0.5f ? "threaded_hole" : "bolt_head");
        ReferenceFit.Fit f = fit.fit(rects);
        // Feature 0 (threaded hole) right, feature 1 (bolt head at 0.41) wrong, feature 2 right.
        assertEquals(2, fit.labelsAgreeing(rects, f));
        assertEquals(-1, ReferenceFit.from(layout).labelsAgreeing(rects, f));
    }

    @Test
    public void theCachingLabellerClassifiesEachBoxOnce() {
        int[] calls = {0};
        ReferenceFit.Labeller l = ReferenceFit.caching(b -> {
            calls[0]++;
            return "screw";
        });
        l.typeOf(box(0.5f, 0.5f));
        l.typeOf(box(0.5f, 0.5f));
        l.typeOf(box(0.6f, 0.5f));
        assertEquals(2, calls[0]);
    }
}
