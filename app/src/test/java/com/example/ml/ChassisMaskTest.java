package com.example.ml;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;

/**
 * Covers the rule that decides whether a region of not-plate pixels is background.
 *
 * <p>Everything else in {@link ChassisMask} is a native OpenCV call the JVM cannot run — the
 * Canny pass, the clutter filter, the connected-component labelling — so the end-to-end
 * behaviour lives in {@code python/background_mask_study.py}, which mirrors the class and
 * scores it against the ground-truth counts. What is pinned here is the one rule that decides
 * what may be deleted from a photograph, which is the part that can silently lose a fastener.
 *
 * <p>Robolectric only because the class logs through {@code android.util.Log}.
 */
@RunWith(RobolectricTestRunner.class)
public class ChassisMaskTest {

    private static final int W = 320, H = 240;

    /** Area, in pixels, of a region occupying the given share of the frame. */
    private static double share(double fraction) {
        return fraction * W * H;
    }

    /**
     * The safety property. An interior region is never background, however dark or busy it
     * is — which is what stops a fastener, the shadow beside one, or a dark recess in the
     * middle of the part from being deleted.
     */
    @Test
    public void anInteriorRegionIsNeverBackground() {
        // Big, but touching no edge: a fastener cluster, or a dark recess mid-plate.
        assertFalse(ChassisMask.isBackgroundRegion(50, 50, 200, 100, share(0.26), W, H));
        // Even one that reaches within a pixel of every edge.
        assertFalse(ChassisMask.isBackgroundRegion(1, 1, W - 2, H - 2, share(0.9), W, H));
    }

    @Test
    public void aRegionTouchingAnyEdgeQualifies() {
        double big = share(0.1);
        assertTrue("left", ChassisMask.isBackgroundRegion(0, 40, 40, 100, big, W, H));
        assertTrue("top", ChassisMask.isBackgroundRegion(40, 0, 100, 40, big, W, H));
        assertTrue("right", ChassisMask.isBackgroundRegion(W - 40, 40, 40, 100, big, W, H));
        assertTrue("bottom", ChassisMask.isBackgroundRegion(40, H - 40, 100, 40, big, W, H));
    }

    /**
     * A sliver along the frame edge is a rounding artefact of the downscale, not a bench.
     * {@code whole screws1.jpg}, the smallest real background in the asset set, covers 4.7%.
     */
    @Test
    public void aSliverAlongTheEdgeIsNotBackground() {
        assertFalse(ChassisMask.isBackgroundRegion(0, 0, W, 1, share(0.004), W, H));
        assertTrue("just above the floor",
                ChassisMask.isBackgroundRegion(0, 0, W, 4, share(0.011), W, H));
    }

    @Test
    public void theFloorIsTheDocumentedOne() {
        // 1% of the frame. Stated as a constant so a change has to be deliberate.
        assertTrue(ChassisMask.MIN_BACKGROUND_FRACTION == 0.01);
        double justUnder = share(ChassisMask.MIN_BACKGROUND_FRACTION) - 1;
        double justOver = share(ChassisMask.MIN_BACKGROUND_FRACTION) + 1;
        assertFalse(ChassisMask.isBackgroundRegion(0, 0, 40, 40, justUnder, W, H));
        assertTrue(ChassisMask.isBackgroundRegion(0, 0, 40, 40, justOver, W, H));
    }

    /**
     * The ceiling exists because the whole method rests on "the middle of the frame is the
     * chassis". A mask covering most of the photograph means that assumption has failed, and
     * the class refuses rather than guessing — see {@link ChassisMask#MAX_BACKGROUND_FRACTION}.
     */
    @Test
    public void theCeilingLeavesRoomForEveryMeasuredCase() {
        // The largest real background in the asset set is three.jpg at 21.8%; the app's own
        // wide shots run 4.7-22.1%. The ceiling has to sit clear of those and well below half.
        assertTrue(ChassisMask.MAX_BACKGROUND_FRACTION > 0.25);
        assertTrue(ChassisMask.MAX_BACKGROUND_FRACTION < 0.5);
    }
}
