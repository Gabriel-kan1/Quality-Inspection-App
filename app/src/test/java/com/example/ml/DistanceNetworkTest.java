package com.example.ml;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/**
 * Covers the pairwise-separation network that proposes correspondences for the positional
 * check: that it is blind to where the part sits and which way up it is, that a member is
 * admitted only when it agrees with every member already accepted, and that it declines to
 * absorb a detection which merely happens to sit near one of them.
 *
 * Plain geometry with no Android types, so it runs without Robolectric.
 */
public class DistanceNetworkTest {

    private static float[][] pts(float... xy) {
        float[][] out = new float[xy.length / 2][];
        for (int i = 0; i < out.length; i++) out[i] = new float[]{xy[2 * i], xy[2 * i + 1]};
        return out;
    }

    /** The same points shifted, turned and rescaled — what a freehand second photo produces. */
    private static float[][] moved(float[][] src, double degrees, double scale,
                                   double dx, double dy) {
        double a = Math.toRadians(degrees);
        float[][] out = new float[src.length][];
        for (int i = 0; i < src.length; i++) {
            out[i] = new float[]{
                    (float) (scale * (src[i][0] * Math.cos(a) - src[i][1] * Math.sin(a)) + dx),
                    (float) (scale * (src[i][0] * Math.sin(a) + src[i][1] * Math.cos(a)) + dy)};
        }
        return out;
    }

    private static final float[][] PART = pts(
            0.15f, 0.15f, 0.85f, 0.20f, 0.30f, 0.85f, 0.85f, 0.80f, 0.45f, 0.50f);

    @Test
    public void separationsAreSymmetricAndZeroOnTheDiagonal() {
        DistanceNetwork n = DistanceNetwork.over(pts(0f, 0f, 3f, 4f));
        assertEquals(5.0, n.between(0, 1), 1e-9);
        assertEquals(5.0, n.between(1, 0), 1e-9);
        assertEquals(0.0, n.between(0, 0), 1e-9);
        assertEquals(2, n.size());
    }

    /**
     * The property the whole approach rests on: separations describe the part, not the photo.
     * Move and turn the part and every distance is unchanged; shoot it closer and they all
     * scale by one common factor.
     */
    @Test
    public void separationsSurviveShiftAndRotationAndScaleTogether() {
        DistanceNetwork ref = DistanceNetwork.over(PART);
        DistanceNetwork shot = DistanceNetwork.over(moved(PART, 37, 0.6, 0.2, -0.1));

        for (int i = 0; i < ref.size(); i++) {
            for (int j = i + 1; j < ref.size(); j++) {
                assertEquals("every separation must scale by the same factor",
                        0.6 * ref.between(i, j), shot.between(i, j), 1e-5);
            }
        }
    }

    @Test
    public void spreadIsTheCharacteristicSizeOfThePointSet() {
        // Four points at (±1, ±1): every one sits sqrt(2) from the centroid.
        DistanceNetwork n = DistanceNetwork.over(pts(1f, 1f, -1f, 1f, 1f, -1f, -1f, -1f));
        assertEquals(Math.sqrt(2), n.spread(), 1e-6);
        assertEquals(0.0, DistanceNetwork.over(pts(2f, 2f)).spread(), 1e-9);
        assertEquals(0.0, DistanceNetwork.over(new float[0][]).spread(), 1e-9);
    }

    /** Two correct pairings must pull in the rest of the part. */
    @Test
    public void growthRecoversTheWholeArrangementFromTwoPairings() {
        DistanceNetwork ref = DistanceNetwork.over(PART);
        DistanceNetwork shot = DistanceNetwork.over(moved(PART, 15, 0.8, 0.05, 0.02));
        DistanceNetwork.Matcher m = new DistanceNetwork.Matcher(ref, shot);

        DistanceNetwork.Consensus c = m.grow(0, 0, 3, 3, 0.8 * 0.12 * ref.spread());
        assertNotNull(c);
        assertEquals(5, c.size);
        assertEquals(0.8, c.scale, 1e-4);
        for (int i = 0; i < 5; i++) {
            assertEquals("each slot must find its own detection", i, c.observedForNode[i]);
        }
        assertTrue("an undistorted shot must agree almost exactly", c.distortion < 1e-4);
    }

    /**
     * The reason each new member is checked against <em>every</em> accepted one rather than
     * against the seed alone.
     *
     * A stray detection can sit at exactly the right distance from one member and be nowhere
     * near where the layout puts it. Checking only the seed would admit it and let it drag the
     * fitted transform; checking all of them refuses it.
     */
    @Test
    public void aDetectionThatAgreesWithOnlyOneMemberIsRefused() {
        // Three collinear slots one unit apart, plus a fourth well off the line.
        float[][] layout = pts(0f, 0f, 1f, 0f, 2f, 0f, 1f, 2f);
        // The first three are found where they belong. The fourth detection is the right
        // distance from slot 0 but on the wrong side entirely.
        float[][] shot = pts(0f, 0f, 1f, 0f, 2f, 0f, -2.236f, 0f);

        DistanceNetwork ref = DistanceNetwork.over(layout);
        DistanceNetwork obs = DistanceNetwork.over(shot);
        assertEquals("the impostor is at slot 3's distance from slot 0",
                ref.between(0, 3), obs.between(0, 3), 1e-3);

        DistanceNetwork.Consensus c =
                new DistanceNetwork.Matcher(ref, obs).grow(0, 0, 2, 2, 0.12 * ref.spread());
        assertNotNull(c);
        assertEquals("only the three genuine correspondences survive", 3, c.size);
        assertEquals(-1, c.observedForNode[3]);
    }

    /**
     * The app's own defect case, seen from the network: a fastener fitted in the wrong hole
     * has the wrong separations from everything else, so it is left out of the agreed set
     * rather than pulling the alignment toward itself.
     *
     * This is what makes the fitted transform trustworthy — it is fitted only over hardware
     * that agrees, so the misplaced one shows up as a large residual instead of being averaged
     * into the answer and hiding itself.
     */
    @Test
    public void hardwareInTheWrongHoleIsLeftOutOfTheAgreedSet() {
        DistanceNetwork ref = DistanceNetwork.over(PART);
        float[][] shot = new float[PART.length][];
        for (int i = 0; i < PART.length; i++) shot[i] = PART[i].clone();
        shot[2] = new float[]{0.55f, 0.78f};   // slot 2's fastener, fitted a quarter-part off

        DistanceNetwork.Consensus c = new DistanceNetwork.Matcher(ref, DistanceNetwork.over(shot))
                .grow(0, 0, 3, 3, 0.12 * ref.spread());
        assertNotNull(c);
        assertEquals("the four correctly fitted positions agree", 4, c.size);
        assertEquals("the misplaced one is not absorbed", -1, c.observedForNode[2]);
    }

    /**
     * The limit that keeps the transform in the pipeline: separations are identical under
     * reflection, so the network alone will happily identify a part with its mirror image.
     *
     * Pinned rather than fixed. {@code LayoutMatcher} fits a proper similarity transform over
     * whatever the network proposes and counts inliers against that, and a mirrored set fails
     * there — which is why the network is used to propose correspondences and never to accept
     * an alignment on its own.
     */
    @Test
    public void theNetworkCannotTellAPartFromItsMirrorImage() {
        DistanceNetwork ref = DistanceNetwork.over(PART);
        float[][] mirrored = new float[PART.length][];
        for (int i = 0; i < PART.length; i++) mirrored[i] = new float[]{-PART[i][0], PART[i][1]};

        DistanceNetwork.Consensus c = new DistanceNetwork.Matcher(ref, DistanceNetwork.over(mirrored))
                .grow(0, 0, 3, 3, 0.12 * ref.spread());
        assertNotNull(c);
        assertEquals("the mirror agrees perfectly on separations alone", 5, c.size);
        assertTrue(c.distortion < 1e-4);
    }

    /** A degenerate seed states nothing, so it must yield nothing rather than a division by zero. */
    @Test
    public void aCoincidentSeedIsRefused() {
        DistanceNetwork ref = DistanceNetwork.over(pts(0f, 0f, 0f, 0f, 1f, 1f));
        DistanceNetwork obs = DistanceNetwork.over(pts(0f, 0f, 0.5f, 0.5f, 1f, 1f));
        assertNull(new DistanceNetwork.Matcher(ref, obs).grow(0, 0, 1, 2, 0.1));
    }
}
