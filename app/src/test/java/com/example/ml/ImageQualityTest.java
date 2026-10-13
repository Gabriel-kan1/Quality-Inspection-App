package com.example.ml;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import java.util.List;

/**
 * Covers the arithmetic and the decision behind the adaptive edge stage's image-quality
 * measurement — everything except the Laplacian itself, which is a native OpenCV call the
 * JVM cannot run.
 *
 * <p>The numbers asserted against are the ones {@code python/adaptive_preprocessing_study.py}
 * measured; the thresholds exist to reproduce that study's crossovers, so a test that pins
 * them is a test that the Java side still agrees with the evidence.
 */
public class ImageQualityTest {

    /** A histogram with {@code count} pixels at each of the given levels. */
    private static long[] histogram(int[] levels, long[] counts) {
        long[] h = new long[256];
        for (int i = 0; i < levels.length; i++) h[levels[i]] += counts[i];
        return h;
    }

    @Test
    public void medianAndP99AreReadFromTheHistogram() {
        // 90% of pixels at 3, 9% at 40, 1% at 200: the median is 3, the 99th percentile 40
        // (the first level whose cumulative count reaches 99%), and the top 1% is ignored.
        ImageQuality q = ImageQuality.fromHistogram(
                histogram(new int[]{3, 40, 200}, new long[]{900, 90, 10}));
        assertEquals(3 * ImageQuality.MAD_TO_SIGMA, q.noiseSigma, 1e-9);
        assertEquals(40, q.edgeContrast, 1e-9);
    }

    @Test
    public void whiteNoiseOfKnownSigmaReadsBackAsThatSigma() {
        // The scaling constant: for the 3x3 Laplacian, noise of sigma s produces an absolute
        // response whose median is 0.6745 * sqrt(20) * s. A histogram whose median sits
        // exactly there must read back as s.
        double s = 10.0;
        int median = (int) Math.round(0.6745 * Math.sqrt(20.0) * s);
        ImageQuality q = ImageQuality.fromHistogram(
                histogram(new int[]{median}, new long[]{1000}));
        assertEquals(s, q.noiseSigma, 0.2);
    }

    @Test
    public void anEmptyHistogramIsNotACrash() {
        ImageQuality q = ImageQuality.fromHistogram(new long[256]);
        assertEquals(0, q.noiseSigma, 0);
        assertEquals(0, q.edgeContrast, 0);
        // Nothing measured reads as an ordinary photograph, not as a soft one: with no
        // evidence the stage runs the rung the ground-truth search found best.
        assertEquals(PreprocessingStrength.MEDIUM, new ImageQuality(0, 200).suggest());
    }

    // ── The decision ─────────────────────────────────────────────────────────

    @Test
    public void aCleanSharpPhotographRunsHybridV6() {
        // stud1.jpg as measured: noise 3.3, edge contrast 70.
        ImageQuality q = new ImageQuality(3.3, 70);
        assertEquals(PreprocessingStrength.MEDIUM, q.suggest());
        assertFalse(q.isNoisy());
        assertFalse(q.isSoft());
    }

    @Test
    public void aNoisyPhotographRunsTheStrongRung() {
        // stud1.jpg with sigma 10 of noise added reads 13.9; three.jpg's textured plate 18.6.
        assertEquals(PreprocessingStrength.STRONG, new ImageQuality(13.9, 186).suggest());
        assertEquals(PreprocessingStrength.STRONG, new ImageQuality(18.6, 255).suggest());
        assertTrue(new ImageQuality(13.9, 186).isNoisy());
    }

    @Test
    public void aSoftPhotographRunsTheLightRung() {
        // The assets blurred 9px read a median edge contrast of 17; at 7px (24) v6 still wins.
        assertEquals(PreprocessingStrength.LIGHT, new ImageQuality(0.7, 17).suggest());
        assertEquals(PreprocessingStrength.MEDIUM, new ImageQuality(1.3, 24).suggest());
        assertTrue(new ImageQuality(0.7, 17).isSoft());
    }

    /** Noise is decided first: grain is the thing to suppress even in a soft photo. */
    @Test
    public void noiseOutranksSoftness() {
        assertEquals(PreprocessingStrength.STRONG, new ImageQuality(15, 15).suggest());
    }

    @Test
    public void thresholdsSitAtTheMeasuredCrossovers() {
        // The textured plates (12-19) tie between the rungs and screws.jpg reads 11.9, so
        // strong starts just above the plates' floor; light needs edges weaker than a 7px blur
        // leaves them.
        assertEquals(12.0, ImageQuality.NOISE_STRONG, 0);
        assertEquals(20.0, ImageQuality.SOFT_CONTRAST, 0);
        assertEquals(PreprocessingStrength.MEDIUM, new ImageQuality(11.9, 255).suggest());
        assertEquals(PreprocessingStrength.STRONG, new ImageQuality(12.0, 255).suggest());
    }

    // ── Eligibility for the size test ────────────────────────────────────────

    /**
     * Every rung goes on the ballot now, at any sharpness.
     *
     * <p>Light used to be withheld above a contrast of 40 because it over-detects on sharp
     * photographs and the size score could not see that. It also withheld light on
     * {@code screws_none.jpg} (contrast 62), where it is the only rung that finds one of the
     * bolts — so the layout came out with five slots instead of six. {@link ReferenceFit} charges
     * unmatched detections on the part now, so over-detection is scored rather than assumed.
     */
    @Test
    public void everyRungIsOnTheBallotWhateverTheSharpness() {
        for (ImageQuality q : new ImageQuality[]{
                new ImageQuality(2.0, 42),     // hole2.jpg, the sharpest-edged clean asset
                new ImageQuality(2.65, 62),    // screws_none.jpg — the case that motivated this
                new ImageQuality(18.6, 255),   // three.jpg, a textured plate
                new ImageQuality(0.7, 17)}) {  // blurred
            List<PreprocessingStrength> c = q.candidates();
            assertEquals(q.describe(), PreprocessingStrength.values().length, c.size());
            assertTrue(q.describe(), c.contains(PreprocessingStrength.LIGHT));
            assertTrue(q.describe(), c.contains(PreprocessingStrength.MEDIUM));
            assertTrue(q.describe(), c.contains(PreprocessingStrength.STRONG));
        }
    }

    @Test
    public void theSuggestedRungIsAlwaysAmongTheCandidates() {
        for (ImageQuality q : new ImageQuality[]{
                new ImageQuality(3, 120), new ImageQuality(15, 200), new ImageQuality(0.5, 15),
                new ImageQuality(15, 15), new ImageQuality(2, 39), new ImageQuality(2, 40)}) {
            assertTrue(q.describe(), q.candidates().contains(q.suggest()));
        }
    }
}
