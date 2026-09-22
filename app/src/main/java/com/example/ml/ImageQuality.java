package com.example.ml;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

/**
 * Two numbers about a photograph that decide how hard its edge stage may denoise: how noisy
 * it is, and how much edge it has to lose.
 *
 * <h3>Noise, measured as noise</h3>
 * The shipped stack's {@code computeAdaptiveBilateralD} buckets the <em>variance</em> of the
 * Laplacian, and {@code python/hybrid_v2_study.py} found that step firing backwards: it handed
 * the least smoothing to the two sharpest photographs. Variance is dominated by the sparse,
 * large responses that edges produce, so a clean, sharp, high-resolution photo has a huge
 * Laplacian variance and no noise at all. On the bundled assets it spans 22-2877 with no noise
 * added, which puts every bucket boundary (200-700) inside the content range.
 *
 * <p>{@link #noiseSigma} is Donoho's robust estimate instead: the <em>median</em> of the
 * Laplacian's absolute response, scaled to the standard deviation of white noise that would
 * produce it. A median ignores the sparse large responses and reads the bulk, which is what
 * noise is. Measured on the same assets it sits at 2-4 for the clean photographs and 12-19 for
 * the textured plates, and tracks synthetic noise almost one-for-one (adding sigma 5/10/20 reads
 * 8.6/14.6/27.2). Dense machining texture reads as noise too — correctly, for this purpose,
 * since texture is what fragments into contours.
 *
 * <h3>Edge contrast</h3>
 * {@link #edgeContrast} is the 99th percentile of the same response: how strong the strongest
 * edges are. A soft or blurred photograph has weak edges, and smoothing it harder loses what
 * little it had — "lower quality" wants a gentler stage, not a stronger one. The clean assets
 * sit at 42-255; a 9px Gaussian blur takes them to about 17.
 *
 * <p>All constants come from {@code python/adaptive_preprocessing_study.py}. Re-derive if the
 * working resolution or the ladder changes.
 */
public final class ImageQuality {

    /**
     * Noise estimate from which {@link PreprocessingStrength#STRONG} is the rung to run.
     * Against ground truth the strong rung overtakes v6 at sigma 7 of added noise (estimate
     * 10.6) and is clearly ahead by sigma 10 (14.6: weighted 20 against 31); the textured
     * plates sit at 12-19 and the two rungs tie on them.
     */
    public static final double NOISE_STRONG = 12.0;
    /**
     * Edge contrast below which {@link PreprocessingStrength#LIGHT} is the rung to run. A 9px
     * blur leaves the assets at a median 17, where light scores 24 against v6's 41; at 7px
     * (24) v6 still wins.
     */
    public static final double SOFT_CONTRAST = 20.0;
    /**
     * Edge contrast below which the light rung may be <em>offered</em> to the size test at all.
     * On anything sharp the light rung is catastrophic — 25 misses and 205 extras on the clean
     * set — in a way a size score cannot see, because it judges the boxes at the slots and not
     * the extras between them. Every clean asset sits above this.
     */
    public static final double LIGHT_ELIGIBLE_CONTRAST = 40.0;

    /**
     * For the 3x3 Laplacian kernel {@code [0 1 0; 1 -4 1; 0 1 0]}, white noise of standard
     * deviation s has a response of standard deviation s*sqrt(20), and the median of the
     * absolute value of a zero-mean normal is 0.6745 of its sigma.
     */
    static final double MAD_TO_SIGMA = 1.0 / (0.6745 * Math.sqrt(20.0));

    /** Robust noise standard deviation, in grey levels. */
    public final double noiseSigma;
    /** 99th percentile of the absolute Laplacian response, in grey levels. */
    public final double edgeContrast;

    ImageQuality(double noiseSigma, double edgeContrast) {
        this.noiseSigma = noiseSigma;
        this.edgeContrast = edgeContrast;
    }

    /**
     * Measures a working-resolution grayscale image. One 16-bit Laplacian, one absolute
     * value and one 256-bin histogram — about 20ms on a desktop, cheaper than a single
     * bilateral filter.
     *
     * @param gray CV_8UC1 at {@link HardwareFeatureExtractor#WORKING_MAX_DIM}; not modified
     */
    public static ImageQuality assess(Mat gray) {
        Mat lap = new Mat();
        Mat abs = new Mat();
        Mat hist = new Mat();
        try {
            Imgproc.Laplacian(gray, lap, CvType.CV_16S, 3);
            // Saturates at 255. The median never gets near it, and a p99 pinned at 255 is
            // read as "very strong edges", which is all the threshold needs from it.
            Core.convertScaleAbs(lap, abs);
            List<Mat> images = new ArrayList<>();
            images.add(abs);
            Imgproc.calcHist(images, new MatOfInt(0), new Mat(), hist, new MatOfInt(256),
                    new MatOfFloat(0f, 256f));
            long[] counts = new long[256];
            for (int i = 0; i < 256; i++) counts[i] = (long) hist.get(i, 0)[0];
            return fromHistogram(counts);
        } finally {
            lap.release();
            abs.release();
            hist.release();
        }
    }

    /**
     * The measurement from a histogram of absolute Laplacian responses, one bin per grey
     * level. Split out so the arithmetic can be tested without OpenCV.
     */
    static ImageQuality fromHistogram(long[] counts) {
        long total = 0;
        for (long c : counts) total += c;
        if (total == 0) return new ImageQuality(0, 0);
        return new ImageQuality(percentile(counts, total, 0.5) * MAD_TO_SIGMA,
                percentile(counts, total, 0.99));
    }

    /** Smallest bin whose cumulative count reaches the fraction — the nearest-rank percentile. */
    private static int percentile(long[] counts, long total, double fraction) {
        long target = (long) Math.ceil(total * fraction);
        long seen = 0;
        for (int i = 0; i < counts.length; i++) {
            seen += counts[i];
            if (seen >= target) return i;
        }
        return counts.length - 1;
    }

    /**
     * The rung this photograph should run on its own evidence. Noise is tested first: a
     * photograph that is both noisy and soft barely exists (noise raises the 99th percentile),
     * and where it did the grain would still be the thing to suppress.
     */
    public PreprocessingStrength suggest() {
        if (noiseSigma >= NOISE_STRONG) return PreprocessingStrength.STRONG;
        if (edgeContrast < SOFT_CONTRAST) return PreprocessingStrength.LIGHT;
        return PreprocessingStrength.MEDIUM;
    }

    /**
     * The rungs a size test may choose between. Medium and strong are never harmful enough to
     * exclude; light is offered only when the edges are genuinely weak. Always contains
     * {@link #suggest()}.
     */
    public List<PreprocessingStrength> candidates() {
        List<PreprocessingStrength> out = new ArrayList<>();
        if (edgeContrast < LIGHT_ELIGIBLE_CONTRAST) out.add(PreprocessingStrength.LIGHT);
        out.add(PreprocessingStrength.MEDIUM);
        out.add(PreprocessingStrength.STRONG);
        return Collections.unmodifiableList(out);
    }

    /** Whether the photograph reads as soft: weak edges, the light rung's territory. */
    public boolean isSoft() {
        return edgeContrast < SOFT_CONTRAST;
    }

    /** Whether the photograph reads as noisy: the strong rung's territory. */
    public boolean isNoisy() {
        return noiseSigma >= NOISE_STRONG;
    }

    public String describe() {
        return String.format(Locale.US, "noise %.1f, edge contrast %.0f", noiseSigma, edgeContrast);
    }

    @Override
    public String toString() {
        return "ImageQuality{" + describe() + "}";
    }
}
