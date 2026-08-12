package com.example.ml;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * A physical size for one detected feature.
 *
 * Measured from the feature's own contour rather than its bounding box. The boxes the
 * detection stage produces are axis-aligned integer rectangles, so a fastener photographed
 * at 45 degrees gets a box up to sqrt(2) larger than the same fastener square-on — a 40%
 * error introduced purely by rotation, which is far larger than any tolerance worth
 * checking. {@code minEnclosingCircle} and {@code minAreaRect} are both rotation-invariant.
 */
public final class FeatureMeasurement {

    public enum Method {
        /** Diameter of the smallest enclosing circle — for round hardware. */
        ENCLOSING_CIRCLE,
        /** Shorter side of the minimum-area rectangle — for elongated hardware. */
        MIN_AREA_RECT
    }

    /** Diameter of the smallest enclosing circle, in mm. */
    public final double circleMm;
    /** Minimum-area rectangle: shorter side, in mm. */
    public final double rectMinorMm;
    /** Minimum-area rectangle: longer side, in mm. */
    public final double rectMajorMm;
    /** Enclosing-circle diameter in pixels, before scaling — the raw geometric quantity. */
    public final double circlePx;

    private FeatureMeasurement(double circleMm, double rectMinorMm, double rectMajorMm, double circlePx) {
        this.circleMm = circleMm;
        this.rectMinorMm = rectMinorMm;
        this.rectMajorMm = rectMajorMm;
        this.circlePx = circlePx;
    }

    /**
     * Both estimators are computed and kept, rather than one being chosen during detection.
     *
     * The choice depends on the feature's type, and the type can change after detection — an
     * operator correcting a label, or confirming one the classifier was unsure of. Selecting
     * at detection time froze the estimator against a label that no longer applied, so a
     * corrected bolt head could still be reported by its rectangle. Choosing at render time
     * keeps the number consistent with whatever the feature is currently called.
     */
    public Method methodFor(String type) {
        return isRoundType(type) ? Method.ENCLOSING_CIRCLE : Method.MIN_AREA_RECT;
    }

    /** Primary dimension for a given type: diameter if round, else the rectangle's short side. */
    public double sizeMmFor(String type) {
        return isRoundType(type) ? circleMm : rectMinorMm;
    }

    /** Compact form for an overlay label, e.g. {@code ⌀12.4mm} or {@code 6.2×14.1mm}. */
    public String shortLabel(String type) {
        return isRoundType(type)
                ? String.format(Locale.US, "⌀%.1fmm", circleMm)
                : String.format(Locale.US, "%.1f×%.1fmm", rectMinorMm, rectMajorMm);
    }

    /**
     * Measures the feature inside {@code box} using the already-computed edge map.
     *
     * @param edges  closed edge image from the detection stage — reused rather than
     *               recomputed so the measurement describes exactly the contour the
     *               detector fired on.
     * @param box    the detection's bounding box, in {@code edges} coordinates.
     * @param scale  pixel-to-millimetre scale; null disables measurement entirely.
     * @return the measurement, or null if no usable contour was found or there is no scale.
     */
    public static FeatureMeasurement measure(Mat edges, Rect box, ScaleCalibration scale) {
        if (edges == null || edges.empty() || scale == null) return null;

        Rect safe = clamp(box, edges.width(), edges.height());
        if (safe == null) return null;

        Mat roi = null;
        try {
            roi = new Mat(edges, safe);
            List<MatOfPoint> contours = new ArrayList<>();
            Mat hierarchy = new Mat();
            Imgproc.findContours(roi, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
            hierarchy.release();
            if (contours.isEmpty()) return null;

            // The box may enclose fragments; the feature is the largest contour within it.
            MatOfPoint largest = null;
            double largestArea = -1;
            for (MatOfPoint c : contours) {
                double a = Imgproc.contourArea(c);
                if (a > largestArea) {
                    largestArea = a;
                    largest = c;
                }
            }
            if (largest == null) return null;

            MatOfPoint2f curve = new MatOfPoint2f(largest.toArray());
            try {
                Point centre = new Point();
                float[] radius = new float[1];
                Imgproc.minEnclosingCircle(curve, centre, radius);
                double diameterPx = radius[0] * 2.0;

                RotatedRect rr = Imgproc.minAreaRect(curve);
                double minor = Math.min(rr.size.width, rr.size.height);
                double major = Math.max(rr.size.width, rr.size.height);

                if (diameterPx <= 0 || minor <= 0) return null;
                return new FeatureMeasurement(scale.toMm(diameterPx), scale.toMm(minor),
                        scale.toMm(major), diameterPx);
            } finally {
                curve.release();
            }
        } catch (Exception e) {
            return null;
        } finally {
            if (roi != null) roi.release();
        }
    }

    /**
     * Round hardware is measured as a circle, everything else by its minimum-area
     * rectangle. Unknown types default to a rectangle, which reports both dimensions and
     * so asserts less about a shape the classifier could not identify.
     */
    public static boolean isRoundType(String type) {
        if (type == null) return false;
        switch (type) {
            case "bolt_head":
            case "hole":
            case "rivet_head":
            case "stud":
                return true;
            default:
                return false;
        }
    }

    private static Rect clamp(Rect r, int width, int height) {
        if (r == null) return null;
        int x = Math.max(0, r.x);
        int y = Math.max(0, r.y);
        int w = Math.min(r.width + Math.min(0, r.x), width - x);
        int h = Math.min(r.height + Math.min(0, r.y), height - y);
        if (w < 3 || h < 3) return null;
        return new Rect(x, y, w, h);
    }
}
