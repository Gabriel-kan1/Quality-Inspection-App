package com.example.ml;

import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Establishes how many millimetres one pixel represents, by locating a printed square
 * marker of known size in the photograph.
 *
 * A photograph carries no inherent scale, so nothing can be measured without a reference
 * object of known dimensions in frame. This is the whole of that step.
 *
 * <h3>What this deliberately does not do (Stage A scope)</h3>
 * It produces a single, uniform millimetres-per-pixel figure and applies it everywhere in
 * the image. That is only strictly valid when the camera is perpendicular to the part and
 * the measured features lie in the same plane, at the same depth, as the marker. Tilt the
 * camera and features further from it will measure small, features nearer will measure
 * large — the error is a function of camera pose, not of the part.
 *
 * The marker's four corners are already sufficient to fix this via a homography
 * ({@code Imgproc.getPerspectiveTransform} + {@code warpPerspective}, both core imgproc —
 * no contrib/ArUco dependency needed). That is Stage B and is intentionally not here: the
 * value of Stage A is establishing that the measurement path works end to end and
 * quantifying its repeatability before adding rectification on top.
 *
 * <h3>Marker</h3>
 * A solid dark square on a light background, printed at a known size, placed flat on the
 * same surface as the parts being measured. Detected geometrically (quadrilateral,
 * convex, four near-equal sides) rather than by any encoded pattern, so any plain square
 * works and no marker dictionary is required.
 */
public final class ScaleCalibration {

    private static final String TAG = "ScaleCalibration";

    /**
     * Printed side length of the calibration marker, in millimetres. Must match the
     * physical marker — every measurement in the app scales linearly with this number, so
     * a wrong value here silently biases every result by the same factor.
     */
    public static double MARKER_SIZE_MM = 25.0;

    /** A candidate must occupy at least this fraction of the image to be considered. */
    private static final double MIN_AREA_FRACTION = 0.0008;
    /** And at most this much, so a large background rectangle isn't mistaken for the marker. */
    private static final double MAX_AREA_FRACTION = 0.25;
    /** Longest side may exceed the shortest by at most this factor, i.e. squareness. */
    private static final double MAX_SIDE_RATIO = 1.25;
    /** Corner angles must be within this many degrees of square. */
    private static final double MAX_ANGLE_DEVIATION_DEG = 20.0;

    public final double mmPerPixel;
    /** Mean marker side length in pixels — the raw measurement the scale derives from. */
    public final double markerSidePx;
    /** Squareness of the detected quad (1.0 = perfect). Low values suggest an oblique shot. */
    public final double squareness;
    /**
     * Where the marker sits in the image, so the detection stage can exclude it. The marker
     * is a high-contrast square lying on the same surface as the parts, so the feature
     * detector finds it readily — left in, it inflates detection counts, gets logged as a
     * classification event, and can be offered to the operator as something to label.
     */
    public final org.opencv.core.Rect markerBounds;

    private ScaleCalibration(double mmPerPixel, double markerSidePx, double squareness,
                             org.opencv.core.Rect markerBounds) {
        this.mmPerPixel = mmPerPixel;
        this.markerSidePx = markerSidePx;
        this.squareness = squareness;
        this.markerBounds = markerBounds;
    }

    /**
     * True when {@code box} is substantially inside the marker — used to drop the marker and
     * any contour fragments of it (corners, inner edges) from the feature list.
     */
    public boolean overlapsMarker(org.opencv.core.Rect box) {
        if (markerBounds == null || box == null) return false;
        int x1 = Math.max(markerBounds.x, box.x);
        int y1 = Math.max(markerBounds.y, box.y);
        int x2 = Math.min(markerBounds.x + markerBounds.width, box.x + box.width);
        int y2 = Math.min(markerBounds.y + markerBounds.height, box.y + box.height);
        if (x2 <= x1 || y2 <= y1) return false;
        double overlap = (double) (x2 - x1) * (y2 - y1);
        double boxArea = (double) box.width * box.height;
        return boxArea > 0 && overlap / boxArea >= 0.5;
    }

    public double toMm(double pixels) {
        return pixels * mmPerPixel;
    }

    /** Human-readable provenance for the results screen and the inspection report. */
    public String describe() {
        return String.format(Locale.US,
                "%.4f mm/px · marker %.1f mm over %.1f px · squareness %.2f",
                mmPerPixel, MARKER_SIZE_MM, markerSidePx, squareness);
    }

    /**
     * Locates the calibration marker and derives the scale, or returns null when no
     * plausible marker is present. A null result must disable measurement rather than
     * fall back to a guessed scale — a fabricated dimension is worse than none.
     *
     * @param src colour source image (RGBA), unmodified by this call.
     */
    public static ScaleCalibration detect(Mat src) {
        if (src == null || src.empty()) return null;

        Mat gray = new Mat();
        Mat binary = new Mat();
        try {
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY);
            // Otsu rather than a fixed cut: the marker is high-contrast by construction, but
            // the ambient level it sits against varies with the same lighting the rest of the
            // pipeline already adapts to.
            Imgproc.threshold(gray, binary, 0, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);

            List<MatOfPoint> contours = new ArrayList<>();
            Mat hierarchy = new Mat();
            Imgproc.findContours(binary, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
            hierarchy.release();

            double imageArea = (double) src.width() * src.height();
            double minArea = imageArea * MIN_AREA_FRACTION;
            double maxArea = imageArea * MAX_AREA_FRACTION;

            ScaleCalibration best = null;
            double bestArea = 0;
            int examined = 0;

            for (MatOfPoint contour : contours) {
                double area = Imgproc.contourArea(contour);
                if (area < minArea || area > maxArea) continue;
                examined++;

                MatOfPoint2f curve = new MatOfPoint2f(contour.toArray());
                MatOfPoint2f approx = new MatOfPoint2f();
                double peri = Imgproc.arcLength(curve, true);
                Imgproc.approxPolyDP(curve, approx, 0.02 * peri, true);
                Point[] pts = approx.toArray();
                curve.release();
                approx.release();

                if (pts.length != 4) continue;

                MatOfPoint quad = new MatOfPoint(pts);
                boolean convex = Imgproc.isContourConvex(quad);
                quad.release();
                if (!convex) continue;

                double[] sides = new double[4];
                for (int i = 0; i < 4; i++) sides[i] = dist(pts[i], pts[(i + 1) % 4]);
                double shortest = Math.min(Math.min(sides[0], sides[1]), Math.min(sides[2], sides[3]));
                double longest = Math.max(Math.max(sides[0], sides[1]), Math.max(sides[2], sides[3]));
                if (shortest <= 0 || longest / shortest > MAX_SIDE_RATIO) continue;

                if (maxAngleDeviation(pts) > MAX_ANGLE_DEVIATION_DEG) continue;

                double meanSide = (sides[0] + sides[1] + sides[2] + sides[3]) / 4.0;
                if (meanSide <= 0) continue;

                // Largest qualifying square wins: a marker is placed deliberately and is
                // normally the most prominent square in frame, whereas incidental squares
                // (fastener heads, panel cut-outs) are smaller.
                if (area > bestArea) {
                    bestArea = area;
                    best = new ScaleCalibration(
                            MARKER_SIZE_MM / meanSide, meanSide, shortest / longest,
                            Imgproc.boundingRect(new MatOfPoint(pts)));
                }
            }

            if (best == null) {
                Log.d(TAG, "No calibration marker found (" + examined
                        + " candidates of suitable area, none square enough).");
            } else {
                Log.d(TAG, "Calibration: " + best.describe());
            }
            return best;
        } catch (Exception e) {
            Log.e(TAG, "Marker detection failed", e);
            return null;
        } finally {
            gray.release();
            binary.release();
        }
    }

    private static double dist(Point a, Point b) {
        return Math.hypot(a.x - b.x, a.y - b.y);
    }

    /** Largest deviation from 90 degrees across the quad's four corners. */
    private static double maxAngleDeviation(Point[] p) {
        double worst = 0;
        for (int i = 0; i < 4; i++) {
            Point prev = p[(i + 3) % 4], cur = p[i], next = p[(i + 1) % 4];
            double ax = prev.x - cur.x, ay = prev.y - cur.y;
            double bx = next.x - cur.x, by = next.y - cur.y;
            double magA = Math.hypot(ax, ay), magB = Math.hypot(bx, by);
            if (magA == 0 || magB == 0) return 180;
            double cos = (ax * bx + ay * by) / (magA * magB);
            cos = Math.max(-1, Math.min(1, cos));
            double deg = Math.toDegrees(Math.acos(cos));
            worst = Math.max(worst, Math.abs(deg - 90));
        }
        return worst;
    }
}
