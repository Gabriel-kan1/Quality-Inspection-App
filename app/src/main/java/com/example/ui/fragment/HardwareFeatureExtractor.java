package com.example.ui.fragment;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.RectF;
import android.net.Uri;
import android.util.Log;

import com.example.ml.ClipEmbedder;
import com.example.ml.ReferenceLibrary;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

public class HardwareFeatureExtractor {
    private static final String TAG = "HardwareFeatureExtractor";

    // ─────────────────────────────────────────────────────────────
    // CONFIG — tune detection here. Grouped to mirror the Python script's
    // CONFIG block. All values are non-final so they can be adjusted at
    // runtime (e.g. from a debug settings screen) without recompiling logic.
    // ─────────────────────────────────────────────────────────────

    // --- Edge detection (computeEdges) ---
    /** CLAHE local-contrast boost. Higher = crisper edges but more texture noise. */
    private static double CLAHE_CLIP_LIMIT = 1.0;
    /** Bilateral filter diameter/sigmas — denoise while preserving edges. */
    private static double BILATERAL_SIGMA_COLOR = 60;
    private static double BILATERAL_SIGMA_SPACE = 20;
    /**
     * Canny thresholds are computed adaptively per-image via Otsu's method rather
     * than fixed, so lighting/exposure differences between photos don't require
     * re-tuning. These multipliers scale the Otsu threshold (low = ratio * high).
     */
    private static double CANNY_LOW_RATIO = 0.5;
    private static double CANNY_HIGH_CAP = 255.0;
    /** Morphology: dilate (rect) to bridge gaps, then close (ellipse) to seal contours. */
    private static int DILATE_KERNEL_SIZE = 5;
    private static int CLOSE_KERNEL_SIZE = 7;

    // --- Geometry filtering (findFeatureBoxes) ---
    private static float MAX_DIMENSION_RATIO = 0.4f;
    private static float MAX_ASPECT_RATIO = 5.0f;
    private static float MIN_ASPECT_RATIO = 0.2f;
    private static int EDGE_MARGIN = 10;
    private static double DYNAMIC_AREA_MULTIPLIER = 0.15;
    private static int PROXIMITY_THRESHOLD = 10;
    private static double FILL_RATIO_THRESHOLD = 0.7;
    /**
     * Minimum circularity (4*pi*area / perimeter^2, 1.0 = perfect circle) required for
     * contours that are roughly as wide as they are tall (aspect ratio within
     * CIRCULARITY_ASPECT_BAND of 1.0). Rejects scratches/seams among the round
     * fastener types (bolt_head, hole, rivet_head, stud) without touching elongated
     * contours, because "screw" in this pipeline is a cylindrical shaft viewed from
     * the side (long and thin, not round) — see TEXT_PROMPTS in the Python script.
     * Applying circularity indiscriminately would filter out real screws. Set to 0
     * to disable entirely.
     */
    private static double MIN_CIRCULARITY = 0.55;
    /** How close to a 1:1 aspect ratio a contour must be before the circularity check applies. */
    private static double CIRCULARITY_ASPECT_BAND = 0.35;

    private static final int CROP_PADDING = 40;   // matches CROP_PADDING in the Python script
    /** Upper bound on boxes carried forward from one image; see the safety valve in findFeatureBoxes. */
    private static int MAX_DETECTIONS = 300;

    static {
        if (!OpenCVLoader.initDebug()) {
            Log.e(TAG, "OpenCV initialization failed.");
        } else {
            Log.d(TAG, "OpenCV initialization succeeded.");
        }
    }

    public static class ExtractedFeature {
        public String type;
        public RectF rectFraction;
        public int color;
        public String label;
        public float confidence;
        /**
         * On-disk path of the persisted crop, or null if it wasn't saved.
         *
         * Deliberately a path rather than a Bitmap: retaining one bitmap per detection made
         * memory scale with contour count, and since Android 8 bitmap pixels live in native
         * memory, that grows anon RSS silently instead of throwing OutOfMemoryError — the app
         * just climbs until the kernel low-memory killer takes it. The UI decodes the handful
         * of thumbnails it actually needs from disk instead.
         */
        public String cropPath;
        /** Full classification detail (scores, runner-up, reason) — drives active-learning selection and event logging. */
        public ReferenceLibrary.Classification classification;

        public ExtractedFeature(String type, RectF rectFraction, int color, String label, float confidence,
                                String cropPath, ReferenceLibrary.Classification classification) {
            this.type = type;
            this.rectFraction = rectFraction;
            this.color = color;
            this.label = label;
            this.confidence = confidence;
            this.cropPath = cropPath;
            this.classification = classification;
        }
    }

    public static class AnalysisResult {
        public List<ExtractedFeature> features;
        public Bitmap processedBitmap;

        public AnalysisResult(List<ExtractedFeature> features, Bitmap processedBitmap) {
            this.features = features;
            this.processedBitmap = processedBitmap;
        }
    }

    /**
     * Receives each detected crop during extraction so it can be persisted, and returns the
     * path it was written to (or null). The crop is recycled immediately after this returns,
     * so implementations must not retain the Bitmap.
     */
    public interface CropSink {
        String save(Bitmap crop, int index);
    }

    /** A detected feature crop (colour bitmap) plus its fractional location in the source image. */
    public static class CropRegion {
        public final Bitmap bitmap;
        public final RectF rectFraction;

        public CropRegion(Bitmap bitmap, RectF rectFraction) {
            this.bitmap = bitmap;
            this.rectFraction = rectFraction;
        }
    }

    /** Computes an adaptive bilateral filter diameter based on the noise level */
    private static int computeAdaptiveBilateralD(Mat gray) {
        // compute Laplacian of the grayscale image
        Mat laplacian = new Mat();
        Imgproc.Laplacian(gray, laplacian, org.opencv.core.CvType.CV_64F);

        // compute mean and standard deviation of the Laplacian
        org.opencv.core.MatOfDouble mean   = new org.opencv.core.MatOfDouble();
        org.opencv.core.MatOfDouble stddev = new org.opencv.core.MatOfDouble();
        Core.meanStdDev(laplacian, mean, stddev);

        // variance = stddev squared — represents overall noise level
        double std           = stddev.toArray()[0];
        double noiseVariance = std * std;

        laplacian.release();
        mean.release();
        stddev.release();

        Log.d(TAG, String.format(Locale.US,
                "Adaptive bilateral D — Laplacian variance: %.1f", noiseVariance));


        if (noiseVariance < 200) {
            // Low noise level
            return 1;
        } else if  (noiseVariance < 300) {
            // Low noise level
            return 2;
        } else if (noiseVariance < 400) {
            // Low noise level
            return 3;
        } else if (noiseVariance < 600) {
            // Low noise level
            return 4;
        } else if (noiseVariance < 700) {
            // Medium noise level
            return 5;
        } else {
            // High noise level
            return 6;
        }
    }

    /** Bundles the intermediate edge Mats produced by {@link #computeEdges(Mat)}. */
    private static final class Edges {
        final Mat gray;
        final Mat edgesDilated;
        final Mat edgesClosed;

        Edges(Mat gray, Mat edgesDilated, Mat edgesClosed) {
            this.gray = gray;
            this.edgesDilated = edgesDilated;
            this.edgesClosed = edgesClosed;
        }

        void release() {
            gray.release();
            edgesDilated.release();
            edgesClosed.release();
        }
    }

    public static AnalysisResult extractFeatures(Context context, String uriString) {
        return extractFeatures(context, uriString, null);
    }

    /**
     * Runs only the OpenCV geometry stage and returns the colour crops of each detected
     * feature (no classification). Used by the in-app Reference Setup builder so the user
     * can label crops to build a {@link ReferenceLibrary}. Callers own the returned bitmaps.
     */
    public static List<CropRegion> extractCrops(Context context, String uriString) {
        List<CropRegion> crops = new ArrayList<>();
        if (uriString == null || uriString.isEmpty()) return crops;

        Bitmap bitmap = loadBitmap(context, uriString, 1280);
        if (bitmap == null) return crops;

        try {
            Mat src = new Mat();
            Utils.bitmapToMat(bitmap, src);
            Edges e = computeEdges(src);

            float w = src.width();
            float h = src.height();
            for (Rect r : findFeatureBoxes(e.edgesClosed, w, h)) {
                Bitmap crop = cropToBitmap(src, r);
                if (crop == null) continue;
                RectF frac = new RectF(
                        r.x / w, r.y / h, (r.x + r.width) / w, (r.y + r.height) / h);
                crops.add(new CropRegion(crop, frac));
            }

            src.release();
            e.release();
        } catch (Exception ex) {
            Log.e(TAG, "extractCrops failure", ex);
        }
        return crops;
    }

    /**
     * Runs the OpenCV extraction pipeline and classifies each detected crop with CLIP
     * against the supplied {@link ReferenceLibrary}. When {@code library} is null or empty
     * (no reference set, or CLIP unavailable) every feature is labelled "unknown".
     */
    public static AnalysisResult extractFeatures(Context context, String uriString, ReferenceLibrary library) {
        return extractFeatures(context, uriString, library, null);
    }

    /**
     * As above, but hands each crop to {@code sink} for persistence before freeing it. Pass a
     * null sink when the caller only needs counts (e.g. the gold-standard reference pass) —
     * nothing is then written to disk and no crop outlives its own loop iteration.
     */
    public static AnalysisResult extractFeatures(Context context, String uriString,
                                                 ReferenceLibrary library, CropSink sink) {
        List<ExtractedFeature> features = new ArrayList<>();
        Bitmap diagnosticBitmap = null;

        if (uriString == null || uriString.isEmpty()) {
            return new AnalysisResult(features, null);
        }

        boolean canClassify = library != null && !library.isEmpty();
        ClipEmbedder embedder = canClassify ? ClipEmbedder.get(context) : null;

        String filename = getFilename(uriString);
        Log.d(TAG, "Running OpenCV Pipeline for: " + filename);

        // Load image with balanced scaling for analysis
        Bitmap bitmap = loadBitmap(context, uriString, 1280); 
        if (bitmap == null) {
            Log.e(TAG, "Failed to load bitmap for URI: " + uriString);
            return new AnalysisResult(features, null);
        }

        try {
            Mat src = new Mat();
            Utils.bitmapToMat(bitmap, src);

            // STEP 2 — preprocessing + edge detection (shared with extractCrops()).
            Edges e = computeEdges(src);

            // --- GENERATE DIAGNOSTIC BITMAP (Green Edges on Gray Background) ---
            Mat rgbaVisual = new Mat();
            Imgproc.cvtColor(e.gray, rgbaVisual, Imgproc.COLOR_GRAY2RGBA);
            // Highlight detected edges in bright green so they are visible on mobile displays.
            rgbaVisual.setTo(new Scalar(0, 255, 0, 255), e.edgesDilated);
            diagnosticBitmap = Bitmap.createBitmap(rgbaVisual.cols(), rgbaVisual.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(rgbaVisual, diagnosticBitmap);

            // STEP 3-6 — OpenCV crop extraction + CLIP classification
            performDynamicDetection(src, e.edgesClosed, embedder, library, sink, features);

            src.release();
            e.release();
            rgbaVisual.release();

        } catch (Exception e) {
            Log.e(TAG, "OpenCV Pipeline failure", e);
        }

        return new AnalysisResult(features, diagnosticBitmap);
    }

    private static void performDynamicDetection(Mat src, Mat edges, ClipEmbedder embedder,
                                                ReferenceLibrary library, CropSink sink,
                                                List<ExtractedFeature> outFeatures) {
        float imgWidth = (float) src.width();
        float imgHeight = (float) src.height();

        List<Rect> merged = findFeatureBoxes(edges, imgWidth, imgHeight);
        boolean canClassify = embedder != null && library != null && !library.isEmpty();

        int boxId = 0;
        for (Rect r : merged) {
            boxId++;

            // One crop is alive at a time: created, embedded, persisted, then freed before the
            // next iteration. Keeping them all would make peak memory scale with contour count.
            Bitmap crop = cropToBitmap(src, r);

            String type = ReferenceLibrary.UNKNOWN;
            float confidence = 0f;
            String label = boxId + ":?";
            ReferenceLibrary.Classification classification = null;
            String cropPath = null;

            try {
                if (canClassify && crop != null) {
                    float[] embedding = embedder.embed(crop);
                    ReferenceLibrary.Classification c = library.classify(embedding);
                    classification = c;
                    type = c.type;
                    confidence = c.score;

                    if (ReferenceLibrary.UNKNOWN.equals(c.type) && !"none".equals(c.closestMatch)) {
                        label = String.format(Locale.US, "%d:? (%s %.2f)", boxId, c.closestMatch, c.score);
                    } else {
                        label = String.format(Locale.US, "%d:%s %.2f", boxId, c.type, c.score);
                    }
                }
                if (sink != null && crop != null) {
                    cropPath = sink.save(crop, boxId);
                }
            } finally {
                if (crop != null) crop.recycle();
            }

            outFeatures.add(new ExtractedFeature(
                type,
                new RectF((float)r.x/imgWidth, (float)r.y/imgHeight, (float)(r.x+r.width)/imgWidth, (float)(r.y+r.height)/imgHeight),
                ReferenceLibrary.colorFor(type),
                label,
                confidence,
                cropPath,
                classification
            ));
        }
    }

    /**
     * Shared OpenCV geometry stage (Steps 3-4 of the Python script): finds contours,
     * applies the dimension/aspect/margin + dynamic-area filters, removes nested boxes
     * and merges nearby fragments. Returns the final feature bounding boxes.
     */
    private static List<Rect> findFeatureBoxes(Mat edges, float imgWidth, float imgHeight) {
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        hierarchy.release();

        List<Rect> validBoxes = new ArrayList<>();
        List<Double> validAreas = new ArrayList<>();

        for (MatOfPoint c : contours) {
            Rect r = Imgproc.boundingRect(c);

            // 1. Max Dimension Check (MAX_DIMENSION_RATIO)
            //if (r.width > imgWidth * MAX_DIMENSION_RATIO || r.height > imgHeight * MAX_DIMENSION_RATIO) continue;

            // 2. Aspect Ratio Check
            float aspect = (float) r.width / r.height;
            //if (aspect > MAX_ASPECT_RATIO || aspect < MIN_ASPECT_RATIO) continue;

            // 3. Edge Margin Check
            //if (r.x < EDGE_MARGIN || r.y < EDGE_MARGIN || (r.x + r.width) > (imgWidth - EDGE_MARGIN) || (r.y + r.height) > (imgHeight - EDGE_MARGIN)) continue;

            double area = Imgproc.contourArea(c);

            // 4. Circularity Check — only for roughly square/round boxes (near 1:1 aspect),
            // so elongated screw-shaft contours are never subject to it (see MIN_CIRCULARITY doc).
            if (MIN_CIRCULARITY > 0 && Math.abs(aspect - 1.0f) <= CIRCULARITY_ASPECT_BAND) {
                double circularity = circularityOf(c, area);
                //if (circularity < MIN_CIRCULARITY) continue;
            }

            validBoxes.add(r);
            validAreas.add(area);
        }

        if (validBoxes.isEmpty()) return new ArrayList<>();

        // 5. Dynamic Area Filtering
        List<Double> sortedAreas = new ArrayList<>(validAreas);
        Collections.sort(sortedAreas, Collections.reverseOrder());
        int topCount = Math.min(3, sortedAreas.size());
        double sumTop = 0;
        for (int i = 0; i < topCount; i++) sumTop += sortedAreas.get(i);
        double referenceAreaValue = sumTop / topCount;
        double dynamicMinArea = Math.max(25.0, referenceAreaValue * DYNAMIC_AREA_MULTIPLIER);

        List<Rect> sizedBoxes = new ArrayList<>();
        for (int i = 0; i < validBoxes.size(); i++) {
            if (validAreas.get(i) >= dynamicMinArea) {
                sizedBoxes.add(validBoxes.get(i));
            }
        }

        // 6. Nested Box Removal & Proximity Merge
        List<Rect> boxes = mergeNearbyBoxes(removeNestedBoxes(sizedBoxes), PROXIMITY_THRESHOLD);

        Log.d(TAG, "findFeatureBoxes: " + contours.size() + " contours -> "
                + validBoxes.size() + " passed geometry -> " + sizedBoxes.size()
                + " passed area -> " + boxes.size() + " after merge");

        // Safety valve. Several geometry filters above are commented out, so a noisy or
        // low-contrast photo can yield an effectively unbounded number of boxes; each one
        // costs a CLIP inference and a PNG write. Keep the largest, which are the ones most
        // likely to be real hardware rather than texture noise.
        if (boxes.size() > MAX_DETECTIONS) {
            Log.w(TAG, "Detection count " + boxes.size() + " exceeds MAX_DETECTIONS ("
                    + MAX_DETECTIONS + "); keeping the largest. Consider re-enabling a geometry filter.");
            Collections.sort(boxes, (a, b) -> Long.compare(
                    (long) b.width * b.height, (long) a.width * a.height));
            boxes = new ArrayList<>(boxes.subList(0, MAX_DETECTIONS));
        }
        return boxes;
    }

    /** 4*pi*area / perimeter^2 — 1.0 for a perfect circle, lower for irregular/elongated shapes. */
    private static double circularityOf(MatOfPoint contour, double area) {
        MatOfPoint2f c2f = new MatOfPoint2f(contour.toArray());
        double perimeter = Imgproc.arcLength(c2f, true);
        c2f.release();
        if (perimeter <= 0) return 0;
        return (4 * Math.PI * area) / (perimeter * perimeter);
    }

    /** Crops the colour source Mat around a box (with CROP_PADDING) and returns an ARGB Bitmap. */
    private static Bitmap cropToBitmap(Mat src, Rect r) {
        int x1 = Math.max(0, r.x - CROP_PADDING);
        int y1 = Math.max(0, r.y - CROP_PADDING);
        int x2 = Math.min(src.cols(), r.x + r.width + CROP_PADDING);
        int y2 = Math.min(src.rows(), r.y + r.height + CROP_PADDING);
        if ((x2 - x1) < 10 || (y2 - y1) < 10) return null;

        Mat sub = new Mat(src, new Rect(x1, y1, x2 - x1, y2 - y1));
        try {
            Bitmap bmp = Bitmap.createBitmap(sub.cols(), sub.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(sub, bmp);
            return bmp;
        } catch (Exception e) {
            Log.e(TAG, "cropToBitmap failed", e);
            return null;
        } finally {
            sub.release();
        }
    }

    /**
     * STEP 2 of the Python script: grayscale, smart glare (black-hat), CLAHE, bilateral,
     * median + Gaussian blurs, Canny, dilate + close. Returns the gray image, the dilated
     * edges (for the diagnostic overlay) and the closed edges (for contour detection).
     * Intermediate Mats are released internally; the caller must release the returned Edges.
     */
    private static Edges computeEdges(Mat src) {
        Mat gray = new Mat();
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY);

        // 1. Smart Glare Detection
        double totalPixels = gray.rows() * gray.cols();
        Mat thresh = new Mat();
        Imgproc.threshold(gray, thresh, 240, 255, Imgproc.THRESH_BINARY);
        double glareRatio = (double) Core.countNonZero(thresh) / totalPixels;
        thresh.release();

        Mat preprocessed = new Mat();
        if (glareRatio > 0.05) {
            Log.d(TAG, "Glare detected: " + glareRatio + ". Applying Black-Hat transform.");
            Mat lightingKernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(75, 75));
            Imgproc.morphologyEx(gray, preprocessed, Imgproc.MORPH_BLACKHAT, lightingKernel);
            lightingKernel.release();
        } else {
            gray.copyTo(preprocessed);
        }

        // 2. CLAHE
        Mat imgClahe = new Mat();
        CLAHE clahe = Imgproc.createCLAHE(CLAHE_CLIP_LIMIT, new Size(16, 16));
        clahe.apply(preprocessed, imgClahe);

        // 3. Bilateral Filter
        Mat imgFiltered = new Mat();
        int adaptiveBilateralD = computeAdaptiveBilateralD(preprocessed);
        Log.d(TAG, "Using adaptive bilateral D = " + adaptiveBilateralD);
        Imgproc.bilateralFilter(imgClahe, imgFiltered, adaptiveBilateralD, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE);

        // 4. Noise Reduction Blurs (median(5) then Gaussian(5x5))
        Mat imgSmoothed = new Mat();
        Imgproc.medianBlur(imgFiltered, imgSmoothed, 5);
        Imgproc.GaussianBlur(imgSmoothed, imgSmoothed, new Size(5, 5), 0);

        // 5. Canny Edge Detection — thresholds derived per-image via Otsu instead of fixed
        // values, so exposure/lighting differences between photos don't need re-tuning.
        Mat otsuMask = new Mat();
        double otsuHigh = Imgproc.threshold(imgSmoothed, otsuMask, 0, CANNY_HIGH_CAP,
                Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);
        otsuMask.release();
        double cannyLow = otsuHigh * CANNY_LOW_RATIO;

        Mat edges = new Mat();
        Imgproc.Canny(imgSmoothed, edges, cannyLow, otsuHigh);

        // 6. Morphology: dilate (rect) then close (ellipse)
        Mat dilateKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT,
                new Size(DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE));
        Mat edgesDilated = new Mat();
        Imgproc.dilate(edges, edgesDilated, dilateKernel);

        Mat closeKernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE,
                new Size(CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE));
        Mat edgesClosed = new Mat();
        Imgproc.morphologyEx(edgesDilated, edgesClosed, Imgproc.MORPH_CLOSE, closeKernel);

        // Release everything the caller does not need.
        preprocessed.release();
        imgClahe.release();
        imgFiltered.release();
        imgSmoothed.release();
        edges.release();
        dilateKernel.release();
        closeKernel.release();

        return new Edges(gray, edgesDilated, edgesClosed);
    }

    private static List<Rect> removeNestedBoxes(List<Rect> boxes) {
        List<Rect> filtered = new ArrayList<>();
        for (int i = 0; i < boxes.size(); i++) {
            Rect r1 = boxes.get(i);
            boolean isInside = false;
            for (int j = 0; j < boxes.size(); j++) {
                if (i == j) continue;
                Rect r2 = boxes.get(j);
                if (r1.x >= r2.x && r1.y >= r2.y && (r1.x + r1.width) <= (r2.x + r2.width) && (r1.y + r1.height) <= (r2.y + r2.height)) {
                    isInside = true;
                    break;
                }
            }
            if (!isInside) filtered.add(r1);
        }
        return filtered;
    }

    private static List<Rect> mergeNearbyBoxes(List<Rect> boxes, int threshold) {
        List<Rect> working = new ArrayList<>(boxes);
        boolean changed = true;
        while (changed) {
            changed = false;
            List<Rect> next = new ArrayList<>();
            boolean[] skipped = new boolean[working.size()];
            for (int i = 0; i < working.size(); i++) {
                if (skipped[i]) continue;
                Rect r1 = working.get(i);
                skipped[i] = true;
                for (int j = i + 1; j < working.size(); j++) {
                    if (skipped[j]) continue;
                    Rect r2 = working.get(j);
                    int hGap = Math.max(0, Math.max(r1.x, r2.x) - Math.min(r1.x + r1.width, r2.x + r2.width));
                    int vGap = Math.max(0, Math.max(r1.y, r2.y) - Math.min(r1.y + r1.height, r2.y + r2.height));
                    if (hGap <= threshold && vGap <= threshold) {
                        int nx = Math.min(r1.x, r2.x);
                        int ny = Math.min(r1.y, r2.y);
                        int nw = Math.max(r1.x + r1.width, r2.x + r2.width) - nx;
                        int nh = Math.max(r1.y + r1.height, r2.y + r2.height) - ny;
                        
                        // Magic Rule: Box Fill Ratio Check (FILL_RATIO_THRESHOLD = 0.7)
                        double combinedArea = (double) nw * nh;
                        double sumOfAreas = (double) r1.width * r1.height + (double) r2.width * r2.height;
                        if (sumOfAreas >= combinedArea * FILL_RATIO_THRESHOLD) {
                            r1 = new Rect(nx, ny, nw, nh);
                            skipped[j] = true;
                            changed = true;
                        }
                    }
                }
                next.add(r1);
            }
            working = next;
        }
        return working;
    }

    /** Loads an asset or content URI downsampled so its longest edge is near {@code maxDim}. */
    public static Bitmap loadBitmap(Context context, String uriString, int maxDim) {
        try {
            InputStream is;
            if (uriString.startsWith("file:///android_asset/")) {
                String path = uriString.replace("file:///android_asset/", "");
                is = context.getAssets().open(path);
            } else {
                is = context.getContentResolver().openInputStream(Uri.parse(uriString));
            }
            if (is == null) return null;
            
            BitmapFactory.Options opts = new BitmapFactory.Options();
            opts.inJustDecodeBounds = true;
            BitmapFactory.decodeStream(is, null, opts);
            is.close();

            int scale = 1;
            if (opts.outWidth > maxDim || opts.outHeight > maxDim) {
                scale = Math.max(opts.outWidth / maxDim, opts.outHeight / maxDim);
            }
            opts.inJustDecodeBounds = false;
            opts.inSampleSize = scale;
            
            // Re-open stream for final decode
            if (uriString.startsWith("file:///android_asset/")) {
                String path = uriString.replace("file:///android_asset/", "");
                is = context.getAssets().open(path);
            } else {
                is = context.getContentResolver().openInputStream(Uri.parse(uriString));
            }
            Bitmap result = BitmapFactory.decodeStream(is, null, opts);
            is.close();
            return result;
        } catch (Exception e) {
            Log.e(TAG, "Bitmap load error: " + e.getMessage());
            return null;
        }
    }

    private static String getFilename(String uri) {
        if (uri == null) return "";
        int slash = uri.lastIndexOf('/');
        return (slash >= 0) ? uri.substring(slash + 1).toLowerCase() : uri.toLowerCase();
    }
}
