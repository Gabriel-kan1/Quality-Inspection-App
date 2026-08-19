package com.example.ml;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.net.Uri;
import android.util.Log;

import androidx.exifinterface.media.ExifInterface;

// FeatureEmbedder, FeatureMeasurement, ReferenceLibrary and ScaleCalibration were imported
// explicitly when this class lived in ui.fragment; it is in com.example.ml now, so they
// resolve without imports.
import com.example.ui.util.AssetBitmaps;

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
    // CONFIG block. These were left non-final for a debug settings screen that was never
    // built, so they were constants without the keyword; restored to final. Add the screen
    // first if runtime tuning is wanted.
    // ─────────────────────────────────────────────────────────────

    /**
     * Longest edge, in pixels, that every image is resized to before analysis.
     *
     * Every other constant in this block is denominated in pixels — kernel sizes, the
     * proximity threshold, the crop padding, the absolute area floor — so they only mean
     * anything against a fixed working resolution. This value is enforced exactly by
     * {@link #loadBitmap}, which resizes after decoding rather than relying on
     * {@code inSampleSize} alone: BitmapFactory rounds that down to a power of two, so
     * asking for 1280 used to yield 2016px from a 4032px source and 1500px from a 3000px
     * one. Two photos of the same part were being measured with different rulers, and the
     * verdict is a detection-count comparison between exactly two such photos.
     */
    public static final int WORKING_MAX_DIM = 1280;

    // --- Edge detection (computeEdges) ---
    /** Fraction of near-white pixels above which the black-hat glare correction runs. */
    private static final double GLARE_RATIO_THRESHOLD = 0.05;
    /** Structuring element for the black-hat lighting correction, in WORKING_MAX_DIM pixels. */
    private static final int GLARE_KERNEL_SIZE = 75;
    /** CLAHE local-contrast boost. Higher = crisper edges but more texture noise. */
    private static final double CLAHE_CLIP_LIMIT = 1.0;
    /** Bilateral filter diameter/sigmas — denoise while preserving edges. */
    private static final double BILATERAL_SIGMA_COLOR = 60;
    private static final double BILATERAL_SIGMA_SPACE = 20;
    /**
     * Canny thresholds are computed adaptively per-image via Otsu's method rather
     * than fixed, so lighting/exposure differences between photos don't require
     * re-tuning. These multipliers scale the Otsu threshold (low = ratio * high).
     */
    private static final double CANNY_LOW_RATIO = 0.5;
    private static final double CANNY_HIGH_CAP = 255.0;
    /** Morphology: dilate (rect) to bridge gaps, then close (ellipse) to seal contours. */
    private static final int DILATE_KERNEL_SIZE = 5;
    private static final int CLOSE_KERNEL_SIZE = 7;

    // --- Geometry filtering (findFeatureBoxes) ---
    /**
     * Aspect band a contour must fall inside to be considered hardware. Measured on real
     * chassis shots: bolt heads land at 0.85-1.30, while structural seams and panel edges
     * run 2.4-4.3. The band is set to separate those two populations. Widen MAX if genuinely
     * elongated features (side-on screw shafts) start being missed.
     */
    private static final float MAX_ASPECT_RATIO = 2.5f;
    private static final float MIN_ASPECT_RATIO = 0.4f;
    private static final double DYNAMIC_AREA_MULTIPLIER = 0.15;
    /**
     * Contours more than this many times the median area are treated as structural
     * (chassis outline, panel edges, shadows) rather than hardware, and are excluded when
     * sizing the dynamic area threshold — see the note in findFeatureBoxes.
     */
    private static final double OUTLIER_AREA_FACTOR = 10.0;
    private static final int PROXIMITY_THRESHOLD = 10;
    private static final double FILL_RATIO_THRESHOLD = 0.7;

    /**
     * Context kept around a detection when cropping it, as a fraction of the box's longer
     * side. Replaces the Python script's fixed 40 px.
     *
     * Fixed padding made the framing depend on how big the feature happened to be in pixels,
     * which is a property of how far away the operator stood rather than of the hardware. A
     * 30x22 box became a 110x102 crop — the fastener 6% of it and background the rest, which
     * is what the encoder then described — while a 201x125 box came out at 44%. The same bolt
     * photographed from two distances produced two differently-framed crops and therefore two
     * different embeddings, so a prototype averaged over them was averaging over framing as
     * much as over hardware.
     *
     * Proportional padding makes the fastener occupy roughly the same share of every crop
     * whatever its pixel size, which is the invariance a scale-sensitive encoder needs. Kept
     * non-zero because a fastener is partly defined by how it sits in its seating, and a crop
     * shaved exactly to the contour throws that away along with any detection jitter.
     */
    private static final double CROP_PADDING_RATIO = 0.15;
    /**
     * Floor on that padding, in pixels. Below a few pixels the ratio rounds to nothing and the
     * crop lands exactly on a contour whose edge is already a pixel or two uncertain.
     */
    private static final int MIN_CROP_PADDING = 2;
    /** Upper bound on boxes carried forward from one image; see the safety valve in findFeatureBoxes. */
    private static final int MAX_DETECTIONS = 300;
    /**
     * Logs every box before and after merging. Verbose — for tuning detection, not normal use.
     *
     * Off by default: at the counts the bundled assets produce this is up to ~110 log lines
     * per image and two per box, charged on every inspection including the reference pass.
     */
    private static final boolean DEBUG_GEOMETRY = false;

    /**
     * Whether the native OpenCV libraries actually loaded.
     *
     * Checked before every pipeline entry point. Without this the class still loads when
     * init fails and each OpenCV call throws {@link UnsatisfiedLinkError} — an {@code Error},
     * not an {@code Exception}, so the catch blocks below miss it, the worker thread dies
     * silently and the results screen waits forever on a callback that never comes.
     */
    private static final boolean OPENCV_READY;

    static {
        boolean ready;
        try {
            // initLocal() is the entry point the official OpenCV Android package documents for
            // a library bundled into the APK; initDebug() is the older name kept for
            // compatibility and says nothing about debug builds despite how it reads.
            ready = OpenCVLoader.initLocal();
        } catch (Throwable t) {
            // A missing or ABI-mismatched .so surfaces here as UnsatisfiedLinkError.
            Log.e(TAG, "OpenCV native libraries could not be loaded.", t);
            ready = false;
        }
        OPENCV_READY = ready;
        if (ready) {
            Log.d(TAG, "OpenCV initialization succeeded.");
        } else {
            Log.e(TAG, "OpenCV initialization failed — detection is disabled for this session.");
        }
    }

    /** True when the detection pipeline can run at all; false means the native libs are missing. */
    public static boolean isAvailable() {
        return OPENCV_READY;
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
        /** Physical size, or null when no calibration marker was found in the photo. */
        public FeatureMeasurement measurement;

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
        /**
         * Scale derived from the calibration marker, or null when none was found. Null means
         * every {@link ExtractedFeature#measurement} is also null — measurement is skipped
         * entirely rather than falling back to an assumed scale.
         */
        public ScaleCalibration calibration;
        /**
         * Whether the glare (black-hat) branch ran for this image.
         *
         * Exposed so a second pass over a different image can be forced down the same branch.
         * The two branches produce structurally different edge maps, and this pipeline's
         * verdict is a count comparison between two independently processed images — a
         * reference at 4.9% glare and an inspection at 5.1% would otherwise disagree on
         * nothing but lighting. See {@link #extractFeatures(Context, String, ReferenceLibrary,
         * CropSink, Boolean)}.
         */
        public boolean glareCorrectionApplied;

        public AnalysisResult(List<ExtractedFeature> features, Bitmap processedBitmap) {
            this(features, processedBitmap, null, false);
        }

        public AnalysisResult(List<ExtractedFeature> features, Bitmap processedBitmap,
                              ScaleCalibration calibration) {
            this(features, processedBitmap, calibration, false);
        }

        public AnalysisResult(List<ExtractedFeature> features, Bitmap processedBitmap,
                              ScaleCalibration calibration, boolean glareCorrectionApplied) {
            this.features = features;
            this.processedBitmap = processedBitmap;
            this.calibration = calibration;
            this.glareCorrectionApplied = glareCorrectionApplied;
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

    /** Bundles the intermediate edge Mats produced by {@link #computeEdges}. */
    private static final class Edges {
        final Mat gray;
        /**
         * Raw Canny output, before any morphology.
         *
         * Localisation deliberately uses the dilated/closed map, because bridging gaps is what
         * makes a fragmented fastener outline into one detectable contour. Measurement must
         * not: dilation grows every boundary outward by roughly the kernel radius, which adds
         * about twice that to any diameter and reads as a systematic over-size. Sizing is
         * therefore taken from this untouched edge map.
         */
        final Mat edgesRaw;
        final Mat edgesDilated;
        final Mat edgesClosed;
        /** Whether the black-hat glare branch ran; see {@link AnalysisResult#glareCorrectionApplied}. */
        final boolean glareCorrectionApplied;

        Edges(Mat gray, Mat edgesRaw, Mat edgesDilated, Mat edgesClosed,
              boolean glareCorrectionApplied) {
            this.gray = gray;
            this.edgesRaw = edgesRaw;
            this.edgesDilated = edgesDilated;
            this.edgesClosed = edgesClosed;
            this.glareCorrectionApplied = glareCorrectionApplied;
        }

        void release() {
            gray.release();
            edgesRaw.release();
            edgesDilated.release();
            edgesClosed.release();
        }
    }

    /** Frees a Mat if it was allocated, tolerating nulls so it can be used from a finally block. */
    private static void releaseQuietly(Mat... mats) {
        for (Mat m : mats) {
            if (m != null) m.release();
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
        if (!OPENCV_READY) {
            Log.e(TAG, "extractCrops skipped — OpenCV is unavailable.");
            return crops;
        }

        Bitmap bitmap = loadBitmap(context, uriString, WORKING_MAX_DIM);
        if (bitmap == null) return crops;

        Mat src = null;
        Edges e = null;
        try {
            src = new Mat();
            Utils.bitmapToMat(bitmap, src);
            e = computeEdges(src, null);

            float w = src.width();
            float h = src.height();
            for (Rect r : findFeatureBoxes(e.edgesClosed, w, h)) {
                Bitmap crop = cropToBitmap(src, r);
                if (crop == null) continue;
                RectF frac = new RectF(
                        r.x / w, r.y / h, (r.x + r.width) / w, (r.y + r.height) / h);
                crops.add(new CropRegion(crop, frac));
            }
        } catch (Throwable ex) {
            // Throwable, not Exception: an OpenCV ABI problem arrives as UnsatisfiedLinkError.
            Log.e(TAG, "extractCrops failure", ex);
        } finally {
            // Native Mats are invisible to the GC and leaking them grows anon RSS with no
            // OutOfMemoryError to show for it, so release on the failure path too.
            releaseQuietly(src);
            if (e != null) e.release();
            bitmap.recycle();
        }
        return crops;
    }

    /**
     * Runs the OpenCV extraction pipeline and classifies each detected crop with the feature encoder
     * against the supplied {@link ReferenceLibrary}. When {@code library} is null or empty
     * (no reference set, or the encoder is unavailable) every feature is labelled "unknown".
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
        return extractFeatures(context, uriString, library, sink, null);
    }

    /**
     * As above, but lets the caller pin the glare branch instead of letting it be decided
     * per-image.
     *
     * @param forceGlareCorrection {@code null} to decide from this image's own glare ratio;
     *        {@code TRUE}/{@code FALSE} to force the black-hat branch on or off. Pass the
     *        first pass's {@link AnalysisResult#glareCorrectionApplied} when running a second
     *        image that has to be counted against the first — the branches produce different
     *        edge maps, and a count comparison across a branch boundary is a lighting
     *        artefact rather than a defect.
     */
    public static AnalysisResult extractFeatures(Context context, String uriString,
                                                 ReferenceLibrary library, CropSink sink,
                                                 Boolean forceGlareCorrection) {
        List<ExtractedFeature> features = new ArrayList<>();
        Bitmap diagnosticBitmap = null;

        if (uriString == null || uriString.isEmpty()) {
            return new AnalysisResult(features, null);
        }
        if (!OPENCV_READY) {
            Log.e(TAG, "extractFeatures skipped — OpenCV is unavailable.");
            return new AnalysisResult(features, null);
        }

        boolean canClassify = library != null && !library.isEmpty();
        FeatureEmbedder embedder = canClassify ? FeatureEmbedder.get(context) : null;

        String filename = getFilename(uriString);
        Log.d(TAG, "Running OpenCV Pipeline for: " + filename);

        // Load image at the fixed working resolution every pixel-denominated constant assumes.
        Bitmap bitmap = loadBitmap(context, uriString, WORKING_MAX_DIM);
        if (bitmap == null) {
            Log.e(TAG, "Failed to load bitmap for URI: " + uriString);
            return new AnalysisResult(features, null);
        }

        Mat src = null;
        Edges e = null;
        Mat rgbaVisual = null;
        boolean glareApplied = false;
        ScaleCalibration calibration = null;
        // Opened once per image, not per crop: constructing a region decoder re-reads the
        // file header, and there can be dozens of detections. Null is a normal result — the
        // source may be smaller than the working image, or not region-decodable at all — and
        // means crops come from the working copy as before.
        SourceCropper sourceCropper = SourceCropper.open(
                context, uriString, bitmap.getWidth(), bitmap.getHeight());
        try {
            src = new Mat();
            Utils.bitmapToMat(bitmap, src);

            // STEP 1 — pixel-to-millimetre scale from the calibration marker, if one is in
            // frame. Runs on the unprocessed source: the marker is high-contrast by design and
            // the glare/CLAHE stage below is tuned for fastener detail, not for finding it.
            calibration = ScaleCalibration.detect(src);

            // STEP 2 — preprocessing + edge detection (shared with extractCrops()).
            e = computeEdges(src, forceGlareCorrection);
            glareApplied = e.glareCorrectionApplied;

            // --- GENERATE DIAGNOSTIC BITMAP (Green Edges on Gray Background) ---
            rgbaVisual = new Mat();
            Imgproc.cvtColor(e.gray, rgbaVisual, Imgproc.COLOR_GRAY2RGBA);
            // Highlight detected edges in bright green so they are visible on mobile displays.
            rgbaVisual.setTo(new Scalar(0, 255, 0, 255), e.edgesDilated);
            diagnosticBitmap = Bitmap.createBitmap(rgbaVisual.cols(), rgbaVisual.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(rgbaVisual, diagnosticBitmap);

            // STEP 3-6 — OpenCV crop extraction + embedding classification + measurement
            performDynamicDetection(src, e.edgesClosed, e.edgesRaw, embedder, library, sink,
                    features, calibration, sourceCropper);

        } catch (Throwable t) {
            // Throwable, not Exception: an OpenCV ABI problem arrives as UnsatisfiedLinkError.
            Log.e(TAG, "OpenCV Pipeline failure", t);
        } finally {
            // Native Mats are invisible to the GC and leaking them grows anon RSS with no
            // OutOfMemoryError to show for it, so release on the failure path too. The region
            // decoder holds a native handle and an open source for the same reason.
            releaseQuietly(src, rgbaVisual);
            if (e != null) e.release();
            if (sourceCropper != null) sourceCropper.close();
            bitmap.recycle();
        }

        return new AnalysisResult(features, diagnosticBitmap, calibration, glareApplied);
    }

    private static void performDynamicDetection(Mat src, Mat edges, Mat edgesRaw,
                                                FeatureEmbedder embedder,
                                                ReferenceLibrary library, CropSink sink,
                                                List<ExtractedFeature> outFeatures,
                                                ScaleCalibration calibration,
                                                SourceCropper sourceCropper) {
        float imgWidth = (float) src.width();
        float imgHeight = (float) src.height();

        List<Rect> merged = findFeatureBoxes(edges, imgWidth, imgHeight);
        boolean canClassify = embedder != null && library != null && !library.isEmpty();

        int boxId = 0;
        int markerBoxes = 0;
        for (Rect r : merged) {
            // The calibration marker is a deliberate part of the scene, not hardware under
            // inspection. Dropping it here keeps it out of the counts, the classification
            // event log and the operator's review queue.
            if (calibration != null && calibration.overlapsMarker(r)) {
                markerBoxes++;
                continue;
            }
            boxId++;

            // One crop is alive at a time: created, embedded, persisted, then freed before the
            // next iteration. Keeping them all would make peak memory scale with contour count.
            Bitmap crop = cropFeature(sourceCropper, src, r);

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

            // Measured after classification, because the type decides whether the feature is
            // measured as a circle or a rectangle — the classifier's only role in sizing.
            FeatureMeasurement measurement = FeatureMeasurement.measure(edgesRaw, r, calibration);
            if (measurement != null) {
                label = label + "  " + measurement.shortLabel(type);
            }

            ExtractedFeature feature = new ExtractedFeature(
                type,
                new RectF((float)r.x/imgWidth, (float)r.y/imgHeight, (float)(r.x+r.width)/imgWidth, (float)(r.y+r.height)/imgHeight),
                ReferenceLibrary.colorFor(type),
                label,
                confidence,
                cropPath,
                classification
            );
            feature.measurement = measurement;
            outFeatures.add(feature);
        }
        if (markerBoxes > 0) {
            Log.d(TAG, "Excluded " + markerBoxes + " detection(s) overlapping the calibration marker.");
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
        int rejectedAspect = 0;

        // Three further geometry filters (max-dimension, edge-margin and circularity) used to
        // sit in this loop, commented out but with their constants and documentation still
        // reading as live configuration. python/detector_ab_study.py confirms they are inert
        // over the bundled assets; they have been removed rather than left as decoration. Git
        // has them if the tuning is ever revisited.
        for (MatOfPoint c : contours) {
            Rect r = Imgproc.boundingRect(c);

            // Aspect Ratio Check
            //
            // Re-enabled: structural seams and panel edges are long, flat contours. Left in,
            // they don't just add noise — a line's bounding box can enclose a nearby bolt head,
            // which then gets deleted by removeNestedBoxes or absorbed by the proximity merge,
            // so real fasteners disappear. The band is wide enough to keep screws, which in this
            // pipeline are shafts viewed side-on and so legitimately elongated.
            float aspect = (float) r.width / r.height;
            if (aspect > MAX_ASPECT_RATIO || aspect < MIN_ASPECT_RATIO) {
                rejectedAspect++;
                continue;
            }

            validBoxes.add(r);
            validAreas.add(Imgproc.contourArea(c));
        }

        if (validBoxes.isEmpty()) return new ArrayList<>();

        // Dynamic Area Filtering
        //
        // The threshold is sized from the largest contours, on the assumption that those are
        // real hardware. That holds for a tight shot, but not for a wider one: there the
        // biggest contours are the chassis outline, panel edges and shadows, which are orders
        // of magnitude larger than any fastener. They drag the reference up until the minimum
        // area exceeds every genuine feature and the frame comes back empty — the further
        // back the camera, the higher the bar, exactly when features are smallest.
        //
        // So structural outliers are discarded before sizing the threshold, using the median
        // as an anchor. A tight shot has no such outliers, so its behaviour is unchanged.
        List<Double> ascending = new ArrayList<>(validAreas);
        Collections.sort(ascending);
        double medianArea = ascending.get(ascending.size() / 2);
        double outlierCeiling = medianArea * OUTLIER_AREA_FACTOR;

        List<Double> referencePool = new ArrayList<>();
        for (double a : validAreas) {
            if (a <= outlierCeiling) referencePool.add(a);
        }
        // Too few left to judge from — fall back rather than size off one or two contours.
        if (referencePool.size() < 3) referencePool = new ArrayList<>(validAreas);

        Collections.sort(referencePool, Collections.reverseOrder());
        int topCount = Math.min(3, referencePool.size());
        double sumTop = 0;
        for (int i = 0; i < topCount; i++) sumTop += referencePool.get(i);
        double referenceAreaValue = sumTop / topCount;
        double dynamicMinArea = Math.max(25.0, referenceAreaValue * DYNAMIC_AREA_MULTIPLIER);

        Log.d(TAG, String.format(Locale.US,
                "area filter: median=%.0f ceiling=%.0f reference=%.0f minArea=%.0f "
                        + "(%d of %d contours excluded as structural)",
                medianArea, outlierCeiling, referenceAreaValue, dynamicMinArea,
                validAreas.size() - referencePool.size(), validAreas.size()));

        List<Rect> sizedBoxes = new ArrayList<>();
        for (int i = 0; i < validBoxes.size(); i++) {
            if (validAreas.get(i) >= dynamicMinArea) {
                sizedBoxes.add(validBoxes.get(i));
            }
        }

        // Safety valve, applied here rather than after the merge.
        //
        // removeNestedBoxes is O(n^2) and mergeNearbyBoxes is O(n^2) per pass repeated
        // until it stabilises, so both are superlinear in a count that nothing upstream
        // bounds — three of the geometry filters above are commented out. Capping the
        // output of those stages, which is what this used to do, leaves their input
        // unbounded and so caps nothing that costs anything.
        //
        // Measured with python/detector_ab_study.py: identical output on all 29 bundled
        // chassis assets (none of which comes close to the cap), and on a dense-noise
        // frame that reaches 1233 boxes it produces the same result in 105ms instead of
        // 384ms. The cap is still applied after the merge as well, since merging can only
        // reduce the count but the post-merge list is what callers pay per-box for.
        if (sizedBoxes.size() > MAX_DETECTIONS) {
            Log.w(TAG, "Box count " + sizedBoxes.size() + " exceeds MAX_DETECTIONS ("
                    + MAX_DETECTIONS + ") before merging; keeping the largest. "
                    + "Consider re-enabling a geometry filter.");
            Collections.sort(sizedBoxes, (a, b) -> Long.compare(
                    (long) b.width * b.height, (long) a.width * a.height));
            sizedBoxes = new ArrayList<>(sizedBoxes.subList(0, MAX_DETECTIONS));
        }

        // 6. Nested Box Removal & Proximity Merge
        List<Rect> deNested = removeNestedBoxes(sizedBoxes);
        List<Rect> boxes = mergeNearbyBoxes(deNested, PROXIMITY_THRESHOLD);

        // Geometry dump: which boxes existed before merging and what they became. Counts alone
        // can't distinguish "lines swallowed bolt heads" from "adjacent bolt heads merged
        // together", and those need opposite fixes.
        if (DEBUG_GEOMETRY) {
            Log.d(TAG, "  pre-merge boxes (" + deNested.size() + "):");
            for (Rect r : deNested) {
                Log.d(TAG, String.format(Locale.US, "    %dx%d at (%d,%d) aspect=%.2f area=%d",
                        r.width, r.height, r.x, r.y, (float) r.width / r.height, r.width * r.height));
            }
            Log.d(TAG, "  post-merge boxes (" + boxes.size() + "):");
            for (Rect r : boxes) {
                Log.d(TAG, String.format(Locale.US, "    %dx%d at (%d,%d) aspect=%.2f area=%d",
                        r.width, r.height, r.x, r.y, (float) r.width / r.height, r.width * r.height));
            }
        }

        Log.d(TAG, "findFeatureBoxes: " + contours.size() + " contours -> "
                + validBoxes.size() + " passed geometry (" + rejectedAspect + " rejected as lines) -> "
                + sizedBoxes.size() + " passed area -> " + boxes.size() + " after merge");

        // Second half of the safety valve: every surviving box costs an encoder inference
        // and a PNG write, so bound what the caller pays for as well as what the merge
        // stages above cost. Keep the largest, which are the ones most likely to be real
        // hardware rather than texture noise.
        if (boxes.size() > MAX_DETECTIONS) {
            Log.w(TAG, "Detection count " + boxes.size() + " exceeds MAX_DETECTIONS ("
                    + MAX_DETECTIONS + "); keeping the largest. Consider re-enabling a geometry filter.");
            Collections.sort(boxes, (a, b) -> Long.compare(
                    (long) b.width * b.height, (long) a.width * a.height));
            boxes = new ArrayList<>(boxes.subList(0, MAX_DETECTIONS));
        }
        return boxes;
    }

    /** Crops the colour source Mat around a box (with CROP_PADDING) and returns an ARGB Bitmap. */
    /**
     * Padding for one detection, scaled to its own size so every crop frames its feature the
     * same way. Square-ish on both axes deliberately: padding each side by its own dimension
     * would stretch an elongated feature's context in the direction it is already long.
     */
    static int cropPaddingFor(Rect r) {
        int longest = Math.max(r.width, r.height);
        return Math.max(MIN_CROP_PADDING, (int) Math.round(longest * CROP_PADDING_RATIO));
    }

    /**
     * One feature crop, taken from the source photo when that carries more detail than the
     * working copy, and from the working copy otherwise.
     *
     * The fallback is not a rare edge case: several bundled assets are already smaller than
     * {@link #WORKING_MAX_DIM}, and a region decode can fail on a stream the platform will
     * not seek. Either way the pipeline continues with the crop it can get.
     */
    private static Bitmap cropFeature(SourceCropper cropper, Mat src, Rect r) {
        int padding = cropPaddingFor(r);
        if (cropper != null) {
            Bitmap fromSource = cropper.crop(
                    new android.graphics.Rect(r.x, r.y, r.x + r.width, r.y + r.height), padding);
            if (fromSource != null) return fromSource;
        }
        return cropToBitmap(src, r);
    }

    private static Bitmap cropToBitmap(Mat src, Rect r) {
        int padding = cropPaddingFor(r);
        int x1 = Math.max(0, r.x - padding);
        int y1 = Math.max(0, r.y - padding);
        int x2 = Math.min(src.cols(), r.x + r.width + padding);
        int y2 = Math.min(src.rows(), r.y + r.height + padding);
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
    private static Edges computeEdges(Mat src, Boolean forceGlareCorrection) {
        Mat gray = new Mat();
        Mat preprocessed = null;
        Mat imgClahe = null;
        Mat imgFiltered = null;
        Mat imgSmoothed = null;
        Mat edges = null;
        Mat dilateKernel = null;
        Mat closeKernel = null;
        Mat edgesDilated = null;
        Mat edgesClosed = null;
        boolean ok = false;

        try {
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY);

            // 1. Smart Glare Detection
            //
            // The caller may pin this rather than let it be decided per-image: the two
            // branches produce structurally different edge maps, so two images compared by
            // detection count have to agree on which one ran. See the forceGlareCorrection
            // parameter on extractFeatures().
            boolean applyGlareCorrection;
            if (forceGlareCorrection != null) {
                applyGlareCorrection = forceGlareCorrection;
                Log.d(TAG, "Glare branch pinned by caller: " + applyGlareCorrection);
            } else {
                double totalPixels = gray.rows() * gray.cols();
                Mat thresh = new Mat();
                Imgproc.threshold(gray, thresh, 240, 255, Imgproc.THRESH_BINARY);
                double glareRatio = (double) Core.countNonZero(thresh) / totalPixels;
                thresh.release();
                applyGlareCorrection = glareRatio > GLARE_RATIO_THRESHOLD;
                Log.d(TAG, String.format(Locale.US, "Glare ratio %.4f (threshold %.2f) — %s",
                        glareRatio, GLARE_RATIO_THRESHOLD,
                        applyGlareCorrection ? "applying Black-Hat" : "no correction"));
            }

            preprocessed = new Mat();
            if (applyGlareCorrection) {
                Mat lightingKernel = Imgproc.getStructuringElement(
                        Imgproc.MORPH_ELLIPSE, new Size(GLARE_KERNEL_SIZE, GLARE_KERNEL_SIZE));
                Imgproc.morphologyEx(gray, preprocessed, Imgproc.MORPH_BLACKHAT, lightingKernel);
                lightingKernel.release();
            } else {
                gray.copyTo(preprocessed);
            }

            // 2. CLAHE
            imgClahe = new Mat();
            CLAHE clahe = Imgproc.createCLAHE(CLAHE_CLIP_LIMIT, new Size(16, 16));
            clahe.apply(preprocessed, imgClahe);

            // 3. Bilateral Filter
            imgFiltered = new Mat();
            int adaptiveBilateralD = computeAdaptiveBilateralD(preprocessed);
            Log.d(TAG, "Using adaptive bilateral D = " + adaptiveBilateralD);
            Imgproc.bilateralFilter(imgClahe, imgFiltered, adaptiveBilateralD, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE);

            // 4. Noise Reduction Blurs (median(5) then Gaussian(5x5))
            imgSmoothed = new Mat();
            Imgproc.medianBlur(imgFiltered, imgSmoothed, 5);
            Imgproc.GaussianBlur(imgSmoothed, imgSmoothed, new Size(5, 5), 0);

            // 5. Canny Edge Detection — thresholds derived per-image via Otsu instead of fixed
            // values, so exposure/lighting differences between photos don't need re-tuning.
            Mat otsuMask = new Mat();
            double otsuHigh = Imgproc.threshold(imgSmoothed, otsuMask, 0, CANNY_HIGH_CAP,
                    Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);
            otsuMask.release();
            double cannyLow = otsuHigh * CANNY_LOW_RATIO;

            edges = new Mat();
            Imgproc.Canny(imgSmoothed, edges, cannyLow, otsuHigh);

            // 6. Morphology: dilate (rect) then close (ellipse)
            dilateKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT,
                    new Size(DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE));
            edgesDilated = new Mat();
            Imgproc.dilate(edges, edgesDilated, dilateKernel);

            closeKernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE,
                    new Size(CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE));
            edgesClosed = new Mat();
            Imgproc.morphologyEx(edgesDilated, edgesClosed, Imgproc.MORPH_CLOSE, closeKernel);

            ok = true;
            // The raw Canny map is handed out rather than freed: measurement sizes features
            // from it, because the dilation below grows every boundary outward and would read
            // as a systematic over-size. See Edges.edgesRaw.
            return new Edges(gray, edges, edgesDilated, edgesClosed, applyGlareCorrection);
        } finally {
            // Intermediates go either way; the four Mats the caller owns are only freed here
            // when we are unwinding on an exception and nobody will ever receive them.
            releaseQuietly(preprocessed, imgClahe, imgFiltered, imgSmoothed,
                    dilateKernel, closeKernel);
            if (!ok) {
                releaseQuietly(gray, edges, edgesDilated, edgesClosed);
            }
        }
    }

    private static List<Rect> removeNestedBoxes(List<Rect> boxes) {
        List<Rect> filtered = new ArrayList<>();
        for (int i = 0; i < boxes.size(); i++) {
            Rect r1 = boxes.get(i);
            boolean isInside = false;
            for (int j = 0; j < boxes.size(); j++) {
                if (i == j) continue;
                Rect r2 = boxes.get(j);
                // Intended to stop a long thin box deleting a compact one inside it: a bolt
                // head sitting on a panel seam is contained by the seam's box but is a
                // separate feature.
                //
                // Currently unreachable. Every box that gets here already passed the same
                // aspect band at the top of findFeatureBoxes, so no container can fail this
                // test — confirmed by python/detector_ab_study.py, which produces byte-identical
                // output on all 29 bundled assets with the guard removed. Kept rather than
                // deleted because it becomes live again the moment the aspect filter is
                // loosened or moved after de-nesting; do not read it as active protection.
                float containerAspect = (float) r2.width / r2.height;
                if (containerAspect > MAX_ASPECT_RATIO || containerAspect < MIN_ASPECT_RATIO) continue;

                if (r1.x >= r2.x && r1.y >= r2.y && (r1.x + r1.width) <= (r2.x + r2.width) && (r1.y + r1.height) <= (r2.y + r2.height)) {
                    isInside = true;
                    break;
                }
            }
            if (!isInside) filtered.add(r1);
        }
        return filtered;
    }

    /**
     * Reunites fragments of a single feature that the edge stage split apart.
     *
     * The fill-ratio guard compares real content against the bounding box, where "content" is
     * the summed area of the original boxes in the group — deliberately not the area of the
     * accumulated bounding box. Using the latter makes the guard self-defeating: once a box has
     * grown, its own area is large enough that any nearby box trivially satisfies the ratio, so
     * merging snowballs. That is how a panel seam and a bolt head could chain outward until a
     * single blob had absorbed fifteen separate detections.
     */
    private static List<Rect> mergeNearbyBoxes(List<Rect> boxes, int threshold) {
        List<Rect> working = new ArrayList<>(boxes);
        // Real content per working box, carried through merges alongside the bounding box.
        List<Double> content = new ArrayList<>();
        for (Rect r : boxes) content.add((double) r.width * r.height);

        boolean changed = true;
        while (changed) {
            changed = false;
            List<Rect> next = new ArrayList<>();
            List<Double> nextContent = new ArrayList<>();
            boolean[] skipped = new boolean[working.size()];
            for (int i = 0; i < working.size(); i++) {
                if (skipped[i]) continue;
                Rect r1 = working.get(i);
                double c1 = content.get(i);
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

                        double combinedArea = (double) nw * nh;
                        double mergedContent = c1 + content.get(j);
                        if (mergedContent >= combinedArea * FILL_RATIO_THRESHOLD) {
                            r1 = new Rect(nx, ny, nw, nh);
                            c1 = mergedContent;
                            skipped[j] = true;
                            changed = true;
                        }
                    }
                }
                next.add(r1);
                nextContent.add(c1);
            }
            working = next;
            content = nextContent;
        }
        return working;
    }

    /**
     * Loads an asset or content URI as an upright bitmap whose longest edge is exactly
     * {@code maxDim} (or smaller, if the source already was).
     *
     * Two things here are load-bearing rather than incidental:
     *
     * <p><b>EXIF orientation is applied.</b> Camera JPEGs and gallery picks record rotation
     * as metadata and leave the pixels in sensor orientation, and {@link BitmapFactory} does
     * not read it. The encoder is not rotation invariant — {@code AlignmentGate.evaluate}
     * requires upright input, and live camera frames are explicitly rotated to match — so a
     * reference photo decoded straight through was being compared 90 degrees out against
     * every frame it was supposed to match.
     *
     * <p><b>The result is scaled to exactly {@code maxDim}.</b> {@code inSampleSize} alone
     * cannot do this: BitmapFactory silently rounds it down to a power of two. Sampling gets
     * within a factor of two cheaply, then one scale step lands it exactly, so the pixel
     * constants downstream mean the same thing for every source resolution.
     */
    public static Bitmap loadBitmap(Context context, String uriString, int maxDim) {
        Bitmap decoded = null;
        Bitmap upright = null;
        try {
            BitmapFactory.Options bounds = new BitmapFactory.Options();
            bounds.inJustDecodeBounds = true;
            try (InputStream is = openStream(context, uriString)) {
                if (is == null) return null;
                BitmapFactory.decodeStream(is, null, bounds);
            }
            if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
                Log.e(TAG, "Could not read image bounds for " + uriString);
                return null;
            }

            int rotation = exifRotationDegrees(context, uriString);

            BitmapFactory.Options opts = new BitmapFactory.Options();
            opts.inSampleSize = AssetBitmaps.sampleSizeFor(bounds.outWidth, bounds.outHeight, maxDim);
            try (InputStream is = openStream(context, uriString)) {
                if (is == null) return null;
                decoded = BitmapFactory.decodeStream(is, null, opts);
            }
            if (decoded == null) return null;

            upright = rotate(decoded, rotation);
            return scaleToLongestEdge(upright, maxDim);
        } catch (Throwable t) {
            Log.e(TAG, "Bitmap load error for " + uriString, t);
            return null;
        } finally {
            // rotate()/scaleToLongestEdge() return the input unchanged when there is nothing
            // to do, so only free intermediates that were genuinely superseded.
            if (upright != null && upright != decoded) {
                if (decoded != null) decoded.recycle();
            }
        }
    }

    /** Package-visible alias so {@link SourceCropper} opens sources exactly the same way. */
    static InputStream open(Context context, String uriString) throws Exception {
        return openStream(context, uriString);
    }

    /** Package-visible alias so {@link SourceCropper} reads the same EXIF rotation. */
    static int exifRotation(Context context, String uriString) {
        return exifRotationDegrees(context, uriString);
    }

    private static InputStream openStream(Context context, String uriString) throws Exception {
        if (uriString.startsWith("file:///android_asset/")) {
            return context.getAssets().open(uriString.substring("file:///android_asset/".length()));
        }
        return context.getContentResolver().openInputStream(Uri.parse(uriString));
    }

    /** EXIF rotation in degrees (0/90/180/270), or 0 when there is none or it can't be read. */
    private static int exifRotationDegrees(Context context, String uriString) {
        try (InputStream is = openStream(context, uriString)) {
            if (is == null) return 0;
            return new ExifInterface(is).getRotationDegrees();
        } catch (Throwable t) {
            // A file with no EXIF segment at all is normal, not an error worth shouting about.
            Log.d(TAG, "No usable EXIF orientation for " + uriString);
            return 0;
        }
    }

    /** Rotates a bitmap, returning the original when there is nothing to do. */
    private static Bitmap rotate(Bitmap src, int degrees) {
        if (src == null || degrees == 0) return src;
        Matrix matrix = new Matrix();
        matrix.postRotate(degrees);
        try {
            return Bitmap.createBitmap(src, 0, 0, src.getWidth(), src.getHeight(), matrix, true);
        } catch (Throwable t) {
            Log.e(TAG, "Failed to apply EXIF rotation of " + degrees + " degrees", t);
            return src;
        }
    }

    /** Scales so the longest edge is exactly {@code maxDim}; returns the original if already smaller. */
    private static Bitmap scaleToLongestEdge(Bitmap src, int maxDim) {
        if (src == null) return null;
        int longest = Math.max(src.getWidth(), src.getHeight());
        if (longest <= maxDim) return src;
        float factor = (float) maxDim / longest;
        int w = Math.max(1, Math.round(src.getWidth() * factor));
        int h = Math.max(1, Math.round(src.getHeight() * factor));
        Bitmap scaled = Bitmap.createScaledBitmap(src, w, h, true);
        if (scaled != src) src.recycle();
        return scaled;
    }

    private static String getFilename(String uri) {
        if (uri == null) return "";
        int slash = uri.lastIndexOf('/');
        return (slash >= 0) ? uri.substring(slash + 1).toLowerCase() : uri.toLowerCase();
    }
}
