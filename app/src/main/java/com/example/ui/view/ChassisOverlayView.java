package com.example.ui.view;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.net.Uri;
import android.util.AttributeSet;
import android.util.Log;
import android.view.View;

import androidx.annotation.Nullable;

import com.example.ml.HardwareFeatureExtractor;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class ChassisOverlayView extends View {

    private static final String TAG = "ChassisOverlayView";

    public static class DetectedFeature {
        public String type;
        public RectF rectFraction; // coordinates from 0.0 to 1.0 relative to view size
        public int color;
        public String label;

        public DetectedFeature(String type, RectF rectFraction, int color, String label) {
            this.type = type;
            this.rectFraction = rectFraction;
            this.color = color;
            this.label = label;
        }
    }

    /**
     * A position the reference layout expects a critical feature at, projected into this
     * photo's coordinates. Drawn only for slots that are empty or hold the wrong hardware —
     * the ones the operator has to go and look at.
     */
    public static class ExpectedSlot {
        public final int slotId;
        /** Centre, fractional (0-1) relative to view size. */
        public final float x;
        public final float y;
        /** True when nothing was found here; false when something wrong was. */
        public final boolean missing;

        public ExpectedSlot(int slotId, float x, float y, boolean missing) {
            this.slotId = slotId;
            this.x = x;
            this.y = y;
            this.missing = missing;
        }
    }

    /** Notified when the user taps a detection, so the host screen can offer a correction. */
    public interface OnFeatureClickListener {
        void onFeatureClick(int index, DetectedFeature feature);
    }

    private Bitmap sourceBitmap = null;
    /**
     * Source scaled to the current view size, cached. onDraw used to call
     * createScaledBitmap on every frame, allocating a fresh view-sized bitmap each
     * time — costly to begin with and much worse now that taps trigger redraws.
     */
    private Bitmap scaledCache = null;
    /**
     * The region excluded as not-chassis, opaque white over transparent, or null.
     *
     * <p>Drawn over the photograph so the operator can see what the detector was not allowed
     * to look at. The detector itself sees the plate's own grey there, not white — filling
     * with white moves the intensity histogram Canny's thresholds come from. See
     * {@code ml/ChassisMask}.
     */
    private Bitmap backgroundOverlay = null;
    private int cachedW = 0, cachedH = 0;

    private final List<DetectedFeature> features = new ArrayList<>();
    private final List<ExpectedSlot> expectedSlots = new ArrayList<>();
    private final Paint paintRect = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint paintText = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint paintBackgroundMesh = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint paintSelected = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint paintSlot = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint paintSlotLabel = new Paint(Paint.ANTI_ALIAS_FLAG);

    private OnFeatureClickListener featureClickListener;
    private int selectedIndex = -1;
    /** Extra hit-test padding in px so small detections stay tappable with a fingertip. */
    private float touchPadding = 0f;

    public ChassisOverlayView(Context context) {
        super(context);
        init();
    }

    public ChassisOverlayView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    public ChassisOverlayView(Context context, @Nullable AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        init();
    }

    private void init() {
        paintRect.setStyle(Paint.Style.STROKE);
        paintRect.setStrokeWidth(5f);

        paintText.setColor(Color.BLACK);
        paintText.setTextSize(26f);
        paintText.setFakeBoldText(true);

        paintBackgroundMesh.setColor(Color.rgb(73, 69, 79)); // #49454F border color representation
        paintBackgroundMesh.setStrokeWidth(2f);
        paintBackgroundMesh.setStyle(Paint.Style.STROKE);

        paintSelected.setStyle(Paint.Style.STROKE);
        paintSelected.setStrokeWidth(9f);
        paintSelected.setColor(Color.WHITE);

        // Dashed, so an expected-but-absent position never reads as a detection. The detection
        // boxes are solid; anything dashed on this overlay is somewhere to look, not something
        // found.
        paintSlot.setStyle(Paint.Style.STROKE);
        paintSlot.setStrokeWidth(5f);
        paintSlot.setPathEffect(new android.graphics.DashPathEffect(new float[]{14f, 10f}, 0f));

        paintSlotLabel.setTextSize(24f);
        paintSlotLabel.setFakeBoldText(true);

        touchPadding = 12f * getResources().getDisplayMetrics().density;
        setClickable(true);
    }

    /**
     * Sets the expected positions to mark up. Pass an empty list to clear them — a stale set
     * drawn after a correction would point the operator at a slot that has since been filled.
     */
    public void setExpectedSlots(List<ExpectedSlot> slots) {
        expectedSlots.clear();
        if (slots != null) expectedSlots.addAll(slots);
        invalidate();
    }

    public void setOnFeatureClickListener(OnFeatureClickListener listener) {
        this.featureClickListener = listener;
    }

    /** Applies a corrected label to one detection and redraws. */
    public void updateFeature(int index, String type, int color, String label) {
        if (index < 0 || index >= features.size()) return;
        DetectedFeature f = features.get(index);
        f.type = type;
        f.color = color;
        f.label = label;
        invalidate();
    }

    public void clearSelection() {
        selectedIndex = -1;
        invalidate();
    }

    public List<DetectedFeature> getFeatures() {
        return features;
    }

    /**
     * Sets the data to be displayed in the overlay.
     * Use this to avoid redundant feature extraction.
     */
    public void setResultData(String uriString, List<HardwareFeatureExtractor.ExtractedFeature> extractedFeatures) {
        setResultData(uriString, extractedFeatures, null);
    }

    /**
     * As above, plus the region excluded as not-chassis, which is drawn white over the photo.
     *
     * @param background {@code AnalysisResult.backgroundOverlay}, or null to show the whole
     *                   photograph as inspected
     */
    public void setResultData(String uriString,
                              List<HardwareFeatureExtractor.ExtractedFeature> extractedFeatures,
                              Bitmap background) {
        loadBitmapFromUri(uriString);
        backgroundOverlay = background;
        features.clear();
        if (extractedFeatures != null) {
            for (HardwareFeatureExtractor.ExtractedFeature ext : extractedFeatures) {
                features.add(new DetectedFeature(ext.type, ext.rectFraction, ext.color, ext.label));
            }
        }
        invalidate();
    }

    public void setImageUri(String uriString) {
        if (uriString == null || "mock_image_uri".equals(uriString)) {
            sourceBitmap = null;
            features.clear();
            invalidate();
            return;
        }

        loadBitmapFromUri(uriString);

        // Apply real feature extraction dynamically
        features.clear();
        HardwareFeatureExtractor.AnalysisResult result = HardwareFeatureExtractor.extractFeatures(getContext(), uriString);
        
        if (result.features != null) {
            for (HardwareFeatureExtractor.ExtractedFeature ext : result.features) {
                features.add(new DetectedFeature(ext.type, ext.rectFraction, ext.color, ext.label));
            }
        }
        invalidate();
    }

    private void loadBitmapFromUri(String uriString) {
        if (scaledCache != null && !scaledCache.isRecycled()) scaledCache.recycle();
        scaledCache = null;
        if (sourceBitmap != null && !sourceBitmap.isRecycled()) sourceBitmap.recycle();

        if (uriString == null || "mock_image_uri".equals(uriString)) {
            sourceBitmap = null;
            return;
        }
        // Downsampled: the chassis photos are camera-resolution and decoding one at full
        // size costs ~46MB, for a view only a few hundred pixels tall.
        //
        // Must be the extractor's working size, not a copy of the number: the feature boxes
        // this view draws are in that coordinate space, so the two drifting apart would
        // offset every box.
        sourceBitmap = HardwareFeatureExtractor.loadBitmap(getContext(), uriString,
                HardwareFeatureExtractor.WORKING_MAX_DIM);
        if (sourceBitmap == null) {
            Log.e(TAG, "Error decoding image file uri: " + uriString);
        }
    }

    @Override
    public boolean onTouchEvent(android.view.MotionEvent event) {
        if (event.getAction() == android.view.MotionEvent.ACTION_UP && !features.isEmpty()) {
            int hit = featureAt(event.getX(), event.getY());
            if (hit >= 0) {
                selectedIndex = hit;
                invalidate();
                performClick();
                if (featureClickListener != null) {
                    featureClickListener.onFeatureClick(hit, features.get(hit));
                }
                return true;
            }
        }
        return super.onTouchEvent(event);
    }

    @Override
    public boolean performClick() {
        return super.performClick();
    }

    /**
     * Index of the detection under the given view coordinate, or -1.
     *
     * Boxes commonly overlap and nest, so the smallest match wins — otherwise a large
     * enclosing box would swallow every tap meant for the small detections inside it.
     */
    private int featureAt(float x, float y) {
        int viewWidth = getWidth();
        int viewHeight = getHeight();
        if (viewWidth == 0 || viewHeight == 0) return -1;

        int best = -1;
        float bestArea = Float.MAX_VALUE;
        for (int i = 0; i < features.size(); i++) {
            RectF fr = features.get(i).rectFraction;
            float left = fr.left * viewWidth - touchPadding;
            float top = fr.top * viewHeight - touchPadding;
            float right = fr.right * viewWidth + touchPadding;
            float bottom = fr.bottom * viewHeight + touchPadding;
            if (x >= left && x <= right && y >= top && y <= bottom) {
                float area = (right - left) * (bottom - top);
                if (area < bestArea) {
                    bestArea = area;
                    best = i;
                }
            }
        }
        return best;
    }

    @Override
    protected void onDetachedFromWindow() {
        super.onDetachedFromWindow();
        if (scaledCache != null && !scaledCache.isRecycled()) scaledCache.recycle();
        scaledCache = null;
        if (sourceBitmap != null && !sourceBitmap.isRecycled()) sourceBitmap.recycle();
        sourceBitmap = null;
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        int viewWidth = getWidth();
        int viewHeight = getHeight();

        if (viewWidth == 0 || viewHeight == 0) return;

        if (sourceBitmap != null) {
            // Rebuild the scaled copy only when the view size actually changes.
            if (scaledCache == null || scaledCache.isRecycled()
                    || cachedW != viewWidth || cachedH != viewHeight) {
                if (scaledCache != null && !scaledCache.isRecycled()) scaledCache.recycle();
                scaledCache = Bitmap.createScaledBitmap(sourceBitmap, viewWidth, viewHeight, true);
                cachedW = viewWidth;
                cachedH = viewHeight;
            }
            canvas.drawBitmap(scaledCache, 0f, 0f, null);
            // What the detector was not allowed to look at. Drawn after the photo and before
            // the boxes, so a box can never be hidden by it.
            if (backgroundOverlay != null && !backgroundOverlay.isRecycled()) {
                canvas.drawBitmap(backgroundOverlay,
                        new android.graphics.Rect(0, 0, backgroundOverlay.getWidth(),
                                backgroundOverlay.getHeight()),
                        new android.graphics.Rect(0, 0, viewWidth, viewHeight), null);
            }
        } else {
            canvas.drawColor(Color.rgb(43, 41, 48)); 

            for (int i = 0; i < viewWidth; i += 60) {
                canvas.drawLine(i, 0, i, viewHeight, paintBackgroundMesh);
            }
            for (int j = 0; j < viewHeight; j += 60) {
                canvas.drawLine(0, j, viewWidth, j, paintBackgroundMesh);
            }
        }

        for (int i = 0; i < features.size(); i++) {
            DetectedFeature feature = features.get(i);
            float left = feature.rectFraction.left * viewWidth;
            float top = feature.rectFraction.top * viewHeight;
            float right = feature.rectFraction.right * viewWidth;
            float bottom = feature.rectFraction.bottom * viewHeight;

            RectF targetRect = new RectF(left, top, right, bottom);

            if (i == selectedIndex) {
                canvas.drawRoundRect(targetRect, 8f, 8f, paintSelected);
            }
            paintRect.setColor(feature.color);
            canvas.drawRoundRect(targetRect, 8f, 8f, paintRect);

            Paint tabPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
            tabPaint.setColor(feature.color);
            tabPaint.setStyle(Paint.Style.FILL);
            canvas.drawRect(left, top - 36f, left + 140f, top, tabPaint);

            canvas.drawText(feature.label, left + 8f, top - 10f, paintText);
        }

        // Expected positions last, so a marker is never hidden under a detection box drawn
        // over it — the whole point is that these stay visible.
        float radius = Math.min(viewWidth, viewHeight) * 0.045f;
        for (ExpectedSlot slot : expectedSlots) {
            float cx = slot.x * viewWidth;
            float cy = slot.y * viewHeight;
            int color = slot.missing ? Color.rgb(248, 113, 113) : Color.rgb(251, 191, 36);
            paintSlot.setColor(color);
            canvas.drawCircle(cx, cy, radius, paintSlot);
            // Cross-hair through the centre: on a busy part a lone ring is easy to lose.
            canvas.drawLine(cx - radius, cy, cx + radius, cy, paintSlot);
            canvas.drawLine(cx, cy - radius, cx, cy + radius, paintSlot);

            String label = "#" + slot.slotId;
            paintSlotLabel.setColor(color);
            canvas.drawText(label, cx + radius + 6f, cy - radius, paintSlotLabel);
        }
    }
}
