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

import com.example.ui.fragment.HardwareFeatureExtractor;

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

    private Bitmap sourceBitmap = null;
    private final List<DetectedFeature> features = new ArrayList<>();
    private final Paint paintRect = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint paintText = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint paintBackgroundMesh = new Paint(Paint.ANTI_ALIAS_FLAG);

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
    }

    public List<DetectedFeature> getFeatures() {
        return features;
    }

    /**
     * Sets the data to be displayed in the overlay.
     * Use this to avoid redundant feature extraction.
     */
    public void setResultData(String uriString, List<HardwareFeatureExtractor.ExtractedFeature> extractedFeatures) {
        loadBitmapFromUri(uriString);
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
        if (uriString == null || "mock_image_uri".equals(uriString)) {
            sourceBitmap = null;
            return;
        }
        try {
            InputStream inputStream;
            if (uriString.startsWith("file:///android_asset/")) {
                // Robust path extraction for assets
                String assetPath = uriString.replace("file:///android_asset/", "");
                inputStream = getContext().getAssets().open(assetPath);
            } else {
                Uri uri = Uri.parse(uriString);
                inputStream = getContext().getContentResolver().openInputStream(uri);
            }
            sourceBitmap = BitmapFactory.decodeStream(inputStream);
            if (inputStream != null) {
                inputStream.close();
            }
        } catch (Exception e) {
            Log.e(TAG, "Error decoding image file uri: " + uriString, e);
            sourceBitmap = null;
        }
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        int viewWidth = getWidth();
        int viewHeight = getHeight();

        if (viewWidth == 0 || viewHeight == 0) return;

        if (sourceBitmap != null) {
            Bitmap scaledBitmap = Bitmap.createScaledBitmap(sourceBitmap, viewWidth, viewHeight, true);
            canvas.drawBitmap(scaledBitmap, 0f, 0f, null);
        } else {
            canvas.drawColor(Color.rgb(43, 41, 48)); 

            for (int i = 0; i < viewWidth; i += 60) {
                canvas.drawLine(i, 0, i, viewHeight, paintBackgroundMesh);
            }
            for (int j = 0; j < viewHeight; j += 60) {
                canvas.drawLine(0, j, viewWidth, j, paintBackgroundMesh);
            }
        }

        for (DetectedFeature feature : features) {
            float left = feature.rectFraction.left * viewWidth;
            float top = feature.rectFraction.top * viewHeight;
            float right = feature.rectFraction.right * viewWidth;
            float bottom = feature.rectFraction.bottom * viewHeight;

            RectF targetRect = new RectF(left, top, right, bottom);

            paintRect.setColor(feature.color);
            canvas.drawRoundRect(targetRect, 8f, 8f, paintRect);

            Paint tabPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
            tabPaint.setColor(feature.color);
            tabPaint.setStyle(Paint.Style.FILL);
            canvas.drawRect(left, top - 36f, left + 140f, top, tabPaint);

            canvas.drawText(feature.label, left + 8f, top - 10f, paintText);
        }
    }
}
