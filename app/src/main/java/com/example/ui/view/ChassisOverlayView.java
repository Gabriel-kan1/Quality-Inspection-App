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

    public void setImageUri(String uriString) {
        if (uriString == null || "mock_image_uri".equals(uriString)) {
            sourceBitmap = null;
            features.clear();
            invalidate();
            return;
        }

        try {
            InputStream inputStream;
            if (uriString.startsWith("file:///android_asset/")) {
                String assetPath = uriString.substring("file:///android_asset/".length());
                inputStream = getContext().getAssets().open(assetPath);
            } else {
                Uri uri = Uri.parse(uriString);
                inputStream = getContext().getContentResolver().openInputStream(uri);
            }
            sourceBitmap = BitmapFactory.decodeStream(inputStream);
            if (inputStream != null) {
                inputStream.close();
            }

            // Apply real feature extraction dynamically based on image content (Python Step 3 to 6)
            features.clear();
            List<com.example.ui.fragment.HardwareFeatureExtractor.ExtractedFeature> extracted = 
                    com.example.ui.fragment.HardwareFeatureExtractor.extractFeatures(getContext(), uriString);
            
            for (com.example.ui.fragment.HardwareFeatureExtractor.ExtractedFeature ext : extracted) {
                features.add(new DetectedFeature(ext.type, ext.rectFraction, ext.color, ext.label));
            }
        } catch (Exception e) {
            Log.e(TAG, "Error decoding image file uri or performing feature extraction: " + uriString, e);
            sourceBitmap = null;
            features.clear();
        }
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        int viewWidth = getWidth();
        int viewHeight = getHeight();

        if (viewWidth == 0 || viewHeight == 0) return;

        // Draw image background or schematic mockup panel
        if (sourceBitmap != null) {
            // Draw scaling real bitmap
            Bitmap scaledBitmap = Bitmap.createScaledBitmap(sourceBitmap, viewWidth, viewHeight, true);
            canvas.drawBitmap(scaledBitmap, 0f, 0f, null);
        } else {
            // No real photo was stored, draw a highly technical metal chassis mesh simulation 
            // perfect for mock/testing presentation representing their actual hardware picture!
            canvas.drawColor(Color.rgb(43, 41, 48)); // Dark grey metallic surface

            // Draw technical cross grid lines
            for (int i = 0; i < viewWidth; i += 60) {
                canvas.drawLine(i, 0, i, viewHeight, paintBackgroundMesh);
            }
            for (int j = 0; j < viewHeight; j += 60) {
                canvas.drawLine(0, j, viewWidth, j, paintBackgroundMesh);
            }

            // Draw beautiful vector cylindrical objects representing PINs and STUDs to aid testing
            Paint partPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
            Paint partDetailPaint = new Paint(Paint.ANTI_ALIAS_FLAG);

            partPaint.setStyle(Paint.Style.FILL);
            partDetailPaint.setStyle(Paint.Style.STROKE);
            partDetailPaint.setStrokeWidth(3f);

            for (DetectedFeature feature : features) {
                float left = feature.rectFraction.left * viewWidth;
                float top = feature.rectFraction.top * viewHeight;
                float right = feature.rectFraction.right * viewWidth;
                float bottom = feature.rectFraction.bottom * viewHeight;
                float cx = (left + right) / 2f;
                float cy = (top + bottom) / 2f;
                float radius = Math.min(right - left, bottom - top) / 2.2f;

                // Outer base plate reflection ring
                partDetailPaint.setColor(Color.rgb(130, 130, 130));
                canvas.drawCircle(cx, cy, radius * 1.2f, partDetailPaint);

                if ("Locator Pin".equals(feature.type)) {
                    // Plain Chamfered Pin Cylinder (metallic grey gradients)
                    partPaint.setColor(Color.rgb(160, 163, 168));
                    canvas.drawCircle(cx, cy, radius, partPaint);

                    // Inner chamfered center hole
                    partPaint.setColor(Color.rgb(80, 83, 88));
                    canvas.drawCircle(cx, cy, radius * 0.35f, partPaint);

                    // Pin highlight ring
                    partDetailPaint.setColor(Color.rgb(220, 225, 230));
                    canvas.drawCircle(cx, cy, radius * 0.8f, partDetailPaint);
                } else {
                    // Threaded Stud Bolt Cylinder (screw ridges & highlight shine)
                    partPaint.setColor(Color.rgb(205, 208, 213));
                    canvas.drawCircle(cx, cy, radius, partPaint);

                    // Screw thread circular divisions
                    partDetailPaint.setColor(Color.rgb(90, 93, 100));
                    canvas.drawCircle(cx, cy, radius * 0.8f, partDetailPaint);
                    canvas.drawCircle(cx, cy, radius * 0.6f, partDetailPaint);
                    canvas.drawCircle(cx, cy, radius * 0.4f, partDetailPaint);

                    // Reflective shiny overlay bar on threaded bolts
                    Paint shinePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
                    shinePaint.setColor(Color.argb(90, 255, 255, 255));
                    shinePaint.setStyle(Paint.Style.FILL);
                    canvas.drawRect(cx - radius * 0.25f, cy - radius, cx + radius * 0.25f, cy + radius, shinePaint);
                }
            }
        }

        // Draw overlay detected bounding boxes
        for (DetectedFeature feature : features) {
            // Convert fractional rectangle to absolute view pixels
            float left = feature.rectFraction.left * viewWidth;
            float top = feature.rectFraction.top * viewHeight;
            float right = feature.rectFraction.right * viewWidth;
            float bottom = feature.rectFraction.bottom * viewHeight;

            RectF targetRect = new RectF(left, top, right, bottom);

            // Draw bounding box
            paintRect.setColor(feature.color);
            canvas.drawRoundRect(targetRect, 8f, 8f, paintRect);

            // Draw label background background tab
            Paint tabPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
            tabPaint.setColor(feature.color);
            tabPaint.setStyle(Paint.Style.FILL);
            canvas.drawRect(left, top - 36f, left + 140f, top, tabPaint);

            // Draw label text
            canvas.drawText(feature.label, left + 8f, top - 10f, paintText);
        }
    }
}
