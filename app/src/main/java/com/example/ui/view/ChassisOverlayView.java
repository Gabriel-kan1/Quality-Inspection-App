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

        // Load standard mock features styled after Geometric Balance HTML parameters
        features.add(new DetectedFeature("Bolt", new RectF(0.15f, 0.20f, 0.35f, 0.38f), Color.rgb(56, 189, 248), "BOLT #01")); // #38BDF8
        features.add(new DetectedFeature("Bolt", new RectF(0.60f, 0.18f, 0.80f, 0.36f), Color.rgb(56, 189, 248), "BOLT #02"));
        features.add(new DetectedFeature("Bolt", new RectF(0.40f, 0.50f, 0.60f, 0.68f), Color.rgb(56, 189, 248), "BOLT #03"));
        features.add(new DetectedFeature("Hole", new RectF(0.18f, 0.65f, 0.32f, 0.79f), Color.rgb(251, 191, 36), "HOLE #01"));  // #FBBF24
        features.add(new DetectedFeature("Screw", new RectF(0.68f, 0.65f, 0.82f, 0.79f), Color.rgb(74, 222, 128), "SCREW #01")); // #4ADE80
    }

    public void setImageUri(String uriString) {
        if (uriString == null || "mock_image_uri".equals(uriString)) {
            sourceBitmap = null;
            invalidate();
            return;
        }

        try {
            Uri uri = Uri.parse(uriString);
            InputStream inputStream = getContext().getContentResolver().openInputStream(uri);
            sourceBitmap = BitmapFactory.decodeStream(inputStream);
            if (inputStream != null) {
                inputStream.close();
            }
        } catch (Exception e) {
            Log.e(TAG, "Error decoding image file uri: " + uriString, e);
            sourceBitmap = null;
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
            // perfect for mock/testing presentation
            canvas.drawColor(Color.rgb(43, 41, 48)); // Dark metal chassis back (#2B2930)

            // Draw technical cross grid lines
            for (int i = 0; i < viewWidth; i += 60) {
                canvas.drawLine(i, 0, i, viewHeight, paintBackgroundMesh);
            }
            for (int j = 0; j < viewHeight; j += 60) {
                canvas.drawLine(0, j, viewWidth, j, paintBackgroundMesh);
            }

            // Draw a subtle wireframe circle in the center matching factory visual guidelines
            Paint guidePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
            guidePaint.setColor(Color.argb(45, 56, 189, 248)); // #38BDF8 with 45 alpha
            guidePaint.setStrokeWidth(4f);
            guidePaint.setStyle(Paint.Style.STROKE);
            canvas.drawCircle(viewWidth / 2f, viewHeight / 2f, Math.min(viewWidth, viewHeight) / 3f, guidePaint);
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
