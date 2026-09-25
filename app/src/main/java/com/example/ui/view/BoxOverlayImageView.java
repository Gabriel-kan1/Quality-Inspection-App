package com.example.ui.view;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.drawable.Drawable;
import android.util.AttributeSet;

import androidx.annotation.Nullable;
import androidx.appcompat.widget.AppCompatImageView;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * An image with numbered boxes drawn over it, the boxes given as fractions of the image.
 *
 * <p>Used by the Edge Lab to show where each method's detections fall on the photograph. The
 * boxes are mapped through the view's own image matrix, so they stay on the pixels they name
 * under any scale type; the numbers match the crop strip beneath and the order an inspection
 * labels its detections in. Several cards can share one source bitmap this way, where baking
 * the boxes into a copy per card would cost a full-size bitmap each.
 */
public class BoxOverlayImageView extends AppCompatImageView {

    private final Paint boxPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint labelBackground = new Paint();
    private final Paint labelPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final RectF scratch = new RectF();
    private List<RectF> boxes = Collections.emptyList();

    public BoxOverlayImageView(Context context) {
        this(context, null);
    }

    public BoxOverlayImageView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        float density = getResources().getDisplayMetrics().density;
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(1.5f * density);
        boxPaint.setColor(Color.rgb(0x38, 0xBD, 0xF8));
        labelBackground.setColor(Color.argb(200, 0, 0, 0));
        labelPaint.setColor(Color.WHITE);
        labelPaint.setTextSize(9f * density);
        labelPaint.setFakeBoldText(true);
    }

    /** Replaces the boxes, as fractions (0-1) of the image. */
    public void setBoxes(@Nullable List<RectF> fractions) {
        boxes = fractions == null ? Collections.<RectF>emptyList() : new ArrayList<>(fractions);
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        Drawable d = getDrawable();
        if (d == null || boxes.isEmpty()) return;
        int iw = d.getIntrinsicWidth();
        int ih = d.getIntrinsicHeight();
        if (iw <= 0 || ih <= 0) return;

        Matrix m = getImageMatrix();
        canvas.save();
        canvas.translate(getPaddingLeft(), getPaddingTop());
        // Numbers only while they can be read; on a map with hundreds of boxes they would
        // bury the photograph, and the strip below still numbers the ones it shows.
        boolean label = boxes.size() <= 60;
        for (int i = 0; i < boxes.size(); i++) {
            RectF f = boxes.get(i);
            scratch.set(f.left * iw, f.top * ih, f.right * iw, f.bottom * ih);
            m.mapRect(scratch);
            canvas.drawRect(scratch, boxPaint);
            if (label) {
                String text = String.valueOf(i + 1);
                float tw = labelPaint.measureText(text);
                float th = labelPaint.getTextSize();
                // Above the box, unless that would leave the image; then just inside it.
                float top = scratch.top - th - 2 < 0 ? scratch.top : scratch.top - th - 2;
                canvas.drawRect(scratch.left, top, scratch.left + tw + 4, top + th + 2,
                        labelBackground);
                canvas.drawText(text, scratch.left + 2, top + th - 1, labelPaint);
            }
        }
        canvas.restore();
    }
}
