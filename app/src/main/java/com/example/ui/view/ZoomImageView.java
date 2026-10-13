package com.example.ui.view;

import android.content.Context;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.graphics.drawable.Drawable;
import android.util.AttributeSet;
import android.view.GestureDetector;
import android.view.MotionEvent;
import android.view.ScaleGestureDetector;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.widget.AppCompatImageView;

/**
 * An image that starts fitted to the view and can be pinched to zoom, dragged when zoomed and
 * double-tapped to zoom in or back out — for reading which box a label belongs to on a photo
 * of small hardware.
 */
public class ZoomImageView extends AppCompatImageView {

    private static final float MAX_ZOOM = 8f;
    private static final float DOUBLE_TAP_ZOOM = 3f;

    private final Matrix matrix = new Matrix();
    private float fitScale = 1f;
    private float zoom = 1f;
    private final ScaleGestureDetector scaler;
    private final GestureDetector gestures;

    public ZoomImageView(@NonNull Context context) {
        this(context, null);
    }

    public ZoomImageView(@NonNull Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        setScaleType(ScaleType.MATRIX);
        scaler = new ScaleGestureDetector(context,
                new ScaleGestureDetector.SimpleOnScaleGestureListener() {
                    @Override
                    public boolean onScale(@NonNull ScaleGestureDetector d) {
                        zoomBy(d.getScaleFactor(), d.getFocusX(), d.getFocusY());
                        return true;
                    }
                });
        gestures = new GestureDetector(context, new GestureDetector.SimpleOnGestureListener() {
            @Override
            public boolean onScroll(MotionEvent e1, @NonNull MotionEvent e2, float dx, float dy) {
                if (zoom <= 1.001f) return false;
                matrix.postTranslate(-dx, -dy);
                clamp();
                return true;
            }

            @Override
            public boolean onDoubleTap(@NonNull MotionEvent e) {
                if (zoom > 1.001f) {
                    fit();
                } else {
                    zoomBy(DOUBLE_TAP_ZOOM, e.getX(), e.getY());
                }
                return true;
            }

            @Override
            public boolean onSingleTapConfirmed(@NonNull MotionEvent e) {
                return performClick();
            }
        });
    }

    @Override
    public boolean performClick() {
        return super.performClick();
    }

    /** Whether the image is zoomed in past its fitted size. */
    public boolean isZoomed() {
        return zoom > 1.001f;
    }

    @Override
    public void setImageDrawable(@Nullable Drawable drawable) {
        super.setImageDrawable(drawable);
        post(this::fit);
    }

    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);
        fit();
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        boolean handled = scaler.onTouchEvent(event);
        handled |= gestures.onTouchEvent(event);
        return handled || super.onTouchEvent(event);
    }

    /** Back to the whole image, fitted and centred. */
    public void fit() {
        Drawable d = getDrawable();
        if (d == null || getWidth() == 0 || getHeight() == 0
                || d.getIntrinsicWidth() <= 0 || d.getIntrinsicHeight() <= 0) {
            return;
        }
        float dw = d.getIntrinsicWidth(), dh = d.getIntrinsicHeight();
        fitScale = Math.min(getWidth() / dw, getHeight() / dh);
        zoom = 1f;
        matrix.reset();
        matrix.postScale(fitScale, fitScale);
        matrix.postTranslate((getWidth() - dw * fitScale) / 2f, (getHeight() - dh * fitScale) / 2f);
        setImageMatrix(matrix);
    }

    private void zoomBy(float factor, float fx, float fy) {
        float next = Math.max(1f, Math.min(MAX_ZOOM, zoom * factor));
        factor = next / zoom;
        zoom = next;
        matrix.postScale(factor, factor, fx, fy);
        clamp();
    }

    /** Keeps the image covering the view when zoomed, centred along any side it is smaller. */
    private void clamp() {
        Drawable d = getDrawable();
        if (d == null) return;
        RectF r = new RectF(0, 0, d.getIntrinsicWidth(), d.getIntrinsicHeight());
        matrix.mapRect(r);
        float dx = 0, dy = 0;
        if (r.width() <= getWidth()) {
            dx = (getWidth() - r.width()) / 2f - r.left;
        } else if (r.left > 0) {
            dx = -r.left;
        } else if (r.right < getWidth()) {
            dx = getWidth() - r.right;
        }
        if (r.height() <= getHeight()) {
            dy = (getHeight() - r.height()) / 2f - r.top;
        } else if (r.top > 0) {
            dy = -r.top;
        } else if (r.bottom < getHeight()) {
            dy = getHeight() - r.bottom;
        }
        matrix.postTranslate(dx, dy);
        setImageMatrix(matrix);
    }
}
