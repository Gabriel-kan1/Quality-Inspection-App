package com.example.ui.view;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.View;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

/**
 * Three stacked lamps showing whether the live frame is aligned well enough to inspect.
 *
 * <p>Only the active lamp is lit; the other two stay dimmed rather than hidden, so the
 * operator can read the state from the position of the light in their peripheral vision
 * without focusing on it — the same reason road traffic lights keep all three housings.
 *
 * <p>The green lamp also carries the hold progress as a ring that fills clockwise, so the
 * three-second dwell is visible rather than a mysterious pause before capture.
 */
public class TrafficLightView extends View {

    public enum State {
        /** Out of tolerance, or nothing recognised in frame. */
        RED,
        /** Within tolerance; the hold is counting. */
        AMBER,
        /** Held for the full dwell; the frame has been accepted. */
        GREEN,
        /** Gate inactive — no reference to compare against. */
        OFF
    }

    private static final int COLOR_RED = 0xFFF87171;
    private static final int COLOR_AMBER = 0xFFFBBF24;
    private static final int COLOR_GREEN = 0xFF4ADE80;
    /** Unlit lamps: dark enough to read as off, light enough to locate. */
    private static final int COLOR_DIM = 0x33FFFFFF;
    private static final int COLOR_HOUSING = 0xCC15151A;

    private static final float DIM_ALPHA = 0.22f;

    private final Paint lampPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint housingPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint glowPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint progressPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final android.graphics.RectF progressArc = new android.graphics.RectF();

    private State state = State.OFF;
    private float holdProgress = 0f;

    public TrafficLightView(Context context) {
        this(context, null);
    }

    public TrafficLightView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        housingPaint.setColor(COLOR_HOUSING);
        progressPaint.setStyle(Paint.Style.STROKE);
        progressPaint.setStrokeCap(Paint.Cap.ROUND);
        progressPaint.setColor(COLOR_GREEN);
    }

    /**
     * @param state    which lamp to light
     * @param progress hold progress 0..1, drawn as a ring on the amber lamp
     */
    public void setState(@NonNull State state, float progress) {
        float clamped = Math.max(0f, Math.min(1f, progress));
        if (this.state == state && Math.abs(this.holdProgress - clamped) < 0.01f) return;
        this.state = state;
        this.holdProgress = clamped;
        invalidate();
    }

    public State getState() {
        return state;
    }

    @Override
    protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec) {
        // Fixed 1:3 aspect so the housing keeps its shape whatever the parent offers.
        int width = resolveSize((int) (28 * getResources().getDisplayMetrics().density), widthMeasureSpec);
        setMeasuredDimension(width, resolveSize(width * 3, heightMeasureSpec));
    }

    @Override
    protected void onDraw(@NonNull Canvas canvas) {
        super.onDraw(canvas);

        float width = getWidth();
        float height = getHeight();
        if (width <= 0 || height <= 0) return;

        float corner = width * 0.4f;
        canvas.drawRoundRect(0, 0, width, height, corner, corner, housingPaint);

        float cellHeight = height / 3f;
        float radius = Math.min(width, cellHeight) * 0.3f;
        float cx = width / 2f;

        drawLamp(canvas, cx, cellHeight * 0.5f, radius, COLOR_RED, state == State.RED);
        drawLamp(canvas, cx, cellHeight * 1.5f, radius, COLOR_AMBER, state == State.AMBER);
        drawLamp(canvas, cx, cellHeight * 2.5f, radius, COLOR_GREEN, state == State.GREEN);

        if (state == State.AMBER && holdProgress > 0f) {
            drawHoldRing(canvas, cx, cellHeight * 1.5f, radius * 1.55f);
        }
    }

    private void drawLamp(Canvas canvas, float cx, float cy, float radius, int color, boolean lit) {
        if (lit) {
            // A soft halo makes the lit lamp readable against a bright camera preview, where
            // a flat filled circle washes out.
            glowPaint.setColor(color);
            glowPaint.setAlpha(70);
            canvas.drawCircle(cx, cy, radius * 1.7f, glowPaint);
            lampPaint.setColor(color);
            lampPaint.setAlpha(255);
        } else {
            lampPaint.setColor(blend(color, COLOR_DIM));
            lampPaint.setAlpha((int) (255 * DIM_ALPHA));
        }
        canvas.drawCircle(cx, cy, radius, lampPaint);
    }

    private void drawHoldRing(Canvas canvas, float cx, float cy, float radius) {
        progressPaint.setStrokeWidth(Math.max(2f, radius * 0.18f));
        progressArc.set(cx - radius, cy - radius, cx + radius, cy + radius);
        // Starts at 12 o'clock so a part-filled ring reads as a countdown, not a random arc.
        canvas.drawArc(progressArc, -90f, 360f * holdProgress, false, progressPaint);
    }

    private static int blend(int color, int overlay) {
        return Color.rgb(
                (Color.red(color) + Color.red(overlay)) / 2,
                (Color.green(color) + Color.green(overlay)) / 2,
                (Color.blue(color) + Color.blue(overlay)) / 2);
    }
}
