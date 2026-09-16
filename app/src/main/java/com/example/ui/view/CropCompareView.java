package com.example.ui.view;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.example.ui.util.ComparisonCrops;

/**
 * Draws a reference crop and an inspection crop side by side, <em>at a common scale on the
 * part</em>, with the box that was compared outlined on each.
 *
 * <h3>Why not two thumbnails</h3>
 * Two image views would each stretch their crop to fit, and a stretched pair says nothing
 * about size — a bolt head half the size of the reference fills its thumbnail just the same.
 * Here the two boxes are drawn in the ratio the size check measured
 * ({@code FeatureLocalisation.SlotFinding.sizeRatio}, which already divides out how close the
 * camera was), so if the outlines look the same size on screen the hardware is the same size
 * on the part, and if one is half again as large that is what the operator sees.
 *
 * The outline is the honest part of the picture. What was compared is the detector's bounding
 * box, not the fastener's true edge, so the box is drawn rather than implied: an operator who
 * sees the reference box hugging a bolt head and the inspection box swallowing its neighbour
 * knows at once that the "size change" is a detection artefact.
 */
public class CropCompareView extends View {

    /** The larger of the two boxes takes this fraction of its panel's shorter side. */
    static final float LARGER_BOX_FRACTION = 0.55f;
    /** Below this a box is drawn at least this fraction of the panel — a legibility floor. */
    static final float MIN_BOX_FRACTION = 0.08f;

    private static final int REFERENCE_OUTLINE = Color.rgb(56, 189, 248);
    private static final int MATCH_OUTLINE = Color.rgb(74, 222, 128);
    private static final int CHANGED_OUTLINE = Color.rgb(251, 146, 60);
    private static final int PANEL_BACKGROUND = Color.rgb(12, 12, 16);
    private static final int PANEL_BORDER = Color.rgb(60, 60, 70);

    private final Paint bitmapPaint = new Paint(Paint.FILTER_BITMAP_FLAG | Paint.ANTI_ALIAS_FLAG);
    private final Paint outlinePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint panelPaint = new Paint();
    private final Paint borderPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint textPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Matrix matrix = new Matrix();
    private final RectF panel = new RectF();
    private final RectF drawnBox = new RectF();

    @Nullable private ComparisonCrops.Crop reference;
    @Nullable private ComparisonCrops.Crop inspection;
    private float sizeRatio = Float.NaN;
    private boolean sizeChanged;
    private String placeholder = "Loading…";

    public CropCompareView(Context context) {
        this(context, null);
    }

    public CropCompareView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        float density = getResources().getDisplayMetrics().density;
        outlinePaint.setStyle(Paint.Style.STROKE);
        outlinePaint.setStrokeWidth(2f * density);
        panelPaint.setColor(PANEL_BACKGROUND);
        borderPaint.setStyle(Paint.Style.STROKE);
        borderPaint.setStrokeWidth(density);
        borderPaint.setColor(PANEL_BORDER);
        textPaint.setColor(Color.rgb(150, 150, 158));
        textPaint.setTextSize(12f * density);
        textPaint.setTextAlign(Paint.Align.CENTER);
    }

    /**
     * Sets both sides. Either crop may be null, in which case that panel says so rather than
     * drawing nothing. The view does not own the bitmaps; the caller recycles them after
     * {@link #clear()}.
     *
     * @param sizeRatio linear size of the inspection box over the reference's, corrected for
     *                  framing; NaN draws both boxes at the same size, which is only honest
     *                  when no comparison was possible and the caller says so elsewhere
     */
    public void setComparison(@Nullable ComparisonCrops.Crop reference,
                              @Nullable ComparisonCrops.Crop inspection,
                              float sizeRatio, boolean sizeChanged) {
        this.reference = reference;
        this.inspection = inspection;
        this.sizeRatio = sizeRatio;
        this.sizeChanged = sizeChanged;
        this.placeholder = "Not available";
        invalidate();
    }

    /** Drops the bitmap references so the caller can recycle them safely. */
    public void clear() {
        reference = null;
        inspection = null;
        invalidate();
    }

    /**
     * Target on-screen size, in pixels, for each box's geometric-mean side given the panel's
     * shorter side: {@code [reference, inspection]}.
     *
     * The larger box is pinned to {@link #LARGER_BOX_FRACTION} of the panel and the other
     * follows in the measured ratio, so the pair always fits and the ratio is always shown
     * true — except that a box that would vanish below {@link #MIN_BOX_FRACTION} is held there,
     * legibly small rather than invisibly so. Pure, so the rule can be tested on its own.
     */
    static float[] targetSizes(float panelShortSide, float sizeRatio) {
        float larger = panelShortSide * LARGER_BOX_FRACTION;
        float floor = panelShortSide * MIN_BOX_FRACTION;
        if (Float.isNaN(sizeRatio) || sizeRatio <= 0f) return new float[]{larger, larger};
        float ref = sizeRatio >= 1f ? larger / sizeRatio : larger;
        float ins = sizeRatio >= 1f ? larger : larger * sizeRatio;
        return new float[]{Math.max(floor, ref), Math.max(floor, ins)};
    }

    @Override
    protected void onDraw(@NonNull Canvas canvas) {
        super.onDraw(canvas);
        float density = getResources().getDisplayMetrics().density;
        float gap = 8f * density;
        float width = getWidth(), height = getHeight();
        float panelWidth = (width - gap) / 2f;
        float shortSide = Math.min(panelWidth, height);
        float[] targets = targetSizes(shortSide, sizeRatio);

        panel.set(0f, 0f, panelWidth, height);
        drawPanel(canvas, reference, targets[0], REFERENCE_OUTLINE);

        panel.set(panelWidth + gap, 0f, width, height);
        drawPanel(canvas, inspection, targets[1], sizeChanged ? CHANGED_OUTLINE : MATCH_OUTLINE);
    }

    private void drawPanel(Canvas canvas, @Nullable ComparisonCrops.Crop crop, float targetSize,
                           int outline) {
        canvas.drawRect(panel, panelPaint);
        if (crop == null || crop.bitmap == null || crop.bitmap.isRecycled()) {
            canvas.drawText(placeholder, panel.centerX(), panel.centerY(), textPaint);
            canvas.drawRect(panel, borderPaint);
            return;
        }
        // Scale so the box's geometric-mean side lands at the target, then place the box's
        // centre at the panel's centre. Everything else in the crop follows at the same scale,
        // which is what makes the two panels comparable.
        float boxMean = (float) Math.sqrt(Math.max(1f, crop.box.width() * crop.box.height()));
        float scale = targetSize / boxMean;
        float dx = panel.centerX() - crop.box.centerX() * scale;
        float dy = panel.centerY() - crop.box.centerY() * scale;

        canvas.save();
        canvas.clipRect(panel);
        matrix.reset();
        matrix.postScale(scale, scale);
        matrix.postTranslate(dx, dy);
        canvas.drawBitmap(crop.bitmap, matrix, bitmapPaint);

        drawnBox.set(crop.box.left * scale + dx, crop.box.top * scale + dy,
                crop.box.right * scale + dx, crop.box.bottom * scale + dy);
        outlinePaint.setColor(outline);
        canvas.drawRect(drawnBox, outlinePaint);
        canvas.restore();
        canvas.drawRect(panel, borderPaint);
    }
}
