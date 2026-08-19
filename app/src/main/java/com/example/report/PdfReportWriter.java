package com.example.report;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Typeface;
import android.text.Layout;
import android.text.StaticLayout;
import android.text.TextPaint;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import java.io.IOException;
import java.io.OutputStream;
import java.util.List;

/**
 * Minimal flowing-document writer that lays out the report a page at a time.
 *
 * <p>This exists to keep report generation free of a third-party PDF library. The reports are a
 * title, a few headed prose blocks and two tables; that is well inside what Canvas plus
 * {@link StaticLayout} can do, and it costs nothing in dependencies, licence terms or CVE
 * exposure. The API is deliberately shaped around this one report rather than being general —
 * content is appended top to bottom and pages break themselves.
 *
 * <p>Geometry is in PostScript points (1/72"), which is the unit a PDF page's width and height
 * are expressed in. Nothing here is in density-independent pixels: a report must come out the
 * same size regardless of which phone drew it.
 *
 * <p>Pages are handed to a {@link PageSink} rather than written directly, so the layout can be
 * tested without the framework PDF backend. See {@link PageSink} for why.
 */
public final class PdfReportWriter {

    /** A4 at 72 points per inch. */
    public static final int A4_WIDTH_PT = 595;
    public static final int A4_HEIGHT_PT = 842;

    private static final float MARGIN_PT = 40f;
    private static final float CELL_PADDING_PT = 6f;
    private static final float TABLE_BORDER_PT = 0.7f;
    private static final int BORDER_COLOR = Color.rgb(190, 190, 190);
    private static final int HEADER_FILL = Color.rgb(128, 128, 128);
    private static final int FOOTER_COLOR = Color.rgb(150, 150, 150);
    private static final float FOOTER_SIZE_PT = 8f;

    /** Text appearance for one block or cell. */
    public static final class Style {
        final float sizePt;
        final boolean bold;
        final int color;
        final Layout.Alignment alignment;

        public Style(float sizePt, boolean bold, int color, Layout.Alignment alignment) {
            this.sizePt = sizePt;
            this.bold = bold;
            this.color = color;
            this.alignment = alignment;
        }

        public Style withAlignment(Layout.Alignment other) {
            return new Style(sizePt, bold, color, other);
        }
    }

    public static final Style TITLE =
            new Style(20f, true, Color.DKGRAY, Layout.Alignment.ALIGN_CENTER);
    public static final Style SECTION =
            new Style(12f, true, Color.DKGRAY, Layout.Alignment.ALIGN_NORMAL);
    public static final Style TABLE_HEADER =
            new Style(12f, true, Color.WHITE, Layout.Alignment.ALIGN_CENTER);
    public static final Style BODY =
            new Style(10f, false, Color.BLACK, Layout.Alignment.ALIGN_NORMAL);
    public static final Style BODY_BOLD =
            new Style(10f, true, Color.BLACK, Layout.Alignment.ALIGN_NORMAL);
    public static final Style PASS =
            new Style(12f, true, Color.rgb(38, 166, 154), Layout.Alignment.ALIGN_NORMAL);

    private final PageSink sink;
    private final float contentWidth;

    @Nullable private Canvas canvas;
    private float cursorY;
    private int pageNumber;

    public PdfReportWriter(@NonNull PageSink sink) {
        this.sink = sink;
        this.contentWidth = A4_WIDTH_PT - 2 * MARGIN_PT;
        startPage();
    }

    // ── Content ─────────────────────────────────────────────────────────────

    /** Appends a wrapped block of text, preceded and followed by the given gaps. */
    public void text(String content, Style style, float spacingBefore, float spacingAfter) {
        if (content == null) content = "";
        cursorY += spacingBefore;
        StaticLayout layout = layout(content, style, contentWidth);
        // A block longer than a whole page has to be allowed to overflow rather than loop
        // forever asking for a fresh page that will never be tall enough.
        if (layout.getHeight() <= usableHeight() && !fits(layout.getHeight())) {
            newPage();
        }
        draw(layout, MARGIN_PT, cursorY);
        cursorY += layout.getHeight() + spacingAfter;
    }

    public void text(String content, Style style) {
        text(content, style, 0f, 0f);
    }

    public void spacer(float points) {
        cursorY += points;
    }

    /**
     * Appends a bordered table. {@code weights} sets the relative column widths, matching the
     * proportional column sizing the report was originally laid out with. The header row is
     * repeated at the top of every page the table spills onto — without that, a table broken
     * across a page boundary loses the only thing saying what its columns mean.
     */
    public void table(String[] headers, float[] weights, List<Row> rows, float spacingBefore) {
        if (headers.length != weights.length) {
            throw new IllegalArgumentException(
                    "headers=" + headers.length + " but weights=" + weights.length);
        }
        cursorY += spacingBefore;

        float[] columnWidths = resolveWidths(weights);
        Row headerRow = new Row(headers, TABLE_HEADER);

        drawRow(headerRow, columnWidths, true);
        for (Row row : rows) {
            float height = rowHeight(row, columnWidths);
            if (!fits(height)) {
                newPage();
                drawRow(headerRow, columnWidths, true);
            }
            drawRow(row, columnWidths, false);
        }
    }

    /** One table row: a cell per column, with an optional per-cell style override. */
    public static final class Row {
        final String[] cells;
        final Style defaultStyle;
        @Nullable private Style[] overrides;
        @Nullable private Layout.Alignment[] alignments;

        public Row(String[] cells, Style defaultStyle) {
            this.cells = cells;
            this.defaultStyle = defaultStyle;
        }

        public Row style(int column, Style style) {
            if (overrides == null) overrides = new Style[cells.length];
            overrides[column] = style;
            return this;
        }

        public Row align(int column, Layout.Alignment alignment) {
            if (alignments == null) alignments = new Layout.Alignment[cells.length];
            alignments[column] = alignment;
            return this;
        }

        Style styleFor(int column) {
            Style base = overrides != null && overrides[column] != null
                    ? overrides[column] : defaultStyle;
            Layout.Alignment override = alignments != null ? alignments[column] : null;
            return override != null ? base.withAlignment(override) : base;
        }
    }

    /** Finishes the document and writes it out. The writer must not be used afterwards. */
    public void writeTo(@NonNull OutputStream out) throws IOException {
        finishPage();
        sink.writeTo(out);
    }

    /** Releases the document without producing output, for the failure path. */
    public void discard() {
        canvas = null;
        sink.discard();
    }

    /** Pages emitted so far, counting the one in progress. Exposed for tests. */
    public int pageCount() {
        return pageNumber;
    }

    // ── Layout plumbing ─────────────────────────────────────────────────────

    private float usableHeight() {
        return A4_HEIGHT_PT - 2 * MARGIN_PT;
    }

    private boolean fits(float height) {
        return cursorY + height <= A4_HEIGHT_PT - MARGIN_PT;
    }

    private void startPage() {
        pageNumber++;
        canvas = sink.beginPage(pageNumber, A4_WIDTH_PT, A4_HEIGHT_PT);
        cursorY = MARGIN_PT;
    }

    private void finishPage() {
        if (canvas == null) return;
        drawFooter();
        sink.finishPage();
        canvas = null;
    }

    private void newPage() {
        finishPage();
        startPage();
    }

    private void drawFooter() {
        if (canvas == null) return;
        TextPaint paint = new TextPaint(Paint.ANTI_ALIAS_FLAG);
        paint.setTypeface(Typeface.SANS_SERIF);
        paint.setTextSize(FOOTER_SIZE_PT);
        paint.setColor(FOOTER_COLOR);
        paint.setTextAlign(Paint.Align.CENTER);
        canvas.drawText("Page " + pageNumber, A4_WIDTH_PT / 2f, A4_HEIGHT_PT - MARGIN_PT / 2f, paint);
    }

    private float[] resolveWidths(float[] weights) {
        float total = 0f;
        for (float w : weights) total += w;
        float[] widths = new float[weights.length];
        for (int i = 0; i < weights.length; i++) {
            widths[i] = contentWidth * weights[i] / total;
        }
        return widths;
    }

    private float rowHeight(Row row, float[] columnWidths) {
        float tallest = 0f;
        for (int i = 0; i < row.cells.length; i++) {
            StaticLayout layout = layout(row.cells[i], row.styleFor(i),
                    columnWidths[i] - 2 * CELL_PADDING_PT);
            tallest = Math.max(tallest, layout.getHeight());
        }
        return tallest + 2 * CELL_PADDING_PT;
    }

    private void drawRow(Row row, float[] columnWidths, boolean header) {
        float height = rowHeight(row, columnWidths);
        if (!fits(height)) newPage();
        if (canvas == null) return;

        float x = MARGIN_PT;
        if (header) {
            Paint fill = new Paint(Paint.ANTI_ALIAS_FLAG);
            fill.setColor(HEADER_FILL);
            canvas.drawRect(MARGIN_PT, cursorY, MARGIN_PT + contentWidth, cursorY + height, fill);
        }

        Paint border = new Paint(Paint.ANTI_ALIAS_FLAG);
        border.setStyle(Paint.Style.STROKE);
        border.setStrokeWidth(TABLE_BORDER_PT);
        border.setColor(BORDER_COLOR);

        for (int i = 0; i < row.cells.length; i++) {
            canvas.drawRect(x, cursorY, x + columnWidths[i], cursorY + height, border);
            StaticLayout layout = layout(row.cells[i], row.styleFor(i),
                    columnWidths[i] - 2 * CELL_PADDING_PT);
            draw(layout, x + CELL_PADDING_PT, cursorY + CELL_PADDING_PT);
            x += columnWidths[i];
        }
        cursorY += height;
    }

    private StaticLayout layout(String content, Style style, float width) {
        TextPaint paint = new TextPaint(Paint.ANTI_ALIAS_FLAG);
        paint.setTypeface(style.bold
                ? Typeface.create(Typeface.SANS_SERIF, Typeface.BOLD)
                : Typeface.SANS_SERIF);
        paint.setTextSize(style.sizePt);
        paint.setColor(style.color);
        // A zero or negative width throws; a column narrower than its own padding is a caller
        // bug, but clamping keeps it from taking the whole report down with it.
        int safeWidth = Math.max(1, Math.round(width));
        return StaticLayout.Builder
                .obtain(content == null ? "" : content, 0,
                        content == null ? 0 : content.length(), paint, safeWidth)
                .setAlignment(style.alignment)
                .setIncludePad(false)
                .build();
    }

    private void draw(StaticLayout layout, float x, float y) {
        if (canvas == null) return;
        canvas.save();
        canvas.translate(x, y);
        layout.draw(canvas);
        canvas.restore();
    }
}
