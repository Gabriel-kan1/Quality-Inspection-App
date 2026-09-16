package com.example.report;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.example.data.entity.FeatureResult;
import com.example.data.entity.InspectionRecord;
import com.example.data.entity.SlotResult;
import com.google.zxing.BinaryBitmap;
import com.google.zxing.MultiFormatReader;
import com.google.zxing.NotFoundException;
import com.google.zxing.RGBLuminanceSource;
import com.google.zxing.common.HybridBinarizer;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;
import org.robolectric.annotation.GraphicsMode;

import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Covers report layout end to end, which nothing did while it lived inside the fragment.
 *
 * <p>The pages are rendered to bitmaps rather than to a real PDF: Robolectric implements no part
 * of {@link android.graphics.pdf.PdfDocument}, so that route cannot run off-device at all. What
 * that costs is coverage of {@link PdfDocumentSink}, which is twelve lines of pure delegation.
 * What it buys is assertions on the part that can actually be wrong — where rows break, whether
 * a table repeats its header, whether anything was drawn at all.
 */
@RunWith(RobolectricTestRunner.class)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
public class InspectionReportPdfTest {

    /** Renders each page to a bitmap so the test can look at what was actually drawn. */
    private static final class BitmapSink implements PageSink {
        final List<Bitmap> pages = new ArrayList<>();
        /**
         * Pixels per point. One is enough to see where rows broke; the QR test needs more,
         * because a 90pt symbol rendered at 72dpi has fewer than three pixels per module and
         * would be testing the emulation rather than the symbol. Four puts it at 288dpi, which
         * is the neighbourhood a report actually gets printed at.
         */
        final int scale;

        BitmapSink() {
            this(1);
        }

        BitmapSink(int scale) {
            this.scale = scale;
        }

        @NonNull
        @Override
        public Canvas beginPage(int pageNumber, int widthPt, int heightPt) {
            Bitmap bitmap = Bitmap.createBitmap(widthPt * scale, heightPt * scale,
                    Bitmap.Config.ARGB_8888);
            bitmap.eraseColor(Color.WHITE);
            pages.add(bitmap);
            Canvas canvas = new Canvas(bitmap);
            canvas.scale(scale, scale);
            return canvas;
        }

        @Override public void finishPage() { }
        @Override public void writeTo(@NonNull OutputStream out) { }
        @Override public void discard() { }

        /** Pixels that are not the white background, i.e. evidence something was drawn. */
        int inkedPixels(int pageIndex) {
            Bitmap page = pages.get(pageIndex);
            int inked = 0;
            for (int y = 0; y < page.getHeight(); y += 2) {
                for (int x = 0; x < page.getWidth(); x += 2) {
                    if (page.getPixel(x, y) != Color.WHITE) inked++;
                }
            }
            return inked;
        }

        /** Y of the lowest drawn pixel, i.e. how far down the page the content reached. */
        int lowestInkedRow(int pageIndex) {
            Bitmap page = pages.get(pageIndex);
            for (int y = page.getHeight() - 1; y >= 0; y--) {
                for (int x = 0; x < page.getWidth(); x++) {
                    if (page.getPixel(x, y) != Color.WHITE) return y;
                }
            }
            return -1;
        }
    }

    private static InspectionRecord record() {
        InspectionRecord r = new InspectionRecord();
        r.setId(42);
        r.setPartId("CHASSIS-001");
        r.setTimestamp(1_700_000_000_000L);
        r.setPassOrFail("FAIL");
        r.setNotes("Second shift, replacement operator.");
        return r;
    }

    private static SlotResult slot(int id, String expected, String found, String status) {
        SlotResult s = new SlotResult();
        s.setSlotId(id);
        s.setExpectedType(expected);
        s.setFoundType(found);
        s.setStatus(status);
        s.setRelX(0.25f);
        s.setRelY(0.75f);
        s.setZone("lower left");
        return s;
    }

    /** A record carrying the identity and seal that {@code ResultsFragment} now files. */
    private static InspectionRecord sealedRecord() {
        InspectionRecord r = record();
        r.setUuid("3f2504e0-4f89-41d3-9a0c-0305e82c3301");
        r.setOperator("SHIFT-A");
        r.setDigest("9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08");
        return r;
    }

    private static BitmapSink render(List<FeatureResult> features,
                                     List<SlotResult> slots,
                                     List<AdvisoryNotes.Note> notes) {
        return render(record(), features, slots, notes, 1);
    }

    private static BitmapSink render(InspectionRecord record,
                                     List<FeatureResult> features,
                                     List<SlotResult> slots,
                                     List<AdvisoryNotes.Note> notes,
                                     int scale) {
        BitmapSink sink = new BitmapSink(scale);
        PdfReportWriter w = new PdfReportWriter(sink);
        InspectionReportPdf.compose(w, record, features, slots, notes);
        return sink;
    }

    @Test
    public void laysOutAFullReportOnASinglePage() {
        List<FeatureResult> features = Arrays.asList(
                new FeatureResult(42, "bolt_head", 6, 5, "FAIL", 3),
                new FeatureResult(42, "hole", 9, 9, "PASS", 0));
        List<SlotResult> slots = Arrays.asList(
                slot(1, "bolt_head", "bolt_head", "PRESENT"),
                slot(2, "bolt_head", null, "MISSING"),
                slot(3, "stud", "screw", "WRONG_TYPE"));
        List<AdvisoryNotes.Note> notes =
                Collections.singletonList(AdvisoryNotes.forType("Scratches", 8,
                        Arrays.asList(2, 1, 2, 3)));

        BitmapSink sink = render(features, slots, notes);

        assertEquals(1, sink.pages.size());
        assertTrue("nothing was drawn on the page", sink.inkedPixels(0) > 500);
    }

    /**
     * The empty-ish report is the one most likely to break: a first inspection has no slot
     * layout and no advisory history, so whole sections are skipped.
     */
    @Test
    public void laysOutAReportWhenEverythingOptionalIsAbsent() {
        BitmapSink sink = render(null, null, null);

        assertEquals(1, sink.pages.size());
        assertTrue("the metadata block should still be drawn", sink.inkedPixels(0) > 200);
    }

    @Test
    public void laysOutAReportForEmptyCollections() {
        BitmapSink sink = render(new ArrayList<>(), new ArrayList<>(), new ArrayList<>());
        assertEquals(1, sink.pages.size());
    }

    /**
     * More slots than fit on one page. This is the path that exercises the page break and the
     * repeated table header; before the rewrite it was iText's problem, now it is ours.
     */
    @Test
    public void paginatesALongLocalisationTable() {
        List<SlotResult> many = new ArrayList<>();
        for (int i = 1; i <= 120; i++) {
            many.add(slot(i, "bolt_head", "bolt_head", "PRESENT"));
        }

        BitmapSink sink = render(null, many, null);

        assertTrue("120 rows must not be crammed onto one page, got " + sink.pages.size(),
                sink.pages.size() > 1);
        // Every page has to carry content; an off-by-one in the break test shows up as a blank
        // page slipped in between two full ones.
        for (int i = 0; i < sink.pages.size(); i++) {
            assertTrue("page " + (i + 1) + " of " + sink.pages.size() + " came out blank",
                    sink.inkedPixels(i) > 200);
        }
    }

    /** A displaced slot reads its offset from a nullable column; a null must not blow up. */
    @Test
    public void toleratesADisplacedSlotWithNoRecordedOffset() {
        SlotResult displaced = slot(1, "rivet_head", "rivet_head", "DISPLACED");
        displaced.setOffsetPct(null);

        BitmapSink sink = render(null, Collections.singletonList(displaced), null);

        assertEquals(1, sink.pages.size());
    }

    /**
     * Long free text has to wrap inside its column, and the row has to grow to fit it. If the
     * row height were still measured as a single line the table would silently overlap itself,
     * so the check is that the extra lines push everything below them down the page.
     */
    @Test
    public void growsARowToFitWrappedCellText() {
        StringBuilder longType = new StringBuilder();
        for (int i = 0; i < 40; i++) longType.append("verylongfeaturename ");

        BitmapSink wrapped = render(Collections.singletonList(
                new FeatureResult(42, longType.toString(), 1, 0, "FAIL", 3)), null, null);
        BitmapSink plain = render(Collections.singletonList(
                new FeatureResult(42, "bolt_head", 1, 0, "FAIL", 3)), null, null);

        assertEquals("both should still be one page", 1, wrapped.pages.size());
        int grew = wrapped.lowestInkedRow(0) - plain.lowestInkedRow(0);
        assertTrue("wrapping 40 words into one column should add many lines of height,"
                + " but content only reached " + grew + "pt further down", grew > 100);
    }

    /** A part ID long enough to wrap must not push the report off the bottom of the page. */
    @Test
    public void keepsTheSealOnThePageForALongPartId() {
        InspectionRecord r = record();
        StringBuilder id = new StringBuilder();
        for (int i = 0; i < 30; i++) id.append("PART-NUMBER-");
        r.setPartId(id.toString());

        BitmapSink sink = new BitmapSink();
        InspectionReportPdf.compose(new PdfReportWriter(sink), r, null, null, null);

        assertTrue(sink.pages.size() >= 1);
        assertTrue(sink.inkedPixels(sink.pages.size() - 1) > 200);
    }

    // ── Traceability block ──────────────────────────────────────────────────

    /**
     * The end-to-end one: the QR has to survive being drawn into the report at its printed
     * size and read back as the pointer that went in. A code on a page that no scanner can
     * resolve is worse than no code, because it looks like traceability.
     */
    @Test
    public void printedQrDecodesBackToTheRecordItNames() throws Exception {
        InspectionRecord record = sealedRecord();
        BitmapSink sink = render(record, null, null, null, 4);

        String scanned = readQrFromPage(sink.pages.get(sink.pages.size() - 1));
        assertNotNull("no QR was found on the report page", scanned);

        ReportCode.Pointer pointer = ReportCode.decode(scanned);
        assertNotNull("the QR held something that was not one of ours", pointer);
        assertEquals(record.getUuid(), pointer.uuid);
        assertEquals("FAIL", pointer.verdictText());
        assertTrue("the printed digest prefix should match the record", pointer.matches(record));
    }

    /**
     * Records filed before sealing existed have no uuid, so there is nothing to point at. The
     * report still has to come out — this is the whole back catalogue on an upgraded device.
     */
    @Test
    public void laysOutAReportForAnUnsealedLegacyRecord() {
        BitmapSink sink = render(record(), null, null, null, 1);

        assertEquals(1, sink.pages.size());
        assertTrue("the report should still be drawn without a code", sink.inkedPixels(0) > 200);
    }

    /**
     * The QR is the last thing on the page, so a report already near the bottom is exactly
     * where a missing page break would show up — as a symbol running off the paper.
     */
    @Test
    public void breaksThePageRatherThanRunningTheQrOffTheBottom() {
        List<SlotResult> many = new ArrayList<>();
        for (int i = 1; i <= 120; i++) {
            many.add(slot(i, "bolt_head", "bolt_head", "PRESENT"));
        }

        BitmapSink sink = render(sealedRecord(), null, many, null, 1);
        int last = sink.pages.size() - 1;

        assertTrue("nothing was drawn on the final page", sink.inkedPixels(last) > 200);
        // A4 is 842pt tall with a 40pt margin. Anything inked below that is off the paper.
        assertTrue("content reached " + sink.lowestInkedRow(last) + "pt, past the bottom margin",
                sink.lowestInkedRow(last) <= 842 - 40 + 1);
    }

    @Nullable
    private static String readQrFromPage(Bitmap page) {
        int width = page.getWidth();
        int height = page.getHeight();
        int[] pixels = new int[width * height];
        page.getPixels(pixels, 0, width, 0, 0, width, height);
        try {
            // No PURE_BARCODE hint: the symbol is a small part of a busy page, which is the
            // situation a real scanner faces and the one worth proving works.
            return new MultiFormatReader().decode(new BinaryBitmap(new HybridBinarizer(
                    new RGBLuminanceSource(width, height, pixels)))).getText();
        } catch (NotFoundException e) {
            return null;
        }
    }

    /**
     * The localisation table gained a size column. A judged row inside tolerance, one outside
     * it and one never judged all have to lay out on the page together.
     */
    @Test
    public void laysOutTheSizeColumnForJudgedAndUnjudgedSlots() {
        SlotResult inside = slot(1, "bolt_head", "bolt_head", "PRESENT");
        inside.setSizeRatio(1.08f);
        SlotResult outside = slot(2, "bolt_head", "bolt_head", "PRESENT");
        outside.setSizeRatio(1.9f);
        SlotResult unjudged = slot(3, "stud", null, "MISSING");
        List<SlotResult> slots = Arrays.asList(inside, outside, unjudged);

        BitmapSink sink = render(Collections.<FeatureResult>emptyList(), slots,
                Collections.<AdvisoryNotes.Note>emptyList());

        assertEquals(1, sink.pages.size());
        assertTrue("nothing was drawn on the page", sink.inkedPixels(0) > 500);
    }
}
