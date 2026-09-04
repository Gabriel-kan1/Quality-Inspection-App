package com.example.report;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;

import androidx.annotation.NonNull;

import com.example.data.entity.FeatureResult;
import com.example.data.entity.InspectionRecord;
import com.example.data.entity.SlotResult;

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

        @NonNull
        @Override
        public Canvas beginPage(int pageNumber, int widthPt, int heightPt) {
            Bitmap bitmap = Bitmap.createBitmap(widthPt, heightPt, Bitmap.Config.ARGB_8888);
            bitmap.eraseColor(Color.WHITE);
            pages.add(bitmap);
            return new Canvas(bitmap);
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

    private static BitmapSink render(List<FeatureResult> features,
                                     List<SlotResult> slots,
                                     List<AdvisoryNotes.Note> notes) {
        BitmapSink sink = new BitmapSink();
        PdfReportWriter w = new PdfReportWriter(sink);
        InspectionReportPdf.compose(w, record(), features, slots, notes);
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
}
