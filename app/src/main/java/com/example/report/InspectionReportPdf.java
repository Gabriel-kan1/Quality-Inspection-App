package com.example.report;

import android.graphics.Bitmap;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.example.data.entity.FeatureResult;
import com.example.data.entity.InspectionRecord;
import com.example.data.entity.SlotResult;
import com.example.ml.FeatureLocalisation;

import android.text.Layout;

import java.io.IOException;
import java.io.OutputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Locale;

/**
 * Lays out the QC inspection report.
 *
 * <p>Separate from the fragment that triggers it so the document can be composed, and asserted
 * on, without a UI: the report is the artefact that leaves the building, and it was previously
 * only reachable by driving the screen.
 */
public final class InspectionReportPdf {

    private static final String TAG = "InspectionReportPdf";

    /**
     * Printed size of the traceability QR, in points. 90pt is about 31mm, which at this
     * payload's symbol size leaves a module comfortably wider than the ~0.5mm a tablet camera
     * needs to resolve.
     */
    private static final float QR_SIZE_PT = 90f;

    /** Pixels the QR is rasterised at before being scaled down into {@link #QR_SIZE_PT}. */
    private static final int QR_RENDER_PX = 360;

    private InspectionReportPdf() { }

    /**
     * Composes the report and writes it to {@code out}. The stream is not closed here; the
     * caller owns it.
     */
    public static void write(@NonNull OutputStream out,
                             @NonNull InspectionRecord record,
                             @Nullable List<FeatureResult> features,
                             @Nullable List<SlotResult> slotResults,
                             @Nullable Collection<AdvisoryNotes.Note> warningNotes)
            throws IOException {
        PdfReportWriter w = new PdfReportWriter(new PdfDocumentSink());
        try {
            compose(w, record, features, slotResults, warningNotes);
            w.writeTo(out);
        } catch (RuntimeException | IOException e) {
            w.discard();
            throw e;
        }
    }

    /**
     * Lays the report out into an already-constructed writer. Split from {@link #write} so tests
     * can compose against a sink that does not need the framework PDF backend.
     */
    static void compose(PdfReportWriter w,
                                InspectionRecord record,
                                @Nullable List<FeatureResult> features,
                                @Nullable List<SlotResult> slotResults,
                                @Nullable Collection<AdvisoryNotes.Note> warningNotes) {
        w.text("CHASSIS QUALITY CONTROL INSPECTION REPORT", PdfReportWriter.TITLE, 0f, 24f);

        // ── Metadata ────────────────────────────────────────────────────────
        w.text("INSPECTION METADATA", PdfReportWriter.SECTION);
        w.text("PART SKU ID: " + record.getPartId(), PdfReportWriter.BODY_BOLD, 10f, 0f);
        w.text("TIMESTAMP: " + new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())
                .format(new Date(record.getTimestamp())), PdfReportWriter.BODY);
        w.text("INSPECTION RESULT STATUS: " + record.getPassOrFail(), PdfReportWriter.PASS);
        String notes = record.getNotes() != null && !record.getNotes().isEmpty()
                ? record.getNotes() : "No supervision notes registered.";
        w.text("OPERATOR NOTES: " + notes, PdfReportWriter.BODY, 0f, 20f);

        // ── Counts, per feature type ────────────────────────────────────────
        List<PdfReportWriter.Row> countRows = new ArrayList<>();
        if (features != null) {
            for (FeatureResult f : features) {
                countRows.add(new PdfReportWriter.Row(new String[]{
                        f.getFeatureType(),
                        String.valueOf(f.getExpectedCount()),
                        String.valueOf(f.getDetectedCount()),
                        f.getStatus()
                }, PdfReportWriter.BODY)
                        .style(0, PdfReportWriter.BODY_BOLD)
                        .align(1, Layout.Alignment.ALIGN_CENTER)
                        .align(2, Layout.Alignment.ALIGN_CENTER)
                        .align(3, Layout.Alignment.ALIGN_CENTER));
            }
        }
        w.table(new String[]{"Feature Type", "Expected Count", "Detected Count", "Status"},
                new float[]{2f, 1f, 1f, 1.2f}, countRows, 15f);

        // ── Localisation, per physical position ─────────────────────────────
        // A separate table from the counts above because the grain differs — that one is per
        // type, this one per physical position.
        if (slotResults != null && !slotResults.isEmpty()) {
            w.text("CRITICAL FEATURE LOCALISATION", PdfReportWriter.SECTION, 24f, 4f);

            int missing = 0, wrong = 0, displaced = 0, sized = 0, sizeChanged = 0;
            for (SlotResult s : slotResults) {
                if ("MISSING".equals(s.getStatus())) missing++;
                else if ("WRONG_TYPE".equals(s.getStatus())) wrong++;
                else if ("DISPLACED".equals(s.getStatus())) displaced++;
                if (s.getSizeRatio() != null) {
                    sized++;
                    if (sizeChanged(s.getSizeRatio())) sizeChanged++;
                }
            }
            w.text(String.format(Locale.getDefault(),
                    "%d of %d expected positions verified, %d missing, %d wrong type,"
                            + " %d out of position. Positions are given as a region of the"
                            + " part with coordinates relative to its own extent.",
                    slotResults.size() - missing - wrong - displaced, slotResults.size(),
                    missing, wrong, displaced), PdfReportWriter.BODY, 0f, 4f);
            // The size column is a comparison with the reference photograph, not a
            // measurement, and the report has to say so: a reader who takes "+48%" for a
            // dimension will look for the millimetres and not find them.
            if (sized > 0) {
                w.text(String.format(Locale.getDefault(),
                        "Size is the detected feature's extent relative to the reference"
                                + " photograph, corrected for framing: %d of %d positions"
                                + " compared, %d beyond the %d%% tolerance. It is a pixel"
                                + " comparison, not a dimension, and does not affect the"
                                + " verdict.",
                        sized, slotResults.size(), sizeChanged,
                        Math.round(FeatureLocalisation.SIZE_TOLERANCE * 100)),
                        PdfReportWriter.BODY, 0f, 10f);
            } else {
                w.text("Size against the reference was not compared for this inspection.",
                        PdfReportWriter.BODY, 0f, 10f);
            }

            List<PdfReportWriter.Row> slotRows = new ArrayList<>();
            for (SlotResult s : slotResults) {
                boolean defect = !"PRESENT".equals(s.getStatus());
                boolean sizeOut = s.getSizeRatio() != null && sizeChanged(s.getSizeRatio());
                slotRows.add(new PdfReportWriter.Row(new String[]{
                        "#" + s.getSlotId(),
                        s.getExpectedType(),
                        String.format(Locale.getDefault(), "%s (%.0f%%, %.0f%%)",
                                s.getZone(), s.getRelX() * 100, s.getRelY() * 100),
                        slotResultText(s),
                        sizeText(s)
                }, PdfReportWriter.BODY)
                        .align(0, Layout.Alignment.ALIGN_CENTER)
                        .style(1, PdfReportWriter.BODY_BOLD)
                        .style(3, defect ? PdfReportWriter.BODY_BOLD : PdfReportWriter.BODY)
                        .align(3, Layout.Alignment.ALIGN_CENTER)
                        .style(4, sizeOut ? PdfReportWriter.BODY_BOLD : PdfReportWriter.BODY)
                        .align(4, Layout.Alignment.ALIGN_CENTER));
            }
            w.table(new String[]{"Slot", "Expected Feature", "Position", "Result", "Size vs Ref"},
                    new float[]{0.6f, 1.6f, 1.7f, 1.3f, 0.9f}, slotRows, 0f);
        }

        // ── Advisory notes ──────────────────────────────────────────────────
        // Prose rather than table rows deliberately: putting a scratch count beside an
        // expected/detected bolt count invites reading it as a defect tally, which is the
        // confusion the severity scale exists to remove.
        if (warningNotes != null && !warningNotes.isEmpty()) {
            w.text("WARNING NOTES (NON-CRITICAL)", PdfReportWriter.SECTION, 24f, 4f);
            w.text("Tracked as trends against this part's inspection history. These findings do"
                            + " not affect the pass/fail verdict above.",
                    PdfReportWriter.BODY, 0f, 10f);

            List<AdvisoryNotes.Note> ordered = new ArrayList<>(warningNotes);
            Collections.sort(ordered, (a, b) -> a.level.ordinal() - b.level.ordinal());
            for (AdvisoryNotes.Note note : ordered) {
                w.text("- " + note.text,
                        note.level == AdvisoryNotes.Level.ELEVATED
                                ? PdfReportWriter.BODY_BOLD : PdfReportWriter.BODY,
                        0f, 4f);
            }
        }

        traceability(w, record);
    }

    /**
     * The block that lets a printed report be traced back to the record it came from.
     *
     * <p>This replaces a footer that read "Factory QC Manager digital signature valid." on
     * every report the app had ever produced. Nothing was signed and no manager was involved;
     * it was decoration asserting an assurance that did not exist, which on a quality document
     * is worse than a blank space. What is printed now is checkable: an identifier that names
     * one row, and a digest state that either says the record still matches what was filed or
     * says plainly that it was never sealed.
     */
    private static void traceability(PdfReportWriter w, InspectionRecord record) {
        w.text("TRACEABILITY", PdfReportWriter.SECTION, 30f, 4f);

        String uuid = record.getUuid();
        w.text("RECORD ID: " + (uuid == null ? "not assigned" : uuid),
                PdfReportWriter.BODY_BOLD);
        w.text("OPERATOR: " + (record.getOperator() == null || record.getOperator().isEmpty()
                        ? "not recorded" : record.getOperator()),
                PdfReportWriter.BODY);

        if (record.getDigest() != null) {
            w.text("INTEGRITY: SEALED (SHA-256 " + record.getDigest() + ")",
                    PdfReportWriter.BODY, 0f, 8f);
        } else {
            // Said out loud rather than left as an absence. A reader comparing two reports has
            // to be able to tell "this one is verifiable" from "this one predates verification".
            w.text("INTEGRITY: NOT SEALED. This inspection was filed before integrity sealing"
                            + " was added and carries no digest to check against.",
                    PdfReportWriter.BODY, 0f, 8f);
        }

        if (uuid == null) return;

        String payload;
        Bitmap qr;
        try {
            payload = ReportCode.encode(record);
            qr = ReportCode.toQrBitmap(payload, QR_RENDER_PX);
        } catch (Exception e) {
            // A report without its code is still a valid report. Losing the whole document
            // because the encoder objected would trade something useful for nothing.
            Log.w(TAG, "QR generation failed; report continues without it", e);
            return;
        }

        w.image(qr, QR_SIZE_PT, Layout.Alignment.ALIGN_NORMAL, 0f, 4f);
        w.text(payload, PdfReportWriter.CAPTION, 0f, 4f);
        // Says where the code works, not just what it does. It identifies a record rather
        // than containing one, so it only resolves on a device that holds this inspection —
        // and a caption that did not say so would send someone scanning a report on the wrong
        // tablet and reading the empty result as a fault.
        w.text("Scan from History > SCAN on the tablet that filed this inspection to open it"
                        + " again. The code carries the record id, the verdict and the first"
                        + " eight bytes of the digest — not the findings themselves.",
                PdfReportWriter.CAPTION, 0f, 0f);
    }

    /** Whether a stored ratio is outside {@link FeatureLocalisation#SIZE_TOLERANCE}, either way. */
    private static boolean sizeChanged(float ratio) {
        return ratio > 0 && Math.abs(Math.log(ratio))
                > Math.log1p(FeatureLocalisation.SIZE_TOLERANCE);
    }

    /** The size column: a signed percentage against the reference, or a dash when unjudged. */
    private static String sizeText(SlotResult s) {
        Float ratio = s.getSizeRatio();
        if (ratio == null || !(ratio > 0)) return "—";
        return String.format(Locale.getDefault(), "%+.0f%%", (ratio - 1f) * 100f);
    }

    private static String slotResultText(SlotResult s) {
        if ("MISSING".equals(s.getStatus())) {
            return "MISSING";
        } else if ("WRONG_TYPE".equals(s.getStatus())) {
            return "WRONG TYPE: " + s.getFoundType();
        } else if ("DISPLACED".equals(s.getStatus())) {
            return String.format(Locale.getDefault(), "%s, %.0f%% OUT OF POSITION",
                    s.getFoundType(),
                    s.getOffsetPct() == null ? 0f : s.getOffsetPct() * 100);
        }
        return "PRESENT";
    }
}
