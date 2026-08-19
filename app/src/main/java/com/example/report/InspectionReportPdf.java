package com.example.report;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.example.data.entity.FeatureResult;
import com.example.data.entity.InspectionRecord;
import com.example.data.entity.SlotResult;

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

            int missing = 0, wrong = 0, displaced = 0;
            for (SlotResult s : slotResults) {
                if ("MISSING".equals(s.getStatus())) missing++;
                else if ("WRONG_TYPE".equals(s.getStatus())) wrong++;
                else if ("DISPLACED".equals(s.getStatus())) displaced++;
            }
            w.text(String.format(Locale.getDefault(),
                    "%d of %d expected positions verified, %d missing, %d wrong type,"
                            + " %d out of position. Positions are given as a region of the"
                            + " part with coordinates relative to its own extent.",
                    slotResults.size() - missing - wrong - displaced, slotResults.size(),
                    missing, wrong, displaced), PdfReportWriter.BODY, 0f, 10f);

            List<PdfReportWriter.Row> slotRows = new ArrayList<>();
            for (SlotResult s : slotResults) {
                boolean defect = !"PRESENT".equals(s.getStatus());
                slotRows.add(new PdfReportWriter.Row(new String[]{
                        "#" + s.getSlotId(),
                        s.getExpectedType(),
                        String.format(Locale.getDefault(), "%s (%.0f%%, %.0f%%)",
                                s.getZone(), s.getRelX() * 100, s.getRelY() * 100),
                        slotResultText(s)
                }, PdfReportWriter.BODY)
                        .align(0, Layout.Alignment.ALIGN_CENTER)
                        .style(1, PdfReportWriter.BODY_BOLD)
                        .style(3, defect ? PdfReportWriter.BODY_BOLD : PdfReportWriter.BODY)
                        .align(3, Layout.Alignment.ALIGN_CENTER));
            }
            w.table(new String[]{"Slot", "Expected Feature", "Position", "Result"},
                    new float[]{0.6f, 1.8f, 1.8f, 1.4f}, slotRows, 0f);
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

        w.text("OFFLINE AUTOMATION SYSTEM LOG GENERATOR SEAL\n"
                        + "Factory QC Manager digital signature valid.",
                PdfReportWriter.BODY.withAlignment(Layout.Alignment.ALIGN_OPPOSITE), 40f, 0f);
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
