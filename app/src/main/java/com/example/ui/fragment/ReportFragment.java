package com.example.ui.fragment;

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;
import androidx.fragment.app.Fragment;
import androidx.navigation.Navigation;

import com.example.R;
import com.example.data.AppDatabase;
import com.example.data.entity.FeatureResult;
import com.example.data.entity.InspectionRecord;
import com.example.data.entity.SlotResult;
import com.example.databinding.FragmentReportBinding;
import com.example.ml.FeatureTypeRegistry;
import com.example.report.AdvisoryNotes;

import com.itextpdf.text.BaseColor;
import com.itextpdf.text.Document;
import com.itextpdf.text.Element;
import com.itextpdf.text.Font;
import com.itextpdf.text.Paragraph;
import com.itextpdf.text.Phrase;
import com.itextpdf.text.pdf.PdfPCell;
import com.itextpdf.text.pdf.PdfPTable;
import com.itextpdf.text.pdf.PdfWriter;

import java.io.File;
import java.io.FileOutputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public class ReportFragment extends Fragment {

    private FragmentReportBinding binding;
    private long inspectionId = -1;
    private InspectionRecord record = null;
    /** Critical-severity results only — the quantitative table. */
    private List<FeatureResult> features = null;
    /** Advisory-severity results, rendered as trend notes instead of counted rows. */
    private List<AdvisoryNotes.Note> warningNotes = null;
    /** Per-position findings for critical features. Empty when no positional check was possible. */
    private List<SlotResult> slotResults = null;
    private File generatedPdfFile = null;

    @Override
    public void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        if (getArguments() != null) {
            inspectionId = getArguments().getLong("inspectionId", -1);
        }
    }

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        binding = FragmentReportBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        binding.btnReportDone.setOnClickListener(v -> {
            Navigation.findNavController(v).navigate(R.id.action_report_to_home);
        });

        binding.btnExportPdf.setOnClickListener(v -> generatePdfReport(true));

        binding.btnShareReport.setOnClickListener(v -> sharePdfReport());

        loadReportDetails();
    }

    private void loadReportDetails() {
        if (inspectionId == -1) return;

        new Thread(() -> {
            AppDatabase db = AppDatabase.getInstance(requireContext());
            record = db.inspectionDao().getInspectionById(inspectionId);

            // Each result is routed by the severity it was filed with (FeatureTypeRegistry.Severity,
            // chosen by the operator when the type was labelled):
            //   CRITICAL -> the quantitative table below
            //   ADVISORY -> warning notes, as a trend against this part's own history
            //   IGNORED  -> nothing; tracked in the DB for learning, absent from the report
            slotResults = db.inspectionDao().getSlotResultsForInspection(inspectionId);

            List<FeatureResult> all = db.inspectionDao().getFeatureResultsForInspection(inspectionId);
            features = new java.util.ArrayList<>();
            warningNotes = new java.util.ArrayList<>();
            for (FeatureResult f : all) {
                FeatureTypeRegistry.Severity severity =
                        FeatureTypeRegistry.Severity.fromLevel(f.getSeverity());
                if (severity == FeatureTypeRegistry.Severity.CRITICAL) {
                    features.add(f);
                } else if (severity == FeatureTypeRegistry.Severity.ADVISORY) {
                    List<Integer> history = db.inspectionDao().getHistoricalDetectedCounts(
                            record == null ? "" : record.getPartId(), f.getFeatureType(),
                            inspectionId, AdvisoryNotes.HISTORY_WINDOW);
                    warningNotes.add(AdvisoryNotes.forType(
                            f.getFeatureType(), f.getDetectedCount(), history));
                }
            }

            requireActivity().runOnUiThread(() -> {
                if (record != null) {
                    binding.tvReportNum.setText("Report ID: QC-RECORD-" + record.getId());
                    binding.tvReportPartId.setText(record.getPartId());

                    SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault());
                    String dateStr = sdf.format(new Date(record.getTimestamp()));
                    binding.tvReportTimestamp.setText(dateStr);

                    binding.tvReportNotesText.setText((record.getNotes() == null || record.getNotes().isEmpty())
                            ? "No additional supervisor notes recorded."
                            : record.getNotes());

                    // Verdict handling
                    if ("PASS".equalsIgnoreCase(record.getPassOrFail())) {
                        binding.tvReportVerdict.setText("PASS");
                        binding.tvReportVerdict.setBackgroundColor(ContextCompat.getColor(requireContext(), R.color.green_pass));
                    } else {
                        binding.tvReportVerdict.setText("FAIL");
                        binding.tvReportVerdict.setBackgroundColor(ContextCompat.getColor(requireContext(), R.color.red_fail));
                    }

                    // Dynamically build features table
                    buildFeatureRows();
                    buildLocalisationRows();
                    buildWarningNotes();
                }
            });
        }).start();
    }

    private void buildFeatureRows() {
        binding.layoutReportFeaturesContainer.removeAllViews();
        if (features == null || features.isEmpty()) {
            TextView tvEmpty = new TextView(getContext());
            tvEmpty.setText("No features recorded.");
            tvEmpty.setTextColor(ContextCompat.getColor(requireContext(), R.color.gray_text_secondary));
            tvEmpty.setTextSize(14f);
            binding.layoutReportFeaturesContainer.addView(tvEmpty);
            return;
        }

        for (FeatureResult f : features) {
            LinearLayout row = new LinearLayout(getContext());
            row.setOrientation(LinearLayout.HORIZONTAL);
            row.setLayoutParams(new LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT));
            row.setPadding(0, 16, 0, 16);

            // Feature Type info
            TextView tvType = new TextView(getContext());
            LinearLayout.LayoutParams lpType = new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1.5f);
            tvType.setLayoutParams(lpType);
            tvType.setText(f.getFeatureType());
            tvType.setTextColor(ContextCompat.getColor(requireContext(), R.color.white));
            tvType.setTextSize(15f);
            tvType.setTypeface(android.graphics.Typeface.defaultFromStyle(android.graphics.Typeface.BOLD));

            // Expected info
            TextView tvExpected = new TextView(getContext());
            LinearLayout.LayoutParams lpExp = new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 0.8f);
            tvExpected.setLayoutParams(lpExp);
            tvExpected.setText(String.valueOf(f.getExpectedCount()));
            tvExpected.setTextColor(ContextCompat.getColor(requireContext(), R.color.gray_text_secondary));
            tvExpected.setTextSize(14f);
            tvExpected.setGravity(Gravity.CENTER);

            // Detected info
            TextView tvDetected = new TextView(getContext());
            LinearLayout.LayoutParams lpDet = new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 0.8f);
            tvDetected.setLayoutParams(lpDet);
            tvDetected.setText(String.valueOf(f.getDetectedCount()));
            tvDetected.setTextColor(ContextCompat.getColor(requireContext(), R.color.white));
            tvDetected.setTextSize(14f);
            tvDetected.setGravity(Gravity.CENTER);

            // Status Badge info
            TextView tvStatus = new TextView(getContext());
            LinearLayout.LayoutParams lpStat = new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 0.9f);
            tvStatus.setLayoutParams(lpStat);
            tvStatus.setText(f.getStatus());
            tvStatus.setTextSize(12f);
            tvStatus.setGravity(Gravity.CENTER);
            tvStatus.setTypeface(android.graphics.Typeface.defaultFromStyle(android.graphics.Typeface.BOLD));

            if ("MATCH".equalsIgnoreCase(f.getStatus()) || "PASS".equalsIgnoreCase(f.getStatus())) {
                tvStatus.setTextColor(ContextCompat.getColor(requireContext(), R.color.green_pass));
            } else {
                tvStatus.setTextColor(ContextCompat.getColor(requireContext(), R.color.red_fail));
            }

            row.addView(tvType);
            row.addView(tvExpected);
            row.addView(tvDetected);
            row.addView(tvStatus);

            // Add Divider
            View divider = new View(getContext());
            divider.setLayoutParams(new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 1));
            divider.setBackgroundColor(ContextCompat.getColor(requireContext(), R.color.gray_text_secondary));
            divider.setAlpha(0.2f);

            binding.layoutReportFeaturesContainer.addView(row);
            binding.layoutReportFeaturesContainer.addView(divider);
        }
    }

    /**
     * Renders one row per expected critical position: which slot, what belongs there, where it
     * is on the part, and what was actually found.
     *
     * The card is hidden entirely when the inspection has no slot rows. Absent rows mean no
     * positional claim was possible — no reference layout, or a photo that could not be
     * registered onto it — and an empty table under this heading would read as "every position
     * checked out", which is the opposite of what it means.
     */
    private void buildLocalisationRows() {
        binding.layoutReportLocalisationContainer.removeAllViews();
        if (slotResults == null || slotResults.isEmpty()) {
            binding.cardReportLocalisation.setVisibility(View.GONE);
            return;
        }
        binding.cardReportLocalisation.setVisibility(View.VISIBLE);

        int missing = 0, wrong = 0;
        for (SlotResult s : slotResults) {
            if ("MISSING".equals(s.getStatus())) missing++;
            else if ("WRONG_TYPE".equals(s.getStatus())) wrong++;
        }
        binding.tvReportLocalisationSummary.setText(String.format(Locale.getDefault(),
                "%d of %d expected positions verified · %d missing · %d wrong type",
                slotResults.size() - missing - wrong, slotResults.size(), missing, wrong));

        for (SlotResult s : slotResults) {
            LinearLayout row = new LinearLayout(getContext());
            row.setOrientation(LinearLayout.HORIZONTAL);
            row.setPadding(0, 14, 0, 14);

            TextView tvSlot = new TextView(getContext());
            tvSlot.setLayoutParams(new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 0.5f));
            tvSlot.setText("#" + s.getSlotId());
            tvSlot.setTextColor(ContextCompat.getColor(requireContext(), R.color.gray_text_secondary));
            tvSlot.setTextSize(14f);

            TextView tvType = new TextView(getContext());
            tvType.setLayoutParams(new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1.6f));
            tvType.setText(s.getExpectedType());
            tvType.setTextColor(ContextCompat.getColor(requireContext(), R.color.white));
            tvType.setTextSize(14f);
            tvType.setTypeface(android.graphics.Typeface.defaultFromStyle(android.graphics.Typeface.BOLD));

            TextView tvWhere = new TextView(getContext());
            tvWhere.setLayoutParams(new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1.6f));
            tvWhere.setText(String.format(Locale.getDefault(), "%s\n(%.0f%%, %.0f%%)",
                    s.getZone(), s.getRelX() * 100, s.getRelY() * 100));
            tvWhere.setTextColor(ContextCompat.getColor(requireContext(), R.color.gray_text_secondary));
            tvWhere.setTextSize(12f);

            TextView tvStatus = new TextView(getContext());
            tvStatus.setLayoutParams(new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1.4f));
            tvStatus.setTextSize(12f);
            tvStatus.setGravity(Gravity.END);
            tvStatus.setTypeface(android.graphics.Typeface.defaultFromStyle(android.graphics.Typeface.BOLD));
            if ("MISSING".equals(s.getStatus())) {
                tvStatus.setText("MISSING");
                tvStatus.setTextColor(ContextCompat.getColor(requireContext(), R.color.red_fail));
            } else if ("WRONG_TYPE".equals(s.getStatus())) {
                tvStatus.setText("WRONG:\n" + s.getFoundType());
                tvStatus.setTextColor(ContextCompat.getColor(requireContext(), R.color.orange_warn));
            } else {
                tvStatus.setText("PRESENT");
                tvStatus.setTextColor(ContextCompat.getColor(requireContext(), R.color.green_pass));
            }

            row.addView(tvSlot);
            row.addView(tvType);
            row.addView(tvWhere);
            row.addView(tvStatus);

            View divider = new View(getContext());
            divider.setLayoutParams(new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 1));
            divider.setBackgroundColor(ContextCompat.getColor(requireContext(), R.color.gray_text_secondary));
            divider.setAlpha(0.2f);

            binding.layoutReportLocalisationContainer.addView(row);
            binding.layoutReportLocalisationContainer.addView(divider);
        }
    }

    /**
     * Renders the advisory findings as one line each, elevated ones first so the note this
     * section exists for is the first thing read. The whole card stays hidden when the
     * inspection produced no advisory types — an empty "warnings" heading reads as a
     * clean bill of health, which is not the same thing as having nothing to trend.
     */
    private void buildWarningNotes() {
        binding.layoutReportWarningsContainer.removeAllViews();
        if (warningNotes == null || warningNotes.isEmpty()) {
            binding.cardReportWarnings.setVisibility(View.GONE);
            return;
        }
        binding.cardReportWarnings.setVisibility(View.VISIBLE);

        List<AdvisoryNotes.Note> ordered = new java.util.ArrayList<>(warningNotes);
        java.util.Collections.sort(ordered, (a, b) -> a.level.ordinal() - b.level.ordinal());

        for (AdvisoryNotes.Note note : ordered) {
            LinearLayout row = new LinearLayout(getContext());
            row.setOrientation(LinearLayout.HORIZONTAL);
            row.setPadding(0, 12, 0, 12);

            // A coloured rule rather than a bullet glyph: it survives the operator's font-size
            // settings, which a fixed-size dot in a TextView does not.
            View bar = new View(getContext());
            LinearLayout.LayoutParams barLp = new LinearLayout.LayoutParams(6,
                    ViewGroup.LayoutParams.MATCH_PARENT);
            barLp.rightMargin = 14;
            bar.setLayoutParams(barLp);
            bar.setBackgroundColor(ContextCompat.getColor(requireContext(),
                    note.level == AdvisoryNotes.Level.ELEVATED
                            ? R.color.amber_advisory : R.color.gray_text_secondary));

            TextView tvNote = new TextView(getContext());
            tvNote.setLayoutParams(new LinearLayout.LayoutParams(0,
                    ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
            tvNote.setText(note.text);
            tvNote.setTextSize(13f);
            tvNote.setTextColor(ContextCompat.getColor(requireContext(),
                    note.level == AdvisoryNotes.Level.ELEVATED
                            ? R.color.white : R.color.gray_text_secondary));

            row.addView(bar);
            row.addView(tvNote);
            binding.layoutReportWarningsContainer.addView(row);
        }
    }

    private void generatePdfReport(boolean showToast) {
        if (record == null) {
            Toast.makeText(getContext(), "Report details still loading. Please hold...", Toast.LENGTH_SHORT).show();
            return;
        }

        try {
            File pdfFolder = requireContext().getExternalFilesDir("reports");
            if (pdfFolder != null && !pdfFolder.exists()) {
                pdfFolder.mkdirs();
            }

            generatedPdfFile = new File(pdfFolder, "QC_REPORT_" + record.getPartId() + "_" + record.getId() + ".pdf");
            Document document = new Document();
            PdfWriter.getInstance(document, new FileOutputStream(generatedPdfFile));

            document.open();

            // Font configurations
            Font titleFont = new Font(Font.FontFamily.HELVETICA, 20, Font.BOLD, BaseColor.DARK_GRAY);
            Font headerFont = new Font(Font.FontFamily.HELVETICA, 12, Font.BOLD, BaseColor.WHITE);
            Font bodyFont = new Font(Font.FontFamily.HELVETICA, 10, Font.NORMAL, BaseColor.BLACK);
            Font boldBodyFont = new Font(Font.FontFamily.HELVETICA, 10, Font.BOLD, BaseColor.BLACK);
            Font passFont = new Font(Font.FontFamily.HELVETICA, 12, Font.BOLD, new BaseColor(38, 166, 154));

            // Document Header
            Paragraph title = new Paragraph("CHASSIS QUALITY CONTROL INSPECTION REPORT", titleFont);
            title.setAlignment(Element.ALIGN_CENTER);
            title.setSpacingAfter(24f);
            document.add(title);

            // Metadata info list
            document.add(new Paragraph("INSPECTION METADATA", headerFont));
            Paragraph metaPart = new Paragraph("PART SKU ID: " + record.getPartId(), boldBodyFont);
            Paragraph metaDate = new Paragraph("TIMESTAMP: " + new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(new Date(record.getTimestamp())), bodyFont);
            Paragraph metaStatus = new Paragraph("INSPECTION RESULT STATUS: " + record.getPassOrFail(), passFont);
            Paragraph metaNotes = new Paragraph("OPERATOR NOTES: " + (record.getNotes() != null && !record.getNotes().isEmpty() ? record.getNotes() : "No supervision notes registered."), bodyFont);

            metaPart.setSpacingBefore(10f);
            metaNotes.setSpacingAfter(20f);

            document.add(metaPart);
            document.add(metaDate);
            document.add(metaStatus);
            document.add(metaNotes);

            // Table breakdown
            PdfPTable table = new PdfPTable(4);
            table.setWidthPercentage(100);
            table.setSpacingBefore(15f);

            float[] columnWidths = {2f, 1f, 1f, 1.2f};
            table.setWidths(columnWidths);

            // Headers cells
            PdfPCell cell1 = new PdfPCell(new Phrase("Feature Type", headerFont));
            cell1.setBackgroundColor(BaseColor.GRAY);
            cell1.setHorizontalAlignment(Element.ALIGN_CENTER);
            cell1.setPadding(8f);

            PdfPCell cell2 = new PdfPCell(new Phrase("Expected Count", headerFont));
            cell2.setBackgroundColor(BaseColor.GRAY);
            cell2.setHorizontalAlignment(Element.ALIGN_CENTER);
            cell2.setPadding(8f);

            PdfPCell cell3 = new PdfPCell(new Phrase("Detected Count", headerFont));
            cell3.setBackgroundColor(BaseColor.GRAY);
            cell3.setHorizontalAlignment(Element.ALIGN_CENTER);
            cell3.setPadding(8f);

            PdfPCell cell4 = new PdfPCell(new Phrase("Status", headerFont));
            cell4.setBackgroundColor(BaseColor.GRAY);
            cell4.setHorizontalAlignment(Element.ALIGN_CENTER);
            cell4.setPadding(8f);

            table.addCell(cell1);
            table.addCell(cell2);
            table.addCell(cell3);
            table.addCell(cell4);

            // Rows data
            if (features != null) {
                for (FeatureResult f : features) {
                    PdfPCell c1 = new PdfPCell(new Phrase(f.getFeatureType(), boldBodyFont));
                    c1.setPadding(6f);
                    PdfPCell c2 = new PdfPCell(new Phrase(String.valueOf(f.getExpectedCount()), bodyFont));
                    c2.setHorizontalAlignment(Element.ALIGN_CENTER);
                    c2.setPadding(6f);
                    PdfPCell c3 = new PdfPCell(new Phrase(String.valueOf(f.getDetectedCount()), bodyFont));
                    c3.setHorizontalAlignment(Element.ALIGN_CENTER);
                    c3.setPadding(6f);
                    PdfPCell c4 = new PdfPCell(new Phrase(f.getStatus(), bodyFont));
                    c4.setHorizontalAlignment(Element.ALIGN_CENTER);
                    c4.setPadding(6f);

                    table.addCell(c1);
                    table.addCell(c2);
                    table.addCell(c3);
                    table.addCell(c4);
                }
            }

            document.add(table);

            // Localisation: which specific expected feature is where, and which one is missing.
            // A separate table from the counts above because the grain differs — that one is per
            // type, this one per physical position.
            if (slotResults != null && !slotResults.isEmpty()) {
                Font locHeaderFont = new Font(Font.FontFamily.HELVETICA, 12, Font.BOLD, BaseColor.DARK_GRAY);
                Paragraph locHeading = new Paragraph("CRITICAL FEATURE LOCALISATION", locHeaderFont);
                locHeading.setSpacingBefore(24f);
                locHeading.setSpacingAfter(4f);
                document.add(locHeading);

                int missing = 0, wrong = 0;
                for (SlotResult s : slotResults) {
                    if ("MISSING".equals(s.getStatus())) missing++;
                    else if ("WRONG_TYPE".equals(s.getStatus())) wrong++;
                }
                Paragraph locBlurb = new Paragraph(String.format(Locale.getDefault(),
                        "%d of %d expected positions verified, %d missing, %d wrong type."
                                + " Positions are given as a region of the part with fractional"
                                + " coordinates from its top-left.",
                        slotResults.size() - missing - wrong, slotResults.size(), missing, wrong),
                        bodyFont);
                locBlurb.setSpacingAfter(10f);
                document.add(locBlurb);

                PdfPTable locTable = new PdfPTable(4);
                locTable.setWidthPercentage(100);
                locTable.setWidths(new float[]{0.6f, 1.8f, 1.8f, 1.4f});
                for (String header : new String[]{"Slot", "Expected Feature", "Position", "Result"}) {
                    PdfPCell hc = new PdfPCell(new Phrase(header, headerFont));
                    hc.setBackgroundColor(BaseColor.GRAY);
                    hc.setHorizontalAlignment(Element.ALIGN_CENTER);
                    hc.setPadding(8f);
                    locTable.addCell(hc);
                }
                for (SlotResult s : slotResults) {
                    boolean defect = !"PRESENT".equals(s.getStatus());
                    PdfPCell c1 = new PdfPCell(new Phrase("#" + s.getSlotId(), bodyFont));
                    c1.setHorizontalAlignment(Element.ALIGN_CENTER);
                    c1.setPadding(6f);
                    PdfPCell c2 = new PdfPCell(new Phrase(s.getExpectedType(), boldBodyFont));
                    c2.setPadding(6f);
                    PdfPCell c3 = new PdfPCell(new Phrase(String.format(Locale.getDefault(),
                            "%s (%.0f%%, %.0f%%)", s.getZone(), s.getRelX() * 100, s.getRelY() * 100), bodyFont));
                    c3.setPadding(6f);
                    String resultText = "MISSING".equals(s.getStatus()) ? "MISSING"
                            : ("WRONG_TYPE".equals(s.getStatus())
                                ? "WRONG TYPE: " + s.getFoundType() : "PRESENT");
                    PdfPCell c4 = new PdfPCell(new Phrase(resultText, defect ? boldBodyFont : bodyFont));
                    c4.setHorizontalAlignment(Element.ALIGN_CENTER);
                    c4.setPadding(6f);

                    locTable.addCell(c1);
                    locTable.addCell(c2);
                    locTable.addCell(c3);
                    locTable.addCell(c4);
                }
                document.add(locTable);
            }

            // Warning notes: advisory findings, as prose rather than table rows. They are
            // deliberately not columns in the table above — putting a scratch count beside an
            // expected/detected bolt count invites reading it as a defect tally, which is the
            // confusion the severity scale exists to remove.
            if (warningNotes != null && !warningNotes.isEmpty()) {
                Font warnHeaderFont = new Font(Font.FontFamily.HELVETICA, 12, Font.BOLD, BaseColor.DARK_GRAY);
                Paragraph warnHeading = new Paragraph("WARNING NOTES (NON-CRITICAL)", warnHeaderFont);
                warnHeading.setSpacingBefore(24f);
                warnHeading.setSpacingAfter(4f);
                document.add(warnHeading);

                Paragraph warnBlurb = new Paragraph(
                        "Tracked as trends against this part's inspection history. These findings do"
                                + " not affect the pass/fail verdict above.", bodyFont);
                warnBlurb.setSpacingAfter(10f);
                document.add(warnBlurb);

                List<AdvisoryNotes.Note> ordered = new java.util.ArrayList<>(warningNotes);
                java.util.Collections.sort(ordered, (a, b) -> a.level.ordinal() - b.level.ordinal());
                for (AdvisoryNotes.Note note : ordered) {
                    Font f = note.level == AdvisoryNotes.Level.ELEVATED ? boldBodyFont : bodyFont;
                    Paragraph p = new Paragraph("- " + note.text, f);
                    p.setSpacingAfter(4f);
                    document.add(p);
                }
            }

            // Seal signature section
            Paragraph seal = new Paragraph("\n\n\n\nOFFLINE AUTOMATION SYSTEM LOG GENERATOR SEAL\nFactory QC Manager digital signature valid.", bodyFont);
            seal.setAlignment(Element.ALIGN_RIGHT);
            document.add(seal);

            document.close();

            if (showToast) {
                Toast.makeText(getContext(), "Report exported to: Internal Storage/reports/QC_REPORT_" + record.getId() + ".pdf", Toast.LENGTH_LONG).show();
            }
        } catch (Exception e) {
            Toast.makeText(getContext(), "Error saving PDF: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    private void sharePdfReport() {
        if (generatedPdfFile == null || !generatedPdfFile.exists()) {
            // Unsaved generate first
            generatePdfReport(false);
        }

        if (generatedPdfFile != null && generatedPdfFile.exists()) {
            Uri pdfUri = FileProvider.getUriForFile(requireContext(), requireContext().getPackageName() + ".fileprovider", generatedPdfFile);

            Intent shareIntent = new Intent(Intent.ACTION_SEND);
            shareIntent.setType("application/pdf");
            shareIntent.putExtra(Intent.EXTRA_STREAM, pdfUri);
            shareIntent.putExtra(Intent.EXTRA_SUBJECT, "Chassis QC Report: " + record.getPartId());
            shareIntent.putExtra(Intent.EXTRA_TEXT, "Automated factory QC validation report PDF generated for chassis SKU ID: " + record.getPartId() + " (Verdict: " + record.getPassOrFail() + ").");
            shareIntent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);

            startActivity(Intent.createChooser(shareIntent, "Share Inspection PDF via:"));
        } else {
            Toast.makeText(getContext(), "Failed to prepare PDF file for sharing.", Toast.LENGTH_SHORT).show();
        }
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}
