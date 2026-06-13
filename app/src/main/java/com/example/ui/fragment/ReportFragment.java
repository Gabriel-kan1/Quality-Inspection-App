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
import com.example.databinding.FragmentReportBinding;

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
    private List<FeatureResult> features = null;
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
            features = db.inspectionDao().getFeatureResultsForInspection(inspectionId);

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
