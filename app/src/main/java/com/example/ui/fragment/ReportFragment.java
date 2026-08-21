package com.example.ui.fragment;

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
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
import com.example.ml.PartFiles;
import com.example.report.AdvisoryNotes;
import com.example.report.InspectionExport;
import com.example.report.InspectionReportPdf;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public class ReportFragment extends Fragment {

    private static final String TAG = "ReportFragment";

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

        binding.btnExportData.setOnClickListener(v -> exportDataFiles());

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
                    // UNVERIFIED is its own verdict — see ResultsFragment.Verdict. Folding it
                    // into the FAIL branch would report a discrepancy that was never measured.
                    if ("PASS".equalsIgnoreCase(record.getPassOrFail())) {
                        binding.tvReportVerdict.setText("PASS");
                        binding.tvReportVerdict.setBackgroundColor(ContextCompat.getColor(requireContext(), R.color.green_pass));
                    } else if ("UNVERIFIED".equalsIgnoreCase(record.getPassOrFail())) {
                        binding.tvReportVerdict.setText("UNVERIFIED");
                        binding.tvReportVerdict.setBackgroundColor(ContextCompat.getColor(requireContext(), R.color.orange_warn));
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
            } else if ("UNVERIFIED".equalsIgnoreCase(f.getStatus())) {
                // Nothing was compared for this row, so it is neither a match nor a defect.
                // Red here would report a discrepancy the inspection never measured.
                tvStatus.setTextColor(ContextCompat.getColor(requireContext(), R.color.gray_text_secondary));
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

        int missing = 0, wrong = 0, displaced = 0;
        for (SlotResult s : slotResults) {
            if ("MISSING".equals(s.getStatus())) missing++;
            else if ("WRONG_TYPE".equals(s.getStatus())) wrong++;
            else if ("DISPLACED".equals(s.getStatus())) displaced++;
        }
        binding.tvReportLocalisationSummary.setText(String.format(Locale.getDefault(),
                "%d of %d expected positions verified · %d missing · %d wrong type · %d out of position",
                slotResults.size() - missing - wrong - displaced, slotResults.size(),
                missing, wrong, displaced));

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
            } else if ("DISPLACED".equals(s.getStatus())) {
                tvStatus.setText(String.format(Locale.getDefault(), "%s%n%.0f%% off",
                        s.getFoundType(), s.getOffsetPct() == null ? 0f : s.getOffsetPct() * 100));
                tvStatus.setTextColor(ContextCompat.getColor(requireContext(), R.color.orange_warn));
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

    /**
     * Base name shared by every artefact this inspection produces, so the PDF, the JSON and
     * the CSV sort together and obviously belong to each other.
     *
     * <p>Keyed on the uuid where there is one. The rowid it falls back to is only meaningful
     * inside this database, which is the wrong property for a filename that exists to be
     * copied somewhere else.
     *
     * <p>The part ID is free text the operator typed, and these paths are handed straight to
     * FileProvider below. Without the sanitiser a separator or a ".." in the ID writes outside
     * the reports folder, so use the same one the rest of the app already does.
     */
    private String reportBaseName() {
        String suffix = record.getUuid() != null ? record.getUuid() : String.valueOf(record.getId());
        return "QC_REPORT_" + PartFiles.safeName(record.getPartId()) + "_" + suffix;
    }

    @Nullable
    private File reportsFolder() {
        File folder = requireContext().getExternalFilesDir("reports");
        if (folder != null && !folder.exists()) {
            folder.mkdirs();
        }
        return folder;
    }

    private void generatePdfReport(boolean showToast) {
        if (record == null) {
            Toast.makeText(getContext(), "Report details still loading. Please hold...", Toast.LENGTH_SHORT).show();
            return;
        }

        File target = new File(reportsFolder(), reportBaseName() + ".pdf");

        try (FileOutputStream out = new FileOutputStream(target)) {
            InspectionReportPdf.write(out, record, features, slotResults, warningNotes);
            generatedPdfFile = target;
        } catch (Exception e) {
            // A partially written file is worse than none: it is a valid-looking path that
            // opens to a corrupt document, and sharePdfReport() would happily send it on.
            Log.e(TAG, "PDF generation failed", e);
            if (target.exists() && !target.delete()) {
                Log.w(TAG, "Could not remove partial report at " + target);
            }
            generatedPdfFile = null;
            Toast.makeText(getContext(), "Error saving PDF: " + e.getMessage(), Toast.LENGTH_LONG).show();
            return;
        }

        if (showToast) {
            Toast.makeText(getContext(), "Report exported to: Internal Storage/reports/"
                    + target.getName(), Toast.LENGTH_LONG).show();
        }
    }

    /**
     * Writes the machine-readable pair — full record as JSON, roll-up row as CSV — and offers
     * them to a share target.
     *
     * <p>Sharing is part of the export rather than a separate step because the folder these
     * land in is app-scoped external storage, which no file manager on a modern Android will
     * show the operator. Writing them and stopping there would produce two files nobody can
     * reach; the FileProvider grant is the only way off the device, and it is the reason that
     * provider exists.
     *
     * <p>The JSON carries the per-slot findings, the CSV deliberately does not — a row whose
     * width depends on how many slots the part has is not a spreadsheet column. See
     * {@link InspectionExport}.
     */
    private void exportDataFiles() {
        if (record == null) {
            Toast.makeText(getContext(), "Report details still loading. Please hold...", Toast.LENGTH_SHORT).show();
            return;
        }

        File folder = reportsFolder();
        File jsonFile = new File(folder, reportBaseName() + ".json");
        File csvFile = new File(folder, reportBaseName() + ".csv");

        try {
            writeText(jsonFile, InspectionExport.toJson(record, features, slotResults));
            writeText(csvFile,
                    InspectionExport.csvHeader() + "\n"
                            + InspectionExport.toCsvRow(record, features, slotResults) + "\n");
        } catch (Exception e) {
            // Same reasoning as the PDF path: a half-written export is a file that parses
            // partly and misleads completely, so neither is left behind.
            Log.e(TAG, "Data export failed", e);
            deleteQuietly(jsonFile);
            deleteQuietly(csvFile);
            Toast.makeText(getContext(), "Error exporting data: " + e.getMessage(),
                    Toast.LENGTH_LONG).show();
            return;
        }

        Toast.makeText(getContext(), "Data exported to: Internal Storage/reports/"
                + jsonFile.getName() + " (+ .csv)", Toast.LENGTH_LONG).show();

        shareFiles(Arrays.asList(jsonFile, csvFile), "*/*",
                "QC Inspection Data: " + record.getPartId(),
                "Machine-readable inspection record for part " + record.getPartId()
                        + " (Verdict: " + record.getPassOrFail() + ", Record ID: "
                        + record.getUuid() + ").",
                "Share Inspection Data via:");
    }

    private static void writeText(File target, String content) throws IOException {
        try (FileOutputStream out = new FileOutputStream(target)) {
            out.write(content.getBytes(StandardCharsets.UTF_8));
        }
    }

    private void deleteQuietly(File file) {
        if (file.exists() && !file.delete()) {
            Log.w(TAG, "Could not remove partial export at " + file);
        }
    }

    private void sharePdfReport() {
        if (generatedPdfFile == null || !generatedPdfFile.exists()) {
            // Unsaved generate first
            generatePdfReport(false);
        }

        if (generatedPdfFile != null && generatedPdfFile.exists()) {
            shareFiles(Collections.singletonList(generatedPdfFile), "application/pdf",
                    "Chassis QC Report: " + record.getPartId(),
                    "Automated factory QC validation report PDF generated for chassis SKU ID: "
                            + record.getPartId() + " (Verdict: " + record.getPassOrFail() + ").",
                    "Share Inspection PDF via:");
        } else {
            Toast.makeText(getContext(), "Failed to prepare PDF file for sharing.", Toast.LENGTH_SHORT).show();
        }
    }

    /**
     * Hands one or more generated files to a share target through the app's FileProvider.
     *
     * <p>Takes a MIME type and a list rather than assuming a single PDF, which is what it used
     * to do. Everything shared here comes out of the same {@code reports/} folder, which is
     * the only path {@code res/xml/file_paths.xml} maps — deliberately, so a URI grant issued
     * by this app can never reach anything else the user owns.
     */
    private void shareFiles(List<File> files, String mimeType, String subject, String body,
                            String chooserTitle) {
        String authority = requireContext().getPackageName() + ".fileprovider";
        ArrayList<Uri> uris = new ArrayList<>(files.size());
        for (File file : files) {
            uris.add(FileProvider.getUriForFile(requireContext(), authority, file));
        }

        Intent shareIntent = uris.size() == 1
                ? new Intent(Intent.ACTION_SEND).putExtra(Intent.EXTRA_STREAM, uris.get(0))
                : new Intent(Intent.ACTION_SEND_MULTIPLE).putParcelableArrayListExtra(
                        Intent.EXTRA_STREAM, uris);
        shareIntent.setType(mimeType);
        shareIntent.putExtra(Intent.EXTRA_SUBJECT, subject);
        shareIntent.putExtra(Intent.EXTRA_TEXT, body);
        shareIntent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);

        startActivity(Intent.createChooser(shareIntent, chooserTitle));
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}
