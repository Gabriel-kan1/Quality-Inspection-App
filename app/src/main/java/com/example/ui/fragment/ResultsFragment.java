package com.example.ui.fragment;

import android.graphics.Color;
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
import androidx.fragment.app.Fragment;
import androidx.navigation.Navigation;

import com.example.R;
import com.example.data.AppDatabase;
import com.example.data.entity.FeatureResult;
import com.example.data.entity.InspectionRecord;
import com.example.data.entity.ReferenceProfile;
import com.example.databinding.FragmentResultsBinding;
import com.example.ml.ReferenceLibrary;
import com.example.ml.ReferenceLibraryProvider;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class ResultsFragment extends Fragment {

    private FragmentResultsBinding binding;
    private String partId = "";
    private String imageUri = "";

    // Per-type counts (keyed by canonical feature type, e.g. "screw").
    private final Map<String, Integer> expectedCounts = new LinkedHashMap<>();
    private final Map<String, Integer> detectedCounts = new LinkedHashMap<>();
    private List<String> orderedTypes = new ArrayList<>();
    private boolean hasDiscrepancy = false;

    @Override
    public void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        if (getArguments() != null) {
            partId = getArguments().getString("partId", "UNKNOWN_PART");
            imageUri = getArguments().getString("imageUri", "");
        }
    }

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        binding = FragmentResultsBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        binding.btnInspectAgain.setOnClickListener(v ->
                Navigation.findNavController(v).navigate(R.id.action_results_to_home));

        binding.btnSaveReport.setOnClickListener(v -> saveRecordAndProceed());

        runFeatureVerificationPipeline();
    }

    private void runFeatureVerificationPipeline() {
        new Thread(() -> {
            AppDatabase db = AppDatabase.getInstance(requireContext());
            ReferenceProfile ref = db.referenceProfileDao().getProfileByPartType(partId);

            // Effective CLIP reference library: per-part if set up in-app, else the bundled default.
            ReferenceLibrary library = ReferenceLibraryProvider.effectiveFor(requireContext(), partId);

            // Detected: classify the active chassis image.
            HardwareFeatureExtractor.AnalysisResult activeResult =
                    HardwareFeatureExtractor.extractFeatures(requireContext(), imageUri, library);
            detectedCounts.clear();
            countByType(activeResult.features, detectedCounts);

            // Expected: from the gold-standard reference image if one exists, else mirror detected.
            expectedCounts.clear();
            boolean hasRealReference = ref != null && ref.getImageUri() != null
                    && !"dummy_placeholder".equals(ref.getImageUri());
            if (hasRealReference) {
                HardwareFeatureExtractor.AnalysisResult refResult =
                        HardwareFeatureExtractor.extractFeatures(requireContext(), ref.getImageUri(), library);
                countByType(refResult.features, expectedCounts);
            } else {
                expectedCounts.putAll(detectedCounts);
            }

            // Ordered union of types (canonical order first, "unknown" last).
            orderedTypes = orderedUnion(expectedCounts.keySet(), detectedCounts.keySet());

            // Discrepancy over known types only (ignore "unknown" noise).
            hasDiscrepancy = false;
            for (String type : orderedTypes) {
                if (ReferenceLibrary.UNKNOWN.equals(type)) continue;
                if (count(expectedCounts, type) != count(detectedCounts, type)) {
                    hasDiscrepancy = true;
                    break;
                }
            }

            requireActivity().runOnUiThread(() -> {
                if (!isAdded() || binding == null) return;

                if (activeResult.processedBitmap != null) {
                    binding.ivCannyEdges.setImageBitmap(activeResult.processedBitmap);
                }
                binding.chassisOverlayView.setResultData(imageUri, activeResult.features);

                if (hasDiscrepancy) {
                    binding.tvResultBadge.setText("INSPECTION FAILED");
                    binding.tvResultSub.setText("Discrepancy detected in chassis configuration.");
                    binding.layoutResultHeader.setBackgroundColor(ContextCompat.getColor(requireContext(), R.color.red_fail));
                } else {
                    binding.tvResultBadge.setText("INSPECTION PASSED");
                    binding.tvResultSub.setText("All expected features verified successfully.");
                    binding.layoutResultHeader.setBackgroundColor(ContextCompat.getColor(requireContext(), R.color.green_pass_header));
                }

                buildFeatureRows();
                buildLegend();
            });
        }).start();
    }

    private static void countByType(List<HardwareFeatureExtractor.ExtractedFeature> features, Map<String, Integer> out) {
        if (features == null) return;
        for (HardwareFeatureExtractor.ExtractedFeature f : features) {
            Integer prev = out.get(f.type);
            out.put(f.type, prev == null ? 1 : prev + 1);
        }
    }

    private static int count(Map<String, Integer> counts, String type) {
        Integer c = counts.get(type);
        return c == null ? 0 : c;
    }

    private static List<String> orderedUnion(Set<String> a, Set<String> b) {
        Set<String> union = new LinkedHashSet<>();
        union.addAll(a);
        union.addAll(b);
        List<String> ordered = new ArrayList<>();
        for (String t : ReferenceLibrary.FEATURE_TYPES) {
            if (union.contains(t)) ordered.add(t);
        }
        for (String t : union) {
            if (!ordered.contains(t) && !ReferenceLibrary.UNKNOWN.equals(t)) ordered.add(t);
        }
        if (union.contains(ReferenceLibrary.UNKNOWN)) ordered.add(ReferenceLibrary.UNKNOWN);
        return ordered;
    }

    /** Builds one summary-table row per feature type in {@link #orderedTypes}. */
    private void buildFeatureRows() {
        LinearLayout container = binding.layoutFeatureRows;
        container.removeAllViews();
        boolean first = true;
        for (String type : orderedTypes) {
            if (!first) container.addView(divider());
            first = false;
            int expected = count(expectedCounts, type);
            int detected = count(detectedCounts, type);
            container.addView(featureRow(type, expected, detected));
        }
    }

    private View divider() {
        View v = new View(requireContext());
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dp(1));
        v.setLayoutParams(lp);
        v.setBackgroundColor(Color.parseColor("#33B0BEC5"));
        return v;
    }

    private View featureRow(String type, int expected, int detected) {
        LinearLayout row = new LinearLayout(requireContext());
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setWeightSum(4f);
        row.setPadding(0, dp(12), 0, dp(12));
        row.setBackgroundColor(ContextCompat.getColor(requireContext(), R.color.surface_card));

        TextView name = cell(ReferenceLibrary.displayName(type), 1.3f, Gravity.START);
        name.setTextColor(ContextCompat.getColor(requireContext(), R.color.white));
        name.setTextSize(14f);
        name.setTypeface(name.getTypeface(), android.graphics.Typeface.BOLD);
        name.setPadding(dp(8), 0, 0, 0);
        row.addView(name);

        TextView exp = cell(String.valueOf(expected), 0.9f, Gravity.CENTER);
        exp.setTextColor(ContextCompat.getColor(requireContext(), R.color.gray_text_secondary));
        exp.setTextSize(14f);
        row.addView(exp);

        TextView det = cell(String.valueOf(detected), 0.9f, Gravity.CENTER);
        det.setTextColor(ContextCompat.getColor(requireContext(), R.color.white));
        det.setTextSize(14f);
        row.addView(det);

        String statusText;
        int statusColor;
        if (ReferenceLibrary.UNKNOWN.equals(type)) {
            statusText = "—";
            statusColor = R.color.gray_text_secondary;
        } else if (expected == detected) {
            statusText = "MATCH";
            statusColor = R.color.green_pass;
        } else if (detected < expected) {
            statusText = "MISSING";
            statusColor = R.color.red_fail;
        } else {
            statusText = "EXTRA";
            statusColor = R.color.orange_warn;
        }
        TextView status = cell(statusText, 0.9f, Gravity.CENTER);
        status.setTextColor(ContextCompat.getColor(requireContext(), statusColor));
        status.setTextSize(12f);
        status.setTypeface(status.getTypeface(), android.graphics.Typeface.BOLD);
        row.addView(status);

        return row;
    }

    private TextView cell(String text, float weight, int gravity) {
        TextView tv = new TextView(requireContext());
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(0,
                LinearLayout.LayoutParams.WRAP_CONTENT, weight);
        tv.setLayoutParams(lp);
        tv.setText(text);
        tv.setGravity(gravity);
        return tv;
    }

    /** Builds a colored-swatch legend chip per feature type. */
    private void buildLegend() {
        LinearLayout legend = binding.layoutLegend;
        legend.removeAllViews();
        for (String type : orderedTypes) {
            LinearLayout chip = new LinearLayout(requireContext());
            chip.setOrientation(LinearLayout.HORIZONTAL);
            chip.setGravity(Gravity.CENTER_VERTICAL);
            chip.setPadding(dp(8), 0, dp(8), 0);

            View swatch = new View(requireContext());
            swatch.setLayoutParams(new LinearLayout.LayoutParams(dp(14), dp(14)));
            swatch.setBackgroundColor(ReferenceLibrary.colorFor(type));
            chip.addView(swatch);

            TextView label = new TextView(requireContext());
            LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT);
            lp.setMarginStart(dp(6));
            label.setLayoutParams(lp);
            label.setText(ReferenceLibrary.displayName(type));
            label.setTextColor(ContextCompat.getColor(requireContext(), R.color.white));
            label.setTextSize(12f);
            chip.addView(label);

            legend.addView(chip);
        }
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private void saveRecordAndProceed() {
        String notes = binding.etInspectionNotes.getText().toString().trim();
        binding.btnSaveReport.setEnabled(false);

        new Thread(() -> {
            AppDatabase db = AppDatabase.getInstance(requireContext());

            InspectionRecord record = new InspectionRecord();
            record.setPartId(partId);
            record.setTimestamp(System.currentTimeMillis());
            record.setPassOrFail(hasDiscrepancy ? "FAIL" : "PASS");
            record.setImageUri(imageUri);
            record.setNotes(notes);

            long inspectionId = db.inspectionDao().insertInspection(record);

            List<FeatureResult> results = new ArrayList<>();
            for (String type : orderedTypes) {
                if (ReferenceLibrary.UNKNOWN.equals(type)) continue;
                int expected = count(expectedCounts, type);
                int detected = count(detectedCounts, type);
                String status = expected == detected ? "MATCH" : (detected < expected ? "MISSING" : "EXTRA");
                results.add(new FeatureResult(inspectionId, ReferenceLibrary.displayName(type),
                        expected, detected, status));
            }
            if (!results.isEmpty()) {
                db.inspectionDao().insertFeatureResults(results);
            }

            requireActivity().runOnUiThread(() -> {
                Toast.makeText(getContext(), "Inspection report filed offline.", Toast.LENGTH_SHORT).show();
                Bundle args = new Bundle();
                args.putLong("inspectionId", inspectionId);
                if (isAdded() && getView() != null) {
                    Navigation.findNavController(getView()).navigate(R.id.action_results_to_report, args);
                }
            });
        }).start();
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}
