package com.example.ui.fragment;

import android.graphics.Color;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
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

import java.util.ArrayList;
import java.util.List;

public class ResultsFragment extends Fragment {

    private FragmentResultsBinding binding;
    private String partId = "";
    private String imageUri = "";
    
    // Dynamic counts
    private int expectedBolts = 3;
    private int expectedHoles = 1;
    private int expectedScrews = 1;
    
    private int detectedBolts = 0;
    private int detectedHoles = 0;
    private int detectedScrews = 0;
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

        // Load overlay image and trigger feature extraction
        binding.chassisOverlayView.setImageUri(imageUri);

        binding.btnInspectAgain.setOnClickListener(v -> {
            Navigation.findNavController(v).navigate(R.id.action_results_to_home);
        });

        binding.btnSaveReport.setOnClickListener(v -> saveRecordAndProceed());

        runFeatureVerificationPipeline();
    }

    private void runFeatureVerificationPipeline() {
        new Thread(() -> {
            AppDatabase db = AppDatabase.getInstance(requireContext());
            ReferenceProfile ref = db.referenceProfileDao().getProfileByPartType(partId);

            // Step 1: Compute Expected counts using Reference setup profile features
            if (ref != null && ref.getImageUri() != null && !"dummy_placeholder".equals(ref.getImageUri())) {
                expectedBolts = 0;
                expectedHoles = 0;
                expectedScrews = 0;
                List<HardwareFeatureExtractor.ExtractedFeature> refFeatures = 
                        HardwareFeatureExtractor.extractFeatures(requireContext(), ref.getImageUri());
                for (HardwareFeatureExtractor.ExtractedFeature f : refFeatures) {
                    if ("Bolt Head".equals(f.type)) expectedBolts++;
                    else if ("Hole".equals(f.type)) expectedHoles++;
                    else if ("Screw".equals(f.type)) expectedScrews++;
                }
            } else {
                // Keep default gold standards (3 bolts, 1 hole, 1 screw)
                expectedBolts = 3;
                expectedHoles = 1;
                expectedScrews = 1;
            }

            // Step 2: Compute Detected counts for selected active chassis
            detectedBolts = 0;
            detectedHoles = 0;
            detectedScrews = 0;
            List<HardwareFeatureExtractor.ExtractedFeature> activeFeatures = 
                    HardwareFeatureExtractor.extractFeatures(requireContext(), imageUri);
            for (HardwareFeatureExtractor.ExtractedFeature f : activeFeatures) {
                if ("Bolt Head".equals(f.type)) detectedBolts++;
                else if ("Hole".equals(f.type)) detectedHoles++;
                else if ("Screw".equals(f.type)) detectedScrews++;
            }

            // Check if there is any mismatch/discrepancy
            hasDiscrepancy = (expectedBolts != detectedBolts) || 
                             (expectedHoles != detectedHoles) || 
                             (expectedScrews != detectedScrews);

            // Push results to UI thread
            requireActivity().runOnUiThread(() -> {
                if (!isAdded() || binding == null) return;

                // Update Header Banner
                if (hasDiscrepancy) {
                    binding.tvResultBadge.setText("INSPECTION FAILED");
                    binding.tvResultSub.setText("Discrepancy detected in chassis configuration.");
                    binding.layoutResultHeader.setBackgroundColor(ContextCompat.getColor(requireContext(), R.color.red_fail));
                } else {
                    binding.tvResultBadge.setText("INSPECTION PASSED");
                    binding.tvResultSub.setText("All expected features verified successfully.");
                    binding.layoutResultHeader.setBackgroundColor(ContextCompat.getColor(requireContext(), R.color.green_pass_header));
                }

                // Update Binders/Counters
                binding.tvExpectedBolts.setText(String.valueOf(expectedBolts));
                binding.tvDetectedBolts.setText(String.valueOf(detectedBolts));
                updateStatusText(binding.tvStatusBolts, expectedBolts, detectedBolts);

                binding.tvExpectedHoles.setText(String.valueOf(expectedHoles));
                binding.tvDetectedHoles.setText(String.valueOf(detectedHoles));
                updateStatusText(binding.tvStatusHoles, expectedHoles, detectedHoles);

                binding.tvExpectedScrews.setText(String.valueOf(expectedScrews));
                binding.tvDetectedScrews.setText(String.valueOf(detectedScrews));
                updateStatusText(binding.tvStatusScrews, expectedScrews, detectedScrews);
            });
        }).start();
    }

    private void updateStatusText(android.widget.TextView tv, int expected, int detected) {
        if (expected == detected) {
            tv.setText("MATCH");
            tv.setTextColor(ContextCompat.getColor(requireContext(), R.color.green_pass));
        } else if (detected < expected) {
            tv.setText("MISSING");
            tv.setTextColor(ContextCompat.getColor(requireContext(), R.color.red_fail));
        } else {
            tv.setText("EXTRA");
            tv.setTextColor(ContextCompat.getColor(requireContext(), R.color.orange_warn));
        }
    }

    private void saveRecordAndProceed() {
        String notes = binding.etInspectionNotes.getText().toString().trim();
        binding.btnSaveReport.setEnabled(false);

        new Thread(() -> {
            AppDatabase db = AppDatabase.getInstance(requireContext());

            // 1. Create standard record
            InspectionRecord record = new InspectionRecord();
            record.setPartId(partId);
            record.setTimestamp(System.currentTimeMillis());
            record.setPassOrFail(hasDiscrepancy ? "FAIL" : "PASS");
            record.setImageUri(imageUri);
            record.setNotes(notes);

            long inspectionId = db.inspectionDao().insertInspection(record);

            // 2. Insert calculated detected dynamic feature results matching our actual CV execution!
            List<FeatureResult> results = new ArrayList<>();
            results.add(new FeatureResult(inspectionId, "Bolt Head", expectedBolts, detectedBolts, 
                    expectedBolts == detectedBolts ? "MATCH" : (detectedBolts < expectedBolts ? "MISSING" : "EXTRA")));
            results.add(new FeatureResult(inspectionId, "Hole", expectedHoles, detectedHoles, 
                    expectedHoles == detectedHoles ? "MATCH" : (detectedHoles < expectedHoles ? "MISSING" : "EXTRA")));
            results.add(new FeatureResult(inspectionId, "Screw", expectedScrews, detectedScrews, 
                    expectedScrews == detectedScrews ? "MATCH" : (detectedScrews < expectedScrews ? "MISSING" : "EXTRA")));

            db.inspectionDao().insertFeatureResults(results);

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
