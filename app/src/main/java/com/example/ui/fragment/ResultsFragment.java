package com.example.ui.fragment;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.navigation.Navigation;

import com.example.R;
import com.example.data.AppDatabase;
import com.example.data.entity.FeatureResult;
import com.example.data.entity.InspectionRecord;
import com.example.databinding.FragmentResultsBinding;

import java.util.ArrayList;
import java.util.List;

public class ResultsFragment extends Fragment {

    private FragmentResultsBinding binding;
    private String partId = "";
    private String imageUri = "";

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

        // Load overlay image or mock representation
        binding.chassisOverlayView.setImageUri(imageUri);

        binding.btnInspectAgain.setOnClickListener(v -> {
            // Take operator back to Barcode Scanner to inspect another one
            Navigation.findNavController(v).navigate(R.id.action_results_to_home);
        });

        binding.btnSaveReport.setOnClickListener(v -> saveRecordAndProceed());
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
            record.setPassOrFail("PASS");
            record.setImageUri(imageUri);
            record.setNotes(notes);

            long inspectionId = db.inspectionDao().insertInspection(record);

            // 2. Insert standard detected feature segments matching mock user mandate:
            // "showing 3 bolt heads detected, 1 hole detected, and 1 screw detected with a PASS result"
            List<FeatureResult> results = new ArrayList<>();
            results.add(new FeatureResult(inspectionId, "Bolt Head", 3, 3, "MATCH"));
            results.add(new FeatureResult(inspectionId, "Hole", 1, 1, "MATCH"));
            results.add(new FeatureResult(inspectionId, "Screw", 1, 1, "MATCH"));

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
