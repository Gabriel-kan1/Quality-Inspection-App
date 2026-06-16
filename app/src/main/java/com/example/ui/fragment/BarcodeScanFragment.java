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
import com.example.data.entity.ReferenceProfile;
import com.example.databinding.FragmentBarcodeScanBinding;

public class BarcodeScanFragment extends Fragment {

    private FragmentBarcodeScanBinding binding;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        binding = FragmentBarcodeScanBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        // Quick Preset fills
        binding.btnPreset1.setOnClickListener(v -> binding.etPartId.setText("CHASSIS-8821-B"));
        binding.btnPreset2.setOnClickListener(v -> binding.etPartId.setText("PART-CHASSIS-A03"));
        binding.btnPreset3.setOnClickListener(v -> binding.etPartId.setText("ENGINE-COV-X42"));

        binding.btnConfirmPart.setOnClickListener(v -> {
            String enteredPartId = binding.etPartId.getText().toString().trim();
            if (enteredPartId.isEmpty()) {
                Toast.makeText(getContext(), "Please enter a valid Part ID", Toast.LENGTH_SHORT).show();
                return;
            }
            checkReferenceAndProceed(enteredPartId);
        });
    }

    private void checkReferenceAndProceed(String partId) {
        new Thread(() -> {
            // Check in AppDatabase if ReferenceProfile exists
            AppDatabase db = AppDatabase.getInstance(requireContext());
            ReferenceProfile profile = db.referenceProfileDao().getProfileByPartType(partId);

            requireActivity().runOnUiThread(() -> {
                Bundle args = new Bundle();
                args.putString("partId", partId);
                
                if (profile != null) {
                    // Profile exists, go directly to Inspection Screen
                    Navigation.findNavController(requireView()).navigate(R.id.action_barcode_to_inspection, args);
                } else {
                    // No profile exists, go to Setup Screen to take/select a reference photo
                    Navigation.findNavController(requireView()).navigate(R.id.action_barcode_to_reference_setup, args);
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
