package com.example.ui.fragment;

import android.Manifest;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;
import androidx.navigation.Navigation;

import com.example.R;
import com.example.data.AppDatabase;
import com.example.data.entity.ReferenceProfile;
import com.example.databinding.FragmentReferenceSetupBinding;
import com.google.common.util.concurrent.ListenableFuture;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ReferenceSetupFragment extends Fragment {

    private static final String TAG = "ReferenceSetupFragment";

    private FragmentReferenceSetupBinding binding;
    private ImageCapture imageCapture;
    private ExecutorService cameraExecutor;
    private ProcessCameraProvider cameraProvider;
    private String partId = "";

    private final ActivityResultLauncher<String> requestPermissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), isGranted -> {
                if (isGranted) {
                    startCamera();
                } else {
                    Toast.makeText(getContext(), "Camera permission required for setting up references", Toast.LENGTH_LONG).show();
                }
            });

    @Override
    public void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        if (getArguments() != null) {
            partId = getArguments().getString("partId", "UNKNOWN_PART");
        }
    }

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        binding = FragmentReferenceSetupBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        cameraExecutor = Executors.newSingleThreadExecutor();

        binding.tvSetupPartId.setText("Part ID: " + partId);

        if (ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA);
        }

        binding.btnCaptureReference.setOnClickListener(v -> takeReferencePhoto());

        binding.btnSkip.setOnClickListener(v -> skipSetupReference());
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(requireContext());
        cameraProviderFuture.addListener(() -> {
            try {
                cameraProvider = cameraProviderFuture.get();
                bindCameraUseCases();
            } catch (Exception e) {
                Log.e(TAG, "Use case binding failed", e);
            }
        }, ContextCompat.getMainExecutor(requireContext()));
    }

    private void bindCameraUseCases() {
        if (cameraProvider == null) return;

        Preview preview = new Preview.Builder().build();
        preview.setSurfaceProvider(binding.previewView.getSurfaceProvider());

        imageCapture = new ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build();

        CameraSelector cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA;

        try {
            cameraProvider.unbindAll();
            cameraProvider.bindToLifecycle(getViewLifecycleOwner(), cameraSelector, preview, imageCapture);
        } catch (Exception e) {
            Log.e(TAG, "Camera use case binding failed", e);
        }
    }

    private void takeReferencePhoto() {
        if (imageCapture == null) return;

        // Create output file for reference image
        File outputDir = requireContext().getExternalFilesDir("references");
        if (outputDir != null && !outputDir.exists()) {
            outputDir.mkdirs();
        }

        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new java.util.Date());
        File photoFile = new File(outputDir, "REF_" + partId + "_" + timeStamp + ".jpg");

        ImageCapture.OutputFileOptions outputOptions = new ImageCapture.OutputFileOptions.Builder(photoFile).build();

        binding.btnCaptureReference.setEnabled(false);
        Toast.makeText(getContext(), "Capturing reference image...", Toast.LENGTH_SHORT).show();

        imageCapture.takePicture(outputOptions, cameraExecutor, new ImageCapture.OnImageSavedCallback() {
            @Override
            public void onImageSaved(@NonNull ImageCapture.OutputFileResults outputFileResults) {
                Uri savedUri = Uri.fromFile(photoFile);
                saveReferenceToDb(savedUri.toString());
            }

            @Override
            public void onError(@NonNull ImageCaptureException exception) {
                requireActivity().runOnUiThread(() -> {
                    binding.btnCaptureReference.setEnabled(true);
                    Toast.makeText(getContext(), "Error capturing image: " + exception.getMessage(), Toast.LENGTH_LONG).show();
                });
            }
        });
    }

    private void saveReferenceToDb(String imageUri) {
        new Thread(() -> {
            AppDatabase db = AppDatabase.getInstance(requireContext());
            ReferenceProfile profile = new ReferenceProfile(partId, imageUri, System.currentTimeMillis());
            db.referenceProfileDao().insertProfile(profile);

            requireActivity().runOnUiThread(() -> {
                Toast.makeText(getContext(), "Reference setup saved successfully!", Toast.LENGTH_SHORT).show();
                proceedToInspection();
            });
        }).start();
    }

    private void skipSetupReference() {
        // Create a dummy reference or proceed without one (defaulting image alignment overlay to none)
        new Thread(() -> {
            AppDatabase db = AppDatabase.getInstance(requireContext());
            // Check if profile exists, if not, save standard dummy placeholder
            ReferenceProfile existing = db.referenceProfileDao().getProfileByPartType(partId);
            if (existing == null) {
                ReferenceProfile profile = new ReferenceProfile(partId, "dummy_placeholder", System.currentTimeMillis());
                db.referenceProfileDao().insertProfile(profile);
            }
            requireActivity().runOnUiThread(this::proceedToInspection);
        }).start();
    }

    private void proceedToInspection() {
        Bundle args = new Bundle();
        args.putString("partId", partId);
        if (isAdded() && getView() != null) {
            Navigation.findNavController(getView()).navigate(R.id.action_reference_setup_to_inspection, args);
        }
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        if (cameraExecutor != null) {
            cameraExecutor.shutdown();
        }
        binding = null;
    }
}
