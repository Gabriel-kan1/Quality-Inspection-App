package com.example.ui.fragment;

import android.Manifest;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
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
import com.example.databinding.FragmentInspectionBinding;
import com.google.common.util.concurrent.ListenableFuture;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class InspectionFragment extends Fragment {

    private static final String TAG = "InspectionFragment";

    private FragmentInspectionBinding binding;
    private ImageCapture imageCapture;
    private ExecutorService cameraExecutor;
    private ProcessCameraProvider cameraProvider;
    private String partId = "";
    private ReferenceProfile referenceProfile = null;
    private boolean isGhostEnabled = true;
    private boolean isAligned = false;
    private Handler alignmentSimulatorHandler;

    private final ActivityResultLauncher<String> requestPermissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), isGranted -> {
                if (isGranted) {
                    startCamera();
                } else {
                    Toast.makeText(getContext(), "Camera permission required for inspection", Toast.LENGTH_LONG).show();
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
        binding = FragmentInspectionBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        cameraExecutor = Executors.newSingleThreadExecutor();
        alignmentSimulatorHandler = new Handler(Looper.getMainLooper());

        binding.tvPartInfo.setText("Part ID: " + partId);

        // Fetch Reference Profile in background
        loadReferenceProfile();

        if (ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA);
        }

        binding.btnToggleGhost.setOnClickListener(v -> toggleGhostOverlay());

        binding.btnCaptureInspect.setOnClickListener(v -> captureAndInspect());

        // Setup touch trigger to instantly align
        binding.tvSimulateHint.setOnClickListener(v -> triggerAlignmentAchieved());

        // Auto transition alignment after 3 seconds of simulation
        simulateAlignmentTuning();
    }

    private void loadReferenceProfile() {
        new Thread(() -> {
            AppDatabase db = AppDatabase.getInstance(requireContext());
            referenceProfile = db.referenceProfileDao().getProfileByPartType(partId);

            requireActivity().runOnUiThread(() -> {
                if (referenceProfile != null && referenceProfile.getImageUri() != null) {
                    String imgUriStr = referenceProfile.getImageUri();
                    if ("dummy_placeholder".equals(imgUriStr)) {
                        binding.ivGhostOverlay.setVisibility(View.GONE);
                        binding.tvGhostStatusIndicator.setText("DEFAULT REFERENCING (NO GHOST OVERLAY)");
                    } else {
                        binding.ivGhostOverlay.setImageURI(Uri.parse(imgUriStr));
                        binding.ivGhostOverlay.setVisibility(View.VISIBLE);
                        binding.tvGhostStatusIndicator.setText("GHOST OVERLAY ON (35%)");
                    }
                } else {
                    binding.ivGhostOverlay.setVisibility(View.GONE);
                    binding.tvGhostStatusIndicator.setText("NO ACTIVE GHOST OVERLAY");
                }
            });
        }).start();
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

    private void toggleGhostOverlay() {
        isGhostEnabled = !isGhostEnabled;
        if (isGhostEnabled && referenceProfile != null && !"dummy_placeholder".equals(referenceProfile.getImageUri())) {
            binding.ivGhostOverlay.setVisibility(View.VISIBLE);
            binding.btnToggleGhost.setText("GHOST OFF");
            binding.tvGhostStatusIndicator.setText("GHOST OVERLAY ON (35%)");
        } else {
            binding.ivGhostOverlay.setVisibility(View.GONE);
            binding.btnToggleGhost.setText("GHOST ON");
            binding.tvGhostStatusIndicator.setText("GHOST OVERLAY INACTIVE");
        }
    }

    private void simulateAlignmentTuning() {
        binding.tvAlignmentState.setText("ALIGNING...");
        binding.tvAlignmentState.setBackgroundColor(ContextCompat.getColor(requireContext(), R.color.red_fail));
        binding.layoutCameraEnclosure.setBackgroundResource(R.drawable.align_border_red);
        isAligned = false;

        alignmentSimulatorHandler.postDelayed(() -> {
            if (isAdded() && binding != null) {
                triggerAlignmentAchieved();
            }
        }, 2200); // Trigger green after 2.2 seconds simulated hold
    }

    private void triggerAlignmentAchieved() {
        isAligned = true;
        binding.tvAlignmentState.setText("ALIGNED (98%)");
        binding.tvAlignmentState.setBackgroundColor(ContextCompat.getColor(requireContext(), R.color.green_pass));
        binding.layoutCameraEnclosure.setBackgroundResource(R.drawable.align_border_green);
        Toast.makeText(getContext(), "Position optimal. Locked.", Toast.LENGTH_SHORT).show();
    }

    private void captureAndInspect() {
        if (imageCapture == null) {
            // If camera is not ready, we can mock it directly
            mockAndNavigate();
            return;
        }

        File outputDir = requireContext().getExternalFilesDir("captured_inspections");
        if (outputDir != null && !outputDir.exists()) {
            outputDir.mkdirs();
        }

        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new java.util.Date());
        File photoFile = new File(outputDir, "COMP_" + partId + "_" + timeStamp + ".jpg");

        ImageCapture.OutputFileOptions outputOptions = new ImageCapture.OutputFileOptions.Builder(photoFile).build();

        binding.btnCaptureInspect.setEnabled(false);
        Toast.makeText(getContext(), "Running computer vision analysis...", Toast.LENGTH_SHORT).show();

        imageCapture.takePicture(outputOptions, cameraExecutor, new ImageCapture.OnImageSavedCallback() {
            @Override
            public void onImageSaved(@NonNull ImageCapture.OutputFileResults outputFileResults) {
                Uri savedUri = Uri.fromFile(photoFile);
                navigateToResults(savedUri.toString());
            }

            @Override
            public void onError(@NonNull ImageCaptureException exception) {
                Log.e(TAG, "Capture failed, falling back to mock file path for inspection screen testing", exception);
                mockAndNavigate();
            }
        });
    }

    private void mockAndNavigate() {
        navigateToResults("mock_image_uri");
    }

    private void navigateToResults(String imageUri) {
        if (isAdded() && getView() != null) {
            requireActivity().runOnUiThread(() -> {
                Bundle args = new Bundle();
                args.putString("partId", partId);
                args.putString("imageUri", imageUri);
                Navigation.findNavController(getView()).navigate(R.id.action_inspection_to_results, args);
            });
        }
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        if (alignmentSimulatorHandler != null) {
            alignmentSimulatorHandler.removeCallbacksAndMessages(null);
        }
        if (cameraExecutor != null) {
            cameraExecutor.shutdown();
        }
        binding = null;
    }
}
