package com.example.ui.fragment;

import android.Manifest;
import android.content.pm.PackageManager;
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
import androidx.camera.core.ExperimentalGetImage;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;
import androidx.navigation.Navigation;

import com.example.R;
import com.example.data.AppDatabase;
import com.example.data.entity.ReferenceProfile;
import com.example.databinding.FragmentBarcodeScanBinding;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.mlkit.vision.barcode.BarcodeScanner;
import com.google.mlkit.vision.barcode.BarcodeScanning;
import com.google.mlkit.vision.barcode.common.Barcode;
import com.google.mlkit.vision.common.InputImage;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

@ExperimentalGetImage
public class BarcodeScanFragment extends Fragment {

    private static final String TAG = "BarcodeScanFragment";

    private FragmentBarcodeScanBinding binding;
    private ExecutorService cameraExecutor;
    private BarcodeScanner barcodeScanner;
    private ProcessCameraProvider cameraProvider;
    private String scannedPartId = null;
    private boolean isScanningActive = true;

    private final ActivityResultLauncher<String> requestPermissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), isGranted -> {
                if (isGranted) {
                    startCamera();
                } else {
                    Toast.makeText(getContext(), "Camera permission required for scanning", Toast.LENGTH_LONG).show();
                }
            });

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        binding = FragmentBarcodeScanBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        cameraExecutor = Executors.newSingleThreadExecutor();
        barcodeScanner = BarcodeScanning.getClient();

        // Check permission
        if (ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA);
        }

        binding.btnRescan.setOnClickListener(v -> resetScanningState());

        binding.btnConfirmPart.setOnClickListener(v -> {
            if (scannedPartId != null) {
                checkReferenceAndProceed(scannedPartId);
            }
        });

        // Simulating scan button to make testing easy without hardware barcode
        binding.btnSkipMock.setOnClickListener(v -> {
            onBarcodeDetected("PART-CHASSIS-A03");
        });
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

        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();

        imageAnalysis.setAnalyzer(cameraExecutor, image -> {
            if (!isScanningActive) {
                image.close();
                return;
            }

            @ExperimentalGetImage
            android.media.Image mediaImage = image.getImage();
            if (mediaImage != null) {
                InputImage inputImage = InputImage.fromMediaImage(mediaImage, image.getImageInfo().getRotationDegrees());
                barcodeScanner.process(inputImage)
                        .addOnSuccessListener(barcodes -> {
                            for (Barcode barcode : barcodes) {
                                String rawValue = barcode.getRawValue();
                                if (rawValue != null && !rawValue.isEmpty()) {
                                    requireActivity().runOnUiThread(() -> onBarcodeDetected(rawValue));
                                    break;
                                }
                            }
                        })
                        .addOnFailureListener(e -> Log.e(TAG, "Barcode scanner error", e))
                        .addOnCompleteListener(task -> image.close());
            } else {
                image.close();
            }
        });

        CameraSelector cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA;

        try {
            cameraProvider.unbindAll();
            cameraProvider.bindToLifecycle(getViewLifecycleOwner(), cameraSelector, preview, imageAnalysis);
        } catch (Exception e) {
            Log.e(TAG, "Binding exceptions", e);
        }
    }

    private void onBarcodeDetected(String partId) {
        scannedPartId = partId;
        isScanningActive = false;

        if (cameraProvider != null) {
            cameraProvider.unbindAll();
        }

        binding.layoutScanningState.setVisibility(View.GONE);
        binding.layoutDetectedState.setVisibility(View.VISIBLE);
        binding.tvDetectedId.setText(partId);
    }

    private void resetScanningState() {
        scannedPartId = null;
        isScanningActive = true;
        binding.layoutScanningState.setVisibility(View.VISIBLE);
        binding.layoutDetectedState.setVisibility(View.GONE);
        startCamera();
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
                    // No profile exists, go to Setup Screen to take a reference photo
                    Navigation.findNavController(requireView()).navigate(R.id.action_barcode_to_reference_setup, args);
                }
            });
        }).start();
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        isScanningActive = false;
        if (cameraExecutor != null) {
            cameraExecutor.shutdown();
        }
        if (barcodeScanner != null) {
            barcodeScanner.close();
        }
        binding = null;
    }
}
