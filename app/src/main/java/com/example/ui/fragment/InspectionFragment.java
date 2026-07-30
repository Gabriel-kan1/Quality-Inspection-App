package com.example.ui.fragment;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.drawable.GradientDrawable;
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
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;
import androidx.navigation.Navigation;

import com.example.R;
import com.example.data.AppDatabase;
import com.example.data.entity.ReferenceProfile;
import com.example.ui.util.AssetBitmaps;
import com.example.databinding.FragmentInspectionBinding;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.Locale;

public class InspectionFragment extends Fragment {

    private static final String TAG = "InspectionFragment";

    private FragmentInspectionBinding binding;
    private String partId = "";
    private ReferenceProfile referenceProfile = null;
    private Uri selectedImageExternalUri = null;
    private boolean isGhostEnabled = true;
    private boolean isAligned = false;
    private Handler alignmentSimulatorHandler;

    private final ActivityResultLauncher<String> selectImageLauncher =
            registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
                if (uri != null) {
                    persistSelectedImage(uri);
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
    public void onViewCreated(@NonNull View view, @NonNull @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        alignmentSimulatorHandler = new Handler(Looper.getMainLooper());

        binding.tvPartInfo.setText("Part ID: " + partId);

        // Load the stored Reference Profile if it exists
        loadReferenceProfile();

        // Scan the custom user_images folder in assets
        scanAssetsFolder();

        // Standard Alignment start
        resetAlignmentState();

        binding.layoutInteractiveTarget.setOnClickListener(v -> launchImageChooser());

        binding.btnCaptureInspect.setOnClickListener(v -> {
            if (selectedImageExternalUri == null) {
                launchImageChooser();
            } else {
                analyzeAndNavigate();
            }
        });

        binding.btnToggleGhost.setOnClickListener(v -> toggleGhostOverlay());

        binding.tvSimulateHint.setOnClickListener(v -> triggerAlignmentAchieved());

        binding.btnUsePresetInspection.setOnClickListener(v -> {
            loadSelectedAsset("chassis_hardware_preset.jpg");
        });
    }

    private void scanAssetsFolder() {
        try {
            String[] files = requireContext().getAssets().list("chassis_images");
            java.util.List<String> imageFiles = new java.util.ArrayList<>();
            if (files != null) {
                for (String file : files) {
                    String lower = file.toLowerCase();
                    if (lower.endsWith(".jpg") || lower.endsWith(".png") || lower.endsWith(".jpeg") || lower.endsWith(".webp") || lower.endsWith(".bmp")) {
                        imageFiles.add(file);
                    }
                }
            }

            if (imageFiles.isEmpty()) {
                binding.tvAssetsFolderStatus.setText("No images found in '/assets/chassis_images/'. Place your .jpg inside!");
                binding.scrollAssets.setVisibility(View.GONE);
            } else {
                binding.tvAssetsFolderStatus.setText("Found " + imageFiles.size() + " image(s) in assets/chassis_images/! Tap to inspect:");
                binding.scrollAssets.setVisibility(View.VISIBLE);
                populateAssetsScroller(imageFiles);
            }
        } catch (Exception e) {
            Log.e(TAG, "Error scanning assets folder", e);
            binding.tvAssetsFolderStatus.setText("Error reading assets/chassis_images folder");
            binding.scrollAssets.setVisibility(View.GONE);
        }
    }

    private void populateAssetsScroller(java.util.List<String> images) {
        binding.layoutAssetsContainer.removeAllViews();
        int margin = (int) (10 * getResources().getDisplayMetrics().density);
        int thumbSize = (int) (48 * getResources().getDisplayMetrics().density);

        for (String filename : images) {
            // Create a gorgeous horizontal layout chip for the item
            android.widget.LinearLayout itemLayout = new android.widget.LinearLayout(getContext());
            itemLayout.setOrientation(android.widget.LinearLayout.HORIZONTAL);
            itemLayout.setGravity(android.view.Gravity.CENTER_VERTICAL);
            itemLayout.setPadding((int)(8 * getResources().getDisplayMetrics().density), 0, (int)(12 * getResources().getDisplayMetrics().density), 0);
            
            // Layout Params with Margins
            android.widget.LinearLayout.LayoutParams params = new android.widget.LinearLayout.LayoutParams(
                    android.widget.LinearLayout.LayoutParams.WRAP_CONTENT,
                    android.widget.LinearLayout.LayoutParams.MATCH_PARENT
            );
            params.setMargins(0, 0, margin, 0);
            itemLayout.setLayoutParams(params);

            // Programmatic Material 3 style background shapes
            GradientDrawable gd = new GradientDrawable();
            gd.setColor(Color.rgb(43, 41, 48)); // elevated surface slate gray
            gd.setCornerRadius(10 * getResources().getDisplayMetrics().density);
            gd.setStroke((int) (1 * getResources().getDisplayMetrics().density), Color.rgb(80, 80, 80));
            itemLayout.setBackground(gd);

            // Thumbnail Image
            android.widget.ImageView imgView = new android.widget.ImageView(getContext());
            android.widget.LinearLayout.LayoutParams imgParams = new android.widget.LinearLayout.LayoutParams(thumbSize, thumbSize);
            imgView.setLayoutParams(imgParams);
            imgView.setScaleType(android.widget.ImageView.ScaleType.CENTER_CROP);
            imgView.setClipToOutline(true);

            // Load and clip asset bitmap, sampled down to the thumbnail size. Decoding these
            // camera-resolution assets at full size costs ~46MB each in native memory.
            Bitmap bitmap = AssetBitmaps.decodeSampled(
                    requireContext(), "chassis_images/" + filename, thumbSize);
            if (bitmap != null) {
                imgView.setImageBitmap(bitmap);
            } else {
                imgView.setImageResource(android.R.drawable.ic_menu_gallery);
            }

            // Material Rounded corners for the thumbnail
            GradientDrawable thumbGd = new GradientDrawable();
            thumbGd.setCornerRadius(6 * getResources().getDisplayMetrics().density);
            imgView.setBackground(thumbGd);

            // File Name Label
            android.widget.TextView label = new android.widget.TextView(getContext());
            label.setText(filename);
            label.setTextSize(10.5f);
            label.setTextColor(Color.WHITE);
            label.setTypeface(android.graphics.Typeface.MONOSPACE);
            label.setPadding((int)(8 * getResources().getDisplayMetrics().density), 0, 0, 0);

            itemLayout.addView(imgView);
            itemLayout.addView(label);

            final String finalFilename = filename;
            itemLayout.setOnClickListener(v -> {
                loadSelectedAsset(finalFilename);
                // Highlight item
                for (int i = 0; i < binding.layoutAssetsContainer.getChildCount(); i++) {
                    View child = binding.layoutAssetsContainer.getChildAt(i);
                    GradientDrawable childGd = (GradientDrawable) child.getBackground();
                    if (child == itemLayout) {
                        childGd.setColor(Color.rgb(26, 70, 128)); // Highlight deep blue
                        childGd.setStroke((int) (1.5 * getResources().getDisplayMetrics().density), Color.rgb(74, 222, 128)); // green highlight border
                    } else {
                        childGd.setColor(Color.rgb(43, 41, 48));
                        childGd.setStroke((int) (1 * getResources().getDisplayMetrics().density), Color.rgb(80, 80, 80));
                    }
                }
            });

            binding.layoutAssetsContainer.addView(itemLayout);
        }
    }

    private void loadSelectedAsset(String filename) {
        try {
            selectedImageExternalUri = Uri.parse("file:///android_asset/chassis_images/" + filename);
            // Preview only — the analysis pipeline re-loads the asset at its own resolution,
            // so there's no reason to hold a 4032x3024 bitmap for an on-screen preview.
            Bitmap bitmap = AssetBitmaps.decodeSampled(
                    requireContext(), "chassis_images/" + filename, 1080);
            if (bitmap != null) binding.ivChassisPreview.setImageBitmap(bitmap);
            binding.ivChassisPreview.setVisibility(View.VISIBLE);
            binding.layoutUploadPlaceholder.setVisibility(View.GONE);

            binding.btnCaptureInspect.setText("ANALYZE CHASSIS");
            binding.btnCaptureInspect.setBackgroundColor(ContextCompat.getColor(requireContext(), R.color.green_pass_header));
            binding.tvOperatorTip.setText("Locking coordinates. Simulating geometric verification analysis...");

            // Trigger simulated tuning
            simulateAlignmentTuning();
            Toast.makeText(getContext(), "Local inspection asset loaded: " + filename, Toast.LENGTH_SHORT).show();
        } catch (Exception e) {
            Toast.makeText(getContext(), "Failed to load asset: " + e.getMessage(), Toast.LENGTH_SHORT).show();
        }
    }

    private void launchImageChooser() {
        showCustomImagePickerDialog();
    }

    private void showCustomImagePickerDialog() {
        try {
            String[] files = requireContext().getAssets().list("chassis_images");
            java.util.List<String> imageFiles = new java.util.ArrayList<>();
            if (files != null) {
                for (String file : files) {
                    String lower = file.toLowerCase();
                    if (lower.endsWith(".jpg") || lower.endsWith(".png") || lower.endsWith(".jpeg") || lower.endsWith(".webp") || lower.endsWith(".bmp")) {
                        imageFiles.add(file);
                    }
                }
            }

            if (imageFiles.isEmpty()) {
                Toast.makeText(getContext(), "No uploaded assets found. Opening system gallery.", Toast.LENGTH_SHORT).show();
                selectImageLauncher.launch("image/*");
                return;
            }

            android.app.Dialog dialog = new android.app.Dialog(requireContext(), android.R.style.Theme_DeviceDefault_NoActionBar_Fullscreen);
            if (dialog.getWindow() != null) {
                dialog.getWindow().setBackgroundDrawable(new android.graphics.drawable.ColorDrawable(Color.argb(240, 15, 15, 18)));
            }

            android.widget.LinearLayout rootLayout = new android.widget.LinearLayout(getContext());
            rootLayout.setOrientation(android.widget.LinearLayout.VERTICAL);
            rootLayout.setPadding(32, 48, 32, 48);
            rootLayout.setBackgroundColor(Color.rgb(20, 20, 25));

            android.widget.TextView titleTv = new android.widget.TextView(getContext());
            titleTv.setText("CHOOSE CHASSIS TO INSPECT");
            titleTv.setTextSize(16f);
            titleTv.setTextColor(Color.rgb(74, 222, 128));
            titleTv.setTypeface(android.graphics.Typeface.DEFAULT_BOLD);
            titleTv.setGravity(android.view.Gravity.CENTER);
            titleTv.setPadding(0, 0, 0, 8);
            rootLayout.addView(titleTv);

            android.widget.TextView subTv = new android.widget.TextView(getContext());
            subTv.setText("Select one of your uploaded files in assets/chassis_images/ directory to perform geometric inspection on it:");
            subTv.setTextSize(11f);
            subTv.setTextColor(Color.LTGRAY);
            subTv.setGravity(android.view.Gravity.CENTER);
            subTv.setPadding(0, 0, 0, 24);
            rootLayout.addView(subTv);

            android.widget.ScrollView scrollView = new android.widget.ScrollView(getContext());
            android.widget.LinearLayout.LayoutParams scrollParams = new android.widget.LinearLayout.LayoutParams(
                    android.widget.LinearLayout.LayoutParams.MATCH_PARENT, 0, 1.0f);
            scrollView.setLayoutParams(scrollParams);

            android.widget.LinearLayout listContainer = new android.widget.LinearLayout(getContext());
            listContainer.setOrientation(android.widget.LinearLayout.VERTICAL);

            int dp8 = (int) (8 * getResources().getDisplayMetrics().density);
            int dp12 = (int) (12 * getResources().getDisplayMetrics().density);
            int thumbSize = (int) (56 * getResources().getDisplayMetrics().density);

            for (String filename : imageFiles) {
                android.widget.LinearLayout rowLayout = new android.widget.LinearLayout(getContext());
                rowLayout.setOrientation(android.widget.LinearLayout.HORIZONTAL);
                rowLayout.setGravity(android.view.Gravity.CENTER_VERTICAL);
                rowLayout.setPadding(dp12, dp12, dp12, dp12);

                android.widget.LinearLayout.LayoutParams rowParams = new android.widget.LinearLayout.LayoutParams(
                        android.widget.LinearLayout.LayoutParams.MATCH_PARENT,
                        android.widget.LinearLayout.LayoutParams.WRAP_CONTENT
                );
                rowParams.setMargins(0, 0, 0, dp12);
                rowLayout.setLayoutParams(rowParams);

                GradientDrawable gd = new GradientDrawable();
                gd.setColor(Color.rgb(35, 35, 40));
                gd.setCornerRadius(12 * getResources().getDisplayMetrics().density);
                gd.setStroke((int) (1 * getResources().getDisplayMetrics().density), Color.rgb(65, 65, 70));
                rowLayout.setBackground(gd);

                android.widget.ImageView imgView = new android.widget.ImageView(getContext());
                android.widget.LinearLayout.LayoutParams imgParams = new android.widget.LinearLayout.LayoutParams(thumbSize, thumbSize);
                imgView.setLayoutParams(imgParams);
                imgView.setScaleType(android.widget.ImageView.ScaleType.CENTER_CROP);
                imgView.setClipToOutline(true);

                Bitmap bitmap = AssetBitmaps.decodeSampled(
                        requireContext(), "chassis_images/" + filename, thumbSize);
                if (bitmap != null) {
                    imgView.setImageBitmap(bitmap);
                } else {
                    imgView.setImageResource(android.R.drawable.ic_menu_gallery);
                }

                GradientDrawable tgd = new GradientDrawable();
                tgd.setCornerRadius(6 * getResources().getDisplayMetrics().density);
                imgView.setBackground(tgd);

                android.widget.LinearLayout textLayout = new android.widget.LinearLayout(getContext());
                textLayout.setOrientation(android.widget.LinearLayout.VERTICAL);
                textLayout.setPadding(dp12, 0, 0, 0);
                android.widget.LinearLayout.LayoutParams textParams = new android.widget.LinearLayout.LayoutParams(
                        0, android.widget.LinearLayout.LayoutParams.WRAP_CONTENT, 1.0f);
                textLayout.setLayoutParams(textParams);

                android.widget.TextView nameTv = new android.widget.TextView(getContext());
                nameTv.setText(filename);
                nameTv.setTextColor(Color.WHITE);
                nameTv.setTextSize(13f);
                nameTv.setTypeface(android.graphics.Typeface.MONOSPACE);
                textLayout.addView(nameTv);

                android.widget.TextView pathTv = new android.widget.TextView(getContext());
                pathTv.setText("/assets/chassis_images/" + filename);
                pathTv.setTextColor(Color.GRAY);
                pathTv.setTextSize(10f);
                textLayout.addView(pathTv);

                android.widget.ImageView arrowImg = new android.widget.ImageView(getContext());
                android.widget.LinearLayout.LayoutParams arrowParams = new android.widget.LinearLayout.LayoutParams(
                        (int) (20 * getResources().getDisplayMetrics().density),
                        (int) (20 * getResources().getDisplayMetrics().density)
                );
                arrowImg.setLayoutParams(arrowParams);
                arrowImg.setImageResource(android.R.drawable.ic_media_play);
                arrowImg.setColorFilter(Color.rgb(74, 222, 128));

                rowLayout.addView(imgView);
                rowLayout.addView(textLayout);
                rowLayout.addView(arrowImg);

                final String finalFilename = filename;
                rowLayout.setOnClickListener(v -> {
                    loadSelectedAsset(finalFilename);
                    dialog.dismiss();
                });

                listContainer.addView(rowLayout);
            }

            scrollView.addView(listContainer);
            rootLayout.addView(scrollView);

            android.widget.LinearLayout controlLayout = new android.widget.LinearLayout(getContext());
            controlLayout.setOrientation(android.widget.LinearLayout.HORIZONTAL);
            controlLayout.setPadding(0, 16, 0, 0);

            android.widget.Button galleryBtn = new android.widget.Button(getContext());
            android.widget.LinearLayout.LayoutParams btnParams1 = new android.widget.LinearLayout.LayoutParams(
                    0, android.widget.LinearLayout.LayoutParams.WRAP_CONTENT, 1.0f);
            btnParams1.rightMargin = dp8;
            galleryBtn.setLayoutParams(btnParams1);
            galleryBtn.setText("OPEN SYSTEM GALLERY");
            galleryBtn.setTextColor(Color.WHITE);
            galleryBtn.setBackgroundColor(Color.rgb(40, 50, 70));
            galleryBtn.setOnClickListener(v -> {
                dialog.dismiss();
                selectImageLauncher.launch("image/*");
            });
            controlLayout.addView(galleryBtn);

            android.widget.Button closeBtn = new android.widget.Button(getContext());
            android.widget.LinearLayout.LayoutParams btnParams2 = new android.widget.LinearLayout.LayoutParams(
                    0, android.widget.LinearLayout.LayoutParams.WRAP_CONTENT, 1.0f);
            btnParams2.leftMargin = dp8;
            closeBtn.setLayoutParams(btnParams2);
            closeBtn.setText("CLOSE");
            closeBtn.setTextColor(Color.WHITE);
            closeBtn.setBackgroundColor(Color.rgb(60, 40, 40));
            closeBtn.setOnClickListener(v -> dialog.dismiss());
            controlLayout.addView(closeBtn);

            rootLayout.addView(controlLayout);
            dialog.setContentView(rootLayout);
            dialog.show();

        } catch (Exception e) {
            Log.e(TAG, "Failed layout popup", e);
            selectImageLauncher.launch("image/*");
        }
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
                        binding.tvGhostStatusIndicator.setText("DEFAULT REFERENCING (NO OVERLAY)");
                    } else {
                        if (imgUriStr.startsWith("file:///android_asset/")) {
                            String assetPath = imgUriStr.substring("file:///android_asset/".length());
                            Bitmap ghost = AssetBitmaps.decodeSampled(requireContext(), assetPath, 1080);
                            if (ghost != null) {
                                binding.ivGhostOverlay.setImageBitmap(ghost);
                            } else {
                                binding.ivGhostOverlay.setImageURI(Uri.parse(imgUriStr));
                            }
                        } else {
                            binding.ivGhostOverlay.setImageURI(Uri.parse(imgUriStr));
                        }
                        binding.ivGhostOverlay.setVisibility(isGhostEnabled ? View.VISIBLE : View.GONE);
                        binding.tvGhostStatusIndicator.setText("GHOST OVERLAY ON (35%)");
                    }
                } else {
                    binding.ivGhostOverlay.setVisibility(View.GONE);
                    binding.tvGhostStatusIndicator.setText("NO ACTIVE GHOST OVERLAY");
                }
            });
        }).start();
    }

    private void persistSelectedImage(Uri inputUri) {
        try {
            InputStream inputStream = requireContext().getContentResolver().openInputStream(inputUri);
            if (inputStream == null) {
                Toast.makeText(getContext(), "Cannot read selected image", Toast.LENGTH_SHORT).show();
                return;
            }

            File outputDir = requireContext().getExternalFilesDir("captured_inspections");
            if (outputDir != null && !outputDir.exists()) {
                outputDir.mkdirs();
            }

            String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new java.util.Date());
            File photoFile = new File(outputDir, "COMP_" + partId + "_" + timeStamp + ".jpg");

            FileOutputStream outputStream = new FileOutputStream(photoFile);
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, bytesRead);
            }
            outputStream.close();
            inputStream.close();

            selectedImageExternalUri = Uri.fromFile(photoFile);

            binding.ivChassisPreview.setImageURI(selectedImageExternalUri);
            binding.ivChassisPreview.setVisibility(View.VISIBLE);
            binding.layoutUploadPlaceholder.setVisibility(View.GONE);

            binding.btnCaptureInspect.setText("ANALYZE CHASSIS");
            binding.btnCaptureInspect.setBackgroundColor(ContextCompat.getColor(requireContext(), R.color.green_pass_header));
            binding.tvOperatorTip.setText("Locking coordinates. Simulating geometric verification analysis...");

            // Trigger simulated tuning
            simulateAlignmentTuning();

        } catch (Exception e) {
            Log.e(TAG, "Failed to copy chassis inspection photo", e);
            Toast.makeText(getContext(), "Error choosing file: " + e.getMessage(), Toast.LENGTH_LONG).show();
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

    private void resetAlignmentState() {
        binding.tvAlignmentState.setText("WAITING USER PHOTO...");
        binding.tvAlignmentState.setBackgroundColor(ContextCompat.getColor(requireContext(), R.color.orange_warn));
        binding.layoutCameraEnclosure.setBackgroundResource(R.drawable.align_border_red);
        isAligned = false;
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
        }, 1800); // Transitions to success in 1.8 seconds
    }

    private void triggerAlignmentAchieved() {
        isAligned = true;
        binding.tvAlignmentState.setText("ALIGNED (98%)");
        binding.tvAlignmentState.setBackgroundColor(ContextCompat.getColor(requireContext(), R.color.green_pass));
        binding.layoutCameraEnclosure.setBackgroundResource(R.drawable.align_border_green);
        binding.tvOperatorTip.setText("Alignment matched reference gold standard! Ready.");
        Toast.makeText(getContext(), "Position matched. Locked.", Toast.LENGTH_SHORT).show();
    }

    private void analyzeAndNavigate() {
        if (selectedImageExternalUri == null) return;

        binding.btnCaptureInspect.setEnabled(false);
        Toast.makeText(getContext(), "Running local geometric verification...", Toast.LENGTH_SHORT).show();

        // 1-second delay to simulate complex computer vision processing
        alignmentSimulatorHandler.postDelayed(() -> navigateToResults(selectedImageExternalUri.toString()), 1000);
    }

    private void navigateToResults(String imageUri) {
        if (isAdded() && getView() != null) {
            Bundle args = new Bundle();
            args.putString("partId", partId);
            args.putString("imageUri", imageUri);
            Navigation.findNavController(getView()).navigate(R.id.action_inspection_to_results, args);
        }
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        if (alignmentSimulatorHandler != null) {
            alignmentSimulatorHandler.removeCallbacksAndMessages(null);
        }
        binding = null;
    }
}
