package com.example.ui.fragment;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.drawable.GradientDrawable;
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
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;
import androidx.navigation.Navigation;

import com.example.R;
import com.example.data.AppDatabase;
import com.example.data.entity.ReferenceProfile;
import com.example.ui.util.AssetBitmaps;
import com.example.databinding.FragmentReferenceSetupBinding;
import com.example.ml.FeatureEmbedder;
import com.example.ml.ReferenceLibrary;
import com.example.ml.ReferenceLibraryProvider;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class ReferenceSetupFragment extends Fragment {

    private static final String TAG = "ReferenceSetupFragment";

    private FragmentReferenceSetupBinding binding;
    private String partId = "";
    private Uri selectedImageExternalUri = null;

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
        binding = FragmentReferenceSetupBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        binding.tvSetupPartId.setText("Part ID: " + partId);

        // Scan the custom user_images folder in assets
        scanAssetsFolder();

        // Click on the upload zone to pick an image
        binding.layoutUploadZone.setOnClickListener(v -> launchImageChooser());

        binding.btnCaptureReference.setOnClickListener(v -> {
            if (selectedImageExternalUri == null) {
                launchImageChooser();
            } else {
                saveReferenceToDb(selectedImageExternalUri.toString());
            }
        });

        binding.btnSkip.setOnClickListener(v -> skipSetupReference());

        binding.btnUsePresetReference.setOnClickListener(v -> {
            loadSelectedAsset("chassis_hardware_preset.jpg");
        });

        binding.btnBuildLibrary.setOnClickListener(v -> {
            if (selectedImageExternalUri == null) {
                Toast.makeText(getContext(), "Choose or load a reference image first.", Toast.LENGTH_SHORT).show();
                launchImageChooser();
            } else {
                startBuildLibrary(selectedImageExternalUri.toString());
            }
        });
    }

    // ── Few-shot library builder ────────────────────────────────────────

    private static final String[] LABEL_OPTIONS =
            {"skip", "bolt_head", "hole", "rivet_head", "screw", "stud"};

    /** Extracts feature crops from the selected reference image, then shows the labelling dialog. */
    private void startBuildLibrary(String uriString) {
        Toast.makeText(getContext(), "Detecting features in reference image...", Toast.LENGTH_SHORT).show();
        new Thread(() -> {
            List<HardwareFeatureExtractor.CropRegion> crops =
                    HardwareFeatureExtractor.extractCrops(requireContext(), uriString);
            requireActivity().runOnUiThread(() -> {
                if (!isAdded() || binding == null) return;
                if (crops.isEmpty()) {
                    Toast.makeText(getContext(), "No features detected in this image.", Toast.LENGTH_LONG).show();
                } else {
                    showLabelDialog(crops);
                }
            });
        }).start();
    }

    /** Fullscreen dialog: one row per crop (thumbnail + type dropdown) + a save button. */
    private void showLabelDialog(List<HardwareFeatureExtractor.CropRegion> crops) {
        android.app.Dialog dialog = new android.app.Dialog(requireContext(),
                android.R.style.Theme_DeviceDefault_NoActionBar_Fullscreen);

        android.widget.LinearLayout root = new android.widget.LinearLayout(getContext());
        root.setOrientation(android.widget.LinearLayout.VERTICAL);
        root.setBackgroundColor(Color.rgb(20, 20, 25));
        root.setPadding(dp(20), dp(28), dp(20), dp(20));

        android.widget.TextView title = new android.widget.TextView(getContext());
        title.setText("LABEL DETECTED FEATURES");
        title.setTextColor(Color.rgb(74, 222, 128));
        title.setTextSize(16f);
        title.setTypeface(android.graphics.Typeface.DEFAULT_BOLD);
        title.setGravity(android.view.Gravity.CENTER);
        root.addView(title);

        android.widget.TextView sub = new android.widget.TextView(getContext());
        sub.setText("Assign a hardware type to each crop (or leave 'skip'). These build the reference library for this part.");
        sub.setTextColor(Color.LTGRAY);
        sub.setTextSize(11f);
        sub.setGravity(android.view.Gravity.CENTER);
        sub.setPadding(0, dp(4), 0, dp(16));
        root.addView(sub);

        android.widget.ScrollView scroll = new android.widget.ScrollView(getContext());
        android.widget.LinearLayout.LayoutParams scrollParams = new android.widget.LinearLayout.LayoutParams(
                android.widget.LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f);
        scroll.setLayoutParams(scrollParams);

        android.widget.LinearLayout list = new android.widget.LinearLayout(getContext());
        list.setOrientation(android.widget.LinearLayout.VERTICAL);

        final List<android.widget.Spinner> spinners = new ArrayList<>();
        int thumb = dp(64);
        for (HardwareFeatureExtractor.CropRegion crop : crops) {
            android.widget.LinearLayout row = new android.widget.LinearLayout(getContext());
            row.setOrientation(android.widget.LinearLayout.HORIZONTAL);
            row.setGravity(android.view.Gravity.CENTER_VERTICAL);
            row.setPadding(dp(8), dp(8), dp(8), dp(8));
            android.widget.LinearLayout.LayoutParams rowLp = new android.widget.LinearLayout.LayoutParams(
                    android.widget.LinearLayout.LayoutParams.MATCH_PARENT,
                    android.widget.LinearLayout.LayoutParams.WRAP_CONTENT);
            rowLp.setMargins(0, 0, 0, dp(10));
            row.setLayoutParams(rowLp);

            android.widget.ImageView img = new android.widget.ImageView(getContext());
            img.setLayoutParams(new android.widget.LinearLayout.LayoutParams(thumb, thumb));
            img.setScaleType(android.widget.ImageView.ScaleType.CENTER_CROP);
            img.setImageBitmap(crop.bitmap);
            row.addView(img);

            android.widget.Spinner spinner = new android.widget.Spinner(getContext());
            android.widget.LinearLayout.LayoutParams spLp = new android.widget.LinearLayout.LayoutParams(
                    0, android.widget.LinearLayout.LayoutParams.WRAP_CONTENT, 1f);
            spLp.setMarginStart(dp(12));
            spinner.setLayoutParams(spLp);
            android.widget.ArrayAdapter<String> adapter = new android.widget.ArrayAdapter<>(
                    requireContext(), android.R.layout.simple_spinner_dropdown_item, LABEL_OPTIONS);
            spinner.setAdapter(adapter);
            row.addView(spinner);
            spinners.add(spinner);

            list.addView(row);
        }
        scroll.addView(list);
        root.addView(scroll);

        android.widget.Button save = new android.widget.Button(getContext());
        save.setText("SAVE LIBRARY");
        save.setTextColor(Color.WHITE);
        save.setBackgroundColor(Color.rgb(26, 70, 128));
        android.widget.LinearLayout.LayoutParams saveLp = new android.widget.LinearLayout.LayoutParams(
                android.widget.LinearLayout.LayoutParams.MATCH_PARENT,
                android.widget.LinearLayout.LayoutParams.WRAP_CONTENT);
        saveLp.setMargins(0, dp(12), 0, 0);
        save.setLayoutParams(saveLp);
        save.setOnClickListener(v -> {
            save.setEnabled(false);
            saveLibrary(crops, spinners, dialog);
        });
        root.addView(save);

        dialog.setContentView(root);
        dialog.show();
    }

    /** Embeds each labelled crop, averages per type, and saves the per-part library. */
    private void saveLibrary(List<HardwareFeatureExtractor.CropRegion> crops,
                             List<android.widget.Spinner> spinners,
                             android.app.Dialog dialog) {
        Toast.makeText(getContext(), "Encoding crops...", Toast.LENGTH_SHORT).show();
        new Thread(() -> {
            FeatureEmbedder embedder = FeatureEmbedder.get(requireContext());
            if (embedder == null) {
                requireActivity().runOnUiThread(() ->
                        Toast.makeText(getContext(),
                                "Feature encoder not available (missing assets/models/feature_encoder.onnx).",
                                Toast.LENGTH_LONG).show());
                return;
            }

            ReferenceLibrary lib = new ReferenceLibrary();
            int labelled = 0;
            for (int i = 0; i < crops.size(); i++) {
                String type = (String) spinners.get(i).getSelectedItem();
                if (type == null || "skip".equals(type)) continue;
                float[] emb = embedder.embed(crops.get(i).bitmap);
                if (emb != null) {
                    lib.addExample(type, emb);
                    labelled++;
                }
            }

            final int count = labelled;
            if (lib.isEmpty()) {
                requireActivity().runOnUiThread(() ->
                        Toast.makeText(getContext(), "No crops were labelled — nothing saved.", Toast.LENGTH_LONG).show());
                return;
            }

            lib.save(ReferenceLibraryProvider.fileForPart(requireContext(), partId));

            // Also persist the reference profile so inspections compute expected counts from this image.
            AppDatabase db = AppDatabase.getInstance(requireContext());
            db.referenceProfileDao().insertProfile(
                    new ReferenceProfile(partId, selectedImageExternalUri.toString(), System.currentTimeMillis()));

            requireActivity().runOnUiThread(() -> {
                if (dialog.isShowing()) dialog.dismiss();
                Toast.makeText(getContext(),
                        "Library saved (" + count + " examples across " + lib.types().size() + " types).",
                        Toast.LENGTH_LONG).show();
                proceedToInspection();
            });
        }).start();
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
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
                binding.tvAssetsFolderStatus.setText("No images found in '/assets/chassis_images/'. Put your .jpg in the folder!");
                binding.scrollAssets.setVisibility(View.GONE);
            } else {
                binding.tvAssetsFolderStatus.setText("Found " + imageFiles.size() + " image(s) in assets/chassis_images/! Tap to load:");
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
            // Preview only — extractCrops() re-loads the asset at its own resolution.
            Bitmap bitmap = AssetBitmaps.decodeSampled(
                    requireContext(), "chassis_images/" + filename, 1080);
            if (bitmap != null) binding.ivReferencePreview.setImageBitmap(bitmap);
            binding.ivReferencePreview.setVisibility(View.VISIBLE);
            binding.layoutUploadPlaceholder.setVisibility(View.GONE);
            binding.btnCaptureReference.setText("SAVE REFERENCE");
            binding.btnCaptureReference.setBackgroundColor(ContextCompat.getColor(requireContext(), R.color.green_pass_header));
            Toast.makeText(getContext(), "Local asset loaded: " + filename, Toast.LENGTH_SHORT).show();
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
            titleTv.setText("CHOOSE REFERENCE CHASSIS PHOTO");
            titleTv.setTextSize(16f);
            titleTv.setTextColor(Color.rgb(74, 222, 128));
            titleTv.setTypeface(android.graphics.Typeface.DEFAULT_BOLD);
            titleTv.setGravity(android.view.Gravity.CENTER);
            titleTv.setPadding(0, 0, 0, 8);
            rootLayout.addView(titleTv);

            android.widget.TextView subTv = new android.widget.TextView(getContext());
            subTv.setText("Select one of your uploaded files in assets/chassis_images/ directory to load it directly:");
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

    private void persistSelectedImage(Uri inputUri) {
        try {
            InputStream inputStream = requireContext().getContentResolver().openInputStream(inputUri);
            if (inputStream == null) {
                Toast.makeText(getContext(), "Cannot read selected image", Toast.LENGTH_SHORT).show();
                return;
            }

            File outputDir = requireContext().getExternalFilesDir("references");
            if (outputDir != null && !outputDir.exists()) {
                outputDir.mkdirs();
            }

            String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new java.util.Date());
            File photoFile = new File(outputDir, "REF_" + partId + "_" + timeStamp + ".jpg");

            FileOutputStream outputStream = new FileOutputStream(photoFile);
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, bytesRead);
            }
            outputStream.close();
            inputStream.close();

            selectedImageExternalUri = Uri.fromFile(photoFile);

            binding.ivReferencePreview.setImageURI(selectedImageExternalUri);
            binding.ivReferencePreview.setVisibility(View.VISIBLE);
            binding.layoutUploadPlaceholder.setVisibility(View.GONE);
            binding.btnCaptureReference.setText("SAVE REFERENCE");
            binding.btnCaptureReference.setBackgroundColor(ContextCompat.getColor(requireContext(), R.color.green_pass_header));
            Toast.makeText(getContext(), "Image uploaded successfully!", Toast.LENGTH_SHORT).show();
        } catch (Exception e) {
            Log.e(TAG, "Failed to copy reference image", e);
            Toast.makeText(getContext(), "Error saving image: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    private void saveReferenceToDb(String imageUri) {
        binding.btnCaptureReference.setEnabled(false);
        new Thread(() -> {
            AppDatabase db = AppDatabase.getInstance(requireContext());
            ReferenceProfile profile = new ReferenceProfile(partId, imageUri, System.currentTimeMillis());
            db.referenceProfileDao().insertProfile(profile);

            requireActivity().runOnUiThread(() -> {
                Toast.makeText(getContext(), "Reference profile established!", Toast.LENGTH_SHORT).show();
                proceedToInspection();
            });
        }).start();
    }

    private void skipSetupReference() {
        new Thread(() -> {
            AppDatabase db = AppDatabase.getInstance(requireContext());
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
        binding = null;
    }
}
