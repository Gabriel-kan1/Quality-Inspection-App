package com.example.ui.fragment;

import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.drawable.GradientDrawable;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.navigation.Navigation;

import com.example.R;
import com.example.data.AppDatabase;
import com.example.data.entity.ReferenceProfile;
import com.example.databinding.FragmentBarcodeScanBinding;
import com.example.ml.HardwareFeatureExtractor;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class BarcodeScanFragment extends Fragment {

    /**
     * Part IDs always offered even with nothing saved against them, so the emulator test
     * shortcuts survive a cleared database.
     */
    private static final String[] PRESET_IDS =
            {"CHASSIS-8821-B", "PART-CHASSIS-A03", "ENGINE-COV-X42"};

    /** Thumbnail edge in px to decode at — small enough that a 4000px reference costs little. */
    private static final int THUMB_PX = 160;

    private FragmentBarcodeScanBinding binding;

    /** One row of the profile list: a part, and the reference photo saved for it if any. */
    private static final class ProfileRow {
        final String partId;
        final String imageUri;
        Bitmap thumb;

        ProfileRow(String partId, String imageUri) {
            this.partId = partId;
            this.imageUri = imageUri;
        }

        boolean hasReference() {
            return imageUri != null && !ReferenceProfile.PLACEHOLDER_URI.equals(imageUri);
        }
    }

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        binding = FragmentBarcodeScanBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        binding.btnConfirmPart.setOnClickListener(v -> {
            String enteredPartId = binding.etPartId.getText().toString().trim();
            if (enteredPartId.isEmpty()) {
                Toast.makeText(getContext(), "Please enter a valid Part ID", Toast.LENGTH_SHORT).show();
                return;
            }
            checkReferenceAndProceed(enteredPartId);
        });

        // Go straight to setup so an already-configured part can be given a new reference photo.
        binding.btnChangeReference.setOnClickListener(v -> {
            String enteredPartId = binding.etPartId.getText().toString().trim();
            if (enteredPartId.isEmpty()) {
                Toast.makeText(getContext(), "Please enter a valid Part ID", Toast.LENGTH_SHORT).show();
                return;
            }
            Bundle args = new Bundle();
            args.putString("partId", enteredPartId);
            Navigation.findNavController(requireView()).navigate(R.id.action_barcode_to_reference_setup, args);
        });

        loadProfiles();
    }

    @Override
    public void onResume() {
        super.onResume();
        // Coming back from reference setup, a part may now have a photo it did not have before.
        loadProfiles();
    }

    /**
     * Reads the saved profiles and decodes their reference thumbnails off the main thread.
     *
     * References can be camera-resolution stills, so decoding them at full size for a 56dp
     * row would cost tens of megabytes each — they are sampled down during decode instead.
     */
    private void loadProfiles() {
        new Thread(() -> {
            List<ProfileRow> rows = new ArrayList<>();
            try {
                AppDatabase db = AppDatabase.getInstance(requireContext());
                // Saved profiles first, newest last-created first, then any preset that has
                // nothing saved yet so the shortcuts remain available on a fresh install.
                Map<String, ProfileRow> byId = new LinkedHashMap<>();
                List<ReferenceProfile> profiles = db.referenceProfileDao().getAllProfiles();
                if (profiles != null) {
                    for (ReferenceProfile p : profiles) {
                        if (p.getPartType() == null) continue;
                        byId.put(p.getPartType(), new ProfileRow(p.getPartType(), p.getImageUri()));
                    }
                }
                for (String preset : PRESET_IDS) {
                    if (!byId.containsKey(preset)) byId.put(preset, new ProfileRow(preset, null));
                }
                rows.addAll(byId.values());

                for (ProfileRow r : rows) {
                    if (r.hasReference()) {
                        r.thumb = HardwareFeatureExtractor.loadBitmap(
                                requireContext(), r.imageUri, THUMB_PX);
                    }
                }
            } catch (Exception e) {
                android.util.Log.e("BarcodeScanFragment", "Failed to load profiles", e);
            }

            final List<ProfileRow> finalRows = rows;
            if (!isAdded()) return;
            requireActivity().runOnUiThread(() -> {
                if (!isAdded() || binding == null) return;
                renderProfiles(finalRows);
            });
        }).start();
    }

    private void renderProfiles(List<ProfileRow> rows) {
        LinearLayout container = binding.layoutProfileRows;
        container.removeAllViews();

        int withReference = 0;
        for (ProfileRow r : rows) if (r.hasReference()) withReference++;
        binding.tvProfilesSubtitle.setText(withReference == 0
                ? "No reference photos saved yet. Pick a part to set one up."
                : withReference + " of " + rows.size() + " parts have a reference photo.");

        for (ProfileRow r : rows) container.addView(buildRow(r));
    }

    private View buildRow(ProfileRow row) {
        LinearLayout item = new LinearLayout(requireContext());
        item.setOrientation(LinearLayout.HORIZONTAL);
        item.setGravity(android.view.Gravity.CENTER_VERTICAL);
        item.setPadding(dp(10), dp(10), dp(14), dp(10));

        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        lp.setMargins(0, 0, 0, dp(8));
        item.setLayoutParams(lp);

        GradientDrawable bg = new GradientDrawable();
        bg.setColor(Color.rgb(43, 41, 48));
        bg.setCornerRadius(dp(12));
        bg.setStroke(dp(1), Color.rgb(80, 80, 90));
        item.setBackground(bg);

        ImageView thumb = new ImageView(requireContext());
        int size = dp(56);
        thumb.setLayoutParams(new LinearLayout.LayoutParams(size, size));
        thumb.setScaleType(ImageView.ScaleType.CENTER_CROP);
        thumb.setClipToOutline(true);
        GradientDrawable thumbBg = new GradientDrawable();
        thumbBg.setColor(Color.rgb(24, 24, 28));
        thumbBg.setCornerRadius(dp(8));
        thumb.setBackground(thumbBg);
        if (row.thumb != null) {
            thumb.setImageBitmap(row.thumb);
        } else {
            thumb.setImageResource(android.R.drawable.ic_menu_camera);
            thumb.setColorFilter(Color.rgb(110, 110, 120));
            thumb.setPadding(dp(14), dp(14), dp(14), dp(14));
        }
        item.addView(thumb);

        LinearLayout text = new LinearLayout(requireContext());
        text.setOrientation(LinearLayout.VERTICAL);
        LinearLayout.LayoutParams textLp =
                new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f);
        textLp.setMarginStart(dp(12));
        text.setLayoutParams(textLp);

        TextView name = new TextView(requireContext());
        name.setText(row.partId);
        name.setTextColor(Color.WHITE);
        name.setTextSize(15f);
        name.setTypeface(android.graphics.Typeface.DEFAULT_BOLD);
        name.setSingleLine(true);
        name.setEllipsize(android.text.TextUtils.TruncateAt.END);
        text.addView(name);

        TextView status = new TextView(requireContext());
        if (row.hasReference()) {
            status.setText("Reference set — " + fileNameOf(row.imageUri));
            status.setTextColor(Color.rgb(120, 200, 150));
        } else {
            status.setText("No reference photo — setup required");
            status.setTextColor(Color.rgb(235, 170, 90));
        }
        status.setTextSize(11.5f);
        status.setSingleLine(true);
        status.setEllipsize(android.text.TextUtils.TruncateAt.END);
        text.addView(status);
        item.addView(text);

        item.setOnClickListener(v -> {
            binding.etPartId.setText(row.partId);
            highlightSelection(row.partId);
        });
        return item;
    }

    /** Marks the row matching the current identifier, so the field and the list agree. */
    private void highlightSelection(String partId) {
        LinearLayout container = binding.layoutProfileRows;
        for (int i = 0; i < container.getChildCount(); i++) {
            View child = container.getChildAt(i);
            if (!(child instanceof LinearLayout)) continue;
            View textCol = ((LinearLayout) child).getChildAt(1);
            if (!(textCol instanceof LinearLayout)) continue;
            View nameView = ((LinearLayout) textCol).getChildAt(0);
            if (!(nameView instanceof TextView)) continue;
            boolean selected = partId.equals(((TextView) nameView).getText().toString());
            GradientDrawable bg = (GradientDrawable) child.getBackground();
            bg.setColor(selected ? Color.rgb(26, 70, 128) : Color.rgb(43, 41, 48));
            bg.setStroke(dp(selected ? 2 : 1),
                    selected ? Color.rgb(74, 222, 128) : Color.rgb(80, 80, 90));
        }
    }

    private static String fileNameOf(String uri) {
        if (uri == null) return "";
        int slash = uri.lastIndexOf('/');
        return slash >= 0 ? uri.substring(slash + 1) : uri;
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private void checkReferenceAndProceed(String partId) {
        new Thread(() -> {
            // Check in AppDatabase if ReferenceProfile exists
            AppDatabase db = AppDatabase.getInstance(requireContext());
            ReferenceProfile profile = db.referenceProfileDao().getProfileByPartType(partId);

            requireActivity().runOnUiThread(() -> {
                Bundle args = new Bundle();
                args.putString("partId", partId);
                
                // A placeholder profile means setup was skipped, so there's no image for the
                // alignment gate to work against and inspection would open already blocked.
                // Treat it the same as having no profile at all and ask for a reference.
                boolean hasReference = profile != null && profile.hasRealImage();

                if (hasReference) {
                    Navigation.findNavController(requireView()).navigate(R.id.action_barcode_to_inspection, args);
                } else {
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
