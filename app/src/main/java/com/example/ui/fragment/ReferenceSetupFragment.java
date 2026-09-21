package com.example.ui.fragment;

import android.graphics.Bitmap;
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
import com.example.data.entity.ExpectedFeature;
import com.example.data.entity.ReferenceProfile;
import com.example.databinding.FragmentReferenceSetupBinding;
import com.example.ml.FeatureLayout;
import com.example.ml.FeatureLocalisation;
import com.example.ml.FeatureTypeRegistry;
import com.example.ml.HardwareFeatureExtractor;
import com.example.ml.PartFiles;
import com.example.ml.ReferenceLibrary;
import com.example.ml.ReferenceLibraryProvider;
import com.example.ml.ThresholdTuner;
import com.example.ui.util.AssetBitmaps;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Reference setup: choose the gold-standard photo, declare how many of each feature the part
 * has, confirm.
 *
 * <p>The counts are the operator's, typed in; they are the expected side of every inspection's
 * PASS/FAIL and never come from a detector. The photograph is detected once, here, for the one
 * thing counts cannot give: where each feature is and how big its box is, which is what the
 * positional check and the dimensional measurement on the results screen work from. The
 * detector's boxes are shown on the preview and saved as the part's {@link FeatureLayout}; the
 * classifier names each box, so there is no labelling dialog. If the detector finds fewer
 * positions than were declared, the screen says so — the counts still stand, and localisation
 * simply tracks the positions it has.
 */
public class ReferenceSetupFragment extends Fragment {

    private static final String TAG = "ReferenceSetupFragment";

    /** Backing image for the "USE BUILT-IN PRESET" shortcut; not currently shipped in assets. */
    private static final String PRESET_ASSET = "chassis_hardware_preset.jpg";

    private FragmentReferenceSetupBinding binding;
    private String partId = "";
    private Uri selectedImageExternalUri = null;

    /** One number field per feature type, keyed by the type's registry key, in row order. */
    private final Map<String, android.widget.EditText> countFields = new LinkedHashMap<>();

    /** True while a confirm is being written, so a double tap cannot file the reference twice. */
    private boolean confirming = false;

    /** The detector's pass over the chosen photo, or null until it has finished. */
    @Nullable
    private HardwareFeatureExtractor.AnalysisResult pendingResult = null;
    /** True from choosing a photo until its detection pass has posted; confirm waits for it. */
    private boolean analysing = false;
    /**
     * Guards against a slow pass over a discarded photo overwriting a newer one's boxes.
     * Incremented every time a photo is chosen; a finishing pass only publishes if its token
     * still matches.
     */
    private int analysisToken = 0;
    /** Largest share of the screen height the preview may take, so the counts stay reachable. */
    private static final float PREVIEW_MAX_HEIGHT_FRACTION = 0.5f;

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

        // The count rows exist before any photo does: a part's bill of features is a property
        // of the part, and the seeded presets arrive with theirs already filled in.
        buildCountRows(new LinkedHashMap<>());
        loadDeclaredCounts();

        binding.btnAddFeatureType.setOnClickListener(v ->
                showCreateTypeDialog(() -> buildCountRows(currentRawCounts())));

        // One button, three states (see confirmLabel): it opens the chooser until a photo is
        // picked, then waits for a non-zero declaration, then confirms both together.
        binding.btnCaptureReference.setOnClickListener(v -> {
            if (selectedImageExternalUri == null) {
                launchImageChooser();
            } else if (total(countsFrom(currentRawCounts())) > 0) {
                confirmReference();
            }
        });
        refreshConfirmButton();

        // The preset image is not in the repo. Offering a button that loads nothing looks like a
        // working shortcut and ends in an empty reference, so only show it if the asset is
        // actually there; the assets browser below already lists whatever is.
        if (assetExists("chassis_images/" + PRESET_ASSET)) {
            binding.btnUsePresetReference.setOnClickListener(v -> loadSelectedAsset(PRESET_ASSET));
        } else {
            binding.btnUsePresetReference.setVisibility(View.GONE);
        }
    }

    // ── The declaration: one count per feature type ─────────────────────

    /**
     * Rebuilds the count rows from the type catalog, restoring {@code keep} into any row it
     * has a value for.
     *
     * <p>Critical types only. Advisory and ignored types never enter the count comparison —
     * that is what those severities mean — so a number typed against one would be stored and
     * never read, which is worse than not asking. A count already declared for a type that has
     * since left the catalog still gets a row, so it is visible rather than silently dropped.
     */
    private void buildCountRows(Map<String, String> keep) {
        binding.layoutExpectedTypeRows.removeAllViews();
        countFields.clear();

        for (FeatureTypeRegistry.TypeInfo t : FeatureTypeRegistry.getAll(requireContext())) {
            if (t.severity != FeatureTypeRegistry.Severity.CRITICAL) continue;
            addCountRow(t.key, t.displayName, keep.get(t.key));
        }
        for (Map.Entry<String, String> e : keep.entrySet()) {
            if (countFields.containsKey(e.getKey())) continue;
            if (countsFrom(java.util.Collections.singletonMap(e.getKey(), e.getValue())).isEmpty()) continue;
            addCountRow(e.getKey(),
                    FeatureTypeRegistry.displayName(requireContext(), e.getKey()), e.getValue());
        }
        // The keyboard covers the lower rows, the total and the confirm button (the screen pans
        // rather than resizes), so Next is how the operator walks down the list — and on the
        // last row it has to become Done, or the keyboard stays up over the button they need.
        android.widget.EditText last = null;
        for (android.widget.EditText f : countFields.values()) last = f;
        if (last != null) {
            last.setImeOptions(android.view.inputmethod.EditorInfo.IME_ACTION_DONE);
            last.setOnEditorActionListener((v, actionId, event) -> {
                if (actionId != android.view.inputmethod.EditorInfo.IME_ACTION_DONE) return false;
                v.clearFocus();
                android.view.inputmethod.InputMethodManager imm =
                        (android.view.inputmethod.InputMethodManager) requireContext()
                                .getSystemService(android.content.Context.INPUT_METHOD_SERVICE);
                if (imm != null) imm.hideSoftInputFromWindow(v.getWindowToken(), 0);
                return true;
            });
        }
        refreshConfirmButton();
    }

    /** One "swatch — name — [ n ]" row, in the legend style the results screen uses. */
    private void addCountRow(String type, String displayName, @Nullable String value) {
        android.widget.LinearLayout row = new android.widget.LinearLayout(requireContext());
        row.setOrientation(android.widget.LinearLayout.HORIZONTAL);
        row.setGravity(android.view.Gravity.CENTER_VERTICAL);
        row.setPadding(0, dp(3), 0, dp(3));

        View swatch = new View(requireContext());
        swatch.setLayoutParams(new android.widget.LinearLayout.LayoutParams(dp(12), dp(12)));
        swatch.setBackgroundColor(ReferenceLibrary.colorFor(type));
        row.addView(swatch);

        android.widget.TextView name = new android.widget.TextView(requireContext());
        android.widget.LinearLayout.LayoutParams nameLp = new android.widget.LinearLayout.LayoutParams(
                0, android.widget.LinearLayout.LayoutParams.WRAP_CONTENT, 1f);
        nameLp.setMarginStart(dp(8));
        name.setLayoutParams(nameLp);
        name.setText(displayName);
        name.setTextColor(ContextCompat.getColor(requireContext(), R.color.white));
        name.setTextSize(12f);
        row.addView(name);

        android.widget.EditText field = new android.widget.EditText(requireContext());
        field.setLayoutParams(new android.widget.LinearLayout.LayoutParams(dp(72), dp(40)));
        field.setBackground(ContextCompat.getDrawable(requireContext(), R.drawable.bg_number_field));
        field.setGravity(android.view.Gravity.CENTER);
        field.setHint("0");
        field.setInputType(android.text.InputType.TYPE_CLASS_NUMBER);
        field.setFilters(new android.text.InputFilter[]{new android.text.InputFilter.LengthFilter(3)});
        field.setImeOptions(android.view.inputmethod.EditorInfo.IME_ACTION_NEXT);
        field.setTextColor(ContextCompat.getColor(requireContext(), R.color.white));
        field.setHintTextColor(ContextCompat.getColor(requireContext(), R.color.gray_text_secondary));
        field.setTextSize(14f);
        field.setTypeface(android.graphics.Typeface.DEFAULT_BOLD);
        field.setTag(type);
        if (value != null && !value.trim().isEmpty()) field.setText(value.trim());
        field.addTextChangedListener(new android.text.TextWatcher() {
            @Override public void beforeTextChanged(CharSequence s, int a, int b, int c) { }
            @Override public void onTextChanged(CharSequence s, int a, int b, int c) { }
            @Override public void afterTextChanged(android.text.Editable s) { refreshConfirmButton(); }
        });
        row.addView(field);

        binding.layoutExpectedTypeRows.addView(row);
        countFields.put(type, field);
    }

    /** What is typed in every row right now, verbatim. */
    private Map<String, String> currentRawCounts() {
        Map<String, String> raw = new LinkedHashMap<>();
        for (Map.Entry<String, android.widget.EditText> e : countFields.entrySet()) {
            raw.put(e.getKey(), e.getValue().getText().toString());
        }
        return raw;
    }

    /** Fills the rows from the part's stored declaration, if it has one. */
    private void loadDeclaredCounts() {
        final android.content.Context context = requireContext().getApplicationContext();
        new Thread(() -> {
            final List<ExpectedFeature> stored =
                    AppDatabase.getInstance(context).expectedFeatureDao().getForPart(partId);
            if (stored.isEmpty() || !isAdded()) return;
            requireActivity().runOnUiThread(() -> {
                if (!isAdded() || binding == null) return;
                Map<String, String> keep = currentRawCounts();
                for (ExpectedFeature f : stored) {
                    keep.put(f.getFeatureType(), String.valueOf(f.getCount()));
                }
                buildCountRows(keep);
            });
        }).start();
    }

    /**
     * The typed counts as numbers. Blank, zero, negative and non-numeric entries are dropped
     * rather than stored: a type with nothing expected has no row, and a stray character must
     * not become an expectation of zero that fails every part with that hardware.
     */
    static Map<String, Integer> countsFrom(Map<String, String> rawByType) {
        Map<String, Integer> counts = new LinkedHashMap<>();
        for (Map.Entry<String, String> e : rawByType.entrySet()) {
            if (e.getKey() == null || e.getValue() == null) continue;
            String raw = e.getValue().trim();
            if (raw.isEmpty()) continue;
            try {
                int n = Integer.parseInt(raw);
                if (n > 0) counts.put(e.getKey(), n);
            } catch (NumberFormatException ignored) {
                // Not a number: nothing declared for this type.
            }
        }
        return counts;
    }

    static int total(Map<String, Integer> counts) {
        int total = 0;
        for (int n : counts.values()) total += n;
        return total;
    }

    /**
     * The confirm button's label for its four states. The button is only enabled in the last
     * one; in the first it is still tappable because it opens the chooser.
     */
    static String confirmLabel(boolean hasImage, int total, boolean detecting) {
        if (!hasImage) return "CHOOSE IMAGE";
        if (total <= 0) return "ENTER FEATURE COUNTS TO CONFIRM";
        if (detecting) return "DETECTING POSITIONS…";
        return String.format(Locale.US, "CONFIRM REFERENCE (%d FEATURE%s)", total,
                total == 1 ? "" : "S");
    }

    static String confirmLabel(boolean hasImage, int total) {
        return confirmLabel(hasImage, total, false);
    }

    /**
     * The line under the explainer: how many positions the detector found against how many
     * features were declared. Positions are what localisation and the size check can track;
     * a shortfall is worth knowing before confirming, because a feature the detector cannot
     * see on the reference has no position to be checked against on any inspection.
     */
    static String positionsText(int positions, int declaredTotal, boolean failed) {
        if (failed) {
            return "Could not detect positions on this photo. Counts will still be checked;"
                    + " position and size will not be.";
        }
        if (positions == 0) {
            return "No feature positions detected on this photo. Counts will still be"
                    + " checked; position and size will not be. A clearer photo may do better.";
        }
        if (declaredTotal > 0 && positions < declaredTotal) {
            return String.format(Locale.US,
                    "Detected %d position%s on this photo for the %d features declared."
                            + " Position and size will be checked for those %d; the counts"
                            + " still stand. A clearer photo may find the rest.",
                    positions, positions == 1 ? "" : "s", declaredTotal, positions);
        }
        if (declaredTotal > 0 && positions > declaredTotal) {
            return String.format(Locale.US,
                    "Detected %d positions on this photo but %d features are declared."
                            + " If the extra boxes are not hardware, that is the detector"
                            + " over-counting; the declared counts still decide PASS/FAIL.",
                    positions, declaredTotal);
        }
        return String.format(Locale.US,
                "Detected %d position%s on this photo. Position and size will be checked.",
                positions, positions == 1 ? "" : "s");
    }

    private void refreshConfirmButton() {
        if (binding == null) return;
        int total = total(countsFrom(currentRawCounts()));
        boolean hasImage = selectedImageExternalUri != null;
        binding.tvExpectedTotal.setText(String.format(Locale.US, "TOTAL: %d", total));
        binding.btnCaptureReference.setText(confirmLabel(hasImage, total, analysing));
        boolean ready = hasImage && total > 0 && !analysing;
        binding.btnCaptureReference.setEnabled(!confirming && (!hasImage || ready));
        binding.btnCaptureReference.setBackgroundColor(ContextCompat.getColor(requireContext(),
                ready ? R.color.green_pass_header : R.color.blue_primary));
        if (hasImage && !analysing) {
            binding.tvReferencePositions.setText(positionsText(
                    pendingResult == null ? 0 : slotTypes(pendingResult.features).size(),
                    total, pendingResult == null));
            binding.tvReferencePositions.setTextColor(ContextCompat.getColor(requireContext(),
                    pendingResult != null && !slotTypes(pendingResult.features).isEmpty()
                            ? R.color.green_pass : R.color.orange_warn));
        }
    }

    // ── The photograph: preview, then one detection pass for positions ───

    /** Both load paths end here: show the photo at its own shape, then detect its positions. */
    private void onImageChosen(String uriString) {
        Bitmap preview = HardwareFeatureExtractor.loadBitmap(requireContext(), uriString, 1080);
        if (preview != null) {
            binding.ivReferencePreview.setImageBitmap(preview);
            fitPreview(preview.getWidth(), preview.getHeight());
        }
        binding.ivReferencePreview.setVisibility(View.VISIBLE);
        binding.overlayReferencePreview.setVisibility(View.GONE);
        binding.layoutUploadPlaceholder.setVisibility(View.GONE);
        binding.tvReferencePositions.setVisibility(View.VISIBLE);
        binding.tvReferencePositions.setText("Detecting feature positions on this photo…");
        binding.tvReferencePositions.setTextColor(
                ContextCompat.getColor(requireContext(), R.color.gray_text_secondary));
        analyseReference(uriString);
    }

    /**
     * Sizes the preview zone to the photograph's aspect, as large as the width allows and no
     * taller than half the screen. The overlay stretches its bitmap to the view, so the view
     * has to be the photo's shape or every box lands off its feature; and a phone-height
     * portrait photo at full width would push the counts and the confirm button off screen.
     */
    private void fitPreview(int imageWidth, int imageHeight) {
        if (imageWidth <= 0 || imageHeight <= 0 || binding == null) return;
        android.util.DisplayMetrics dm = getResources().getDisplayMetrics();
        int maxW = binding.layoutPreviewContainer.getWidth() - dp(32);
        if (maxW <= 0) maxW = dm.widthPixels - dp(32);
        int maxH = Math.round(dm.heightPixels * PREVIEW_MAX_HEIGHT_FRACTION);
        float scale = Math.min((float) maxW / imageWidth, (float) maxH / imageHeight);
        int w = Math.max(1, Math.round(imageWidth * scale));
        int h = Math.max(1, Math.round(imageHeight * scale));
        android.widget.FrameLayout.LayoutParams lp =
                (android.widget.FrameLayout.LayoutParams) binding.layoutUploadZone.getLayoutParams();
        lp.width = w;
        lp.height = h;
        lp.gravity = android.view.Gravity.CENTER_HORIZONTAL;
        binding.layoutUploadZone.setLayoutParams(lp);
    }

    /**
     * Runs the detector over the reference once, for positions and box sizes only.
     *
     * <p>Same call, library and tuned thresholds as an inspection, so the boxes saved here are
     * the boxes an inspection will be matched against. The counts are not touched by this:
     * they are what the operator typed, whatever the detector makes of the photo.
     */
    private void analyseReference(String uriString) {
        final int token = ++analysisToken;
        final android.content.Context context = requireContext().getApplicationContext();
        pendingResult = null;
        analysing = true;
        refreshConfirmButton();

        new Thread(() -> {
            HardwareFeatureExtractor.AnalysisResult result = null;
            try {
                ReferenceLibrary library = ReferenceLibraryProvider.effectiveFor(context, partId);
                if (library == null) library = new ReferenceLibrary();
                ThresholdTuner.applyTo(context, partId, library);
                // Null sink: nothing here needs the crops, only the boxes.
                result = HardwareFeatureExtractor.extractFeatures(context, uriString, library, null);
                if (result.processedBitmap != null) {
                    result.processedBitmap.recycle();
                    result.processedBitmap = null;
                }
            } catch (Throwable t) {
                Log.e(TAG, "Reference detection failed for " + partId, t);
            }
            final HardwareFeatureExtractor.AnalysisResult done = result;
            if (!isAdded()) return;
            requireActivity().runOnUiThread(() -> {
                if (!isAdded() || binding == null || token != analysisToken) return;
                pendingResult = done;
                analysing = false;
                if (done != null && !done.features.isEmpty()) {
                    binding.ivReferencePreview.setVisibility(View.GONE);
                    binding.overlayReferencePreview.setVisibility(View.VISIBLE);
                    binding.overlayReferencePreview.setResultData(uriString, done.features);
                }
                refreshConfirmButton();
            });
        }).start();
    }

    /**
     * The slot type for each detection, index-aligned with {@code features}, or null where the
     * detection makes no slot.
     *
     * <p>The classifier's nearest prototype is used whether or not its score cleared the
     * acceptance threshold — the same choice the old labelling dialog made when it pre-filled
     * each row. A reference is one photo of a part that is known to be correct, so the
     * nearest type is very likely the right one, and a fresh part's library rejects almost
     * everything as "unknown", which would leave the layout empty. Only critical types make
     * slots: localisation is a claim about hardware with a place it belongs, and an advisory
     * scratch has none.
     */
    private List<String> slotTypes(List<HardwareFeatureExtractor.ExtractedFeature> features) {
        return slotTypes(features, type -> FeatureTypeRegistry.isCritical(requireContext(), type));
    }

    public static List<String> slotTypes(List<HardwareFeatureExtractor.ExtractedFeature> features,
                                  java.util.function.Predicate<String> isCritical) {
        List<String> types = new ArrayList<>();
        if (features == null) return types;
        for (HardwareFeatureExtractor.ExtractedFeature f : features) {
            ReferenceLibrary.Classification c = f.classification;
            String type = c != null && c.closestMatch != null && !"none".equals(c.closestMatch)
                    ? c.closestMatch : f.type;
            if (type == null || ReferenceLibrary.UNKNOWN.equals(type) || !isCritical.test(type)) {
                type = null;
            }
            if (type != null) types.add(type);
        }
        return types;
    }

    /**
     * The layout to save for this reference: one slot per critical detection, with its box,
     * in the working frame the boxes were measured in. Null when there is nothing to track.
     */
    @Nullable
    private FeatureLayout layoutFrom(HardwareFeatureExtractor.AnalysisResult result, String uri) {
        if (result == null || result.features == null) return null;
        List<android.graphics.RectF> rects = new ArrayList<>();
        List<String> types = new ArrayList<>();
        for (HardwareFeatureExtractor.ExtractedFeature f : result.features) {
            ReferenceLibrary.Classification c = f.classification;
            String type = c != null && c.closestMatch != null && !"none".equals(c.closestMatch)
                    ? c.closestMatch : f.type;
            if (type == null || ReferenceLibrary.UNKNOWN.equals(type)
                    || !FeatureTypeRegistry.isCritical(requireContext(), type)) {
                type = ReferenceLibrary.UNKNOWN; // skipped by layoutFrom
            }
            rects.add(f.rectFraction);
            types.add(type);
        }
        FeatureLayout layout = FeatureLocalisation.layoutFrom(rects, types, null);
        if (layout.isEmpty()) return null;
        layout.setReferenceUri(uri);
        layout.setImageSize(result.workingWidth, result.workingHeight);
        layout.setEdgeMethodId(result.edgeMethodId);
        return layout;
    }

    /**
     * Files the photo and the declaration together. Both writes sit in one transaction so a
     * part can never be left with a new photo and the previous part's counts, or the reverse.
     */
    private void confirmReference() {
        final Map<String, Integer> counts = countsFrom(currentRawCounts());
        if (selectedImageExternalUri == null || total(counts) <= 0 || confirming || analysing) return;
        final String uri = selectedImageExternalUri.toString();
        final android.content.Context context = requireContext().getApplicationContext();
        // Built on the UI thread: it reads the type registry through the fragment's context.
        final FeatureLayout layout = layoutFrom(pendingResult, uri);
        confirming = true;
        refreshConfirmButton();

        new Thread(() -> {
            boolean saved = false;
            try {
                AppDatabase db = AppDatabase.getInstance(context);
                List<ExpectedFeature> rows = new java.util.ArrayList<>();
                for (Map.Entry<String, Integer> e : counts.entrySet()) {
                    rows.add(new ExpectedFeature(partId, e.getKey(), e.getValue()));
                }
                db.runInTransaction(() -> {
                    db.expectedFeatureDao().replaceForPart(partId, rows);
                    db.referenceProfileDao().insertProfile(
                            new ReferenceProfile(partId, uri, System.currentTimeMillis()));
                });
                // The positions detected on this photo replace whatever layout the part had
                // — including one hand-labelled under the old flow against the same photo,
                // which would otherwise keep reporting a slot the operator has just declared
                // away as MISSING. No positions at all means no layout, and the old one goes.
                if (layout != null) {
                    layout.save(context, partId);
                } else {
                    File stale = FeatureLayout.fileFor(context, partId);
                    if (stale.exists() && !stale.delete()) {
                        Log.w(TAG, "Could not remove the superseded layout " + stale);
                    }
                }
                saved = true;
            } catch (Exception e) {
                Log.e(TAG, "Failed to confirm reference for " + partId, e);
            }
            final boolean ok = saved;
            if (!isAdded()) return;
            requireActivity().runOnUiThread(() -> {
                if (!isAdded() || binding == null) return;
                confirming = false;
                if (!ok) {
                    refreshConfirmButton();
                    Toast.makeText(getContext(), "Could not save the reference. Try again.",
                            Toast.LENGTH_LONG).show();
                    return;
                }
                Toast.makeText(getContext(), String.format(Locale.US,
                        "Reference confirmed — %d feature%s declared.", total(counts),
                        total(counts) == 1 ? "" : "s"), Toast.LENGTH_LONG).show();
                proceedToInspection();
            });
        }).start();
    }

    /**
     * Name + severity for a type the catalog does not have yet, mirroring the review dialog on
     * the results screen so the scale means the same thing wherever it is set. A part whose
     * hardware is not in the catalog cannot be declared otherwise.
     */
    private void showCreateTypeDialog(Runnable onCreated) {
        android.app.Dialog d = new android.app.Dialog(requireContext());

        android.widget.LinearLayout root = new android.widget.LinearLayout(getContext());
        root.setOrientation(android.widget.LinearLayout.VERTICAL);
        root.setBackgroundColor(Color.rgb(28, 27, 31));
        root.setPadding(dp(20), dp(20), dp(20), dp(20));

        android.widget.TextView title = new android.widget.TextView(getContext());
        title.setText("NEW FEATURE TYPE");
        title.setTextColor(Color.WHITE);
        title.setTextSize(14f);
        title.setTypeface(android.graphics.Typeface.DEFAULT_BOLD);
        root.addView(title);

        android.widget.EditText nameInput = new android.widget.EditText(getContext());
        nameInput.setHint("Name (e.g. Scratch)");
        nameInput.setTextColor(Color.WHITE);
        nameInput.setHintTextColor(Color.rgb(180, 180, 188));

        // Enter finishes the name instead of opening a second line: this field holds one short
        // label, and a stray newline would otherwise ride along into the type key.
        nameInput.setSingleLine(true);
        nameInput.setImeOptions(android.view.inputmethod.EditorInfo.IME_ACTION_DONE);
        nameInput.setOnEditorActionListener((v, actionId, event) -> {
            if (actionId != android.view.inputmethod.EditorInfo.IME_ACTION_DONE
                    && actionId != android.view.inputmethod.EditorInfo.IME_NULL) return false;
            v.clearFocus();
            android.view.inputmethod.InputMethodManager imm =
                    (android.view.inputmethod.InputMethodManager) requireContext()
                            .getSystemService(android.content.Context.INPUT_METHOD_SERVICE);
            if (imm != null) imm.hideSoftInputFromWindow(v.getWindowToken(), 0);
            return true;
        });
        root.addView(nameInput);

        android.widget.TextView heading = new android.widget.TextView(getContext());
        heading.setText("How should this type be reported?");
        heading.setTextColor(Color.WHITE);
        heading.setTextSize(11f);
        heading.setPadding(0, dp(14), 0, dp(4));
        root.addView(heading);

        android.widget.SeekBar seek = new android.widget.SeekBar(getContext());
        seek.setMax(FeatureTypeRegistry.Severity.maxLevel());
        root.addView(seek);

        android.widget.LinearLayout ends = new android.widget.LinearLayout(getContext());
        ends.setOrientation(android.widget.LinearLayout.HORIZONTAL);
        android.widget.TextView left = new android.widget.TextView(getContext());
        left.setText("CRITICAL");
        left.setTextSize(9.5f);
        left.setTextColor(Color.rgb(180, 180, 188));
        left.setLayoutParams(new android.widget.LinearLayout.LayoutParams(
                0, android.widget.LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        android.widget.TextView right = new android.widget.TextView(getContext());
        right.setText("IGNORED");
        right.setTextSize(9.5f);
        right.setGravity(android.view.Gravity.END);
        right.setTextColor(Color.rgb(180, 180, 188));
        right.setLayoutParams(new android.widget.LinearLayout.LayoutParams(
                0, android.widget.LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        ends.addView(left);
        ends.addView(right);
        root.addView(ends);

        android.widget.TextView value = new android.widget.TextView(getContext());
        value.setTextColor(Color.WHITE);
        value.setTextSize(11.5f);
        value.setPadding(dp(10), dp(8), dp(10), dp(8));
        root.addView(value);

        android.widget.Button create = new android.widget.Button(getContext());
        create.setText("CREATE");
        create.setTextColor(Color.WHITE);
        create.setBackgroundColor(Color.rgb(26, 70, 128));
        create.setEnabled(false);
        root.addView(create);

        final FeatureTypeRegistry.Severity[] severity = {null};
        final Runnable refresh = () -> {
            if (severity[0] == null) {
                value.setText("Not set — drag the slider to choose.");
                value.setBackgroundColor(Color.rgb(50, 50, 55));
            } else {
                int bg;
                switch (severity[0]) {
                    case CRITICAL: bg = Color.rgb(120, 40, 44); break;
                    case ADVISORY: bg = Color.rgb(112, 82, 26); break;
                    default:       bg = Color.rgb(58, 58, 66); break;
                }
                value.setText(severity[0].label.toUpperCase(Locale.US) + " — " + severity[0].blurb);
                value.setBackgroundColor(bg);
            }
            create.setEnabled(nameInput.getText().toString().trim().length() > 0 && severity[0] != null);
        };
        refresh.run();

        seek.setOnSeekBarChangeListener(new android.widget.SeekBar.OnSeekBarChangeListener() {
            @Override public void onProgressChanged(android.widget.SeekBar bar, int progress, boolean fromUser) {
                severity[0] = FeatureTypeRegistry.Severity.fromLevel(
                        FeatureTypeRegistry.Severity.maxLevel() - progress);
                refresh.run();
            }
            @Override public void onStartTrackingTouch(android.widget.SeekBar bar) {}
            @Override public void onStopTrackingTouch(android.widget.SeekBar bar) {}
        });
        nameInput.addTextChangedListener(new android.text.TextWatcher() {
            @Override public void beforeTextChanged(CharSequence s, int a, int b, int c) {}
            @Override public void onTextChanged(CharSequence s, int a, int b, int c) {}
            @Override public void afterTextChanged(android.text.Editable s) { refresh.run(); }
        });

        create.setOnClickListener(v -> {
            String displayName = nameInput.getText().toString().trim();
            String key = FeatureTypeRegistry.keyFor(displayName);
            if (key.isEmpty() || severity[0] == null) return;
            FeatureTypeRegistry.upsert(requireContext(), key, displayName, severity[0]);
            d.dismiss();
            if (onCreated != null) onCreated.run();
        });

        d.setContentView(root);
        d.show();
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
                    String lower = file.toLowerCase(java.util.Locale.ROOT);
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

    /** Whether a bundled asset is really present, so a missing one fails loudly instead of quietly. */
    private boolean assetExists(String path) {
        try (InputStream in = requireContext().getAssets().open(path)) {
            return in != null;
        } catch (Exception e) {
            return false;
        }
    }

    private void loadSelectedAsset(String filename) {
        if (!assetExists("chassis_images/" + filename)) {
            // Without this the preview stays blank and a reference pointing at a file that is
            // not there could be confirmed.
            Toast.makeText(getContext(), "Missing bundled image: " + filename, Toast.LENGTH_LONG).show();
            return;
        }
        try {
            selectedImageExternalUri = Uri.parse("file:///android_asset/chassis_images/" + filename);
            onImageChosen(selectedImageExternalUri.toString());
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
                    String lower = file.toLowerCase(java.util.Locale.ROOT);
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
            File photoFile = new File(outputDir, "REF_" + PartFiles.safeName(partId) + "_" + timeStamp + ".jpg");

            FileOutputStream outputStream = new FileOutputStream(photoFile);
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, bytesRead);
            }
            outputStream.close();
            inputStream.close();

            selectedImageExternalUri = Uri.fromFile(photoFile);
            onImageChosen(selectedImageExternalUri.toString());
            Toast.makeText(getContext(), "Image uploaded successfully!", Toast.LENGTH_SHORT).show();
        } catch (Exception e) {
            Log.e(TAG, "Failed to copy reference image", e);
            Toast.makeText(getContext(), "Error saving image: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
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
