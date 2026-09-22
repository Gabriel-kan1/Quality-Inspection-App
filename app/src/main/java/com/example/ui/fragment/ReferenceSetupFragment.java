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
import com.example.data.entity.ClassificationEvent;
import com.example.data.entity.ReferenceProfile;
import com.example.ui.util.AssetBitmaps;
import com.example.databinding.FragmentReferenceSetupBinding;
import com.example.ml.HardwareFeatureExtractor;
import com.example.ml.FeatureEmbedder;
import com.example.ml.FeatureLayout;
import com.example.ml.FeatureTypeRegistry;
import com.example.ml.LayoutMatcher;
import com.example.ml.LearningCropStore;
import com.example.ml.PartFiles;
import com.example.ml.ReferenceLibrary;
import com.example.ml.ReferenceLibraryProvider;
import com.example.ml.ThresholdTuner;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class ReferenceSetupFragment extends Fragment {

    private static final String TAG = "ReferenceSetupFragment";

    /** Backing image for the "USE BUILT-IN PRESET" shortcut; not currently shipped in assets. */
    private static final String PRESET_ASSET = "chassis_hardware_preset.jpg";

    private FragmentReferenceSetupBinding binding;
    private String partId = "";
    private Uri selectedImageExternalUri = null;
    /**
     * Guards against a slow analysis of a discarded image overwriting a newer one's results.
     * Incremented every time a reference is chosen; a finishing pass only publishes if its
     * token still matches.
     */
    private int analysisToken = 0;

    /** Last analysed reference: its detections, and one crop per detection for labelling. */
    private HardwareFeatureExtractor.AnalysisResult pendingResult = null;
    /**
     * Thresholds the classifier was using when {@link #pendingResult} was produced. The
     * library is reloaded at save time, so these have to be captured here or the logged
     * events would carry the defaults rather than the values that actually decided them.
     */
    private float pendingSimilarityThreshold = ReferenceLibrary.DEFAULT_SIMILARITY_THRESHOLD;
    private float pendingMarginThreshold = ReferenceLibrary.DEFAULT_MARGIN_THRESHOLD;
    /**
     * Crops kept alive for the labelling dialog, keyed by the extractor's 1-based box id.
     * Keyed rather than a list because a detection whose crop failed to decode still produces
     * a feature, and a plain append would silently shift every later thumbnail by one.
     */
    private final java.util.Map<Integer, Bitmap> pendingCrops = new java.util.LinkedHashMap<>();

    /**
     * Longest side kept for a retained crop. Large enough that {@link FeatureEmbedder} sees
     * essentially what it would have seen from the full-size crop (it resizes to 224 itself),
     * small enough that a few dozen of them do not put tens of megabytes into native memory —
     * where, on Android 8+, a leak grows the process until the kernel kills it rather than
     * raising OutOfMemoryError.
     */
    private static final int CROP_KEEP_PX = 256;

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

        // Saving a reference goes through labelling: an unclassified detection contributes
        // nothing to the counts, the landmarks or the positional report, so a reference saved
        // without this step silently defines much less than it appears to.
        binding.btnCaptureReference.setOnClickListener(v -> {
            if (selectedImageExternalUri == null) {
                launchImageChooser();
            } else if (pendingResult == null) {
                Toast.makeText(getContext(), "Still analysing this image — one moment.",
                        Toast.LENGTH_SHORT).show();
            } else {
                showClassifyAllDialog(this::proceedToInspection);
            }
        });

        // The preset image is not in the repo. Offering a button that loads nothing looks like a
        // working shortcut and ends in an empty reference, so only show it if the asset is
        // actually there; the assets browser below already lists whatever is.
        if (assetExists("chassis_images/" + PRESET_ASSET)) {
            binding.btnUsePresetReference.setOnClickListener(v -> loadSelectedAsset(PRESET_ASSET));
        } else {
            binding.btnUsePresetReference.setVisibility(View.GONE);
        }

        // Same flow as SAVE REFERENCE — labelling and library building were always one act,
        // and having them on two buttons let an operator save a reference whose crops had
        // never been labelled at all.
        binding.btnBuildLibrary.setOnClickListener(v -> {
            if (selectedImageExternalUri == null) {
                Toast.makeText(getContext(), "Choose or load a reference image first.", Toast.LENGTH_SHORT).show();
                launchImageChooser();
            } else if (pendingResult == null) {
                Toast.makeText(getContext(), "Still analysing this image — one moment.",
                        Toast.LENGTH_SHORT).show();
            } else {
                showClassifyAllDialog(this::proceedToInspection);
            }
        });
    }


    // ── Reference preview: the same pass an inspection runs ──────────────

    /**
     * Detects and classifies the chosen reference exactly as an inspection would, then draws
     * the result on the same overlay the results screen uses.
     *
     * A reference photo is not a picture, it is the definition of what "correct" means for
     * this part: its detections become the expected counts and the localisation layout. Until
     * now the operator chose one blind and only discovered what it contained by running an
     * inspection against it — which is how a reference yielding three usable features instead
     * of five went unnoticed.
     *
     * Deliberately the same call, library and tuned thresholds as
     * {@code ResultsFragment.runFeatureVerificationPipeline}. A preview classified by a
     * different configuration than the real pass would be worse than none.
     */
    private void analyseReference(String uriString) {
        final int token = ++analysisToken;
        final android.content.Context context = requireContext().getApplicationContext();

        binding.cardReferenceAnalysis.setVisibility(View.VISIBLE);
        binding.tvReferenceAnalysisStatus.setText("Analysing reference image...");
        binding.layoutReferenceTypeRows.removeAllViews();
        binding.tvReferenceLocalisationReadiness.setText("");

        new Thread(() -> {
            ReferenceLibrary library = ReferenceLibraryProvider.effectiveFor(context, partId);
            if (library == null) library = new ReferenceLibrary();
            ThresholdTuner.applyTo(context, partId, library);

            // The sink keeps a scaled copy of every crop instead of writing one. Nothing is
            // persisted until the operator has labelled it: an unlabelled reference crop is not
            // evidence of anything, whereas a labelled one is written to the learning store at
            // save time alongside its event row. The labelling step needs the pixels either
            // way, and the extractor frees each crop as soon as this returns.
            final java.util.Map<Integer, Bitmap> crops = new java.util.LinkedHashMap<>();
            final float simThreshold = library.getSimilarityThreshold();
            final float marginThreshold = library.getMarginThreshold();
            HardwareFeatureExtractor.AnalysisResult result =
                    HardwareFeatureExtractor.extractFeatures(context, uriString, library,
                            (crop, index) -> {
                                Bitmap kept = scaleForKeeping(crop);
                                if (kept != null) crops.put(index, kept);
                                return null;
                            });
            if (result.processedBitmap != null) result.processedBitmap.recycle();

            requireActivity().runOnUiThread(() -> {
                if (!isAdded() || binding == null || token != analysisToken) {
                    for (Bitmap b : crops.values()) b.recycle();
                    return;
                }
                releasePendingCrops();
                pendingResult = result;
                pendingSimilarityThreshold = simThreshold;
                pendingMarginThreshold = marginThreshold;
                pendingCrops.putAll(crops);
                showReferenceAnalysis(uriString, result);
            });
        }).start();
    }

    /** Aspect-preserving copy bounded to {@link #CROP_KEEP_PX}, or the original if already small. */
    private static Bitmap scaleForKeeping(Bitmap crop) {
        if (crop == null || crop.isRecycled()) return null;
        int w = crop.getWidth(), h = crop.getHeight();
        if (w <= 0 || h <= 0) return null;
        int longest = Math.max(w, h);
        if (longest <= CROP_KEEP_PX) return crop.copy(Bitmap.Config.ARGB_8888, false);
        float scale = (float) CROP_KEEP_PX / longest;
        return Bitmap.createScaledBitmap(crop,
                Math.max(1, Math.round(w * scale)), Math.max(1, Math.round(h * scale)), true);
    }

    private void releasePendingCrops() {
        for (Bitmap b : pendingCrops.values()) {
            if (b != null && !b.isRecycled()) b.recycle();
        }
        pendingCrops.clear();
        pendingResult = null;
    }

    /** Renders the overlay, the per-type breakdown, and whether localisation will be possible. */
    private void showReferenceAnalysis(String uriString,
                                       HardwareFeatureExtractor.AnalysisResult result) {
        binding.ivReferencePreview.setVisibility(View.GONE);
        binding.overlayReferencePreview.setVisibility(View.VISIBLE);
        binding.overlayReferencePreview.setResultData(uriString, result.features);

        if (result.features.isEmpty()) {
            binding.tvReferenceAnalysisStatus.setText(
                    "No features detected in this image. It will define an empty reference —"
                            + " nothing to compare an inspection against.");
            binding.tvReferenceLocalisationReadiness.setText("");
            return;
        }

        // Counts by type, in the order the types were first seen.
        java.util.Map<String, Integer> counts = new java.util.LinkedHashMap<>();
        for (HardwareFeatureExtractor.ExtractedFeature f : result.features) {
            Integer prev = counts.get(f.type);
            counts.put(f.type, prev == null ? 1 : prev + 1);
        }

        binding.tvReferenceAnalysisStatus.setText(String.format(Locale.getDefault(),
                "%d feature%s detected across %d type%s. These become the expected counts for"
                        + " every inspection of this part.",
                result.features.size(), result.features.size() == 1 ? "" : "s",
                counts.size(), counts.size() == 1 ? "" : "s"));

        int landmarks = 0;
        int criticalSlots = 0;
        binding.layoutReferenceTypeRows.removeAllViews();
        for (java.util.Map.Entry<String, Integer> e : counts.entrySet()) {
            String type = e.getKey();
            int count = e.getValue();
            boolean unknown = ReferenceLibrary.UNKNOWN.equals(type);
            FeatureTypeRegistry.Severity severity =
                    unknown ? null : FeatureTypeRegistry.severityOf(requireContext(), type);

            // Mirrors the layout builder: localisation sees critical hardware and nothing
            // else, so those are the only detections that become landmarks.
            if (!unknown && severity == FeatureTypeRegistry.Severity.CRITICAL) {
                landmarks += count;
                criticalSlots += count;
            }

            binding.layoutReferenceTypeRows.addView(
                    typeRow(type, count, severity, unknown));
        }

        binding.tvReferenceLocalisationReadiness.setText(readinessText(landmarks, criticalSlots));
        binding.tvReferenceLocalisationReadiness.setTextColor(
                LayoutMatcher.tooSmallToRegister(landmarks) || criticalSlots == 0
                        ? ContextCompat.getColor(requireContext(), R.color.orange_warn)
                        : ContextCompat.getColor(requireContext(), R.color.green_pass));
    }

    /** One "swatch — name — count — severity" row, matching the results screen's legend style. */
    private View typeRow(String type, int count, FeatureTypeRegistry.Severity severity, boolean unknown) {
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
        name.setText(unknown ? "Unclassified"
                : FeatureTypeRegistry.displayName(requireContext(), type));
        name.setTextColor(ContextCompat.getColor(requireContext(), R.color.white));
        name.setTextSize(12f);
        row.addView(name);

        android.widget.TextView tail = new android.widget.TextView(requireContext());
        tail.setText(unknown
                ? count + "  ·  not a landmark"
                : count + "  ·  " + severity.label.toLowerCase(Locale.US));
        tail.setTextColor(ContextCompat.getColor(requireContext(), R.color.gray_text_secondary));
        tail.setTextSize(11f);
        row.addView(tail);

        return row;
    }

    /**
     * Says up front whether this reference can support the positional check, and why not when
     * it cannot. Discovering that only after filing an inspection is what made a weaker
     * reference photo look like a broken feature.
     */
    private String readinessText(int landmarks, int criticalSlots) {
        if (criticalSlots == 0) {
            return "No critical features here, so inspections of this part will have no"
                    + " positional check and nothing to pass or fail on. Re-grade a type as"
                    + " critical, or choose a photo where the hardware is clearer.";
        }
        if (LayoutMatcher.tooSmallToRegister(criticalSlots)) {
            return String.format(Locale.getDefault(),
                    "Only %d critical feature%s — too few to register a position against, so"
                            + " localisation will be unavailable. Localisation uses critical"
                            + " hardware only; choose a photo where more of it is clearly"
                            + " visible, or grade another type as critical.",
                    criticalSlots, criticalSlots == 1 ? "" : "s");
        }
        return String.format(Locale.getDefault(),
                "%d critical position%s tracked. Localisation will be available, and tolerates"
                        + " up to %d of them being missing.",
                criticalSlots, criticalSlots == 1 ? "" : "s",
                criticalSlots - LayoutMatcher.requiredInliers(criticalSlots));
    }

    // ── Reference labelling: every detection gets an explicit type ───────

    /** Sentinel option that opens the "create a new type" sub-dialog. */
    private static final String NEW_TYPE_KEY = "__new__";

    /**
     * Spinner colours for the labelling dialog: a white field with dark grey text.
     *
     * Set on the collapsed row and the popup together. The platform's dropdown item styling
     * takes its text colour from the theme, which on this dark fullscreen dialog is light —
     * so a white popup left at the default renders near-invisible text.
     */
    private static final int SPINNER_BG = Color.WHITE;
    private static final int SPINNER_TEXT = Color.rgb(60, 60, 66);

    /**
     * The type to open a row on: the nearest prototype regardless of score, or null when there
     * was nothing to compare against (no encoder, or an empty library on a part's first setup).
     * {@code "none"} is the sentinel {@code closestMatch} carries in that case.
     */
    private static String closestUsable(ReferenceLibrary.Classification c) {
        if (c == null) return null;
        if (c.closestMatch != null && !"none".equals(c.closestMatch)) return c.closestMatch;
        return ReferenceLibrary.UNKNOWN.equals(c.type) ? null : c.type;
    }

    /** Rebuilds every row's dropdown after the type catalog changes. Set while the dialog is up. */
    private Runnable pendingRebuildOptions = null;

    /**
     * Makes the operator name every feature the detector found in the reference.
     *
     * A reference photo defines what "correct" means for the part: its detections become the
     * expected counts, the registration landmarks and the reported positions. Anything left
     * unclassified is dead weight — not counted, not a landmark, not reportable — so a
     * reference saved with two thirds of its detections unknown quietly produces a far weaker
     * check than the operator thinks they set up. Requiring an answer for each one is what
     * makes the reference mean what it appears to mean.
     *
     * The classifier's own guess pre-selects each row it was confident enough to accept, so
     * this is confirmation rather than data entry.
     */
    private void showClassifyAllDialog(Runnable onSaved) {
        if (pendingResult == null || pendingResult.features.isEmpty()) {
            Toast.makeText(getContext(), "No features detected — nothing to classify.",
                    Toast.LENGTH_LONG).show();
            return;
        }

        android.app.Dialog dialog = new android.app.Dialog(requireContext(),
                android.R.style.Theme_DeviceDefault_NoActionBar_Fullscreen);

        android.widget.LinearLayout root = new android.widget.LinearLayout(getContext());
        root.setOrientation(android.widget.LinearLayout.VERTICAL);
        root.setBackgroundColor(Color.rgb(20, 20, 25));
        root.setPadding(dp(20), dp(28), dp(20), dp(20));

        android.widget.TextView title = new android.widget.TextView(getContext());
        title.setText("REVIEW DETECTED FEATURES");
        title.setTextColor(Color.rgb(74, 222, 128));
        title.setTextSize(16f);
        title.setTypeface(android.graphics.Typeface.DEFAULT_BOLD);
        title.setGravity(android.view.Gravity.CENTER);
        root.addView(title);

        android.widget.TextView sub = new android.widget.TextView(getContext());
        sub.setTextColor(Color.LTGRAY);
        sub.setTextSize(11f);
        sub.setGravity(android.view.Gravity.CENTER);
        sub.setPadding(0, dp(4), 0, dp(14));
        root.addView(sub);

        android.widget.ScrollView scroll = new android.widget.ScrollView(getContext());
        scroll.setLayoutParams(new android.widget.LinearLayout.LayoutParams(
                android.widget.LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f));
        android.widget.LinearLayout list = new android.widget.LinearLayout(getContext());
        list.setOrientation(android.widget.LinearLayout.VERTICAL);
        scroll.addView(list);
        root.addView(scroll);

        android.widget.Button save = new android.widget.Button(getContext());
        save.setTextColor(Color.WHITE);
        save.setBackgroundColor(Color.rgb(26, 70, 128));
        android.widget.LinearLayout.LayoutParams saveLp = new android.widget.LinearLayout.LayoutParams(
                android.widget.LinearLayout.LayoutParams.MATCH_PARENT,
                android.widget.LinearLayout.LayoutParams.WRAP_CONTENT);
        saveLp.setMargins(0, dp(12), 0, 0);
        save.setLayoutParams(saveLp);
        root.addView(save);

        final List<HardwareFeatureExtractor.ExtractedFeature> features = pendingResult.features;
        final String[] chosen = new String[features.size()];
        final List<android.widget.Spinner> spinners = new ArrayList<>();
        final List<List<String>> optionKeysPerRow = new ArrayList<>();

        final Runnable refreshSave = () -> {
            int done = 0;
            for (String c : chosen) if (c != null) done++;
            save.setEnabled(done == chosen.length);
            save.setText(done == chosen.length
                    ? "SAVE REFERENCE (" + done + " classified)"
                    : "CLASSIFY ALL TO CONTINUE (" + done + "/" + chosen.length + ")");
            sub.setText(done == chosen.length
                    ? "Pre-filled with the closest match for each crop — correct any that are"
                            + " wrong. These define the expected features for this part."
                    : "Give every crop a type. Anything left unnamed is excluded from the counts,"
                            + " the alignment landmarks and the positional report.");
        };

        int thumb = dp(64);
        for (int i = 0; i < features.size(); i++) {
            final int index = i;
            HardwareFeatureExtractor.ExtractedFeature f = features.get(i);

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
            Bitmap crop = pendingCrops.get(index + 1);
            if (crop != null && !crop.isRecycled()) img.setImageBitmap(crop);
            row.addView(img);

            android.widget.LinearLayout col = new android.widget.LinearLayout(getContext());
            col.setOrientation(android.widget.LinearLayout.VERTICAL);
            android.widget.LinearLayout.LayoutParams colLp = new android.widget.LinearLayout.LayoutParams(
                    0, android.widget.LinearLayout.LayoutParams.WRAP_CONTENT, 1f);
            colLp.setMarginStart(dp(12));
            col.setLayoutParams(colLp);

            android.widget.TextView guess = new android.widget.TextView(getContext());
            guess.setTextColor(Color.rgb(150, 150, 158));
            guess.setTextSize(10f);
            ReferenceLibrary.Classification c = f.classification;
            guess.setText(c == null || "none".equals(c.closestMatch)
                    ? "#" + (index + 1) + " · no suggestion"
                    : String.format(Locale.US, "#%d · closest: %s (%.2f)", index + 1,
                            FeatureTypeRegistry.displayName(requireContext(), c.closestMatch), c.score));
            col.addView(guess);

            android.widget.Spinner spinner = new android.widget.Spinner(getContext());
            spinner.setLayoutParams(new android.widget.LinearLayout.LayoutParams(
                    android.widget.LinearLayout.LayoutParams.MATCH_PARENT,
                    android.widget.LinearLayout.LayoutParams.WRAP_CONTENT));
            spinner.setPopupBackgroundDrawable(
                    new android.graphics.drawable.ColorDrawable(SPINNER_BG));
            col.addView(spinner);
            row.addView(col);
            list.addView(row);

            spinners.add(spinner);
            optionKeysPerRow.add(new ArrayList<>());
            // Default every row to the nearest prototype, whether or not the score cleared the
            // threshold, so the operator reviews and corrects rather than naming each crop from
            // scratch. closestMatch rather than type: type is "unknown" for anything rejected,
            // which is most of a busy reference and would leave the operator doing all the work.
            //
            // The cost is that a low-confidence guess is saved as a confirmed label unless
            // somebody notices — the per-row score is shown above each dropdown for exactly
            // that reason. Only a crop the classifier could not rank at all stays unset.
            chosen[index] = closestUsable(c);
        }

        // Adapter population is a separate pass so creating a type can rebuild every row's
        // options at once when the catalog gains an entry.
        final Runnable rebuildOptions = () -> {
            List<FeatureTypeRegistry.TypeInfo> types = FeatureTypeRegistry.getAll(requireContext());
            for (int i = 0; i < spinners.size(); i++) {
                List<String> keys = optionKeysPerRow.get(i);
                List<String> labels = new ArrayList<>();
                keys.clear();
                keys.add(null);
                labels.add("— choose a type —");
                for (FeatureTypeRegistry.TypeInfo t : types) {
                    keys.add(t.key);
                    labels.add(t.severity == FeatureTypeRegistry.Severity.CRITICAL
                            ? t.displayName
                            : t.displayName + " (" + t.severity.label.toLowerCase(Locale.US) + ")");
                }
                keys.add(NEW_TYPE_KEY);
                labels.add("+ Create new type...");

                // A preset the catalog does not offer — a prototype whose type was never
                // registered — would show as "choose a type" while still counting as answered,
                // enabling save on a row nobody actually set. Drop it back to unset instead.
                if (chosen[i] != null && !keys.contains(chosen[i])) chosen[i] = null;

                android.widget.Spinner sp = spinners.get(i);
                android.widget.ArrayAdapter<String> adapter = new android.widget.ArrayAdapter<String>(
                        requireContext(), android.R.layout.simple_spinner_dropdown_item, labels) {
                    @NonNull @Override
                    public View getView(int position, View convertView, @NonNull ViewGroup parent) {
                        return paint(super.getView(position, convertView, parent));
                    }
                    @Override
                    public View getDropDownView(int position, View convertView, @NonNull ViewGroup parent) {
                        return paint(super.getDropDownView(position, convertView, parent));
                    }
                    private View paint(View v) {
                        v.setBackgroundColor(SPINNER_BG);
                        if (v instanceof android.widget.TextView) {
                            ((android.widget.TextView) v).setTextColor(SPINNER_TEXT);
                        }
                        return v;
                    }
                };
                sp.setOnItemSelectedListener(null);
                sp.setAdapter(adapter);
                int sel = chosen[i] == null ? 0 : Math.max(0, keys.indexOf(chosen[i]));
                sp.setSelection(sel);

                final int rowIndex = i;
                sp.setOnItemSelectedListener(new android.widget.AdapterView.OnItemSelectedListener() {
                    @Override public void onItemSelected(android.widget.AdapterView<?> parent, View v, int pos, long id) {
                        String key = optionKeysPerRow.get(rowIndex).get(pos);
                        if (NEW_TYPE_KEY.equals(key)) {
                            showCreateTypeDialog(rowIndex, chosen);
                            return;
                        }
                        chosen[rowIndex] = key;
                        refreshSave.run();
                    }
                    @Override public void onNothingSelected(android.widget.AdapterView<?> parent) {}
                });
            }
            refreshSave.run();
        };
        pendingRebuildOptions = rebuildOptions;
        rebuildOptions.run();

        save.setOnClickListener(v -> {
            save.setEnabled(false);
            saveClassifiedReference(chosen, dialog, onSaved);
        });

        dialog.setOnDismissListener(x -> pendingRebuildOptions = null);
        dialog.setContentView(root);
        dialog.setCancelable(true);
        dialog.show();
    }

    /**
     * Name + severity for a type the catalog does not have yet, mirroring the review dialog on
     * the results screen so the scale means the same thing wherever it is set.
     */
    private void showCreateTypeDialog(int rowIndex, String[] chosen) {
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
            chosen[rowIndex] = key;
            d.dismiss();
        });

        // Backing out must not leave the row showing "+ Create new type..." as though it were an
        // answer, so the dropdowns are rebuilt on either exit path.
        d.setOnDismissListener(x -> {
            if (pendingRebuildOptions != null) pendingRebuildOptions.run();
        });

        d.setContentView(root);
        d.show();
    }

    /**
     * Embeds every labelled crop into this part's library, saves it, and records the reference
     * profile so inspections compute their expected counts from this image.
     */
    private void saveClassifiedReference(String[] chosen, android.app.Dialog dialog, Runnable onSaved) {
        final android.content.Context context = requireContext().getApplicationContext();
        final String uri = selectedImageExternalUri.toString();
        final List<Bitmap> crops = new ArrayList<>();
        for (int i = 0; i < chosen.length; i++) crops.add(pendingCrops.get(i + 1));
        final List<HardwareFeatureExtractor.ExtractedFeature> features = pendingResult.features;
        final int workingWidth = pendingResult.workingWidth;
        final int workingHeight = pendingResult.workingHeight;
        final String edgeMethodId = pendingResult.edgeMethodId;
        // Under the adaptive method, which rung the reference's boxes came out of. An
        // inspection may legitimately run another; the results screen names both.
        final String preprocessingRung = pendingResult.preprocessing == null ? null
                : pendingResult.preprocessing.rung.label;
        final float simThreshold = pendingSimilarityThreshold;
        final float marginThreshold = pendingMarginThreshold;

        Toast.makeText(getContext(), "Encoding labelled crops...", Toast.LENGTH_SHORT).show();
        new Thread(() -> {
            FeatureEmbedder embedder = FeatureEmbedder.get(context);
            if (embedder == null) {
                requireActivity().runOnUiThread(() -> Toast.makeText(getContext(),
                        "Feature encoder unavailable (missing assets/models/feature_encoder.onnx).",
                        Toast.LENGTH_LONG).show());
                return;
            }

            // Extends the part's existing library rather than replacing it: labels confirmed
            // during past inspections are scarce, hard-won examples, and setting a new reference
            // photo must not discard them.
            ReferenceLibrary lib = ReferenceLibrary.load(
                    ReferenceLibraryProvider.fileForPart(context, partId));
            if (lib == null) lib = new ReferenceLibrary();

            int labelled = 0;
            for (int i = 0; i < chosen.length; i++) {
                Bitmap crop = crops.get(i);
                if (chosen[i] == null || crop == null || crop.isRecycled()) continue;
                float[] emb = embedder.embed(crop);
                if (emb != null) {
                    lib.addExample(chosen[i], emb);
                    labelled++;
                }
            }
            lib.save(ReferenceLibraryProvider.fileForPart(context, partId));

            // The labels are the reference, not just training data. Persisting them as the
            // part's layout is what makes an inspection compare against what the operator
            // actually said, instead of re-classifying the reference photo every run and
            // getting a different answer as the library drifts — a borderline crop scoring
            // either side of the threshold changed the expected counts and the slot list
            // without the photo changing at all.
            FeatureLayout layout = new FeatureLayout();
            layout.setReferenceUri(uri);
            // The frame the slot fractions are measured in, so a box can be turned back into
            // pixels when an inspection asks whether a slot is big enough to size-check.
            layout.setImageSize(workingWidth, workingHeight);
            // The edge method these boxes came out of. An inspection run under a different
            // one is compared against boxes a different detector drew, and is told so.
            layout.setEdgeMethodId(edgeMethodId);
            layout.setPreprocessingRung(preprocessingRung);
            for (int i = 0; i < chosen.length; i++) {
                if (chosen[i] == null) continue;
                // Critical types only, matching what the inspection side puts forward as
                // candidates. Localisation is a claim about hardware with a place it is
                // supposed to be; an advisory scratch has none, and a background patch is not
                // a feature at all. A slot either side of that line could never be paired with
                // anything, so it would sit in the layout as a permanent non-match, dragging
                // the inlier count down and raising the bar it is measured against at once.
                // One reference ended up with 18 of its 24 slots being bare surface.
                if (FeatureTypeRegistry.severityOf(context, chosen[i])
                        != FeatureTypeRegistry.Severity.CRITICAL) continue;
                android.graphics.RectF r = features.get(i).rectFraction;
                if (r == null) continue;
                // The box size goes in with the position. It is what an inspection compares the
                // detection at this slot against, scaled by the alignment, so the part's
                // hardware can be size-checked without a calibration marker in frame.
                layout.addSlot(chosen[i], (r.left + r.right) / 2f, (r.top + r.bottom) / 2f,
                        r.width(), r.height());
            }
            layout.save(context, partId);

            AppDatabase db = AppDatabase.getInstance(context);
            db.referenceProfileDao().insertProfile(
                    new ReferenceProfile(partId, uri, System.currentTimeMillis()));

            // Record every label as a reviewed classification event, exactly as an inspection
            // review is recorded. These labels used to vanish into the prototypes: the operator
            // would spend a reference setup naming sixteen crops, and NoiseGate and
            // ThresholdTuner — which fit only from ClassificationEvent — never saw one of them.
            // The classification stored is the one the operator was shown, made against the
            // library *before* these examples were added, so a replay asks the right question.
            logReferenceLabels(context, db, chosen, crops, features, simThreshold, marginThreshold);

            final int count = labelled;
            final int typeCount = lib.types().size();
            requireActivity().runOnUiThread(() -> {
                if (dialog.isShowing()) dialog.dismiss();
                Toast.makeText(getContext(), "Reference saved — " + count
                        + " examples across " + typeCount + " types.", Toast.LENGTH_LONG).show();
                if (onSaved != null) onSaved.run();
            });
        }).start();
    }

    /**
     * One reviewed {@link ClassificationEvent} per labelled crop, with the crop persisted to
     * the learning store so the row is a complete record rather than a label with no pixels.
     *
     * Skipped crops are not logged. An inspection logs every detection because a later review
     * may come back to it; nothing ever comes back to a reference crop the operator declined
     * to name, and an unreviewed row that can never be reviewed would only inflate the
     * "logged detections" count for no one's benefit.
     *
     * Best-effort: a failure here must not lose the reference the operator just saved.
     */
    private void logReferenceLabels(android.content.Context context, AppDatabase db,
                                    String[] chosen, List<Bitmap> crops,
                                    List<HardwareFeatureExtractor.ExtractedFeature> features,
                                    float simThreshold, float marginThreshold) {
        try {
            long now = System.currentTimeMillis();
            List<ClassificationEvent> pending = new ArrayList<>();
            for (int i = 0; i < chosen.length; i++) {
                if (chosen[i] == null || i >= features.size()) continue;
                ReferenceLibrary.Classification c = features.get(i).classification;
                Bitmap crop = i < crops.size() ? crops.get(i) : null;

                ClassificationEvent e = new ClassificationEvent();
                e.setPartId(partId);
                e.setTimestamp(now);
                // Box ids are 1-based in the extractor's labels; keep the same numbering.
                e.setCropPath(LearningCropStore.save(context, partId, crop, i + 1));
                e.setPredictedType(c != null ? c.closestMatch : ReferenceLibrary.UNKNOWN);
                e.setPredictedScore(c != null ? c.score : 0f);
                e.setRunnerUpType(c != null ? c.runnerUp : null);
                e.setRunnerUpScore(c != null ? c.runnerUpScore : -1f);
                e.setReason(c != null ? c.reason.name() : ReferenceLibrary.Reason.NO_LIBRARY.name());
                e.setSimilarityThreshold(simThreshold);
                e.setMarginThreshold(marginThreshold);
                e.setEncoderId(FeatureEmbedder.ENCODER_ID);
                e.setConfirmedType(chosen[i]);
                e.setReviewed(true);
                e.setReviewSource(ClassificationEvent.SOURCE_REFERENCE_SETUP);
                pending.add(e);
            }
            db.runInTransaction(() -> {
                for (ClassificationEvent e : pending) db.classificationEventDao().insert(e);
            });
        } catch (Exception ex) {
            android.util.Log.e("ReferenceSetup", "Failed to log reference labels", ex);
        }
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
            // Without this the load runs on a non-existent file, the preview stays blank and
            // analyseReference() reports zero detections as though the part were empty.
            Toast.makeText(getContext(), "Missing bundled image: " + filename, Toast.LENGTH_LONG).show();
            return;
        }
        try {
            selectedImageExternalUri = Uri.parse("file:///android_asset/chassis_images/" + filename);
            // Preview only — analyseReference() re-loads the asset at its own resolution.
            Bitmap bitmap = AssetBitmaps.decodeSampled(
                    requireContext(), "chassis_images/" + filename, 1080);
            if (bitmap != null) binding.ivReferencePreview.setImageBitmap(bitmap);
            // Plain preview first so the choice registers immediately; the overlay replaces it
            // when the classification pass finishes a second or two later.
            binding.ivReferencePreview.setVisibility(View.VISIBLE);
            binding.overlayReferencePreview.setVisibility(View.GONE);
            binding.layoutUploadPlaceholder.setVisibility(View.GONE);
            binding.btnCaptureReference.setText("SAVE REFERENCE");
            binding.btnCaptureReference.setBackgroundColor(ContextCompat.getColor(requireContext(), R.color.green_pass_header));
            analyseReference(selectedImageExternalUri.toString());
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

            binding.ivReferencePreview.setImageURI(selectedImageExternalUri);
            binding.ivReferencePreview.setVisibility(View.VISIBLE);
            binding.overlayReferencePreview.setVisibility(View.GONE);
            binding.layoutUploadPlaceholder.setVisibility(View.GONE);
            binding.btnCaptureReference.setText("SAVE REFERENCE");
            binding.btnCaptureReference.setBackgroundColor(ContextCompat.getColor(requireContext(), R.color.green_pass_header));
            analyseReference(selectedImageExternalUri.toString());
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

    private void proceedToInspection() {
        Bundle args = new Bundle();
        args.putString("partId", partId);
        if (isAdded() && getView() != null) {
            Navigation.findNavController(getView()).navigate(R.id.action_reference_setup_to_inspection, args);
        }
    }

    @Override
    public void onDestroyView() {
        // Crops are native-memory bitmaps; on Android 8+ leaking them grows the process until
        // the kernel kills it rather than raising OutOfMemoryError.
        releasePendingCrops();
        super.onDestroyView();
        binding = null;
    }
}
