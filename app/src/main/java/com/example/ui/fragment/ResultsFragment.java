package com.example.ui.fragment;

import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.Bundle;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;
import androidx.navigation.Navigation;

import com.example.R;
import com.example.data.AppDatabase;
import com.example.data.entity.ClassificationEvent;
import com.example.data.entity.FeatureResult;
import com.example.data.entity.InspectionRecord;
import com.example.data.entity.ReferenceProfile;
import com.example.databinding.FragmentResultsBinding;
import com.example.ml.ActiveLearningPolicy;
import com.example.ml.ClipEmbedder;
import com.example.ml.FeatureTypeRegistry;
import com.example.ml.LearningCropStore;
import com.example.ml.ReferenceLibrary;
import com.example.ml.ReferenceLibraryProvider;
import com.example.ml.ThresholdTuner;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;

public class ResultsFragment extends Fragment {

    private FragmentResultsBinding binding;
    private String partId = "";
    private String imageUri = "";

    // Per-type counts (keyed by canonical feature type, e.g. "screw").
    private final Map<String, Integer> expectedCounts = new LinkedHashMap<>();
    private final Map<String, Integer> detectedCounts = new LinkedHashMap<>();
    private List<String> orderedTypes = new ArrayList<>();
    private boolean hasDiscrepancy = false;
    private boolean hasRealReference = false;
    // Effective reference library for this part — mutated in place as the user labels
    // "unknown" detections live, then persisted so future inspections benefit.
    private ReferenceLibrary activeLibrary;

    // ── Self-learning loop state ──
    /** Row ids of this run's logged classification events, so they can be tied to the saved inspection. */
    private final List<Long> loggedEventIds = new ArrayList<>();
    /** Event row id per feature index, so a review can update the right row in place. */
    private final Map<Integer, Long> eventIdByFeature = new LinkedHashMap<>();
    private ThresholdTuner.Tuned appliedThresholds;
    private int reviewsThisRun = 0;
    /** Kept so overlay taps can correct a detection after the results screen is built. */
    private HardwareFeatureExtractor.AnalysisResult analysisResult;

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

        binding.btnInspectAgain.setOnClickListener(v ->
                Navigation.findNavController(v).navigate(R.id.action_results_to_home));

        binding.btnSaveReport.setOnClickListener(v -> saveRecordAndProceed());

        runFeatureVerificationPipeline();
    }

    private void runFeatureVerificationPipeline() {
        new Thread(() -> {
            AppDatabase db = AppDatabase.getInstance(requireContext());
            ReferenceProfile ref = db.referenceProfileDao().getProfileByPartType(partId);

            // Effective CLIP reference library: per-part if set up in-app, else the bundled default,
            // else a fresh empty one (first run for this part — everything starts "unknown").
            ReferenceLibrary library = ReferenceLibraryProvider.effectiveFor(requireContext(), partId);
            activeLibrary = library != null ? library : new ReferenceLibrary();

            // Self-learning: apply thresholds fitted from this part's own confirmed history
            // (if enough of it exists) instead of the hand-picked defaults. Must happen
            // before classification so the whole run uses the tuned values.
            appliedThresholds = ThresholdTuner.applyTo(requireContext(), partId, activeLibrary);

            // Detected: classify the active chassis image. Crops are written straight to the
            // learning store and freed inside the extractor, so nothing scales with box count.
            HardwareFeatureExtractor.AnalysisResult activeResult =
                    HardwareFeatureExtractor.extractFeatures(requireContext(), imageUri, activeLibrary,
                            (crop, index) -> LearningCropStore.save(requireContext(), partId, crop, index));

            // Log every detection as a classification event — reviewed or not. Unreviewed
            // rows still carry the scores, so a later review or export can use them.
            logClassificationEvents(db, activeResult.features);

            // Expected: from the gold-standard reference image if one exists, else mirror detected
            // (computed after live-labelling below, since labelling can change detected types).
            expectedCounts.clear();
            hasRealReference = ref != null && ref.getImageUri() != null
                    && !"dummy_placeholder".equals(ref.getImageUri());
            if (hasRealReference) {
                // Null sink: the reference pass only contributes counts, so its crops are never
                // written or retained. This pass previously doubled peak crop memory for nothing.
                HardwareFeatureExtractor.AnalysisResult refResult =
                        HardwareFeatureExtractor.extractFeatures(requireContext(), ref.getImageUri(),
                                activeLibrary, null);
                countByType(refResult.features, expectedCounts);
            }

            // Active learning: rather than prompting on every unknown, ask only about the
            // detections where a human answer would actually move the classifier, plus a
            // small sample of confident ones to keep the tuning set unbiased.
            List<ReferenceLibrary.Classification> classifications = new ArrayList<>();
            for (HardwareFeatureExtractor.ExtractedFeature f : activeResult.features) {
                classifications.add(f.classification);
            }
            List<ActiveLearningPolicy.Candidate> candidates =
                    ActiveLearningPolicy.select(classifications, activeLibrary, new Random());

            requireActivity().runOnUiThread(() -> {
                if (!isAdded() || binding == null) return;
                resolveCandidatesSequentially(activeResult, candidates, 0,
                        // Refit thresholds with this run's answers included, so the next
                        // inspection of this part starts from a better decision boundary.
                        () -> retuneThresholds(() -> finalizeResults(activeResult)));
            });
        }).start();
    }

    /** Persists one {@link ClassificationEvent} (and its crop) per detection. */
    private void logClassificationEvents(AppDatabase db,
                                          List<HardwareFeatureExtractor.ExtractedFeature> features) {
        loggedEventIds.clear();
        eventIdByFeature.clear();
        long now = System.currentTimeMillis();

        // Build the rows first — crop writes are file I/O and shouldn't hold a DB transaction open.
        List<ClassificationEvent> pending = new ArrayList<>();
        for (int i = 0; i < features.size(); i++) {
            HardwareFeatureExtractor.ExtractedFeature f = features.get(i);
            ReferenceLibrary.Classification c = f.classification;

            ClassificationEvent e = new ClassificationEvent();
            e.setPartId(partId);
            e.setTimestamp(now);
            // Already written to disk by the extractor's CropSink during detection.
            e.setCropPath(f.cropPath);
            // Store the winning prototype, not the reported label, so ThresholdTuner can
            // replay "what if the threshold had been lower" against this row later.
            e.setPredictedType(c != null ? c.closestMatch : ReferenceLibrary.UNKNOWN);
            e.setPredictedScore(c != null ? c.score : 0f);
            e.setRunnerUpType(c != null ? c.runnerUp : null);
            e.setRunnerUpScore(c != null ? c.runnerUpScore : -1f);
            e.setReason(c != null ? c.reason.name() : ReferenceLibrary.Reason.NO_LIBRARY.name());
            e.setSimilarityThreshold(activeLibrary.getSimilarityThreshold());
            e.setMarginThreshold(activeLibrary.getMarginThreshold());
            e.setReviewed(false);
            pending.add(e);
        }

        // One transaction for the whole batch rather than N implicit ones.
        db.runInTransaction(() -> {
            for (int i = 0; i < pending.size(); i++) {
                long id = db.classificationEventDao().insert(pending.get(i));
                loggedEventIds.add(id);
                eventIdByFeature.put(i, id);
            }
        });
    }

    /** Computes final counts/discrepancy (post live-labelling) and renders the results screen. */
    private void finalizeResults(HardwareFeatureExtractor.AnalysisResult activeResult) {
        if (!isAdded() || binding == null) return;
        analysisResult = activeResult;

        detectedCounts.clear();
        countByType(activeResult.features, detectedCounts);
        if (!hasRealReference) {
            expectedCounts.putAll(detectedCounts);
        }

        // Ordered union of types (canonical order first, "unknown" last).
        orderedTypes = orderedUnion(expectedCounts.keySet(), detectedCounts.keySet());

        // Discrepancy over known types only (ignore "unknown" noise).
        hasDiscrepancy = false;
        for (String type : orderedTypes) {
            if (ReferenceLibrary.UNKNOWN.equals(type)) continue;
            if (count(expectedCounts, type) != count(detectedCounts, type)) {
                hasDiscrepancy = true;
                break;
            }
        }

        if (activeResult.processedBitmap != null) {
            binding.ivCannyEdges.setImageBitmap(activeResult.processedBitmap);
        }
        binding.chassisOverlayView.setResultData(imageUri, activeResult.features);
        // Any detection can be corrected by tapping it, not just the ones the app chose to
        // ask about — the operator can see a wrong label the classifier was confident about.
        binding.chassisOverlayView.setOnFeatureClickListener((index, f) -> correctFeatureAt(index));

        if (hasDiscrepancy) {
            binding.tvResultBadge.setText("INSPECTION FAILED");
            binding.tvResultSub.setText("Discrepancy detected in chassis configuration.");
            binding.layoutResultHeader.setBackgroundColor(ContextCompat.getColor(requireContext(), R.color.red_fail));
        } else {
            binding.tvResultBadge.setText("INSPECTION PASSED");
            binding.tvResultSub.setText("All expected features verified successfully.");
            binding.layoutResultHeader.setBackgroundColor(ContextCompat.getColor(requireContext(), R.color.green_pass_header));
        }

        buildFeatureRows();
        buildLegend();
        buildLearningStatus();
    }

    /** Surfaces the state of the learning loop so it isn't an invisible black box. */
    private void buildLearningStatus() {
        new Thread(() -> {
            int reviewedTotal = 0;
            int loggedTotal = 0;
            try {
                AppDatabase db = AppDatabase.getInstance(requireContext());
                reviewedTotal = db.classificationEventDao().countReviewedForPart(partId);
                loggedTotal = db.classificationEventDao().countForPart(partId);
            } catch (Exception e) {
                android.util.Log.e("ResultsFragment", "Failed to read learning stats", e);
            }

            final int reviewed = reviewedTotal;
            final int logged = loggedTotal;
            requireActivity().runOnUiThread(() -> {
                if (!isAdded() || binding == null) return;

                StringBuilder sb = new StringBuilder();
                sb.append(String.format(Locale.US,
                        "Labelled examples for this part: %d confirmed of %d logged detections.\n",
                        reviewed, logged));
                sb.append(String.format(Locale.US, "You reviewed %d this run.\n", reviewsThisRun));

                if (appliedThresholds != null) {
                    sb.append(String.format(Locale.US,
                            "Thresholds auto-tuned from your history: similarity %.2f, margin %.3f "
                                    + "(%.0f%% accurate on %d labels, vs %.0f%% at the defaults).",
                            appliedThresholds.similarityThreshold,
                            appliedThresholds.marginThreshold,
                            appliedThresholds.accuracy * 100f,
                            appliedThresholds.sampleSize,
                            appliedThresholds.baselineAccuracy * 100f));
                } else {
                    int needed = Math.max(0, ThresholdTuner.MIN_EVENTS - reviewed);
                    sb.append(String.format(Locale.US,
                            "Using default thresholds (similarity %.2f, margin %.3f). ",
                            activeLibrary.getSimilarityThreshold(), activeLibrary.getMarginThreshold()));
                    sb.append(needed > 0
                            ? needed + " more confirmed labels before auto-tuning kicks in."
                            : "No tuned pair beat the defaults yet — defaults are still winning.");
                }
                binding.tvLearningStatus.setText(sb.toString());
            });
        }).start();
    }

    /** Opens the identify dialog for a detection the operator tapped on the overlay. */
    private void correctFeatureAt(int index) {
        if (!isAdded() || binding == null || analysisResult == null) return;
        if (index < 0 || index >= analysisResult.features.size()) return;

        HardwareFeatureExtractor.ExtractedFeature feature = analysisResult.features.get(index);
        ActiveLearningPolicy.Candidate candidate = ActiveLearningPolicy.userInitiated(index);
        showIdentifyDialog(feature, candidate, 1, 1, () -> {
            if (!isAdded() || binding == null) return;
            binding.chassisOverlayView.updateFeature(
                    index, feature.type, ReferenceLibrary.colorFor(feature.type), feature.label);
            binding.chassisOverlayView.clearSelection();
            refreshAfterCorrection();
        });
    }

    /**
     * Recomputes counts and re-renders the summary after a correction. Deliberately does not
     * re-run detection or reload the overlay bitmap — only the labels changed.
     */
    private void refreshAfterCorrection() {
        detectedCounts.clear();
        countByType(analysisResult.features, detectedCounts);
        if (!hasRealReference) {
            expectedCounts.clear();
            expectedCounts.putAll(detectedCounts);
        }
        orderedTypes = orderedUnion(expectedCounts.keySet(), detectedCounts.keySet());

        hasDiscrepancy = false;
        for (String type : orderedTypes) {
            if (ReferenceLibrary.UNKNOWN.equals(type)) continue;
            if (count(expectedCounts, type) != count(detectedCounts, type)) {
                hasDiscrepancy = true;
                break;
            }
        }

        if (hasDiscrepancy) {
            binding.tvResultBadge.setText("INSPECTION FAILED");
            binding.tvResultSub.setText("Discrepancy detected in chassis configuration.");
            binding.layoutResultHeader.setBackgroundColor(ContextCompat.getColor(requireContext(), R.color.red_fail));
        } else {
            binding.tvResultBadge.setText("INSPECTION PASSED");
            binding.tvResultSub.setText("All expected features verified successfully.");
            binding.layoutResultHeader.setBackgroundColor(ContextCompat.getColor(requireContext(), R.color.green_pass_header));
        }

        buildFeatureRows();
        buildLegend();
        buildLearningStatus();
    }

    // ── Live "identify unknown feature" flow ───────────────────────────────────

    /** Walks the active-learning candidate queue one dialog at a time, then runs onDone. */
    private void resolveCandidatesSequentially(HardwareFeatureExtractor.AnalysisResult result,
                                                List<ActiveLearningPolicy.Candidate> candidates,
                                                int index, Runnable onDone) {
        if (!isAdded() || binding == null) return;
        if (index >= candidates.size()) {
            onDone.run();
            return;
        }
        ActiveLearningPolicy.Candidate candidate = candidates.get(index);
        HardwareFeatureExtractor.ExtractedFeature feature = result.features.get(candidate.featureIndex);
        showIdentifyDialog(feature, candidate, index + 1, candidates.size(),
                () -> resolveCandidatesSequentially(result, candidates, index + 1, onDone));
    }

    /** Fullscreen dialog: crop thumbnail + pick an existing type, create a new one, or skip. */
    private void showIdentifyDialog(HardwareFeatureExtractor.ExtractedFeature feature,
                                     ActiveLearningPolicy.Candidate candidate,
                                     int position, int total, Runnable onResolved) {
        boolean isSpotCheck = candidate.kind == ActiveLearningPolicy.Kind.SPOT_CHECK;
        boolean isUserInitiated = candidate.kind == ActiveLearningPolicy.Kind.USER_INITIATED;
        // Both of these start from the current label rather than a blank slate.
        boolean preselectCurrent = isSpotCheck || isUserInitiated;

        android.app.Dialog dialog = new android.app.Dialog(requireContext(),
                android.R.style.Theme_DeviceDefault_NoActionBar_Fullscreen);

        android.widget.LinearLayout root = new android.widget.LinearLayout(getContext());
        root.setOrientation(android.widget.LinearLayout.VERTICAL);
        root.setBackgroundColor(Color.rgb(20, 20, 25));
        root.setPadding(dp(20), dp(28), dp(20), dp(20));

        TextView title = new TextView(getContext());
        if (isUserInitiated) {
            title.setText("CORRECT FEATURE");
            title.setTextColor(Color.rgb(56, 189, 248));
        } else {
            title.setText((isSpotCheck ? "SPOT CHECK (" : "IDENTIFY FEATURE (") + position + "/" + total + ")");
            title.setTextColor(isSpotCheck ? Color.rgb(250, 204, 21) : Color.rgb(74, 222, 128));
        }
        title.setTextSize(16f);
        title.setTypeface(android.graphics.Typeface.DEFAULT_BOLD);
        title.setGravity(Gravity.CENTER);
        root.addView(title);

        TextView sub = new TextView(getContext());
        if (isUserInitiated) {
            sub.setText("Currently recorded as \"" + FeatureTypeRegistry.displayName(requireContext(), feature.type)
                    + "\". Pick the correct type — your answer is added to this part's reference set.");
        } else if (isSpotCheck) {
            sub.setText("Confirm or correct this one. Spot checks keep the auto-tuning honest — "
                    + "without them the app only ever learns from its own mistakes.");
        } else {
            sub.setText(explainUncertainty(feature.classification));
        }
        sub.setTextColor(Color.LTGRAY);
        sub.setTextSize(11f);
        sub.setGravity(Gravity.CENTER);
        sub.setPadding(0, dp(4), 0, dp(16));
        root.addView(sub);

        android.widget.ImageView thumb = new android.widget.ImageView(getContext());
        int thumbSize = dp(140);
        android.widget.LinearLayout.LayoutParams thumbLp = new android.widget.LinearLayout.LayoutParams(thumbSize, thumbSize);
        thumbLp.gravity = Gravity.CENTER_HORIZONTAL;
        thumb.setLayoutParams(thumbLp);
        thumb.setScaleType(android.widget.ImageView.ScaleType.CENTER_CROP);
        // Decoded on demand for this one dialog and released when it closes, so at most one
        // crop bitmap is in memory at a time regardless of how many detections there were.
        final Bitmap thumbBitmap = decodeCrop(feature.cropPath);
        if (thumbBitmap != null) thumb.setImageBitmap(thumbBitmap);
        dialog.setOnDismissListener(d -> {
            thumb.setImageDrawable(null);
            if (thumbBitmap != null && !thumbBitmap.isRecycled()) thumbBitmap.recycle();
        });
        root.addView(thumb);

        // Show what the classifier thought, so the operator has context for the call.
        ReferenceLibrary.Classification c = feature.classification;
        if (c != null && c.closestMatch != null && !"none".equals(c.closestMatch)) {
            TextView guess = new TextView(getContext());
            StringBuilder sb = new StringBuilder();
            sb.append(String.format(Locale.US, "Best match: %s (%.2f)",
                    FeatureTypeRegistry.displayName(requireContext(), c.closestMatch), c.score));
            if (c.runnerUp != null) {
                sb.append(String.format(Locale.US, "   ·   Runner-up: %s (%.2f)",
                        FeatureTypeRegistry.displayName(requireContext(), c.runnerUp), c.runnerUpScore));
            }
            guess.setText(sb.toString());
            guess.setTextColor(Color.rgb(150, 150, 160));
            guess.setTextSize(10.5f);
            guess.setGravity(Gravity.CENTER);
            guess.setPadding(0, dp(10), 0, 0);
            root.addView(guess);
        }

        // ── Type spinner: skip, then every known type, then "create new" ──
        List<FeatureTypeRegistry.TypeInfo> knownTypes = FeatureTypeRegistry.getAll(requireContext());
        List<String> optionLabels = new ArrayList<>();
        List<String> optionKeys = new ArrayList<>();
        optionLabels.add(preselectCurrent ? "Skip (leave as is)" : "Skip (leave as unknown)");
        optionKeys.add(null);
        for (FeatureTypeRegistry.TypeInfo t : knownTypes) {
            optionLabels.add(t.displayName + (t.critical ? "" : " (non-critical)"));
            optionKeys.add(t.key);
        }
        optionLabels.add("+ Create new type...");
        optionKeys.add("__new__");

        android.widget.Spinner spinner = new android.widget.Spinner(getContext());
        android.widget.LinearLayout.LayoutParams spinnerLp = new android.widget.LinearLayout.LayoutParams(
                android.widget.LinearLayout.LayoutParams.MATCH_PARENT, android.widget.LinearLayout.LayoutParams.WRAP_CONTENT);
        spinnerLp.setMargins(0, dp(16), 0, dp(8));
        spinner.setLayoutParams(spinnerLp);
        android.widget.ArrayAdapter<String> adapter = new android.widget.ArrayAdapter<>(
                requireContext(), android.R.layout.simple_spinner_dropdown_item, optionLabels);
        spinner.setAdapter(adapter);
        // Start on the current type — the common case is "yes, that's right", so confirming
        // should be one tap rather than a hunt through the list.
        if (preselectCurrent && feature.type != null) {
            int preselect = optionKeys.indexOf(feature.type);
            if (preselect >= 0) spinner.setSelection(preselect);
        }
        root.addView(spinner);

        // ── "Create new type" panel: name + explicit critical/non-critical choice ──
        android.widget.LinearLayout newTypePanel = new android.widget.LinearLayout(getContext());
        newTypePanel.setOrientation(android.widget.LinearLayout.VERTICAL);
        newTypePanel.setVisibility(View.GONE);
        newTypePanel.setPadding(0, dp(8), 0, 0);

        android.widget.EditText nameInput = new android.widget.EditText(getContext());
        nameInput.setHint("New feature type name (e.g. Scratch)");
        nameInput.setTextColor(Color.WHITE);
        nameInput.setHintTextColor(Color.GRAY);
        newTypePanel.addView(nameInput);

        TextView criticalLabel = new TextView(getContext());
        criticalLabel.setText("Should this type be included in the report?");
        criticalLabel.setTextColor(Color.LTGRAY);
        criticalLabel.setTextSize(11f);
        criticalLabel.setPadding(0, dp(12), 0, dp(6));
        newTypePanel.addView(criticalLabel);

        android.widget.LinearLayout criticalRow = new android.widget.LinearLayout(getContext());
        criticalRow.setOrientation(android.widget.LinearLayout.HORIZONTAL);

        android.widget.Button criticalBtn = new android.widget.Button(getContext());
        criticalBtn.setText("CRITICAL");
        android.widget.LinearLayout.LayoutParams critLp = new android.widget.LinearLayout.LayoutParams(
                0, android.widget.LinearLayout.LayoutParams.WRAP_CONTENT, 1f);
        critLp.rightMargin = dp(6);
        criticalBtn.setLayoutParams(critLp);

        android.widget.Button nonCriticalBtn = new android.widget.Button(getContext());
        nonCriticalBtn.setText("NON-CRITICAL");
        android.widget.LinearLayout.LayoutParams nonCritLp = new android.widget.LinearLayout.LayoutParams(
                0, android.widget.LinearLayout.LayoutParams.WRAP_CONTENT, 1f);
        nonCriticalBtn.setLayoutParams(nonCritLp);

        criticalRow.addView(criticalBtn);
        criticalRow.addView(nonCriticalBtn);
        newTypePanel.addView(criticalRow);
        root.addView(newTypePanel);

        // Explicit tri-state (null = not chosen yet) — never defaulted, per user requirement.
        Boolean[] criticalChoice = {null};
        Runnable refreshCriticalButtons = () -> {
            boolean isCritical = Boolean.TRUE.equals(criticalChoice[0]);
            boolean isNonCritical = Boolean.FALSE.equals(criticalChoice[0]);
            criticalBtn.setBackgroundColor(isCritical ? Color.rgb(26, 100, 60) : Color.rgb(50, 50, 55));
            nonCriticalBtn.setBackgroundColor(isNonCritical ? Color.rgb(100, 70, 26) : Color.rgb(50, 50, 55));
            criticalBtn.setTextColor(Color.WHITE);
            nonCriticalBtn.setTextColor(Color.WHITE);
        };
        refreshCriticalButtons.run();

        android.widget.Button save = new android.widget.Button(getContext());
        save.setText("CONFIRM");
        save.setTextColor(Color.WHITE);
        save.setBackgroundColor(Color.rgb(26, 70, 128));
        android.widget.LinearLayout.LayoutParams saveLp = new android.widget.LinearLayout.LayoutParams(
                android.widget.LinearLayout.LayoutParams.MATCH_PARENT, android.widget.LinearLayout.LayoutParams.WRAP_CONTENT);
        saveLp.setMargins(0, dp(20), 0, 0);
        save.setLayoutParams(saveLp);

        Runnable refreshSaveEnabled = () -> {
            int selected = spinner.getSelectedItemPosition();
            boolean isNew = selected >= 0 && "__new__".equals(optionKeys.get(selected));
            boolean valid = !isNew
                    || (nameInput.getText().toString().trim().length() > 0 && criticalChoice[0] != null);
            save.setEnabled(valid);
        };

        criticalBtn.setOnClickListener(v -> { criticalChoice[0] = Boolean.TRUE; refreshCriticalButtons.run(); refreshSaveEnabled.run(); });
        nonCriticalBtn.setOnClickListener(v -> { criticalChoice[0] = Boolean.FALSE; refreshCriticalButtons.run(); refreshSaveEnabled.run(); });
        nameInput.addTextChangedListener(new android.text.TextWatcher() {
            @Override public void beforeTextChanged(CharSequence s, int a, int b, int c) {}
            @Override public void onTextChanged(CharSequence s, int a, int b, int c) {}
            @Override public void afterTextChanged(android.text.Editable s) { refreshSaveEnabled.run(); }
        });

        spinner.setOnItemSelectedListener(new android.widget.AdapterView.OnItemSelectedListener() {
            @Override public void onItemSelected(android.widget.AdapterView<?> parent, View v, int pos, long id) {
                boolean isNew = "__new__".equals(optionKeys.get(pos));
                newTypePanel.setVisibility(isNew ? View.VISIBLE : View.GONE);
                refreshSaveEnabled.run();
            }
            @Override public void onNothingSelected(android.widget.AdapterView<?> parent) {}
        });
        refreshSaveEnabled.run();

        save.setOnClickListener(v -> {
            save.setEnabled(false);
            int selected = spinner.getSelectedItemPosition();
            String key = optionKeys.get(selected);

            if (key == null) {
                // Skipped. Still recorded as ground truth "unknown" for an unknown-prompt —
                // the operator looking at it and declining to name it is real signal. Skipping
                // a spot check or a tap-correction is genuine abstention ("never mind"), so
                // those record nothing rather than asserting the label is wrong.
                if (!isSpotCheck && !isUserInitiated) {
                    recordReview(candidate, ReferenceLibrary.UNKNOWN, () -> {
                        dialog.dismiss();
                        onResolved.run();
                    });
                } else {
                    dialog.dismiss();
                    onResolved.run();
                }
                return;
            }

            String typeKey;
            if ("__new__".equals(key)) {
                String displayName = nameInput.getText().toString().trim();
                typeKey = FeatureTypeRegistry.keyFor(displayName);
                boolean critical = Boolean.TRUE.equals(criticalChoice[0]);
                if (typeKey.isEmpty()) {
                    save.setEnabled(true);
                    return;
                }
                FeatureTypeRegistry.upsert(requireContext(), typeKey, displayName, critical);
            } else {
                typeKey = key;
            }

            final String confirmedType = typeKey;
            applyLabelAndPersist(feature, confirmedType, () ->
                    recordReview(candidate, confirmedType, () -> {
                        dialog.dismiss();
                        onResolved.run();
                    }));
        });
        root.addView(save);

        dialog.setContentView(root);
        dialog.setCancelable(false);
        dialog.show();
    }

    /** Loads a persisted crop from disk, or null if there's no path or it can't be read. */
    private static Bitmap decodeCrop(String path) {
        if (path == null) return null;
        try {
            return android.graphics.BitmapFactory.decodeFile(path);
        } catch (Throwable t) {
            android.util.Log.e("ResultsFragment", "Failed to decode crop " + path, t);
            return null;
        }
    }

    /** Plain-language reason this detection needs a human, based on how it failed to classify. */
    private String explainUncertainty(ReferenceLibrary.Classification c) {
        if (c == null || c.reason == ReferenceLibrary.Reason.NO_LIBRARY) {
            return "Nothing has been labelled for this part yet. Naming this crop starts the reference library.";
        }
        if (c.reason == ReferenceLibrary.Reason.AMBIGUOUS) {
            return "Two feature types scored almost identically here, so the app won't guess. "
                    + "Your answer teaches it to tell them apart.";
        }
        if (c.score < ActiveLearningPolicy.NOISE_FLOOR) {
            return "This doesn't resemble anything in the library. It may be a feature type "
                    + "you haven't defined yet — or just noise you can skip.";
        }
        return "This landed just short of the confidence threshold. Confirming it is the most "
                + "useful correction you can make right now.";
    }

    /** Writes the human's answer back onto the logged event — the label half of the training pair. */
    private void recordReview(ActiveLearningPolicy.Candidate candidate, String confirmedType, Runnable onDone) {
        Long eventId = eventIdByFeature.get(candidate.featureIndex);
        if (eventId == null) {
            onDone.run();
            return;
        }
        new Thread(() -> {
            try {
                AppDatabase db = AppDatabase.getInstance(requireContext());
                String source;
                if (candidate.kind == ActiveLearningPolicy.Kind.SPOT_CHECK) {
                    source = ClassificationEvent.SOURCE_SPOT_CHECK;
                } else if (candidate.kind == ActiveLearningPolicy.Kind.USER_INITIATED) {
                    source = ClassificationEvent.SOURCE_OVERLAY_TAP;
                } else {
                    source = ClassificationEvent.SOURCE_UNKNOWN_PROMPT;
                }
                db.classificationEventDao().recordReview(eventId, confirmedType, source);
                reviewsThisRun++;
            } catch (Exception ex) {
                android.util.Log.e("ResultsFragment", "Failed to record review", ex);
            }
            requireActivity().runOnUiThread(() -> {
                if (isAdded()) onDone.run();
            });
        }).start();
    }

    /**
     * Refits per-part thresholds against the (now larger) confirmed history. Runs after the
     * review pass so this inspection's answers count toward the next one.
     */
    private void retuneThresholds(Runnable onDone) {
        new Thread(() -> {
            ThresholdTuner.Tuned fitted = null;
            try {
                AppDatabase db = AppDatabase.getInstance(requireContext());
                List<ClassificationEvent> reviewed =
                        db.classificationEventDao().getReviewedForPart(partId);
                fitted = ThresholdTuner.fit(reviewed);
                if (fitted != null) {
                    ThresholdTuner.save(requireContext(), partId, fitted);
                }
            } catch (Exception e) {
                android.util.Log.e("ResultsFragment", "Threshold retune failed", e);
            }
            final ThresholdTuner.Tuned result = fitted;
            requireActivity().runOnUiThread(() -> {
                if (!isAdded()) return;
                if (result != null) appliedThresholds = result;
                onDone.run();
            });
        }).start();
    }

    /** Relabels the feature in place, feeds the crop back into the reference library, and persists it. */
    private void applyLabelAndPersist(HardwareFeatureExtractor.ExtractedFeature feature, String typeKey, Runnable onDone) {
        new Thread(() -> {
            feature.type = typeKey;
            feature.color = ReferenceLibrary.colorFor(typeKey);
            String prefix = feature.label != null && feature.label.contains(":")
                    ? feature.label.substring(0, feature.label.indexOf(':')) : "?";
            feature.label = prefix + ":" + typeKey + " (labeled)";

            // Re-decode the crop just long enough to embed it into the prototype, then free it.
            Bitmap crop = decodeCrop(feature.cropPath);
            if (crop != null) {
                try {
                    ClipEmbedder embedder = ClipEmbedder.get(requireContext());
                    if (embedder != null) {
                        float[] emb = embedder.embed(crop);
                        if (emb != null) {
                            activeLibrary.addExample(typeKey, emb);
                            activeLibrary.save(ReferenceLibraryProvider.fileForPart(requireContext(), partId));
                        }
                    }
                } finally {
                    crop.recycle();
                }
            }

            requireActivity().runOnUiThread(() -> {
                if (isAdded()) onDone.run();
            });
        }).start();
    }

    private static void countByType(List<HardwareFeatureExtractor.ExtractedFeature> features, Map<String, Integer> out) {
        if (features == null) return;
        for (HardwareFeatureExtractor.ExtractedFeature f : features) {
            Integer prev = out.get(f.type);
            out.put(f.type, prev == null ? 1 : prev + 1);
        }
    }

    private static int count(Map<String, Integer> counts, String type) {
        Integer c = counts.get(type);
        return c == null ? 0 : c;
    }

    private static List<String> orderedUnion(Set<String> a, Set<String> b) {
        Set<String> union = new LinkedHashSet<>();
        union.addAll(a);
        union.addAll(b);
        List<String> ordered = new ArrayList<>();
        for (String t : ReferenceLibrary.FEATURE_TYPES) {
            if (union.contains(t)) ordered.add(t);
        }
        for (String t : union) {
            if (!ordered.contains(t) && !ReferenceLibrary.UNKNOWN.equals(t)) ordered.add(t);
        }
        if (union.contains(ReferenceLibrary.UNKNOWN)) ordered.add(ReferenceLibrary.UNKNOWN);
        return ordered;
    }

    /** Builds one summary-table row per feature type in {@link #orderedTypes}. */
    private void buildFeatureRows() {
        LinearLayout container = binding.layoutFeatureRows;
        container.removeAllViews();
        boolean first = true;
        for (String type : orderedTypes) {
            if (!first) container.addView(divider());
            first = false;
            int expected = count(expectedCounts, type);
            int detected = count(detectedCounts, type);
            container.addView(featureRow(type, expected, detected));
        }
    }

    private View divider() {
        View v = new View(requireContext());
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dp(1));
        v.setLayoutParams(lp);
        v.setBackgroundColor(Color.parseColor("#33B0BEC5"));
        return v;
    }

    private View featureRow(String type, int expected, int detected) {
        LinearLayout row = new LinearLayout(requireContext());
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setWeightSum(4f);
        row.setPadding(0, dp(12), 0, dp(12));
        row.setBackgroundColor(ContextCompat.getColor(requireContext(), R.color.surface_card));

        TextView name = cell(FeatureTypeRegistry.displayName(requireContext(), type), 1.3f, Gravity.START);
        name.setTextColor(ContextCompat.getColor(requireContext(), R.color.white));
        name.setTextSize(14f);
        name.setTypeface(name.getTypeface(), android.graphics.Typeface.BOLD);
        name.setPadding(dp(8), 0, 0, 0);
        row.addView(name);

        TextView exp = cell(String.valueOf(expected), 0.9f, Gravity.CENTER);
        exp.setTextColor(ContextCompat.getColor(requireContext(), R.color.gray_text_secondary));
        exp.setTextSize(14f);
        row.addView(exp);

        TextView det = cell(String.valueOf(detected), 0.9f, Gravity.CENTER);
        det.setTextColor(ContextCompat.getColor(requireContext(), R.color.white));
        det.setTextSize(14f);
        row.addView(det);

        String statusText;
        int statusColor;
        if (ReferenceLibrary.UNKNOWN.equals(type)) {
            statusText = "—";
            statusColor = R.color.gray_text_secondary;
        } else if (expected == detected) {
            statusText = "MATCH";
            statusColor = R.color.green_pass;
        } else if (detected < expected) {
            statusText = "MISSING";
            statusColor = R.color.red_fail;
        } else {
            statusText = "EXTRA";
            statusColor = R.color.orange_warn;
        }
        TextView status = cell(statusText, 0.9f, Gravity.CENTER);
        status.setTextColor(ContextCompat.getColor(requireContext(), statusColor));
        status.setTextSize(12f);
        status.setTypeface(status.getTypeface(), android.graphics.Typeface.BOLD);
        row.addView(status);

        return row;
    }

    private TextView cell(String text, float weight, int gravity) {
        TextView tv = new TextView(requireContext());
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(0,
                LinearLayout.LayoutParams.WRAP_CONTENT, weight);
        tv.setLayoutParams(lp);
        tv.setText(text);
        tv.setGravity(gravity);
        return tv;
    }

    /** Builds a colored-swatch legend chip per feature type. */
    private void buildLegend() {
        LinearLayout legend = binding.layoutLegend;
        legend.removeAllViews();
        for (String type : orderedTypes) {
            LinearLayout chip = new LinearLayout(requireContext());
            chip.setOrientation(LinearLayout.HORIZONTAL);
            chip.setGravity(Gravity.CENTER_VERTICAL);
            chip.setPadding(dp(8), 0, dp(8), 0);

            View swatch = new View(requireContext());
            swatch.setLayoutParams(new LinearLayout.LayoutParams(dp(14), dp(14)));
            swatch.setBackgroundColor(ReferenceLibrary.colorFor(type));
            chip.addView(swatch);

            TextView label = new TextView(requireContext());
            LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT);
            lp.setMarginStart(dp(6));
            label.setLayoutParams(lp);
            label.setText(FeatureTypeRegistry.displayName(requireContext(), type));
            label.setTextColor(ContextCompat.getColor(requireContext(), R.color.white));
            label.setTextSize(12f);
            chip.addView(label);

            legend.addView(chip);
        }
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private void saveRecordAndProceed() {
        String notes = binding.etInspectionNotes.getText().toString().trim();
        binding.btnSaveReport.setEnabled(false);

        new Thread(() -> {
            AppDatabase db = AppDatabase.getInstance(requireContext());

            InspectionRecord record = new InspectionRecord();
            record.setPartId(partId);
            record.setTimestamp(System.currentTimeMillis());
            record.setPassOrFail(hasDiscrepancy ? "FAIL" : "PASS");
            record.setImageUri(imageUri);
            record.setNotes(notes);

            long inspectionId = db.inspectionDao().insertInspection(record);

            List<FeatureResult> results = new ArrayList<>();
            for (String type : orderedTypes) {
                if (ReferenceLibrary.UNKNOWN.equals(type)) continue;
                int expected = count(expectedCounts, type);
                int detected = count(detectedCounts, type);
                String status = expected == detected ? "MATCH" : (detected < expected ? "MISSING" : "EXTRA");
                boolean critical = FeatureTypeRegistry.isCritical(requireContext(), type);
                results.add(new FeatureResult(inspectionId, FeatureTypeRegistry.displayName(requireContext(), type),
                        expected, detected, status, critical));
            }
            if (!results.isEmpty()) {
                db.inspectionDao().insertFeatureResults(results);
            }

            // Attach this run's classification events to the saved inspection so the
            // learning history stays traceable back to a specific report.
            if (!loggedEventIds.isEmpty()) {
                db.classificationEventDao().assignInspection(inspectionId, loggedEventIds);
            }

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
