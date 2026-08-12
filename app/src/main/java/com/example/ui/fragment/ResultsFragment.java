package com.example.ui.fragment;

import android.content.Context;
import android.content.SharedPreferences;
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
import com.example.data.entity.SlotResult;
import com.example.databinding.FragmentResultsBinding;
import com.example.ui.view.ChassisOverlayView;
import com.example.ml.ActiveLearningPolicy;
import com.example.ml.FeatureEmbedder;
import com.example.ml.FeatureLayout;
import com.example.ml.FeatureLocalisation;
import com.example.ml.FeatureTypeRegistry;
import com.example.ml.LearningCropStore;
import com.example.ml.ReferenceLibrary;
import com.example.ml.ReferenceLibraryProvider;
import com.example.ml.ScaleCalibration;
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

    /** Remembers whether the edge preview is expanded, so the choice survives the next inspection. */
    private static final String PREFS = "results_view_prefs";
    private static final String KEY_CANNY_EXPANDED = "canny_edges_expanded";

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

    // ── Positional check state ──
    /**
     * Expected arrangement of critical features, derived from the reference pass. Null when
     * this part has no reference photo, which is what makes localisation unavailable rather
     * than merely empty.
     */
    private FeatureLayout referenceLayout;
    /** Latest positional findings. Recomputed after every live label correction. */
    private FeatureLocalisation.Result localisation;

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

        setUpCannyEdgesToggle();

        runFeatureVerificationPipeline();
    }

    /**
     * The edge preview is a diagnostic view rather than something an inspector needs on
     * every run, so it collapses away. The bitmap is still set on the (hidden) ImageView
     * when the pipeline finishes — expanding just reveals it, no re-analysis.
     */
    private void setUpCannyEdgesToggle() {
        SharedPreferences prefs = requireContext().getSharedPreferences(PREFS, Context.MODE_PRIVATE);
        applyCannyEdgesExpanded(prefs.getBoolean(KEY_CANNY_EXPANDED, false));

        binding.headerCannyEdges.setOnClickListener(v -> {
            boolean expanded = binding.ivCannyEdges.getVisibility() != View.VISIBLE;
            applyCannyEdgesExpanded(expanded);
            prefs.edit().putBoolean(KEY_CANNY_EXPANDED, expanded).apply();
        });
    }

    private void applyCannyEdgesExpanded(boolean expanded) {
        binding.ivCannyEdges.setVisibility(expanded ? View.VISIBLE : View.GONE);
        // Filled triangle pointing down when open, right when closed.
        binding.tvCannyEdgesChevron.setText(expanded ? "▾" : "▸");
    }

    private void runFeatureVerificationPipeline() {
        new Thread(() -> {
            AppDatabase db = AppDatabase.getInstance(requireContext());
            ReferenceProfile ref = db.referenceProfileDao().getProfileByPartType(partId);

            // Effective reference library: per-part if set up in-app, else the bundled default,
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
                // Null sink: the reference pass only contributes counts and slot positions, so its
                // crops are never written or retained. This pass previously doubled peak crop
                // memory for nothing.
                HardwareFeatureExtractor.AnalysisResult refResult =
                        HardwareFeatureExtractor.extractFeatures(requireContext(), ref.getImageUri(),
                                activeLibrary, null);
                countByType(refResult.features, expectedCounts);
                referenceLayout = buildLayout(refResult.features);
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
            // Which embedding space produced the scores above, so a later threshold fit can
            // exclude events from a retired encoder instead of averaging across both.
            e.setEncoderId(FeatureEmbedder.ENCODER_ID);
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

        // Positional check before the verdict: a misplaced fastener is a failure the count
        // comparison above cannot see.
        localisation = runLocalisation();
        hasDiscrepancy = computeDiscrepancy();

        if (activeResult.processedBitmap != null) {
            binding.ivCannyEdges.setImageBitmap(activeResult.processedBitmap);
        }
        binding.chassisOverlayView.setResultData(imageUri, activeResult.features);
        binding.chassisOverlayView.setExpectedSlots(missingSlotMarkers());
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
        buildLocalisation();
        buildMeasurements();
        buildLearningStatus();
    }

    /**
     * Ghost markers for expected positions that are empty or hold the wrong hardware, in this
     * photo's coordinates. Present slots are left off: the operator can already see those as
     * detection boxes, and doubling every one with a marker would bury the few that matter.
     */
    private List<ChassisOverlayView.ExpectedSlot> missingSlotMarkers() {
        List<ChassisOverlayView.ExpectedSlot> markers = new ArrayList<>();
        if (localisation == null || !localisation.aligned) return markers;
        for (FeatureLocalisation.SlotFinding f : localisation.findings) {
            if (f.status == FeatureLocalisation.Status.PRESENT) continue;
            if (!f.hasProjection()) continue;
            markers.add(new ChassisOverlayView.ExpectedSlot(f.displayNumber, f.projectedX, f.projectedY,
                    f.status == FeatureLocalisation.Status.MISSING));
        }
        return markers;
    }

    /**
     * Reports where each expected critical feature is, and which specific one is missing.
     *
     * When the photo could not be registered onto the reference layout the section says so
     * explicitly instead of rendering empty. An unalignable shot and a part with every
     * position correct look identical in an empty list, and only one of them is good news.
     */
    private void buildLocalisation() {
        LinearLayout container = binding.layoutLocalisationRows;
        container.removeAllViews();

        if (localisation == null || !localisation.aligned) {
            binding.tvLocalisationStatus.setText(localisation == null
                    ? "Positional check not run." : localisation.summary());
            return;
        }

        binding.tvLocalisationStatus.setText(localisation.summary());

        for (FeatureLocalisation.SlotFinding f : localisation.findings) {
            LinearLayout row = new LinearLayout(requireContext());
            row.setOrientation(LinearLayout.HORIZONTAL);
            row.setPadding(0, dp(5), 0, dp(5));

            TextView slot = new TextView(requireContext());
            slot.setLayoutParams(new LinearLayout.LayoutParams(dp(38), LinearLayout.LayoutParams.WRAP_CONTENT));
            slot.setText("#" + f.displayNumber);
            slot.setTextColor(ContextCompat.getColor(requireContext(), R.color.gray_text_secondary));
            slot.setTextSize(13f);

            TextView desc = new TextView(requireContext());
            desc.setLayoutParams(new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
            String expected = FeatureTypeRegistry.displayName(requireContext(), f.expectedType);
            desc.setText(expected + " — " + f.position());
            desc.setTextColor(ContextCompat.getColor(requireContext(), R.color.white));
            desc.setTextSize(13f);

            TextView status = new TextView(requireContext());
            status.setLayoutParams(new LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT));
            status.setTextSize(12f);
            status.setTypeface(android.graphics.Typeface.defaultFromStyle(android.graphics.Typeface.BOLD));
            switch (f.status) {
                case MISSING:
                    status.setText("MISSING");
                    status.setTextColor(ContextCompat.getColor(requireContext(), R.color.red_fail));
                    break;
                case WRONG_TYPE:
                    status.setText("WRONG: " + FeatureTypeRegistry.displayName(requireContext(), f.foundType));
                    status.setTextColor(ContextCompat.getColor(requireContext(), R.color.orange_warn));
                    break;
                default:
                    status.setText("PRESENT");
                    status.setTextColor(ContextCompat.getColor(requireContext(), R.color.green_pass));
                    break;
            }

            row.addView(slot);
            row.addView(desc);
            row.addView(status);
            container.addView(row);
        }
    }

    /**
     * Reports physical sizes, or explains why there are none. A missing calibration marker
     * must read as "not measured" rather than silently showing an empty section — an absent
     * dimension is recoverable, a fabricated one is not.
     */
    private void buildMeasurements() {
        LinearLayout container = binding.layoutMeasurementRows;
        container.removeAllViews();

        ScaleCalibration cal = analysisResult != null ? analysisResult.calibration : null;
        if (cal == null) {
            binding.tvCalibrationStatus.setText(String.format(Locale.US,
                    "No calibration marker found — dimensions not measured.\n"
                            + "Place a %.0f mm square marker flat beside the part, in frame.",
                    ScaleCalibration.MARKER_SIZE_MM));
            return;
        }

        int measured = 0;
        for (HardwareFeatureExtractor.ExtractedFeature f : analysisResult.features) {
            if (f.measurement == null) continue;
            measured++;
            LinearLayout row = new LinearLayout(requireContext());
            row.setOrientation(LinearLayout.HORIZONTAL);
            row.setPadding(0, dp(4), 0, dp(4));

            TextView name = new TextView(requireContext());
            name.setLayoutParams(new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
            name.setText(FeatureTypeRegistry.displayName(requireContext(), f.type));
            name.setTextColor(ContextCompat.getColor(requireContext(), R.color.white));
            name.setTextSize(13f);
            row.addView(name);

            TextView value = new TextView(requireContext());
            value.setText(f.measurement.shortLabel(f.type));
            value.setTypeface(android.graphics.Typeface.MONOSPACE);
            value.setTextColor(ContextCompat.getColor(requireContext(), R.color.white));
            value.setTextSize(13f);
            row.addView(value);

            container.addView(row);
        }

        binding.tvCalibrationStatus.setText(String.format(Locale.US,
                "%d of %d detections measured.\n%s",
                measured, analysisResult.features.size(), cal.describe()));

        if (cal.squareness < 0.90) {
            TextView warn = new TextView(requireContext());
            warn.setText("Marker looks oblique — measurements assume a perpendicular shot, "
                    + "so accuracy is reduced. Re-shoot square to the part.");
            warn.setTextColor(ContextCompat.getColor(requireContext(), R.color.orange_warn));
            warn.setTextSize(11.5f);
            warn.setPadding(0, dp(8), 0, 0);
            container.addView(warn);
        }
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

        // Re-run the positional check: relabelling a detection changes which slots it can
        // occupy, so a correction can turn a "missing" slot into a matched one and back.
        localisation = runLocalisation();
        hasDiscrepancy = computeDiscrepancy();

        if (hasDiscrepancy) {
            binding.tvResultBadge.setText("INSPECTION FAILED");
            binding.tvResultSub.setText("Discrepancy detected in chassis configuration.");
            binding.layoutResultHeader.setBackgroundColor(ContextCompat.getColor(requireContext(), R.color.red_fail));
        } else {
            binding.tvResultBadge.setText("INSPECTION PASSED");
            binding.tvResultSub.setText("All expected features verified successfully.");
            binding.layoutResultHeader.setBackgroundColor(ContextCompat.getColor(requireContext(), R.color.green_pass_header));
        }

        binding.chassisOverlayView.setExpectedSlots(missingSlotMarkers());
        buildFeatureRows();
        buildLegend();
        buildLocalisation();
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
        // Only affects the wording of the skip option now; nothing is pre-selected.
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
        sub.setTextColor(Color.WHITE);
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
            guess.setTextColor(Color.WHITE);
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
            optionLabels.add(t.severity == FeatureTypeRegistry.Severity.CRITICAL
                    ? t.displayName
                    : t.displayName + " (" + t.severity.label.toLowerCase(java.util.Locale.US) + ")");
            optionKeys.add(t.key);
        }
        optionLabels.add("+ Create new type...");
        optionKeys.add("__new__");

        android.widget.Spinner spinner = new android.widget.Spinner(getContext());
        android.widget.LinearLayout.LayoutParams spinnerLp = new android.widget.LinearLayout.LayoutParams(
                android.widget.LinearLayout.LayoutParams.MATCH_PARENT, android.widget.LinearLayout.LayoutParams.WRAP_CONTENT);
        spinnerLp.setMargins(0, dp(16), 0, dp(8));
        spinner.setLayoutParams(spinnerLp);
        // Both the collapsed row and the dropdown popup default to the platform theme's colours,
        // which fight this dialog's dark background: theme text is dark-on-dark in the row, and
        // the popup is white, so white text disappears into it. Style both ends explicitly —
        // dark surfaces with white text — matching the card styling used elsewhere in the app.
        android.graphics.drawable.GradientDrawable spinnerBg = new android.graphics.drawable.GradientDrawable();
        spinnerBg.setColor(Color.rgb(43, 41, 48));
        spinnerBg.setCornerRadius(dp(10));
        spinnerBg.setStroke(dp(1), Color.rgb(95, 95, 105));
        spinner.setBackground(spinnerBg);
        spinner.setPopupBackgroundDrawable(
                new android.graphics.drawable.ColorDrawable(Color.rgb(32, 32, 38)));

        // Per-type similarity, so the operator can see how close the runners-up were rather than
        // just the top two. Rows are built by hand instead of using the platform item layout:
        // the score has to sit hard right, which a single TextView can't do reliably.
        final Map<String, Float> typeScores = (c != null && c.scores != null) ? c.scores : new LinkedHashMap<>();
        android.widget.ArrayAdapter<String> adapter = new android.widget.ArrayAdapter<String>(
                requireContext(), android.R.layout.simple_spinner_dropdown_item, optionLabels) {
            private View row(int position) {
                return spinnerRow(optionLabels.get(position), typeScores.get(optionKeys.get(position)));
            }
            @NonNull @Override
            public View getView(int position, View convertView, @NonNull ViewGroup parent) {
                return row(position);
            }
            @Override
            public View getDropDownView(int position, View convertView, @NonNull ViewGroup parent) {
                return row(position);
            }
        };
        spinner.setAdapter(adapter);

        // Open on the classifier's best match rather than Skip, so confirming a correct guess is
        // a single tap. Note this does anchor the operator toward agreeing — and those agreements
        // are the ground truth the tuner fits against — so the MIN_ACCURACY gate in ThresholdTuner
        // is what now stops a run of easy confirmations from driving the thresholds somewhere bad.
        String preferred = (c != null && c.closestMatch != null && !"none".equals(c.closestMatch))
                ? c.closestMatch : feature.type;
        if (preferred != null) {
            int preselect = optionKeys.indexOf(preferred);
            if (preselect >= 0) spinner.setSelection(preselect);
        }
        root.addView(spinner);

        // ── "Create new type" panel: just the name. Severity has its own panel below,
        //    because an existing type needs to be re-gradeable too — otherwise a catch-all
        //    like "surface" is stuck at whatever it was first saved as, with no screen
        //    anywhere in the app to change it.
        android.widget.LinearLayout newTypePanel = new android.widget.LinearLayout(getContext());
        newTypePanel.setOrientation(android.widget.LinearLayout.VERTICAL);
        newTypePanel.setVisibility(View.GONE);
        newTypePanel.setPadding(0, dp(8), 0, 0);

        android.widget.EditText nameInput = new android.widget.EditText(getContext());
        nameInput.setHint("New feature type name (e.g. Scratch)");
        nameInput.setTextColor(Color.WHITE);
        nameInput.setHintTextColor(Color.rgb(180, 180, 188));
        newTypePanel.addView(nameInput);
        root.addView(newTypePanel);

        // ── Severity scale ──
        // Laid out critical-on-the-left so the slider reads as "how much does this matter",
        // descending. That inverts the stored level, hence the maxLevel() - progress mapping
        // on every read and write below.
        android.widget.LinearLayout severityPanel = new android.widget.LinearLayout(getContext());
        severityPanel.setOrientation(android.widget.LinearLayout.VERTICAL);
        severityPanel.setVisibility(View.GONE);
        severityPanel.setPadding(0, dp(14), 0, 0);

        TextView severityHeading = new TextView(getContext());
        severityHeading.setText("How should this type be reported?");
        severityHeading.setTextColor(Color.WHITE);
        severityHeading.setTextSize(11f);
        severityPanel.addView(severityHeading);

        android.widget.SeekBar severitySeek = new android.widget.SeekBar(getContext());
        severitySeek.setMax(FeatureTypeRegistry.Severity.maxLevel());
        android.widget.LinearLayout.LayoutParams seekLp = new android.widget.LinearLayout.LayoutParams(
                android.widget.LinearLayout.LayoutParams.MATCH_PARENT,
                android.widget.LinearLayout.LayoutParams.WRAP_CONTENT);
        seekLp.setMargins(0, dp(6), 0, 0);
        severitySeek.setLayoutParams(seekLp);
        severityPanel.addView(severitySeek);

        android.widget.LinearLayout endLabels = new android.widget.LinearLayout(getContext());
        endLabels.setOrientation(android.widget.LinearLayout.HORIZONTAL);
        TextView leftEnd = new TextView(getContext());
        leftEnd.setText("CRITICAL");
        leftEnd.setTextColor(Color.rgb(180, 180, 188));
        leftEnd.setTextSize(9.5f);
        leftEnd.setLayoutParams(new android.widget.LinearLayout.LayoutParams(
                0, android.widget.LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        TextView rightEnd = new TextView(getContext());
        rightEnd.setText("IGNORED");
        rightEnd.setTextColor(Color.rgb(180, 180, 188));
        rightEnd.setTextSize(9.5f);
        rightEnd.setGravity(Gravity.END);
        rightEnd.setLayoutParams(new android.widget.LinearLayout.LayoutParams(
                0, android.widget.LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        endLabels.addView(leftEnd);
        endLabels.addView(rightEnd);
        severityPanel.addView(endLabels);

        TextView severityValue = new TextView(getContext());
        severityValue.setTextColor(Color.WHITE);
        severityValue.setTextSize(11.5f);
        severityValue.setPadding(dp(10), dp(8), dp(10), dp(8));
        android.widget.LinearLayout.LayoutParams valueLp = new android.widget.LinearLayout.LayoutParams(
                android.widget.LinearLayout.LayoutParams.MATCH_PARENT,
                android.widget.LinearLayout.LayoutParams.WRAP_CONTENT);
        valueLp.setMargins(0, dp(8), 0, 0);
        severityValue.setLayoutParams(valueLp);
        severityPanel.addView(severityValue);
        root.addView(severityPanel);

        // Null = the operator has not chosen yet. A SeekBar always has a position, so the
        // unset state is tracked separately rather than read off the widget; a brand-new type
        // must not inherit whatever stop the thumb happened to start on.
        FeatureTypeRegistry.Severity[] severityChoice = {null};
        Runnable refreshSeverity = () -> {
            FeatureTypeRegistry.Severity s = severityChoice[0];
            if (s == null) {
                severityValue.setText("Not set — drag the slider to choose.");
                severityValue.setBackgroundColor(Color.rgb(50, 50, 55));
                return;
            }
            int bg;
            switch (s) {
                case CRITICAL: bg = Color.rgb(120, 40, 44); break;
                case ADVISORY: bg = Color.rgb(112, 82, 26); break;
                default:       bg = Color.rgb(58, 58, 66); break;
            }
            severityValue.setText(s.label.toUpperCase(java.util.Locale.US) + " — " + s.blurb);
            severityValue.setBackgroundColor(bg);
        };
        refreshSeverity.run();

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
                    || (nameInput.getText().toString().trim().length() > 0 && severityChoice[0] != null);
            save.setEnabled(valid);
        };

        severitySeek.setOnSeekBarChangeListener(new android.widget.SeekBar.OnSeekBarChangeListener() {
            @Override public void onProgressChanged(android.widget.SeekBar bar, int progress, boolean fromUser) {
                severityChoice[0] = FeatureTypeRegistry.Severity.fromLevel(
                        FeatureTypeRegistry.Severity.maxLevel() - progress);
                refreshSeverity.run();
                refreshSaveEnabled.run();
            }
            @Override public void onStartTrackingTouch(android.widget.SeekBar bar) {}
            @Override public void onStopTrackingTouch(android.widget.SeekBar bar) {}
        });
        nameInput.addTextChangedListener(new android.text.TextWatcher() {
            @Override public void beforeTextChanged(CharSequence s, int a, int b, int c) {}
            @Override public void onTextChanged(CharSequence s, int a, int b, int c) {}
            @Override public void afterTextChanged(android.text.Editable s) { refreshSaveEnabled.run(); }
        });

        spinner.setOnItemSelectedListener(new android.widget.AdapterView.OnItemSelectedListener() {
            @Override public void onItemSelected(android.widget.AdapterView<?> parent, View v, int pos, long id) {
                String key = optionKeys.get(pos);
                boolean isNew = "__new__".equals(key);
                boolean isSkip = key == null;
                newTypePanel.setVisibility(isNew ? View.VISIBLE : View.GONE);
                severityPanel.setVisibility(isSkip ? View.GONE : View.VISIBLE);

                if (isNew) {
                    // Deliberately left unset so the operator has to state it, which is what
                    // keeps the scale meaningful. setProgress fires the listener, so the null
                    // is restored afterwards rather than before.
                    severitySeek.setProgress(0);
                    severityChoice[0] = null;
                } else if (!isSkip) {
                    FeatureTypeRegistry.Severity current =
                            FeatureTypeRegistry.severityOf(requireContext(), key);
                    severityChoice[0] = current;
                    severitySeek.setProgress(FeatureTypeRegistry.Severity.maxLevel() - current.level);
                }
                refreshSeverity.run();
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
                if (typeKey.isEmpty() || severityChoice[0] == null) {
                    save.setEnabled(true);
                    return;
                }
                FeatureTypeRegistry.upsert(requireContext(), typeKey, displayName, severityChoice[0]);
            } else {
                typeKey = key;
                // Confirming a label is also the moment the operator can re-grade the type,
                // so persist a moved slider. Only on an actual change: upsert rewrites the
                // whole catalog file, and this dialog runs once per reviewed crop.
                FeatureTypeRegistry.Severity chosen = severityChoice[0];
                if (chosen != null && chosen != FeatureTypeRegistry.severityOf(requireContext(), typeKey)) {
                    FeatureTypeRegistry.upsert(requireContext(), typeKey,
                            FeatureTypeRegistry.displayName(requireContext(), typeKey), chosen);
                }
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

    /**
     * One spinner row: type name on the left, similarity right-aligned. Score is null for the
     * Skip and "create new" entries, and for any type with no prototype to compare against.
     */
    private View spinnerRow(String label, Float score) {
        LinearLayout row = new LinearLayout(requireContext());
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER_VERTICAL);
        row.setPadding(dp(14), dp(12), dp(14), dp(12));

        TextView name = new TextView(requireContext());
        name.setLayoutParams(new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        name.setText(label);
        name.setTextColor(Color.WHITE);
        name.setTextSize(14f);
        name.setSingleLine(true);
        name.setEllipsize(android.text.TextUtils.TruncateAt.END);
        row.addView(name);

        TextView value = new TextView(requireContext());
        value.setLayoutParams(new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT));
        value.setText(score == null ? "" : String.format(Locale.US, "%.2f", score));
        // Monospace keeps the decimals in a straight column down the list.
        value.setTypeface(android.graphics.Typeface.MONOSPACE);
        value.setTextColor(Color.WHITE);
        value.setTextSize(14f);
        value.setPadding(dp(12), 0, 0, 0);
        row.addView(value);

        return row;
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
                    FeatureEmbedder embedder = FeatureEmbedder.get(requireContext());
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

    /**
     * Builds the registration layout from the reference pass: every detection whose type the
     * operator treats as a real feature, i.e. anything not graded
     * {@link FeatureTypeRegistry.Severity#IGNORED}.
     *
     * Deliberately wider than what gets reported. Recovering the transform is pure geometry
     * and every stable landmark helps, so restricting the layout to critical types only threw
     * away evidence and shrank the layout below the size an alignment can be believed at —
     * a reference with two scratches among five detections stopped working entirely. Ignored
     * types stay out because they are background by the operator's own declaration, and
     * background moves between shots.
     */
    private FeatureLayout buildLayout(List<HardwareFeatureExtractor.ExtractedFeature> refFeatures) {
        if (refFeatures == null || refFeatures.isEmpty()) return null;
        Set<String> registrationTypes = new LinkedHashSet<>();
        for (HardwareFeatureExtractor.ExtractedFeature f : refFeatures) {
            if (f.type == null) continue;
            if (FeatureTypeRegistry.severityOf(requireContext(), f.type)
                    != FeatureTypeRegistry.Severity.IGNORED) {
                registrationTypes.add(f.type);
            }
        }
        List<android.graphics.RectF> rects = new ArrayList<>();
        List<String> types = new ArrayList<>();
        for (HardwareFeatureExtractor.ExtractedFeature f : refFeatures) {
            rects.add(f.rectFraction);
            types.add(f.type);
        }
        FeatureLayout layout = FeatureLocalisation.layoutFrom(rects, types, registrationTypes);
        return layout.isEmpty() ? null : layout;
    }

    /**
     * Runs the positional check. Registration uses every non-ignored detection; only critical
     * slots are reported, which is the scope the localisation feature was asked for.
     *
     * Unknown detections are passed through as candidates on purpose: {@code LayoutMatcher}
     * allows them to occupy a slot, so a fastener that is physically present but which the
     * classifier failed to recognise reads as "present" rather than as missing hardware — the
     * less misleading of the two errors.
     */
    private FeatureLocalisation.Result runLocalisation() {
        if (referenceLayout == null) {
            return FeatureLocalisation.localise(null, null, null, null);
        }
        List<android.graphics.RectF> rects = new ArrayList<>();
        List<String> types = new ArrayList<>();
        Set<String> criticalTypes = new LinkedHashSet<>();
        for (HardwareFeatureExtractor.ExtractedFeature f : analysisResult.features) {
            if (f.type == null) continue;
            FeatureTypeRegistry.Severity severity =
                    FeatureTypeRegistry.severityOf(requireContext(), f.type);
            if (severity == FeatureTypeRegistry.Severity.CRITICAL) criticalTypes.add(f.type);
            boolean eligible = ReferenceLibrary.UNKNOWN.equals(f.type)
                    || severity != FeatureTypeRegistry.Severity.IGNORED;
            if (!eligible) continue;
            rects.add(f.rectFraction);
            types.add(f.type);
        }
        // Critical types the reference expects, even if none was detected this run — otherwise
        // a slot whose feature is entirely absent would be filtered out of its own report.
        for (FeatureLayout.Slot s : referenceLayout.slots()) {
            if (FeatureTypeRegistry.isCritical(requireContext(), s.type)) criticalTypes.add(s.type);
        }
        return FeatureLocalisation.localise(referenceLayout, rects, types, criticalTypes);
    }

    /**
     * Whether this run counts as a failure: any count mismatch on a
     * {@link FeatureTypeRegistry.Severity#CRITICAL} type, or — when the photo could be
     * registered onto the reference layout — any expected position that is empty or holds the
     * wrong hardware.
     *
     * The positional half is what catches a defect counting cannot see: two screws expected
     * and two found, but one of them in the wrong hole. It only contributes when alignment
     * succeeded; an unalignable photo yields no positional claim at all rather than a fail.
     *
     * "unknown" is skipped as before — it is unclassified noise, not a finding. Advisory and
     * ignored types are skipped for the same reason they stay out of the quantitative table:
     * a scratch more than last time is a trend note, and a differently-shaped patch of bare
     * surface is nothing at all. Before the scale existed both of those failed the part,
     * which is how a catch-all type like "surface" could turn a good part red.
     */
    private boolean computeDiscrepancy() {
        for (String type : orderedTypes) {
            if (ReferenceLibrary.UNKNOWN.equals(type)) continue;
            if (!FeatureTypeRegistry.isCritical(requireContext(), type)) continue;
            if (count(expectedCounts, type) != count(detectedCounts, type)) return true;
        }
        return localisation != null && localisation.aligned && localisation.hasPositionalDefect();
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
                int severity = FeatureTypeRegistry.severityOf(requireContext(), type).level;
                results.add(new FeatureResult(inspectionId, FeatureTypeRegistry.displayName(requireContext(), type),
                        expected, detected, status, severity));
            }
            if (!results.isEmpty()) {
                db.inspectionDao().insertFeatureResults(results);
            }

            // Positional findings, when the photo could be registered. Nothing is written for an
            // unaligned run: absent rows mean "no positional claim", which the report renders as
            // such, whereas rows saying PRESENT would assert a check that never happened.
            if (localisation != null && localisation.aligned) {
                List<SlotResult> slotRows = new ArrayList<>();
                for (FeatureLocalisation.SlotFinding f : localisation.findings) {
                    // displayNumber, not slotId: the raw layout id has gaps where advisory
                    // landmarks were filtered out, and the layout itself is never persisted, so
                    // the number the operator saw is the only one that stays meaningful.
                    slotRows.add(new SlotResult(inspectionId, f.displayNumber,
                            FeatureTypeRegistry.displayName(requireContext(), f.expectedType),
                            f.foundType == null ? null
                                    : FeatureTypeRegistry.displayName(requireContext(), f.foundType),
                            f.status.name(), f.x, f.y, f.relX, f.relY,
                            FeatureLocalisation.zone(f.relX, f.relY)));
                }
                if (!slotRows.isEmpty()) {
                    db.inspectionDao().insertSlotResults(slotRows);
                }
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
