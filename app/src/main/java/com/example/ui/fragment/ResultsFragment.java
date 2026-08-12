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
import com.example.ml.HardwareFeatureExtractor;
import com.example.ml.ActiveLearningPolicy;
import com.example.ml.FeatureEmbedder;
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

    /**
     * How many of a part's most recent reviews the threshold fit is allowed to see.
     *
     * <p>Generous next to ThresholdTuner's 20-event minimum, so the fit is never starved,
     * but bounded — the grid search is ~2000 points replayed over every event on every
     * inspection, and without a cap that cost climbs forever.
     */
    private static final int TUNING_WINDOW_EVENTS = 2000;

    /**
     * How long a part's reviewed, unfiled classification events are kept.
     *
     * <p>Events attached to a saved inspection are never dropped — they are the audit trail
     * behind a filed report. Unreviewed ones are never dropped either, since they are still
     * actionable. This only clears reviewed events that already contributed their answer to
     * the prototypes and thresholds and are now well outside the tuning window.
     */
    private static final long EVENT_RETENTION_MS = 180L * 24 * 60 * 60 * 1000; // ~6 months

    /** Disk budget for one part's learning crops. See LearningCropStore.prune. */
    private static final long CROP_BUDGET_BYTES = 64L * 1024 * 1024;

    private FragmentResultsBinding binding;
    private String partId = "";
    private String imageUri = "";
    /**
     * Set when the inspected image was a still that failed the alignment gate; null for
     * live captures, which cannot reach this screen without holding the gate green.
     *
     * <p>A misframed still is still analysed on purpose — it cannot be re-shot from here —
     * but the detections come off an image the app already judged out of tolerance, so the
     * results must not present themselves as if they came off a good one.
     */
    @Nullable private String alignmentWarning;

    // Per-type counts (keyed by canonical feature type, e.g. "screw").
    private final Map<String, Integer> expectedCounts = new LinkedHashMap<>();
    private final Map<String, Integer> detectedCounts = new LinkedHashMap<>();
    private List<String> orderedTypes = new ArrayList<>();
    private volatile Verdict verdict = Verdict.UNVERIFIED;
    private volatile boolean hasRealReference = false;

    /**
     * How an inspection ended.
     *
     * <p>{@link #UNVERIFIED} is deliberately not a kind of pass. Without a reference image
     * there is no expected configuration to compare against, so nothing was verified — and
     * saying so is the only honest option. This screen used to mirror the detected counts
     * into the expected ones in that case, which made the comparison trivially true and
     * reported "All expected features verified successfully" for a part whose expected
     * features had never been established.
     */
    enum Verdict { PASSED, FAILED, UNVERIFIED }

    // Effective reference library for this part — mutated in place as the user labels
    // "unknown" detections live, then persisted so future inspections benefit.
    private volatile ReferenceLibrary activeLibrary;

    // ── Self-learning loop state ──
    /** Row ids of this run's logged classification events, so they can be tied to the saved inspection. */
    private final List<Long> loggedEventIds = new ArrayList<>();
    /** Event row id per feature index, so a review can update the right row in place. */
    private final Map<Integer, Long> eventIdByFeature = new LinkedHashMap<>();
    // Written on the pipeline/review threads and read on the UI thread. The handler posts
    // give most of these a happens-before edge, but the fields are cheap to publish properly
    // and the counter below is incremented off a thread that never posts before it is read.
    private volatile ThresholdTuner.Tuned appliedThresholds;
    private final java.util.concurrent.atomic.AtomicInteger reviewsThisRun =
            new java.util.concurrent.atomic.AtomicInteger();
    /** Kept so overlay taps can correct a detection after the results screen is built. */
    private HardwareFeatureExtractor.AnalysisResult analysisResult;

    /**
     * Posts background results back to the UI.
     *
     * Deliberately a Handler rather than {@code requireActivity().runOnUiThread(...)}: the
     * pipeline outlives a back press, and requireActivity() throws IllegalStateException once
     * the fragment detaches — on a background thread, where it takes the process down.
     */
    private final android.os.Handler mainHandler =
            new android.os.Handler(android.os.Looper.getMainLooper());

    /** Runs {@code action} on the main thread, dropping it if the fragment is already gone. */
    private void postToUi(Runnable action) {
        mainHandler.post(() -> {
            if (!isAdded() || binding == null) return;
            action.run();
        });
    }

    @Override
    public void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        if (getArguments() != null) {
            partId = getArguments().getString("partId", "UNKNOWN_PART");
            imageUri = getArguments().getString("imageUri", "");
            alignmentWarning = getArguments().getString("alignmentWarning");
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
        // Resolved before the thread starts: the pipeline runs for several seconds, and
        // requireContext()/requireActivity() throw once the fragment detaches — a back press
        // mid-analysis would otherwise crash on a background thread.
        final android.content.Context context = requireContext().getApplicationContext();
        new Thread(() -> {
            AppDatabase db = AppDatabase.getInstance(context);
            ReferenceProfile ref = db.referenceProfileDao().getProfileByPartType(partId);

            // Effective reference library: per-part if set up in-app, else the bundled default,
            // else a fresh empty one (first run for this part — everything starts "unknown").
            ReferenceLibrary library = ReferenceLibraryProvider.effectiveFor(context, partId);
            activeLibrary = library != null ? library : new ReferenceLibrary();

            // Self-learning: apply thresholds fitted from this part's own confirmed history
            // (if enough of it exists) instead of the hand-picked defaults. Must happen
            // before classification so the whole run uses the tuned values.
            appliedThresholds = ThresholdTuner.applyTo(context, partId, activeLibrary);

            // Detected: classify the active chassis image. Crops are written straight to the
            // learning store and freed inside the extractor, so nothing scales with box count.
            HardwareFeatureExtractor.AnalysisResult activeResult =
                    HardwareFeatureExtractor.extractFeatures(context, imageUri, activeLibrary,
                            (crop, index) -> LearningCropStore.save(context, partId, crop, index));

            // Log every detection as a classification event — reviewed or not. Unreviewed
            // rows still carry the scores, so a later review or export can use them.
            logClassificationEvents(db, activeResult.features);

            // Expected: from the gold-standard reference image if one exists, else mirror detected
            // (computed after live-labelling below, since labelling can change detected types).
            expectedCounts.clear();
            hasRealReference = ref != null && ref.hasRealImage();
            if (hasRealReference) {
                // Null sink: the reference pass only contributes counts, so its crops are never
                // written or retained. This pass previously doubled peak crop memory for nothing.
                //
                // The glare branch is pinned to whatever the inspection image used. These two
                // counts are about to be subtracted from each other, so they have to come out
                // of the same pipeline — otherwise a reference just under the glare threshold
                // and an inspection just over it disagree on lighting alone and the part is
                // reported as failed.
                HardwareFeatureExtractor.AnalysisResult refResult =
                        HardwareFeatureExtractor.extractFeatures(context, ref.getImageUri(),
                                activeLibrary, null, activeResult.glareCorrectionApplied);
                countByType(refResult.features, expectedCounts);
                if (refResult.processedBitmap != null) {
                    // The reference pass only contributes counts; its diagnostic bitmap is a
                    // full-size ARGB_8888 nobody displays.
                    refResult.processedBitmap.recycle();
                }
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

            // Housekeeping last: this run's crops and events are already written, so pruning
            // here bounds the dataset without racing anything the UI is about to read.
            pruneLearningData(context, db);

            postToUi(() -> {
                if (!isAdded() || binding == null) return;
                resolveCandidatesSequentially(activeResult, candidates, 0,
                        // Refit thresholds with this run's answers included, so the next
                        // inspection of this part starts from a better decision boundary.
                        () -> retuneThresholds(() -> finalizeResults(activeResult)));
            });
        }).start();
    }

    /**
     * Bounds the on-device learning dataset for this part.
     *
     * <p>Every detection writes a crop and a row, and nothing removed either, so a part
     * inspected regularly grew both without limit — up to a few hundred PNGs per inspection.
     * Best-effort: failing to prune must never fail an inspection, so anything thrown here is
     * logged and swallowed.
     */
    private void pruneLearningData(android.content.Context context, AppDatabase db) {
        try {
            db.classificationEventDao().deleteStaleForPart(
                    partId, System.currentTimeMillis() - EVENT_RETENTION_MS);
            // Crops behind an unreviewed event are the only evidence a later review could
            // look at, so they survive the byte cap regardless of age.
            Set<String> keep = new java.util.HashSet<>(
                    db.classificationEventDao().getUnreviewedCropPaths(partId));
            LearningCropStore.prune(context, partId, CROP_BUDGET_BYTES, keep);
        } catch (Exception e) {
            android.util.Log.e("ResultsFragment", "Learning-data prune failed", e);
        }
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

        if (activeResult.processedBitmap != null) {
            binding.ivCannyEdges.setImageBitmap(activeResult.processedBitmap);
        }
        binding.chassisOverlayView.setResultData(imageUri, activeResult.features);
        // Any detection can be corrected by tapping it, not just the ones the app chose to
        // ask about — the operator can see a wrong label the classifier was confident about.
        binding.chassisOverlayView.setOnFeatureClickListener((index, f) -> correctFeatureAt(index));

        recomputeAndRender();
    }

    /**
     * The verdict rule, as a pure function so it can be tested without a fragment.
     *
     * <p>Counts are compared per type; {@code unknown} is skipped because unclassified
     * detections are noise on both sides rather than evidence of a discrepancy.
     */
    static Verdict verdictFor(boolean hasRealReference,
                              Map<String, Integer> expected,
                              Map<String, Integer> detected,
                              Iterable<String> types) {
        if (!hasRealReference) return Verdict.UNVERIFIED;
        for (String type : types) {
            if (ReferenceLibrary.UNKNOWN.equals(type)) continue;
            if (count(expected, type) != count(detected, type)) return Verdict.FAILED;
        }
        return Verdict.PASSED;
    }

    /**
     * Recomputes the counts and verdict from {@link #analysisResult} and re-renders the
     * summary. Called once when the pipeline finishes and again after every live correction,
     * which only changes labels — never the detections or the overlay bitmap.
     */
    private void recomputeAndRender() {
        detectedCounts.clear();
        countByType(analysisResult.features, detectedCounts);

        // With no reference, "expected" is unknowable. Mirroring detected keeps the summary
        // table readable as a plain inventory of what was found; the verdict below does not
        // consult it, and the rows render their expected/status cells as "—" instead of
        // claiming a match. See Verdict.UNVERIFIED.
        if (!hasRealReference) {
            expectedCounts.clear();
            expectedCounts.putAll(detectedCounts);
        }

        // Ordered union of types (canonical order first, "unknown" last).
        orderedTypes = orderedUnion(expectedCounts.keySet(), detectedCounts.keySet());
        verdict = verdictFor(hasRealReference, expectedCounts, detectedCounts, orderedTypes);
        renderVerdict();

        buildFeatureRows();
        buildLegend();
        buildLearningStatus();
    }

    /** Paints the result header for the current {@link #verdict}. */
    private void renderVerdict() {
        final String badge;
        final String sub;
        final int color;
        switch (verdict) {
            case FAILED:
                badge = "INSPECTION FAILED";
                sub = "Discrepancy detected in chassis configuration.";
                color = R.color.red_fail;
                break;
            case UNVERIFIED:
                badge = "UNVERIFIED";
                sub = "No reference image for this part — features detected, nothing compared.";
                color = R.color.orange_warn;
                break;
            case PASSED:
            default:
                badge = "INSPECTION PASSED";
                sub = "All expected features verified successfully.";
                color = R.color.green_pass_header;
                break;
        }
        binding.tvResultBadge.setText(badge);
        binding.tvResultSub.setText(sub);
        binding.layoutResultHeader.setBackgroundColor(
                ContextCompat.getColor(requireContext(), color));

        if (alignmentWarning != null && !alignmentWarning.isEmpty()) {
            binding.tvAlignmentWarning.setText("⚠ IMAGE FAILED ALIGNMENT CHECK — "
                    + alignmentWarning + ". Detections may be unreliable.");
            binding.tvAlignmentWarning.setVisibility(View.VISIBLE);
        } else {
            binding.tvAlignmentWarning.setVisibility(View.GONE);
        }
    }

    /** Surfaces the state of the learning loop so it isn't an invisible black box. */
    private void buildLearningStatus() {
        final android.content.Context context = requireContext().getApplicationContext();
        new Thread(() -> {
            int reviewedTotal = 0;
            int loggedTotal = 0;
            try {
                AppDatabase db = AppDatabase.getInstance(context);
                reviewedTotal = db.classificationEventDao().countReviewedForPart(partId);
                loggedTotal = db.classificationEventDao().countForPart(partId);
            } catch (Exception e) {
                android.util.Log.e("ResultsFragment", "Failed to read learning stats", e);
            }

            final int reviewed = reviewedTotal;
            final int logged = loggedTotal;
            postToUi(() -> {
                if (!isAdded() || binding == null) return;

                StringBuilder sb = new StringBuilder();
                sb.append(String.format(Locale.US,
                        "Labelled examples for this part: %d confirmed of %d logged detections.\n",
                        reviewed, logged));
                sb.append(String.format(Locale.US, "You reviewed %d this run.\n", reviewsThisRun.get()));

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
        recomputeAndRender();
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

        // ── "Create new type" panel: name + explicit critical/non-critical choice ──
        android.widget.LinearLayout newTypePanel = new android.widget.LinearLayout(getContext());
        newTypePanel.setOrientation(android.widget.LinearLayout.VERTICAL);
        newTypePanel.setVisibility(View.GONE);
        newTypePanel.setPadding(0, dp(8), 0, 0);

        android.widget.EditText nameInput = new android.widget.EditText(getContext());
        nameInput.setHint("New feature type name (e.g. Scratch)");
        nameInput.setTextColor(Color.WHITE);
        nameInput.setHintTextColor(Color.rgb(180, 180, 188));
        newTypePanel.addView(nameInput);

        TextView criticalLabel = new TextView(getContext());
        criticalLabel.setText("Should this type be included in the report?");
        criticalLabel.setTextColor(Color.WHITE);
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
        final android.content.Context context = requireContext().getApplicationContext();
        new Thread(() -> {
            try {
                AppDatabase db = AppDatabase.getInstance(context);
                String source;
                if (candidate.kind == ActiveLearningPolicy.Kind.SPOT_CHECK) {
                    source = ClassificationEvent.SOURCE_SPOT_CHECK;
                } else if (candidate.kind == ActiveLearningPolicy.Kind.USER_INITIATED) {
                    source = ClassificationEvent.SOURCE_OVERLAY_TAP;
                } else {
                    source = ClassificationEvent.SOURCE_UNKNOWN_PROMPT;
                }
                db.classificationEventDao().recordReview(eventId, confirmedType, source);
                reviewsThisRun.incrementAndGet();
            } catch (Exception ex) {
                android.util.Log.e("ResultsFragment", "Failed to record review", ex);
            }
            postToUi(() -> {
                if (isAdded()) onDone.run();
            });
        }).start();
    }

    /**
     * Refits per-part thresholds against the (now larger) confirmed history. Runs after the
     * review pass so this inspection's answers count toward the next one.
     */
    private void retuneThresholds(Runnable onDone) {
        final android.content.Context context = requireContext().getApplicationContext();
        new Thread(() -> {
            ThresholdTuner.Tuned fitted = null;
            try {
                AppDatabase db = AppDatabase.getInstance(context);
                List<ClassificationEvent> reviewed = db.classificationEventDao()
                        .getRecentReviewedForPart(partId, TUNING_WINDOW_EVENTS);
                fitted = ThresholdTuner.fit(reviewed);
                if (fitted != null) {
                    ThresholdTuner.save(context, partId, fitted);
                }
            } catch (Exception e) {
                android.util.Log.e("ResultsFragment", "Threshold retune failed", e);
            }
            final ThresholdTuner.Tuned result = fitted;
            postToUi(() -> {
                if (!isAdded()) return;
                if (result != null) appliedThresholds = result;
                onDone.run();
            });
        }).start();
    }

    /** Relabels the feature in place, feeds the crop back into the reference library, and persists it. */
    private void applyLabelAndPersist(HardwareFeatureExtractor.ExtractedFeature feature, String typeKey, Runnable onDone) {
        final android.content.Context context = requireContext().getApplicationContext();
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
                    FeatureEmbedder embedder = FeatureEmbedder.get(context);
                    if (embedder != null) {
                        float[] emb = embedder.embed(crop);
                        if (emb != null) {
                            activeLibrary.addExample(typeKey, emb);
                            activeLibrary.save(ReferenceLibraryProvider.fileForPart(context, partId));
                        }
                    }
                } finally {
                    crop.recycle();
                }
            }

            postToUi(() -> {
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

        // With no reference there is no expected count — showing the mirrored detected
        // figure here would read as a confirmed match.
        boolean comparable = verdict != Verdict.UNVERIFIED;
        TextView exp = cell(comparable ? String.valueOf(expected) : "—", 0.9f, Gravity.CENTER);
        exp.setTextColor(ContextCompat.getColor(requireContext(), R.color.gray_text_secondary));
        exp.setTextSize(14f);
        row.addView(exp);

        TextView det = cell(String.valueOf(detected), 0.9f, Gravity.CENTER);
        det.setTextColor(ContextCompat.getColor(requireContext(), R.color.white));
        det.setTextSize(14f);
        row.addView(det);

        String statusText;
        int statusColor;
        if (ReferenceLibrary.UNKNOWN.equals(type) || !comparable) {
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
        // The filed report is the artefact that outlives this screen, so the caveat has to
        // travel with it — a PDF that omits "this image failed the alignment check" reads
        // as a clean inspection.
        if (alignmentWarning != null && !alignmentWarning.isEmpty()) {
            String caveat = "Alignment check failed on the inspected image: " + alignmentWarning + ".";
            notes = notes.isEmpty() ? caveat : caveat + "\n\n" + notes;
        }
        final String finalNotes = notes;
        binding.btnSaveReport.setEnabled(false);

        final android.content.Context context = requireContext().getApplicationContext();
        new Thread(() -> {
            AppDatabase db = AppDatabase.getInstance(context);

            InspectionRecord record = new InspectionRecord();
            record.setPartId(partId);
            record.setTimestamp(System.currentTimeMillis());
            record.setPassOrFail(verdict == Verdict.FAILED ? "FAIL"
                    : verdict == Verdict.UNVERIFIED ? "UNVERIFIED" : "PASS");
            record.setImageUri(imageUri);
            record.setNotes(finalNotes);

            long inspectionId = db.inspectionDao().insertInspection(record);

            List<FeatureResult> results = new ArrayList<>();
            for (String type : orderedTypes) {
                if (ReferenceLibrary.UNKNOWN.equals(type)) continue;
                int expected = count(expectedCounts, type);
                int detected = count(detectedCounts, type);
                // Mirrors the on-screen table: with no reference the expected count is the
                // detected one echoed back, so calling every row a MATCH would put a claim
                // in the filed report that the inspection never established.
                String status = verdict == Verdict.UNVERIFIED ? "UNVERIFIED"
                        : expected == detected ? "MATCH"
                        : (detected < expected ? "MISSING" : "EXTRA");
                boolean critical = FeatureTypeRegistry.isCritical(context, type);
                results.add(new FeatureResult(inspectionId, FeatureTypeRegistry.displayName(context, type),
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

            postToUi(() -> {
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
        // Drop queued background results: their runnables capture views, and the pipeline
        // routinely outlives this fragment.
        mainHandler.removeCallbacksAndMessages(null);
        // Not recycled — ivCannyEdges still holds it until the view hierarchy is collected,
        // and freeing it here risks drawing a recycled bitmap during teardown.
        analysisResult = null;
        binding = null;
    }
}
