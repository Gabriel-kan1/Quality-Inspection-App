package com.example.ui.fragment;

import android.content.Context;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.Gravity;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;

import com.example.R;
import com.example.databinding.FragmentEdgeLabBinding;
import com.example.databinding.ItemEdgeMethodBinding;
import com.example.ml.EdgeMethodLab;
import com.example.ml.HardwareFeatureExtractor;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Debug screen: one photograph, every edge operator, side by side — and for each, the boxes
 * and crops an inspection would take from its map.
 *
 * <p>The detector's edge stage has never been visible anywhere in the app. An inspection shows
 * the boxes that came out the far end, which is the last place a bad edge map is diagnosable —
 * by then a missed fastener and a fastener rejected on aspect ratio look identical. This screen
 * shows the stage itself, and shows it for the alternatives at the same time, so "Canny is
 * struggling with this lighting" becomes something to look at rather than something to assert.
 * Each card then follows its map through to what an inspection would do with it — where the
 * surviving boxes fall on the photograph and the crops the classifier would be handed — and a
 * header card states what the photograph measures as, and so which rung the adaptive stage
 * would run on it.
 *
 * <p>The comparison itself files nothing and touches no part record: it takes an image from
 * the gallery or from the bundled test photographs, and everything it produces lives until the
 * screen is left. The one thing it does change is the edge stage inspections run — each card
 * offers USE FOR INSPECTIONS, which persists that method as the detector's edge operator for
 * every inspection and reference setup that follows ({@link EdgeMethodLab#select}). See
 * {@link EdgeMethodLab} for the methods and for why they are scored the way they are.
 */
public class EdgeLabFragment extends Fragment {

    private static final String TAG = "EdgeLabFragment";

    /** Longest edge of the source preview. Big enough to judge framing, small enough to hold. */
    private static final int SOURCE_PREVIEW_PX = 720;

    /** Folder of bundled test photographs the TEST PHOTO button offers. */
    private static final String ASSET_FOLDER = "chassis_images";

    private FragmentEdgeLabBinding binding;

    private final Handler mainHandler = new Handler(Looper.getMainLooper());

    /**
     * Bitmaps currently held by the result cards and the source preview.
     *
     * <p>Nine edge maps plus a preview is tens of megabytes of native memory, and a second run
     * would strand the first set with nothing left holding a reference at a size the GC has no
     * urgency about. They are recycled explicitly when replaced and when the screen goes away.
     */
    private final List<Bitmap> displayedBitmaps = new ArrayList<>();

    /**
     * Incremented on every run so a slow first comparison cannot render over a faster second
     * one. The worker captures the value it started with and drops its results if it has moved.
     */
    private int runGeneration = 0;

    /** The cards on screen, so a choice on one can update the badges on all of them. */
    private final List<ItemEdgeMethodBinding> cards = new ArrayList<>();
    private final List<EdgeMethodLab.MethodResult> shownResults = new ArrayList<>();

    /** The shown run, so the verdict can be restated after a choice. */
    @Nullable private EdgeMethodLab.Comparison shownComparison;
    /**
     * The part every method is scored against, or null. Kept across runs: an operator
     * comparing photographs of one part should not have to pick it every time.
     */
    @Nullable private String scoringPart;
    /** The photograph on screen, so changing the part can re-run it. */
    @Nullable private String shownUri;
    @Nullable private String shownLabel;
    /** The notes about masking and the marker that follow the verdict; empty when neither. */
    private String shownQualityExtra = "";

    private final ActivityResultLauncher<String> pickImageLauncher =
            registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
                if (uri != null) compare(uri.toString(), fileNameOf(uri));
            });

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        binding = FragmentEdgeLabBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        binding.btnPickImage.setOnClickListener(v -> pickImageLauncher.launch("image/*"));
        binding.btnPickAsset.setOnClickListener(v -> showAssetChooser());
        binding.btnPickPart.setOnClickListener(v -> showPartChooser());
        showPart();
        showActiveMethod();

        if (!HardwareFeatureExtractor.isAvailable()) {
            // Every method on this screen is an OpenCV call, so without the native libraries
            // there is nothing to offer. Saying so beats nine identical failure cards.
            binding.tvEmpty.setText("OpenCV did not load on this device, so no edge method can "
                    + "run. This is the same failure that disables detection during an "
                    + "inspection.");
            binding.btnPickImage.setEnabled(false);
            binding.btnPickAsset.setEnabled(false);
        }
    }

    /** The header line naming the method inspections currently run. */
    private void showActiveMethod() {
        if (binding == null) return;
        String name = EdgeMethodLab.selectedName(requireContext());
        binding.tvActiveMethod.setText(EdgeMethodLab.defaultSelected(requireContext())
                ? "INSPECTIONS RUN: " + name + " (default)"
                : "INSPECTIONS RUN: " + name);
    }

    /**
     * Makes one method the edge stage for inspections and refreshes every card's badge and
     * button, so the screen shows the new state without re-running the comparison.
     */
    private void useMethod(EdgeMethodLab.MethodResult result) {
        EdgeMethodLab.select(requireContext(), result.id);
        String active = EdgeMethodLab.selectedId(requireContext());
        for (int i = 0; i < cards.size() && i < shownResults.size(); i++) {
            markInUse(cards.get(i), shownResults.get(i), active);
        }
        showActiveMethod();
        showQuality();
        Toast.makeText(getContext(), "Inspections and reference setup now use: " + result.name
                + ". Re-run reference setup for parts whose reference was made with another"
                + " method.", Toast.LENGTH_LONG).show();
    }

    private static void markInUse(ItemEdgeMethodBinding card, EdgeMethodLab.MethodResult result,
                                  String activeId) {
        boolean inUse = result.id.equals(activeId);
        card.tvMethodInUse.setVisibility(inUse ? View.VISIBLE : View.GONE);
        card.btnUseMethod.setEnabled(!inUse && !result.failed());
        card.btnUseMethod.setText(inUse ? "IN USE FOR INSPECTIONS"
                : result.failed() ? "DID NOT RUN ON THIS PHOTO" : "USE FOR INSPECTIONS");
        // In use outranks shipped for the stroke: the operator wants to find the card that is
        // running, and the SHIPPED badge still says which one that would be by default.
        card.cardMethod.setStrokeWidth(inUse ? 3 : (result.isShippedPipeline ? 2 : 0));
    }

    private void showPart() {
        binding.btnPickPart.setText(scoringPart == null ? "SCORE AGAINST PART: NONE"
                : "SCORE AGAINST PART: " + scoringPart);
    }

    /**
     * Offers the parts that have a labelled reference. Choosing one re-runs the photograph on
     * screen, since every method's score changes with it.
     */
    private void showPartChooser() {
        List<String> parts = EdgeMethodLab.partsWithReference(requireContext());
        if (parts.isEmpty()) {
            Toast.makeText(getContext(), "No part has a labelled reference yet. Set one up from "
                    + "a new inspection first.", Toast.LENGTH_LONG).show();
            return;
        }
        List<String> items = new ArrayList<>();
        items.add("None (starting point only)");
        items.addAll(parts);
        new AlertDialog.Builder(requireContext())
                .setTitle("Score every method against")
                .setItems(items.toArray(new String[0]), (d, which) -> {
                    scoringPart = which == 0 ? null : parts.get(which - 1);
                    showPart();
                    if (shownUri != null) compare(shownUri, shownLabel);
                })
                .setNegativeButton("Cancel", null)
                .show();
    }

    /** Offers the photographs bundled in assets, which are the images the studies were run on. */
    private void showAssetChooser() {
        List<String> images = new ArrayList<>();
        try {
            String[] files = requireContext().getAssets().list(ASSET_FOLDER);
            if (files != null) {
                for (String file : files) {
                    String lower = file.toLowerCase(Locale.US);
                    if (lower.endsWith(".jpg") || lower.endsWith(".jpeg") || lower.endsWith(".png")
                            || lower.endsWith(".webp") || lower.endsWith(".bmp")) {
                        images.add(file);
                    }
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "Could not list " + ASSET_FOLDER, e);
        }

        if (images.isEmpty()) {
            Toast.makeText(getContext(),
                    "No bundled test photographs — use CHOOSE IMAGE instead.",
                    Toast.LENGTH_SHORT).show();
            return;
        }

        String[] names = images.toArray(new String[0]);
        new AlertDialog.Builder(requireContext())
                .setTitle("Bundled test photograph")
                .setItems(names, (dialog, which) ->
                        compare("file:///android_asset/" + ASSET_FOLDER + "/" + names[which],
                                names[which]))
                .setNegativeButton("Cancel", null)
                .show();
    }

    /** Runs every method over one image and rebuilds the cards from the results. */
    private void compare(String uriString, String label) {
        final int generation = ++runGeneration;
        shownUri = uriString;
        shownLabel = label;
        final String part = scoringPart;

        clearResults();
        binding.tvSourceName.setText(label);
        binding.cardSource.setVisibility(View.VISIBLE);
        binding.tvEmpty.setVisibility(View.GONE);
        binding.layoutProgress.setVisibility(View.VISIBLE);
        binding.tvStatus.setText(String.format(Locale.US, "Running %d methods on %s%s…",
                EdgeMethodLab.catalogueSize(), label,
                part == null ? "" : " and scoring each against " + part));

        // Resolved here, on the main thread, and captured. requireContext() from the worker
        // would throw the moment the tab is left mid-comparison — which is exactly when a
        // several-second run is most likely to still be going.
        final Context context = requireContext().getApplicationContext();

        new Thread(() -> {
            Bitmap preview = HardwareFeatureExtractor.loadBitmap(
                    context, uriString, SOURCE_PREVIEW_PX);
            EdgeMethodLab.Comparison comparison = EdgeMethodLab.runAll(context, uriString, part);

            mainHandler.post(() -> {
                if (binding == null || !isAdded() || generation != runGeneration) {
                    // Superseded or gone. These bitmaps will never reach a view, and nothing
                    // else holds them, so they have to be freed here.
                    recycle(preview);
                    for (Bitmap b : comparison.bitmaps()) recycle(b);
                    return;
                }
                render(preview, comparison);
            });
        }, "edge-method-lab").start();
    }

    private void render(@Nullable Bitmap preview, EdgeMethodLab.Comparison comparison) {
        binding.layoutProgress.setVisibility(View.GONE);
        List<EdgeMethodLab.MethodResult> results = comparison.results;

        if (preview != null) {
            binding.ivSource.setImageBitmap(preview);
            displayedBitmaps.add(preview);
        }
        // Edge maps and crops, each once: a crop several methods found is one shared bitmap.
        displayedBitmaps.addAll(comparison.bitmaps());

        StringBuilder extra = new StringBuilder();
        if (comparison.backgroundFraction > 0) {
            extra.append(String.format(Locale.US, "\n\n%.1f%% of the frame was background and "
                    + "was painted out before any method ran, as an inspection does.",
                    100 * comparison.backgroundFraction));
        }
        if (comparison.markerFound) {
            extra.append("\n\nA calibration marker is in frame; boxes on it are left out of "
                    + "every card, as an inspection leaves them out.");
        }
        shownComparison = comparison;
        shownQualityExtra = extra.toString();
        showQuality();

        if (results.isEmpty()) {
            binding.tvEmpty.setText("Nothing came back for that image — it could not be decoded, "
                    + "or OpenCV is unavailable. Check logcat for the reason.");
            binding.tvEmpty.setVisibility(View.VISIBLE);
            return;
        }

        binding.tvReadingNote.setVisibility(View.VISIBLE);

        LayoutInflater inflater = LayoutInflater.from(requireContext());
        String active = EdgeMethodLab.selectedId(requireContext());
        for (EdgeMethodLab.MethodResult result : results) {
            ItemEdgeMethodBinding card =
                    ItemEdgeMethodBinding.inflate(inflater, binding.containerResults, false);

            card.tvMethodName.setText(result.name);
            String metrics = result.metricsLine();
            if (!comparison.scores.isEmpty() && !result.failed()
                    && comparison.scores.containsKey(result.id)) {
                metrics += Double.isNaN(result.referenceScore)
                        ? "  ·  not alignable to reference"
                        : String.format(Locale.US, "  ·  vs reference %.3f", result.referenceScore);
            }
            card.tvMethodMetrics.setText(metrics);
            // What a per-photograph method decided here — the adaptive card names its rung.
            card.tvMethodNote.setText(result.note);
            card.tvMethodNote.setVisibility(result.note == null ? View.GONE : View.VISIBLE);
            card.tvMethodSummary.setText(result.summary);
            card.tvMethodBadge.setVisibility(
                    result.isShippedPipeline ? View.VISIBLE : View.GONE);
            card.tvMethodPicked.setText(comparison.scores.isEmpty()
                    ? "ADAPTIVE STARTS HERE (NO REFERENCE TO SCORE AGAINST)"
                    : "ADAPTIVE RUNS THIS: CLOSEST TO THE REFERENCE");
            card.tvMethodPicked.setVisibility(result.picked ? View.VISIBLE : View.GONE);
            // A stroke rather than a different fill: the shipped row has to stand out without
            // reading as a verdict about which method won.
            markInUse(card, result, active);
            card.btnUseMethod.setOnClickListener(v -> useMethod(result));
            cards.add(card);
            shownResults.add(result);

            if (result.edgeMap != null) {
                card.ivMethodEdges.setImageBitmap(result.edgeMap);
            } else {
                card.ivMethodEdges.setVisibility(View.GONE);
            }
            showCrops(card, result, preview);

            binding.containerResults.addView(card.getRoot());
        }
    }

    /**
     * States which method the adaptive stage would run on this photo and why. Restated after a
     * USE FOR INSPECTIONS tap, since its last line depends on the choice.
     */
    private void showQuality() {
        if (binding == null) return;
        EdgeMethodLab.Comparison c = shownComparison;
        if (c == null || c.quality == null || c.chosenId == null) {
            binding.cardQuality.setVisibility(View.GONE);
            return;
        }
        binding.tvQualityTitle.setText(c.scores.isEmpty()
                ? "Adaptive starts with: " + EdgeMethodLab.nameOf(c.chosenId)
                : "Adaptive runs: " + EdgeMethodLab.nameOf(c.chosenId));
        StringBuilder body = new StringBuilder();
        for (String line : EdgeMethodLab.verdict(c.quality,
                EdgeMethodLab.selectedId(requireContext()), c.partId, c.partProblem,
                c.priorDescription, c.scores, c.incumbentId, c.chosenId)) {
            if (body.length() > 0) body.append('\n');
            body.append(line);
        }
        body.append(shownQualityExtra);
        binding.tvQualityBody.setText(body);
        binding.cardQuality.setVisibility(View.VISIBLE);
    }

    /**
     * Fills a card's crop section: the photograph with this method's boxes over it, then the
     * crops themselves, numbered alike. Both reuse bitmaps already tracked for recycling.
     */
    private void showCrops(ItemEdgeMethodBinding card, EdgeMethodLab.MethodResult result,
                           @Nullable Bitmap preview) {
        if (result.failed()) {
            card.tvCropsHeading.setVisibility(View.GONE);
            card.tvCropsLine.setVisibility(View.GONE);
            card.ivMethodBoxes.setVisibility(View.GONE);
            card.scrollCrops.setVisibility(View.GONE);
            return;
        }
        card.tvCropsLine.setText(result.cropsLine());
        if (preview != null) {
            card.ivMethodBoxes.setImageBitmap(preview);
            card.ivMethodBoxes.setBoxes(result.boxes);
        } else {
            card.ivMethodBoxes.setVisibility(View.GONE);
        }
        if (result.crops.isEmpty()) {
            card.scrollCrops.setVisibility(View.GONE);
            return;
        }

        Context context = requireContext();
        float density = getResources().getDisplayMetrics().density;
        int tile = Math.round(72 * density);
        int gap = Math.round(6 * density);
        int black = ContextCompat.getColor(context, R.color.black);
        int grey = ContextCompat.getColor(context, R.color.gray_text_secondary);
        for (int i = 0; i < result.crops.size(); i++) {
            LinearLayout cell = new LinearLayout(context);
            cell.setOrientation(LinearLayout.VERTICAL);
            cell.setGravity(Gravity.CENTER_HORIZONTAL);
            LinearLayout.LayoutParams cellParams = new LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
            cellParams.setMarginEnd(gap);

            Bitmap crop = result.crops.get(i);
            View picture;
            if (crop != null) {
                ImageView image = new ImageView(context);
                image.setScaleType(ImageView.ScaleType.FIT_CENTER);
                image.setImageBitmap(crop);
                image.setContentDescription("Crop " + (i + 1));
                picture = image;
            } else {
                // The inspection skips a crop it cannot read; say so rather than leave a gap.
                TextView none = new TextView(context);
                none.setGravity(Gravity.CENTER);
                none.setText("no crop");
                none.setTextSize(10);
                none.setTextColor(grey);
                picture = none;
            }
            picture.setBackgroundColor(black);
            cell.addView(picture, new LinearLayout.LayoutParams(tile, tile));

            TextView number = new TextView(context);
            number.setText(String.valueOf(i + 1));
            number.setTextSize(10);
            number.setTextColor(grey);
            cell.addView(number);

            card.containerCrops.addView(cell, cellParams);
        }
    }

    private void clearResults() {
        shownComparison = null;
        shownQualityExtra = "";
        binding.cardQuality.setVisibility(View.GONE);
        binding.containerResults.removeAllViews();
        cards.clear();
        shownResults.clear();
        binding.ivSource.setImageDrawable(null);
        binding.tvReadingNote.setVisibility(View.GONE);
        for (Bitmap bitmap : displayedBitmaps) recycle(bitmap);
        displayedBitmaps.clear();
    }

    private static void recycle(@Nullable Bitmap bitmap) {
        if (bitmap != null && !bitmap.isRecycled()) bitmap.recycle();
    }

    /** Best-effort display name for a picked image; the full content URI is unreadable. */
    private String fileNameOf(Uri uri) {
        String path = uri.getLastPathSegment();
        if (path == null) return uri.toString();
        int slash = path.lastIndexOf('/');
        return slash >= 0 ? path.substring(slash + 1) : path;
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        // Bumping the generation retires any comparison still running: its results will be
        // recycled on arrival rather than posted at a binding that no longer exists.
        runGeneration++;
        mainHandler.removeCallbacksAndMessages(null);
        for (Bitmap bitmap : displayedBitmaps) recycle(bitmap);
        displayedBitmaps.clear();
        cards.clear();
        shownResults.clear();
        binding = null;
    }
}
