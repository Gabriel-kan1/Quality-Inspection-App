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
import android.view.ViewGroup;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;
import androidx.fragment.app.Fragment;

import com.example.databinding.FragmentEdgeLabBinding;
import com.example.databinding.ItemEdgeMethodBinding;
import com.example.ml.EdgeMethodLab;
import com.example.ml.HardwareFeatureExtractor;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Debug screen: one photograph, every edge operator, side by side.
 *
 * <p>The detector's edge stage has never been visible anywhere in the app. An inspection shows
 * the boxes that came out the far end, which is the last place a bad edge map is diagnosable —
 * by then a missed fastener and a fastener rejected on aspect ratio look identical. This screen
 * shows the stage itself, and shows it for the alternatives at the same time, so "Canny is
 * struggling with this lighting" becomes something to look at rather than something to assert.
 *
 * <p>Deliberately outside the inspection flow: it writes nothing, files nothing and touches no
 * part record. It takes an image from the gallery or from the bundled test photographs, and
 * everything it produces lives until the screen is left. See {@link EdgeMethodLab} for the
 * methods and for why they are scored the way they are.
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

        clearResults();
        binding.tvSourceName.setText(label);
        binding.cardSource.setVisibility(View.VISIBLE);
        binding.tvEmpty.setVisibility(View.GONE);
        binding.layoutProgress.setVisibility(View.VISIBLE);
        binding.tvStatus.setText(String.format(Locale.US, "Running %d methods on %s…",
                EdgeMethodLab.catalogueSize(), label));

        // Resolved here, on the main thread, and captured. requireContext() from the worker
        // would throw the moment the tab is left mid-comparison — which is exactly when a
        // several-second run is most likely to still be going.
        final Context context = requireContext().getApplicationContext();

        new Thread(() -> {
            Bitmap preview = HardwareFeatureExtractor.loadBitmap(
                    context, uriString, SOURCE_PREVIEW_PX);
            List<EdgeMethodLab.MethodResult> results = EdgeMethodLab.runAll(context, uriString);

            mainHandler.post(() -> {
                if (binding == null || !isAdded() || generation != runGeneration) {
                    // Superseded or gone. These bitmaps will never reach a view, and nothing
                    // else holds them, so they have to be freed here.
                    recycle(preview);
                    for (EdgeMethodLab.MethodResult r : results) recycle(r.edgeMap);
                    return;
                }
                render(preview, results);
            });
        }, "edge-method-lab").start();
    }

    private void render(@Nullable Bitmap preview, List<EdgeMethodLab.MethodResult> results) {
        binding.layoutProgress.setVisibility(View.GONE);

        if (preview != null) {
            binding.ivSource.setImageBitmap(preview);
            displayedBitmaps.add(preview);
        }

        if (results.isEmpty()) {
            binding.tvEmpty.setText("Nothing came back for that image — it could not be decoded, "
                    + "or OpenCV is unavailable. Check logcat for the reason.");
            binding.tvEmpty.setVisibility(View.VISIBLE);
            return;
        }

        binding.tvReadingNote.setVisibility(View.VISIBLE);

        LayoutInflater inflater = LayoutInflater.from(requireContext());
        for (EdgeMethodLab.MethodResult result : results) {
            ItemEdgeMethodBinding card =
                    ItemEdgeMethodBinding.inflate(inflater, binding.containerResults, false);

            card.tvMethodName.setText(result.name);
            card.tvMethodMetrics.setText(result.metricsLine());
            card.tvMethodSummary.setText(result.summary);
            card.tvMethodBadge.setVisibility(
                    result.isShippedPipeline ? View.VISIBLE : View.GONE);
            // A stroke rather than a different fill: the shipped row has to stand out without
            // reading as a verdict about which method won.
            card.cardMethod.setStrokeWidth(result.isShippedPipeline ? 2 : 0);

            if (result.edgeMap != null) {
                card.ivMethodEdges.setImageBitmap(result.edgeMap);
                displayedBitmaps.add(result.edgeMap);
            } else {
                card.ivMethodEdges.setVisibility(View.GONE);
            }

            binding.containerResults.addView(card.getRoot());
        }
    }

    private void clearResults() {
        binding.containerResults.removeAllViews();
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
        binding = null;
    }
}
