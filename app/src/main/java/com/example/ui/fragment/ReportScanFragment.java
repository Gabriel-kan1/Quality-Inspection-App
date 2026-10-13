package com.example.ui.fragment;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;
import androidx.navigation.Navigation;

import com.example.R;
import com.example.data.AppDatabase;
import com.example.data.entity.InspectionRecord;
import com.example.data.entity.PartProfile;
import com.example.databinding.FragmentReportScanBinding;
import com.example.report.ReportCode;
import com.google.zxing.BinaryBitmap;
import com.google.zxing.DecodeHintType;
import com.google.zxing.MultiFormatReader;
import com.google.zxing.NotFoundException;
import com.google.zxing.PlanarYUVLuminanceSource;
import com.google.zxing.Result;
import com.google.zxing.common.HybridBinarizer;

import java.nio.ByteBuffer;
import java.util.EnumMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import com.example.ui.util.Notice;

/**
 * Reads the QR printed on an inspection report and opens that inspection.
 *
 * <p>Until this existed the code was write-only: the report carried a pointer and nothing in
 * the app could follow it, so the caption promising "scan to retrieve this inspection" was a
 * promise the software could not keep.
 *
 * <p>What the code carries is an identifier, not the report — see {@link ReportCode}. So this
 * screen resolves rather than decodes: the uuid is looked up in the local database, and a code
 * from another device correctly finds nothing. That is a real answer ("not filed on this
 * tablet"), not a failure, and it is reported as such.
 */
public class ReportScanFragment extends Fragment {

    private static final String TAG = "ReportScanFragment";

    private FragmentReportScanBinding binding;
    private ExecutorService analysisExecutor;
    @Nullable private ProcessCameraProvider cameraProvider;

    /**
     * Latched the moment a scan is acted on, so a decoder running at frame rate cannot fire the
     * same navigation several times before the transition completes.
     */
    private final AtomicBoolean handled = new AtomicBoolean(false);

    /**
     * The last payload that was rejected, so holding the camera over the wrong barcode explains
     * itself once instead of once per frame.
     */
    @Nullable private volatile String lastRejected;

    private final MultiFormatReader reader = new MultiFormatReader();

    private final ActivityResultLauncher<String> cameraPermissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), granted -> {
                if (granted) {
                    startCamera();
                } else {
                    status("Camera permission denied. Reports can still be opened from the"
                            + " history list.");
                }
            });

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        binding = FragmentReportScanBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        Map<DecodeHintType, Object> hints = new EnumMap<>(DecodeHintType.class);
        // TRY_HARDER costs time per frame and buys reads on codes that are angled, dim or
        // partly obscured — which is the normal condition of a label on a workshop bench.
        hints.put(DecodeHintType.TRY_HARDER, Boolean.TRUE);
        reader.setHints(hints);

        analysisExecutor = Executors.newSingleThreadExecutor();

        if (ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA);
        }
    }

    // ── Camera ──────────────────────────────────────────────────────────────

    private void startCamera() {
        if (!isAdded() || binding == null) return;
        com.google.common.util.concurrent.ListenableFuture<ProcessCameraProvider> future =
                ProcessCameraProvider.getInstance(requireContext());
        future.addListener(() -> {
            if (!isAdded() || binding == null) return;
            try {
                cameraProvider = future.get();
                bindUseCases();
            } catch (Exception e) {
                Log.e(TAG, "Camera provider unavailable", e);
                status("Camera unavailable. Reports can still be opened from the history list.");
            }
        }, ContextCompat.getMainExecutor(requireContext()));
    }

    private void bindUseCases() {
        Preview preview = new Preview.Builder().build();
        preview.setSurfaceProvider(binding.previewScan.getSurfaceProvider());

        // Left as the default YUV_420_888 rather than the RGBA the inspection pipeline asks
        // for: the decoder wants luminance only, and the Y plane is already exactly that, so
        // this avoids a colour conversion for data that would be thrown away.
        //
        // No target rotation either. A QR carries its own orientation in three finder
        // patterns, so rotating the buffer to match the display would be work with no effect
        // on whether it reads.
        ImageAnalysis analysis = new ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();
        analysis.setAnalyzer(analysisExecutor, this::analyzeFrame);

        try {
            cameraProvider.unbindAll();
            cameraProvider.bindToLifecycle(getViewLifecycleOwner(),
                    CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis);
        } catch (Exception e) {
            Log.e(TAG, "Failed to bind camera use cases", e);
            status("Camera could not start: " + e.getMessage());
        }
    }

    private void analyzeFrame(@NonNull ImageProxy image) {
        try {
            if (handled.get()) return;
            Result result = decode(image);
            if (result != null) {
                final String text = result.getText();
                postToUi(() -> onScanned(text));
            }
        } catch (Exception e) {
            Log.w(TAG, "Frame decode failed", e);
        } finally {
            // Always, on every path. A frame that is not closed stalls the whole analysis
            // pipeline, and the screen simply stops seeing anything.
            image.close();
        }
    }

    @Nullable
    private Result decode(ImageProxy image) {
        byte[] luminance = luminancePlane(image);
        int width = image.getWidth();
        int height = image.getHeight();
        PlanarYUVLuminanceSource source = new PlanarYUVLuminanceSource(
                luminance, width, height, 0, 0, width, height, false);
        try {
            return reader.decodeWithState(new BinaryBitmap(new HybridBinarizer(source)));
        } catch (NotFoundException e) {
            // The overwhelmingly common case: this frame has no code in it.
            return null;
        } finally {
            reader.reset();
        }
    }

    /**
     * Copies the Y plane into a tightly packed buffer.
     *
     * <p>The plane cannot be handed to ZXing as-is. Camera hardware pads each row out to its
     * own stride, and some devices interleave the luminance samples, so the raw buffer is
     * usually wider than the image and occasionally not contiguous at all. Reading it directly
     * produces a sheared picture that never decodes.
     */
    private static byte[] luminancePlane(ImageProxy image) {
        ImageProxy.PlaneProxy plane = image.getPlanes()[0];
        return packLuminance(plane.getBuffer(), image.getWidth(), image.getHeight(),
                plane.getRowStride(), plane.getPixelStride());
    }

    /**
     * The stride arithmetic, separated from {@link ImageProxy} so it can be tested.
     *
     * <p>This is the part with something to get wrong, and getting it wrong does not throw —
     * it produces a sheared image that simply never decodes, which looks identical to "the
     * camera cannot see the code". Hence {@code ReportScanDecodeTest}.
     */
    static byte[] packLuminance(ByteBuffer buffer, int width, int height,
                                int rowStride, int pixelStride) {
        byte[] out = new byte[width * height];
        byte[] row = new byte[rowStride];
        int written = 0;
        for (int y = 0; y < height; y++) {
            buffer.position(y * rowStride);
            buffer.get(row, 0, Math.min(rowStride, buffer.remaining()));
            if (pixelStride == 1) {
                System.arraycopy(row, 0, out, written, width);
            } else {
                for (int x = 0; x < width; x++) {
                    out[written + x] = row[x * pixelStride];
                }
            }
            written += width;
        }
        return out;
    }

    // ── Resolving a scan ────────────────────────────────────────────────────

    private void onScanned(String text) {
        if (!isAdded() || binding == null || handled.get()) return;

        ReportCode.Pointer pointer = ReportCode.decode(text);
        if (pointer == null) {
            explainForeignCode(text);
            return;
        }
        // Latch before the lookup, not after: the analyser is still running and would
        // otherwise queue a second navigation while this one is on a background thread.
        if (!handled.compareAndSet(false, true)) return;
        status("Report " + pointer.uuid.substring(0, 8) + "… — looking it up");
        resolve(pointer);
    }

    private void resolve(ReportCode.Pointer pointer) {
        AppDatabase db = AppDatabase.getInstance(requireContext().getApplicationContext());
        new Thread(() -> {
            InspectionRecord record;
            try {
                record = db.inspectionDao().getInspectionByUuid(pointer.uuid);
            } catch (Exception e) {
                Log.e(TAG, "Lookup failed", e);
                postToUi(() -> {
                    handled.set(false);
                    status("Could not read the database: " + e.getMessage());
                });
                return;
            }
            postToUi(() -> openOrExplain(pointer, record));
        }).start();
    }

    private void openOrExplain(ReportCode.Pointer pointer, @Nullable InspectionRecord record) {
        if (!isAdded() || binding == null) return;

        if (record == null) {
            // The code is well formed and names a real inspection — just not one filed here.
            // Saying so is the honest answer, and re-arming lets the operator try another.
            handled.set(false);
            status("That report was not filed on this device.\nRecord " + pointer.uuid);
            return;
        }

        // The integrity check, finally doing something. A mismatch means the printout and the
        // stored record disagree, which is worth saying out loud before the report is opened.
        if (pointer.digestPrefix != null && !pointer.matches(record)) {
            Notice.show(getContext(), "Warning: this printout does not match the filed record.");
        }

        Bundle args = new Bundle();
        args.putLong("inspectionId", record.getId());
        Navigation.findNavController(requireView())
                .navigate(R.id.action_scan_to_report, args);
    }

    /**
     * Explains a code that is not one of ours.
     *
     * <p>Worth a lookup rather than a flat rejection: the most likely thing an operator points
     * this at by mistake is the part's own label, and "that is a part, not a report" is a far
     * more useful answer than "unrecognised". Deliberately read-only — parts are added from the
     * registry on purpose, and creating one from an arbitrary scan is exactly the silent
     * near-duplicate problem the typed part field was removed to stop.
     */
    private void explainForeignCode(String text) {
        if (text.equals(lastRejected)) return;
        lastRejected = text;

        AppDatabase db = AppDatabase.getInstance(requireContext().getApplicationContext());
        new Thread(() -> {
            PartProfile part = null;
            try {
                part = db.partProfileDao().getPartProfileByPartId(text);
            } catch (Exception e) {
                Log.w(TAG, "Part lookup failed", e);
            }
            final boolean isKnownPart = part != null;
            postToUi(() -> {
                if (isKnownPart) {
                    status("That is the label for part " + text + ", not an inspection report"
                            + " code. Report codes are printed on the report itself.");
                } else {
                    status("Not an inspection report code:\n" + ellipsis(text));
                }
            });
        }).start();
    }

    private static String ellipsis(String text) {
        return text.length() <= 60 ? text : text.substring(0, 60) + "…";
    }

    // ── Plumbing ────────────────────────────────────────────────────────────

    private void status(String message) {
        if (binding != null) binding.tvScanStatus.setText(message);
    }

    private void postToUi(Runnable action) {
        if (!isAdded()) return;
        requireActivity().runOnUiThread(() -> {
            if (isAdded() && binding != null) action.run();
        });
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        if (cameraProvider != null) cameraProvider.unbindAll();
        cameraProvider = null;
        if (analysisExecutor != null) analysisExecutor.shutdown();
        analysisExecutor = null;
        binding = null;
    }
}
