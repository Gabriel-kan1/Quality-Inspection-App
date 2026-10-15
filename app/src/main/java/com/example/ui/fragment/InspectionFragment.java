package com.example.ui.fragment;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.PorterDuff;
import android.graphics.drawable.GradientDrawable;
import android.hardware.display.DisplayManager;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.util.Log;
import android.view.Display;
import android.view.LayoutInflater;
import android.view.Surface;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;
import androidx.navigation.Navigation;

import com.example.R;
import com.example.data.AppDatabase;
import com.example.data.entity.ReferenceProfile;
import com.example.ml.GhostOutline;
import com.example.ml.HardwareFeatureExtractor;
import com.example.ui.GhostMode;
import com.example.ml.AlignmentGate;
import com.example.ml.HoldTimer;
import com.example.ml.ProcessingTimes;
import com.example.ml.PartFiles;
import com.example.ui.util.AssetBitmaps;
import com.example.ui.view.TrafficLightView;
import com.example.databinding.FragmentInspectionBinding;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import com.example.ui.util.Notice;

/**
 * Capture screen for an inspection.
 *
 * <p>The camera feed is gated: each frame is scored against the part's reference photo by
 * {@link AlignmentGate}, and the shutter only fires once the operator has held the phone
 * inside tolerance for {@link HoldTimer#DEFAULT_HOLD_MS}. Capture is automatic at that
 * point, because asking for a button press at the decisive moment reintroduces exactly the
 * shake the hold was there to eliminate.
 */
public class InspectionFragment extends Fragment {

    private static final String TAG = "InspectionFragment";

    /** Where the frames the gate scores come from. */
    private enum Mode {
        /** Live camera, full traffic light with the three-second hold. */
        LIVE,
        /** A saved still. Scored once and lit red or green, but no hold — a static image
         *  cannot drift, so a dwell over it would be theatre. */
        STILL
    }

    private FragmentInspectionBinding binding;
    private String partId = "";
    private ReferenceProfile referenceProfile = null;
    private Uri selectedImageExternalUri = null;
    private GhostMode ghostMode = GhostMode.OUTLINE;

    /** The reference photograph at working resolution, for {@link GhostMode#TINTED} and {@link GhostMode#PLAIN}. */
    @Nullable private Bitmap ghostPhoto;
    /** Its edges alone on a transparent ground, for {@link GhostMode#OUTLINE}; null if OpenCV is down. */
    @Nullable private Bitmap ghostOutline;

    /**
     * Opacity of the photograph ghost. Was 0.35 untinted; the MULTIPLY tint multiplies every
     * channel down toward one hue, so the same alpha read as a fainter shadow, and 0.45 is the
     * lowest value at which its fastener heads stay readable over a bright plate.
     */
    private static final float TINTED_GHOST_ALPHA = 0.45f;
    /** Opacity of the untinted photograph ghost: the value it shipped with before any of this. */
    private static final float PLAIN_GHOST_ALPHA = 0.35f;
    /**
     * Opacity of the outline ghost. Near-opaque on purpose: it is a few pixels of line over an
     * otherwise untouched live image, so there is nothing for it to obscure and no reason for
     * it to be faint.
     */
    private static final float OUTLINE_GHOST_ALPHA = 0.9f;

    /** Written on the main thread, read by the analysis thread to drop stale frames. */
    private volatile Mode mode = Mode.LIVE;
    private boolean referenceMissing = false;
    private boolean debugReadoutVisible = false;
    /**
     * Why the chosen still failed the alignment gate, or null when it passed or when no
     * still has been scored. Main thread only, and carried to the results screen so a
     * verdict computed on a badly framed image is labelled as such.
     */
    @Nullable private String stillGateWarning = null;

    private Handler mainHandler;
    private ExecutorService analysisExecutor;
    private ProcessCameraProvider cameraProvider;
    private ImageCapture imageCapture;
    private ImageAnalysis imageAnalysis;

    /**
     * MainActivity declares configChanges="orientation|screenSize", so a rotation never
     * recreates this fragment and {@link #bindCameraUseCases()} never runs again. Without
     * this listener the target rotation captured at bind time would stay pinned to whichever
     * way up the phone happened to be when the camera started, and every later capture would
     * be handed to the encoder sideways. Both use cases accept a new rotation in place, so
     * there is nothing to rebind.
     */
    @Nullable private DisplayManager.DisplayListener displayListener;
    private int previewDisplayId = Display.INVALID_DISPLAY;

    /** Built off the reference photo once, then read from the analysis thread only. */
    private volatile AlignmentGate alignmentGate;
    private final HoldTimer holdTimer = new HoldTimer();
    /** Latches the moment the dwell completes so the shutter fires exactly once. */
    private final AtomicBoolean captureInFlight = new AtomicBoolean(false);

    private final ActivityResultLauncher<String> selectImageLauncher =
            registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
                if (uri != null) {
                    persistSelectedImage(uri);
                }
            });

    private final ActivityResultLauncher<String> cameraPermissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), granted -> {
                if (granted) {
                    startCamera();
                } else {
                    // Without a camera there is no live gate, so fall back to the still path
                    // rather than leaving the operator on a dead screen.
                    Notice.show(getContext(), "Camera denied — inspect a saved image instead.");
                    switchToStillMode();
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
        binding = FragmentInspectionBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @NonNull @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        mainHandler = new Handler(Looper.getMainLooper());
        analysisExecutor = Executors.newSingleThreadExecutor();
        // The results screen reports the gate's time per frame for this capture only.
        ProcessingTimes.resetGate();

        binding.tvPartInfo.setText("Part ID: " + partId);

        // Scan the custom user_images folder in assets
        scanAssetsFolder();

        setLight(TrafficLightView.State.OFF, 0f, "PREPARING ALIGNMENT CHECK", null);

        binding.layoutInteractiveTarget.setOnClickListener(v -> {
            if (mode == Mode.STILL) launchImageChooser();
        });

        binding.btnCaptureInspect.setOnClickListener(v -> {
            if (mode == Mode.LIVE) {
                // Live capture is driven by the gate, not by this button.
            } else if (selectedImageExternalUri == null) {
                launchImageChooser();
            } else {
                analyzeAndNavigate();
            }
        });

        binding.btnToggleGhost.setOnClickListener(v -> toggleGhostOverlay());
        renderGhost();

        binding.tvModeToggle.setOnClickListener(v -> {
            if (mode == Mode.LIVE) {
                switchToStillMode();
            } else {
                switchToLiveMode();
            }
        });

        // Long-press the hint to expose the raw alignment numbers while tuning thresholds.
        binding.tvAlignmentHint.setOnLongClickListener(v -> {
            debugReadoutVisible = !debugReadoutVisible;
            binding.tvAlignmentDebug.setVisibility(debugReadoutVisible ? View.VISIBLE : View.GONE);
            return true;
        });

        binding.btnGoToReferenceSetup.setOnClickListener(v -> {
            Bundle args = new Bundle();
            args.putString("partId", partId);
            if (isAdded() && getView() != null) {
                Navigation.findNavController(getView())
                        .navigate(R.id.action_inspection_to_reference_setup, args);
            }
        });

        // The same choice reference setup offers: the gallery, the camera, or a bundled photo.
        binding.btnUsePresetInspection.setOnClickListener(v -> launchImageChooser());

        // Everything else waits on the reference: without one there is nothing to gate against.
        loadReferenceProfile();
    }

    // ── Reference profile & gate construction ───────────────────────────

    private void loadReferenceProfile() {
        // Indexing the reference takes long enough that the fragment can be gone by the time
        // it finishes, so hold the application context rather than reaching for the fragment's.
        final android.content.Context context = requireContext().getApplicationContext();
        new Thread(() -> {
            AppDatabase db = AppDatabase.getInstance(context);
            ReferenceProfile profile = db.referenceProfileDao().getProfileByPartType(partId);

            String uri = profile != null ? profile.getImageUri() : null;
            boolean usable = profile != null && profile.hasRealImage();

            // Build the gate on this background thread — encoding the reference is an ONNX
            // inference, far too slow to run on the main thread.
            AlignmentGate gate = null;
            if (usable) {
                Bitmap reference = HardwareFeatureExtractor.loadBitmap(
                        context, uri, HardwareFeatureExtractor.WORKING_MAX_DIM);
                if (reference != null) {
                    gate = AlignmentGate.fromReference(context, reference);
                    reference.recycle();
                }
            }

            final AlignmentGate builtGate = gate;
            final boolean gateUsable = usable;
            mainHandler.post(() -> {
                if (!isAdded() || binding == null) return;
                referenceProfile = profile;
                alignmentGate = builtGate;
                applyGhostOverlay(gateUsable ? uri : null);

                if (!gateUsable) {
                    blockForMissingReference("This part has no reference photo.");
                } else if (builtGate == null) {
                    // Either the feature encoder is missing from assets or the reference
                    // could not be encoded. Without it every frame would score the same,
                    // so the gate would be decorative rather than a check.
                    blockForMissingReference("The feature encoder could not read this reference photo.");
                } else {
                    requestCameraAndStart();
                }
            });
        }).start();
    }

    /** Puts the screen into the blocked state: no gate, no capture, only a route to setup. */
    private void blockForMissingReference(String reason) {
        referenceMissing = true;
        binding.layoutNoReference.setVisibility(View.VISIBLE);
        binding.layoutUploadPlaceholder.setVisibility(View.GONE);
        binding.previewCamera.setVisibility(View.GONE);
        binding.viewTrafficLight.setVisibility(View.GONE);
        binding.tvAlignmentHint.setVisibility(View.GONE);
        binding.tvModeToggle.setVisibility(View.GONE);
        binding.cardAssetsFolderInspect.setVisibility(View.GONE);

        binding.tvAlignmentState.setText("BLOCKED");
        binding.tvAlignmentState.setBackgroundColor(
                ContextCompat.getColor(requireContext(), R.color.orange_warn));
        binding.layoutCameraEnclosure.setBackgroundResource(R.drawable.align_border_idle);

        binding.tvOperatorTip.setText(reason + " Alignment cannot be verified, so inspection is blocked.");
        binding.btnCaptureInspect.setEnabled(false);
        binding.btnCaptureInspect.setText("REFERENCE REQUIRED");
        binding.btnCaptureInspect.setBackgroundColor(
                ContextCompat.getColor(requireContext(), R.color.color_border));
        binding.btnToggleGhost.setEnabled(false);
    }

    /**
     * Loads the reference and prepares both ghost renderings of it.
     *
     * <p>Both are built here, once, rather than on each toggle: the outline is a full edge
     * stage at working resolution, which is too slow to run on a button press and pointless
     * to run twice. The photograph is loaded at {@link HardwareFeatureExtractor#WORKING_MAX_DIM}
     * rather than the old 1080 so the outline's pixel-denominated kernels mean what the study
     * tuned them to mean.
     */
    private void applyGhostOverlay(@Nullable String imageUri) {
        recycleGhosts();
        if (imageUri == null) {
            renderGhost();
            return;
        }
        ExecutorService executor = analysisExecutor;
        if (executor == null || executor.isShutdown()) return;
        final android.content.Context context = requireContext().getApplicationContext();
        final int tint = ContextCompat.getColor(context, R.color.ghost_tint);
        executor.execute(() -> {
            Bitmap photo = HardwareFeatureExtractor.loadBitmap(
                    context, imageUri, HardwareFeatureExtractor.WORKING_MAX_DIM);
            if (photo == null) return;
            Bitmap outline = GhostOutline.render(photo, tint);
            mainHandler.post(() -> {
                if (!isAdded() || binding == null) {
                    photo.recycle();
                    if (outline != null) outline.recycle();
                    return;
                }
                recycleGhosts();
                ghostPhoto = photo;
                ghostOutline = outline;
                renderGhost();
            });
        });
    }

    /**
     * Puts the current mode on screen: the right bitmap, the right treatment, the right labels.
     *
     * <p>The renderings need different treatments. The tinted photograph uses MULTIPLY, which
     * keeps its luminance detail while pushing every pixel toward one hue, and is shown
     * translucent because it covers the whole frame; the plain photograph is the same bitmap
     * with no filter at the opacity it always had. The outline is already painted in the
     * accent hue on a transparent ground, so a colour filter would only darken it; it gets
     * no filter and is shown near-opaque because there is nothing behind its lines to protect.
     * Alignment state deliberately stays on the traffic light and the border; neither
     * rendering's colour ever means pass or fail.
     */
    private void renderGhost() {
        if (binding == null) return;
        binding.btnToggleGhost.setText(ghostMode.buttonLabel());

        if (ghostPhoto == null) {
            binding.ivGhostOverlay.setVisibility(View.GONE);
            binding.tvGhostStatusIndicator.setText("NO ACTIVE GHOST OVERLAY");
            return;
        }
        if (!ghostMode.showsOverlay()) {
            binding.ivGhostOverlay.setVisibility(View.GONE);
            binding.tvGhostStatusIndicator.setText(ghostMode.statusLabel());
            return;
        }

        // OUTLINE falls back to the photograph when OpenCV could not produce an outline, and
        // says so: a silent fallback would leave the operator wondering why the wireframe
        // they saw yesterday is a photograph today.
        boolean outline = ghostMode == GhostMode.OUTLINE && ghostOutline != null;
        if (outline) {
            binding.ivGhostOverlay.clearColorFilter();
            binding.ivGhostOverlay.setImageBitmap(ghostOutline);
            binding.ivGhostOverlay.setAlpha(OUTLINE_GHOST_ALPHA);
            binding.tvGhostStatusIndicator.setText(ghostMode.statusLabel());
        } else if (ghostMode == GhostMode.PLAIN) {
            binding.ivGhostOverlay.clearColorFilter();
            binding.ivGhostOverlay.setImageBitmap(ghostPhoto);
            binding.ivGhostOverlay.setAlpha(PLAIN_GHOST_ALPHA);
            binding.tvGhostStatusIndicator.setText(ghostMode.statusLabel());
        } else {
            binding.ivGhostOverlay.setColorFilter(
                    ContextCompat.getColor(requireContext(), R.color.ghost_tint),
                    PorterDuff.Mode.MULTIPLY);
            binding.ivGhostOverlay.setImageBitmap(ghostPhoto);
            binding.ivGhostOverlay.setAlpha(TINTED_GHOST_ALPHA);
            binding.tvGhostStatusIndicator.setText(ghostMode == GhostMode.OUTLINE
                    ? "GHOST OVERLAY: TINTED (no outline - OpenCV unavailable)"
                    : ghostMode.statusLabel());
        }
        binding.ivGhostOverlay.setVisibility(View.VISIBLE);
    }

    /** Frees both renderings. Two working-resolution bitmaps are ~10MB of native memory. */
    private void recycleGhosts() {
        if (binding != null) binding.ivGhostOverlay.setImageDrawable(null);
        if (ghostPhoto != null && !ghostPhoto.isRecycled()) ghostPhoto.recycle();
        if (ghostOutline != null && !ghostOutline.isRecycled()) ghostOutline.recycle();
        ghostPhoto = null;
        ghostOutline = null;
    }

    /**
     * Decodes {@code uriString} on the analysis thread and shows it in {@code target}.
     *
     * <p>Replaces {@code ImageView.setImageURI}, which decodes synchronously on whatever
     * thread calls it — for a gallery pick that is a 4032x3024 JPEG, roughly 46MB of native
     * memory and a visible stall on the UI thread. {@link HardwareFeatureExtractor#loadBitmap}
     * samples it down on the way in, and applies the EXIF rotation that {@code setImageURI}
     * ignores, so the preview now matches the orientation the gate actually scores.
     *
     * <p>Shares {@link #analysisExecutor} rather than spawning a thread: this is a one-off per
     * pick or toggle, and camera frames are dropped rather than queued, so it interleaves
     * without backing anything up.
     */
    private void loadIntoAsync(ImageView target, String uriString, int maxDim) {
        ExecutorService executor = analysisExecutor;
        if (executor == null || executor.isShutdown()) return;
        // Resolved before the worker starts: requireContext() throws once the fragment
        // detaches, and a decode easily outlives a back press.
        final android.content.Context context = requireContext().getApplicationContext();
        executor.execute(() -> {
            Bitmap bitmap = HardwareFeatureExtractor.loadBitmap(context, uriString, maxDim);
            if (bitmap == null) return;
            mainHandler.post(() -> {
                if (!isAdded() || binding == null) {
                    bitmap.recycle();
                    return;
                }
                target.setImageBitmap(bitmap);
            });
        });
    }

    // ── Live camera ─────────────────────────────────────────────────────

    private void requestCameraAndStart() {
        if (ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA);
        }
    }

    private void startCamera() {
        if (!isAdded() || binding == null || referenceMissing) return;
        mode = Mode.LIVE;
        applyModeChrome();

        com.google.common.util.concurrent.ListenableFuture<ProcessCameraProvider> future =
                ProcessCameraProvider.getInstance(requireContext());
        future.addListener(() -> {
            if (!isAdded() || binding == null) return;
            try {
                cameraProvider = future.get();
                bindCameraUseCases();
            } catch (Exception e) {
                Log.e(TAG, "Camera provider unavailable", e);
                Notice.show(getContext(), "Camera unavailable — inspect a saved image instead.");
                switchToStillMode();
            }
        }, ContextCompat.getMainExecutor(requireContext()));
    }

    private void bindCameraUseCases() {
        Preview preview = new Preview.Builder().build();
        preview.setSurfaceProvider(binding.previewCamera.getSurfaceProvider());

        // Both use cases are pinned to the display's current rotation so the frame the gate
        // approves and the still that gets analysed are oriented the same way. The pinning has
        // to be kept up to date as the phone turns — see registerDisplayListener().
        int targetRotation = currentDisplayRotation();

        ImageAnalysis analysis = new ImageAnalysis.Builder()
                // Alignment is about where the phone is right now, so a backlog of stale
                // frames is worse than useless — always score the newest one.
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                // The encoder wants colour, and RGBA lets toBitmap() hand one over without
                // a YUV conversion of our own.
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setTargetRotation(targetRotation)
                .build();
        analysis.setAnalyzer(analysisExecutor, this::analyzeFrame);

        imageCapture = new ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                // Without this the saved JPEG records orientation in EXIF and leaves the
                // pixels in sensor orientation, so the analysis pass downstream sees a
                // sideways image of the part the gate just approved.
                .setTargetRotation(targetRotation)
                .build();

        try {
            cameraProvider.unbindAll();
            cameraProvider.bindToLifecycle(getViewLifecycleOwner(),
                    CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis, imageCapture);
            imageAnalysis = analysis;
            registerDisplayListener();
            holdTimer.reset();
            AlignmentGate gate = alignmentGate;
            if (gate != null) gate.resetMotion();
            captureInFlight.set(false);
            setLight(TrafficLightView.State.RED, 0f, "POINT AT THE PART", null);
            binding.tvOperatorTip.setText(
                    "Frame the area like the ghost and hold steady for 3 seconds. Capture is automatic.");
        } catch (Exception e) {
            Log.e(TAG, "Failed to bind camera use cases", e);
            Notice.show(getContext(), "Camera bind failed: " + e.getMessage());
            switchToStillMode();
        }
    }

    /** The rotation the preview is currently being shown at, or ROTATION_0 before it attaches. */
    private int currentDisplayRotation() {
        if (binding == null) return Surface.ROTATION_0;
        Display display = binding.previewCamera.getDisplay();
        return display != null ? display.getRotation() : Surface.ROTATION_0;
    }

    /**
     * Keeps the bound use cases pointed at the display's real rotation for as long as the
     * camera is live. Idempotent, so a rebind does not stack listeners.
     */
    private void registerDisplayListener() {
        if (displayListener != null || binding == null) return;

        Display display = binding.previewCamera.getDisplay();
        previewDisplayId = display != null ? display.getDisplayId() : Display.INVALID_DISPLAY;

        displayListener = new DisplayManager.DisplayListener() {
            @Override public void onDisplayAdded(int displayId) { }
            @Override public void onDisplayRemoved(int displayId) { }

            @Override
            public void onDisplayChanged(int displayId) {
                // Fires for brightness and refresh-rate changes too, so re-read rather than
                // assume the orientation actually moved. Setting the same value is a no-op.
                if (displayId != previewDisplayId || binding == null) return;
                int rotation = currentDisplayRotation();
                if (imageCapture != null) imageCapture.setTargetRotation(rotation);
                if (imageAnalysis != null) imageAnalysis.setTargetRotation(rotation);
            }
        };

        DisplayManager dm = (DisplayManager) requireContext().getSystemService(Context.DISPLAY_SERVICE);
        if (dm != null) {
            dm.registerDisplayListener(displayListener, mainHandler);
        } else {
            displayListener = null;
        }
    }

    private void unregisterDisplayListener() {
        if (displayListener == null) return;
        Context context = getContext();
        if (context != null) {
            DisplayManager dm = (DisplayManager) context.getSystemService(Context.DISPLAY_SERVICE);
            if (dm != null) dm.unregisterDisplayListener(displayListener);
        }
        displayListener = null;
        previewDisplayId = Display.INVALID_DISPLAY;
    }

    /**
     * Scores one preview frame. Runs on {@link #analysisExecutor}; touches no views except
     * through {@link #mainHandler}.
     */
    private void analyzeFrame(@NonNull ImageProxy image) {
        Bitmap frame = null;
        Bitmap upright = null;
        try {
            AlignmentGate gate = alignmentGate;
            if (gate == null || mode != Mode.LIVE || captureInFlight.get()) return;

            frame = image.toBitmap();
            upright = rotateToUpright(frame, image.getImageInfo().getRotationDegrees());

            // Read before evaluate: it times the motion rate, which must not include
            // the encoder passes evaluate is about to spend.
            long now = SystemClock.elapsedRealtime();
            long gateStart = ProcessingTimes.now();
            AlignmentGate.Reading reading = gate.evaluate(upright, now);
            ProcessingTimes.recordGateFrame(ProcessingTimes.now() - gateStart);
            float progress = holdTimer.update(reading.inTolerance, now);
            boolean passed = holdTimer.isSatisfied();

            mainHandler.post(() -> onReading(reading, progress, passed));
        } catch (Exception e) {
            Log.e(TAG, "Frame analysis failed", e);
        } finally {
            if (upright != null && upright != frame) upright.recycle();
            if (frame != null) frame.recycle();
            image.close();
        }
    }

    /**
     * Rotates a frame upright. The encoder is not rotation invariant and the reference was
     * encoded upright, so a sideways frame would score as a different area entirely.
     */
    private static Bitmap rotateToUpright(Bitmap frame, int rotationDegrees) {
        if (rotationDegrees == 0) return frame;
        android.graphics.Matrix matrix = new android.graphics.Matrix();
        matrix.postRotate(rotationDegrees);
        return Bitmap.createBitmap(frame, 0, 0, frame.getWidth(), frame.getHeight(), matrix, true);
    }

    /** Applies one frame's verdict to the traffic light, and fires the shutter on a pass. */
    private void onReading(AlignmentGate.Reading reading, float progress, boolean passed) {
        if (!isAdded() || binding == null || mode != Mode.LIVE) return;

        if (passed) {
            if (captureInFlight.compareAndSet(false, true)) {
                setLight(TrafficLightView.State.GREEN, 1f, "ALIGNED — CAPTURING", reading.debugLine());
                captureAlignedFrame();
            }
            return;
        }

        if (reading.inTolerance) {
            long remaining = holdTimer.remainingMs();
            setLight(TrafficLightView.State.AMBER, progress,
                    String.format(Locale.US, "HOLD STILL — %.1fs", remaining / 1000f),
                    reading.debugLine());
        } else {
            // The hint already carries the figure belonging to its own fault. Appending the
            // area score to every hint put "78% match" beside a rejection for being too
            // close, which reads as a contradiction rather than as advice.
            setLight(TrafficLightView.State.RED, 0f, reading.hint, reading.debugLine());
        }
    }

    /** Takes the full-resolution still once the dwell has completed. */
    private void captureAlignedFrame() {
        if (imageCapture == null) {
            captureInFlight.set(false);
            return;
        }

        File outputDir = requireContext().getExternalFilesDir("captured_inspections");
        if (outputDir != null && !outputDir.exists()) {
            outputDir.mkdirs();
        }
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault())
                .format(new java.util.Date());
        File photoFile = new File(outputDir, "COMP_" + PartFiles.safeName(partId) + "_" + timeStamp + ".jpg");

        ImageCapture.OutputFileOptions options =
                new ImageCapture.OutputFileOptions.Builder(photoFile).build();

        imageCapture.takePicture(options, ContextCompat.getMainExecutor(requireContext()),
                new ImageCapture.OnImageSavedCallback() {
                    @Override
                    public void onImageSaved(@NonNull ImageCapture.OutputFileResults results) {
                        if (!isAdded() || binding == null) return;
                        selectedImageExternalUri = Uri.fromFile(photoFile);
                        binding.tvOperatorTip.setText("Alignment held. Running geometric verification...");
                        navigateToResults(selectedImageExternalUri.toString());
                    }

                    @Override
                    public void onError(@NonNull ImageCaptureException e) {
                        Log.e(TAG, "Auto-capture failed", e);
                        if (!isAdded() || binding == null) return;
                        Notice.show(getContext(), "Capture failed: " + e.getMessage());
                        // Let the operator re-earn the green rather than stranding the gate.
                        holdTimer.reset();
                        captureInFlight.set(false);
                    }
                });
    }

    // ── Still-image path ────────────────────────────────────────────────

    private void switchToStillMode() {
        if (referenceMissing) return;
        mode = Mode.STILL;
        if (cameraProvider != null) cameraProvider.unbindAll();
        // Nothing is bound to re-orient any more; bindCameraUseCases() registers again if the
        // operator goes back to the live feed.
        unregisterDisplayListener();
        imageCapture = null;
        imageAnalysis = null;
        holdTimer.reset();
        captureInFlight.set(false);
        applyModeChrome();
        if (selectedImageExternalUri != null) {
            scoreStillImage(selectedImageExternalUri.toString());
        } else {
            setLight(TrafficLightView.State.OFF, 0f, "CHOOSE AN IMAGE TO SCORE", null);
        }
    }

    private void switchToLiveMode() {
        if (referenceMissing) return;
        selectedImageExternalUri = null;
        binding.ivChassisPreview.setVisibility(View.GONE);
        binding.ivChassisPreview.setImageDrawable(null);
        requestCameraAndStart();
    }

    /** Shows or hides the chrome that only makes sense in one of the two modes. */
    private void applyModeChrome() {
        boolean live = mode == Mode.LIVE;
        binding.previewCamera.setVisibility(live ? View.VISIBLE : View.GONE);
        binding.viewTrafficLight.setVisibility(View.VISIBLE);
        binding.tvAlignmentHint.setVisibility(View.VISIBLE);
        binding.cardAssetsFolderInspect.setVisibility(live ? View.GONE : View.VISIBLE);
        binding.layoutUploadPlaceholder.setVisibility(
                !live && selectedImageExternalUri == null ? View.VISIBLE : View.GONE);
        binding.tvModeToggle.setText(live
                ? "Inspect a saved image instead"
                : "Back to live camera gate");

        // Kept enabled in live mode so a press explains why nothing happened, rather than
        // leaving the operator prodding a dead button.
        binding.btnCaptureInspect.setEnabled(true);
        if (live) {
            binding.btnCaptureInspect.setText("AUTO-CAPTURE ON GREEN");
            binding.btnCaptureInspect.setBackgroundColor(
                    ContextCompat.getColor(requireContext(), R.color.color_border));
        } else {
            binding.btnCaptureInspect.setText(
                    selectedImageExternalUri == null ? "CHOOSE IMAGE" : "ANALYZE CHASSIS");
            binding.btnCaptureInspect.setBackgroundColor(ContextCompat.getColor(requireContext(),
                    selectedImageExternalUri == null ? R.color.blue_primary : R.color.green_pass_header));
        }
    }

    /**
     * Scores a chosen still once. No dwell — the light reports how well the saved image is
     * framed, and a red one is a warning rather than a block, so deliberately misaligned
     * assets stay usable for testing the detection pipeline.
     */
    private void scoreStillImage(String uriString) {
        AlignmentGate gate = alignmentGate;
        if (gate == null) return;

        setLight(TrafficLightView.State.OFF, 0f, "SCORING IMAGE...", null);
        // Stale until this image has actually been scored; a failure to score must not
        // inherit the previous image's verdict.
        stillGateWarning = null;
        // Runs on the analysis executor rather than a fresh thread: AlignmentGate reuses
        // scratch Mats across calls, so every use of it has to stay on one thread.
        ExecutorService executor = analysisExecutor;
        if (executor == null || executor.isShutdown()) return;
        // Resolved here rather than inside the worker: requireContext() throws once the
        // fragment detaches, and decoding plus scoring easily outlives a back press.
        final android.content.Context context = requireContext().getApplicationContext();
        executor.execute(() -> {
            Bitmap bitmap = HardwareFeatureExtractor.loadBitmap(
                    context, uriString, HardwareFeatureExtractor.WORKING_MAX_DIM);
            if (bitmap == null) return;
            AlignmentGate.Reading reading;
            try {
                // A still has no predecessor, so the motion estimate must not carry over
                // from whatever was scored before it.
                gate.resetMotion();
                reading = gate.evaluate(bitmap, SystemClock.elapsedRealtime());
                // The reading otherwise exists only on the traffic light. A wrong TOO FAR on
                // two near-identical photographs took a study to diagnose because nothing
                // recorded the framing ratio the gate had actually read.
                Log.d(TAG, "Still gate for " + uriString + ": " + reading.hint + " ["
                        + reading.debugLine() + "]");
            } catch (Exception e) {
                Log.e(TAG, "Still scoring failed", e);
                return;
            } finally {
                bitmap.recycle();
            }

            final AlignmentGate.Reading result = reading;
            mainHandler.post(() -> {
                if (!isAdded() || binding == null || mode != Mode.STILL) return;
                if (result.inTolerance) {
                    setLight(TrafficLightView.State.GREEN, 1f,
                            "SAME AREA, FRAMED LIKE THE REFERENCE", result.debugLine());
                    binding.tvOperatorTip.setText("Saved image matches the reference area and"
                            + " framing (area " + result.matchPercent() + "%). Ready to analyze.");
                } else {
                    // The hint names the one thing that is wrong and by how much, so it is
                    // the whole message here too rather than a prefix to a second number.
                    setLight(TrafficLightView.State.RED, 0f, result.hint, result.debugLine());
                    binding.tvOperatorTip.setText(result.hint
                            + ". You can still analyze it, but detections may be unreliable.");
                    // Carried to the results screen: analysis is still allowed here, so the
                    // warning has to travel with the image rather than stopping at this screen.
                    stillGateWarning = result.hint;
                }
            });
        });
    }

    /** Drives every piece of alignment chrome from a single place so they can't disagree. */
    private void setLight(TrafficLightView.State state, float progress,
                          String hint, @Nullable String debug) {
        if (binding == null) return;
        binding.viewTrafficLight.setState(state, progress);
        binding.tvAlignmentHint.setText(hint);

        int borderRes;
        int chipColor;
        String chipText;
        switch (state) {
            case GREEN:
                borderRes = R.drawable.align_border_green;
                chipColor = R.color.green_pass;
                chipText = "ALIGNED";
                break;
            case AMBER:
                borderRes = R.drawable.align_border_amber;
                chipColor = R.color.orange_warn;
                chipText = "HOLDING";
                break;
            case RED:
                borderRes = R.drawable.align_border_red;
                chipColor = R.color.red_fail;
                chipText = "NOT ALIGNED";
                break;
            default:
                borderRes = R.drawable.align_border_idle;
                chipColor = R.color.color_border;
                chipText = "STANDBY";
                break;
        }
        binding.layoutCameraEnclosure.setBackgroundResource(borderRes);
        binding.tvAlignmentState.setText(chipText);
        binding.tvAlignmentState.setBackgroundColor(ContextCompat.getColor(requireContext(), chipColor));

        binding.tvAlignmentDebug.setText(debug == null ? "" : debug);
        binding.tvAlignmentDebug.setVisibility(
                debugReadoutVisible && debug != null ? View.VISIBLE : View.GONE);
    }

    // ── Asset browser (unchanged behaviour, still mode only) ────────────

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
                binding.tvAssetsFolderStatus.setText("No images found in '/assets/chassis_images/'. Place your .jpg inside!");
                binding.scrollAssets.setVisibility(View.GONE);
            } else {
                binding.tvAssetsFolderStatus.setText("Found " + imageFiles.size() + " image(s) in assets/chassis_images/! Tap to inspect:");
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
        if (referenceMissing) return;
        try {
            // Clear first so the mode switch doesn't queue a pointless score of the image
            // that is about to be replaced.
            if (mode == Mode.LIVE) {
                selectedImageExternalUri = null;
                switchToStillMode();
            }
            selectedImageExternalUri = Uri.parse("file:///android_asset/chassis_images/" + filename);
            // Preview only — the analysis pipeline re-loads the asset at its own resolution,
            // so there's no reason to hold a 4032x3024 bitmap for an on-screen preview.
            Bitmap bitmap = AssetBitmaps.decodeSampled(
                    requireContext(), "chassis_images/" + filename, 1080);
            if (bitmap != null) binding.ivChassisPreview.setImageBitmap(bitmap);
            binding.ivChassisPreview.setVisibility(View.VISIBLE);
            binding.layoutUploadPlaceholder.setVisibility(View.GONE);

            applyModeChrome();
            scoreStillImage(selectedImageExternalUri.toString());
        } catch (Exception e) {
            Notice.show(getContext(), "Failed to load asset: " + e.getMessage());
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

            android.app.Dialog dialog = new android.app.Dialog(requireContext(), android.R.style.Theme_DeviceDefault_NoActionBar_Fullscreen);
            if (dialog.getWindow() != null) {
                dialog.getWindow().setBackgroundDrawable(new android.graphics.drawable.ColorDrawable(Color.argb(240, 15, 15, 18)));
            }

            android.widget.LinearLayout rootLayout = new android.widget.LinearLayout(getContext());
            rootLayout.setOrientation(android.widget.LinearLayout.VERTICAL);
            rootLayout.setPadding(32, 48, 32, 48);
            rootLayout.setBackgroundColor(Color.rgb(20, 20, 25));

            android.widget.TextView titleTv = new android.widget.TextView(getContext());
            titleTv.setText("CHOOSE CHASSIS TO INSPECT");
            titleTv.setTextSize(16f);
            titleTv.setTextColor(Color.rgb(74, 222, 128));
            titleTv.setTypeface(android.graphics.Typeface.DEFAULT_BOLD);
            titleTv.setGravity(android.view.Gravity.CENTER);
            titleTv.setPadding(0, 0, 0, 8);
            rootLayout.addView(titleTv);

            android.widget.TextView subTv = new android.widget.TextView(getContext());
            subTv.setText(imageFiles.isEmpty()
                    ? "Choose a photo from the gallery, or take one with the camera."
                    : "Choose a photo from the gallery, take one with the camera, or use one of"
                            + " the bundled test photos below:");
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

            // The live camera rather than the phone's camera app: its photo is taken by the
            // alignment gate, so it is framed like the reference before it is inspected.
            android.widget.Button cameraBtn = new android.widget.Button(getContext());
            android.widget.LinearLayout.LayoutParams cameraParams =
                    new android.widget.LinearLayout.LayoutParams(
                            android.widget.LinearLayout.LayoutParams.MATCH_PARENT,
                            android.widget.LinearLayout.LayoutParams.WRAP_CONTENT);
            cameraParams.topMargin = 16;
            cameraBtn.setLayoutParams(cameraParams);
            cameraBtn.setText("TAKE A PHOTO");
            cameraBtn.setTextColor(Color.WHITE);
            cameraBtn.setBackgroundColor(ContextCompat.getColor(requireContext(),
                    R.color.green_pass_header));
            cameraBtn.setOnClickListener(v -> {
                dialog.dismiss();
                switchToLiveMode();
            });
            rootLayout.addView(cameraBtn);

            rootLayout.addView(controlLayout);
            com.example.ui.util.SystemBars.keepClear(rootLayout);
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
                Notice.show(getContext(), "Cannot read selected image");
                return;
            }

            File outputDir = requireContext().getExternalFilesDir("captured_inspections");
            if (outputDir != null && !outputDir.exists()) {
                outputDir.mkdirs();
            }

            String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new java.util.Date());
            File photoFile = new File(outputDir, "COMP_" + PartFiles.safeName(partId) + "_" + timeStamp + ".jpg");

            FileOutputStream outputStream = new FileOutputStream(photoFile);
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, bytesRead);
            }
            outputStream.close();
            inputStream.close();

            if (mode == Mode.LIVE) {
                selectedImageExternalUri = null;
                switchToStillMode();
            }
            selectedImageExternalUri = Uri.fromFile(photoFile);

            loadIntoAsync(binding.ivChassisPreview, selectedImageExternalUri.toString(),
                    HardwareFeatureExtractor.WORKING_MAX_DIM);
            binding.ivChassisPreview.setVisibility(View.VISIBLE);
            binding.layoutUploadPlaceholder.setVisibility(View.GONE);

            applyModeChrome();
            scoreStillImage(selectedImageExternalUri.toString());

        } catch (Exception e) {
            Log.e(TAG, "Failed to copy chassis inspection photo", e);
            Notice.show(getContext(), "Error choosing file: " + e.getMessage());
        }
    }

    private void toggleGhostOverlay() {
        ghostMode = ghostMode.next();
        renderGhost();
    }

    private void analyzeAndNavigate() {
        if (selectedImageExternalUri == null) return;

        binding.btnCaptureInspect.setEnabled(false);
        navigateToResults(selectedImageExternalUri.toString());
    }

    private void navigateToResults(String imageUri) {
        if (isAdded() && getView() != null) {
            Bundle args = new Bundle();
            args.putString("partId", partId);
            args.putString("imageUri", imageUri);
            // Only ever set on the still path: a live capture cannot reach here without
            // holding the gate green for three seconds, so there is nothing to warn about.
            args.putString("alignmentWarning", stillGateWarning);
            Navigation.findNavController(getView()).navigate(R.id.action_inspection_to_results, args);
        }
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        unregisterDisplayListener();
        if (mainHandler != null) {
            mainHandler.removeCallbacksAndMessages(null);
        }
        if (cameraProvider != null) {
            cameraProvider.unbindAll();
            cameraProvider = null;
        }
        imageCapture = null;
        imageAnalysis = null;

        // Order matters: the gate owns native OpenCV buffers the GC knows nothing about, and
        // freeing them under a frame that is still being scored is a native crash rather than
        // an exception. Drain the analysis thread first, then release.
        AlignmentGate gate = alignmentGate;
        alignmentGate = null;
        if (analysisExecutor != null) {
            analysisExecutor.shutdown();
            try {
                analysisExecutor.awaitTermination(1, java.util.concurrent.TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            analysisExecutor = null;
        }
        if (gate != null) gate.release();
        recycleGhosts();
        binding = null;
    }
}
