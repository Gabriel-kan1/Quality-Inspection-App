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
import com.example.ui.util.MethodBrief;
import com.example.databinding.FragmentReferenceSetupBinding;
import com.example.ml.AdaptivePreprocessing;
import com.example.ml.HardwareFeatureExtractor;
import com.example.ml.FeatureEmbedder;
import com.example.ml.EdgeMethodLab;
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
import com.example.ui.util.Notice;

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
    /**
     * The edge method the operator pinned for this reference, or null to let the adaptive
     * stage start from its own suggestion. Only meaningful during setup; see
     * {@code showMethodChoice}.
     */
    private String pendingMethod;
    /**
     * How many features the operator said this reference photo holds, or 0 if they did not say.
     * When set, setup runs every method and starts from the one whose count is nearest it.
     */
    private int expectedCount = 0;
    /** Whether the method on screen was picked by {@code expectedCount} rather than the operator. */
    private boolean pendingAutoPicked = false;
    /**
     * A photo is chosen but not analysed: the card is asking for the count, which decides the
     * edge method the analysis runs with.
     */
    private boolean awaitingCount = false;
    /**
     * Every method's boxes for the photo at {@link #surveyUri}, as fractions. They depend on
     * the photo alone, so changing the count, flipping between methods or using another one
     * reuses them instead of running every method again.
     */
    private String surveyUri;
    private java.util.Map<String, java.util.List<android.graphics.RectF>> methodBoxes;
    /** The order the arrows step through the methods in: nearest the count first. */
    private final java.util.List<String> methodOrder = new ArrayList<>();
    /**
     * The method whose boxes are on the photo: the analysed one in use, or another being
     * previewed. Null until an analysis has finished.
     */
    private String shownMethod;
    /** Why the method in use is in use; shown whenever its boxes are on the photo. */
    private CharSequence inUseNote;
    private boolean inUseExact;
    /** An analysis is running: its result, not the one on screen, is what would be saved. */
    private boolean analysing = false;

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

    /** Where this screen keeps its own display choices. */
    private static final String UI_PREFS = "reference_setup_ui";
    /** Whether the method drop-down is open; open until the operator folds it away. */
    private static final String KEY_ABOUT_OPEN = "method_about_open";

    /** Saved-state key for {@link #pendingCapture}: the camera app can outlive this process. */
    private static final String STATE_PENDING_CAPTURE = "pending_capture";

    /**
     * The file the camera app was asked to write the reference photo into, until it answers.
     * Kept across a process restart: a phone short of memory may kill this app while the
     * camera is open, and the photo must still be found when it returns.
     */
    private File pendingCapture;

    private final ActivityResultLauncher<Uri> takePictureLauncher =
            registerForActivityResult(new ActivityResultContracts.TakePicture(),
                    saved -> onPhotoTaken(Boolean.TRUE.equals(saved)));

    private final ActivityResultLauncher<String> cameraPermissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), granted -> {
                if (granted) {
                    launchCamera();
                } else {
                    Notice.show(getContext(),
                            "The camera permission is needed to take a"
                            + " reference photo. A photo can still be chosen from the gallery.");
                }
            });

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
        if (savedInstanceState != null) {
            String path = savedInstanceState.getString(STATE_PENDING_CAPTURE);
            if (path != null) pendingCapture = new File(path);
        }
    }

    @Override
    public void onSaveInstanceState(@NonNull Bundle outState) {
        super.onSaveInstanceState(outState);
        if (pendingCapture != null) {
            outState.putString(STATE_PENDING_CAPTURE, pendingCapture.getAbsolutePath());
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
            } else if (analysisReady()) {
                showClassifyAllDialog(this::proceedToInspection);
            }
        });

        binding.btnFindMethod.setOnClickListener(v -> findMethodForCount());
        binding.btnSkipCount.setOnClickListener(v -> skipCount());
        binding.etExpectedCount.setOnEditorActionListener((field, actionId, event) -> {
            // Done on the soft keyboard, or Enter on a hardware one (which arrives twice, as
            // the key goes down and up, and should run the match once).
            boolean enter = event != null
                    && event.getKeyCode() == android.view.KeyEvent.KEYCODE_ENTER;
            if (actionId != android.view.inputmethod.EditorInfo.IME_ACTION_DONE && !enter) {
                return false;
            }
            if (!enter || event.getAction() == android.view.KeyEvent.ACTION_DOWN) {
                findMethodForCount();
            }
            return true;
        });

        binding.tvMethodAboutHeader.setOnClickListener(v -> {
            uiPrefs().edit().putBoolean(KEY_ABOUT_OPEN, !aboutOpen()).apply();
            if (shownMethod != null) renderAbout();
        });
        binding.btnSaveReferenceToGallery.setOnClickListener(v -> {
            if (selectedImageExternalUri == null) return;
            com.example.ui.util.GallerySaver.savePhoto(requireContext(),
                    selectedImageExternalUri.toString(),
                    com.example.ui.util.GallerySaver.name("REFERENCE", partId));
        });
        binding.btnMethodPrev.setOnClickListener(v -> stepMethod(-1));
        binding.btnMethodNext.setOnClickListener(v -> stepMethod(1));
        binding.layoutMethodLabel.setOnClickListener(v -> chooseMethodFromList());
        binding.btnUseMethod.setOnClickListener(v -> {
            if (shownMethod != null && selectedImageExternalUri != null) {
                analyseReference(selectedImageExternalUri.toString(), shownMethod);
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
                Notice.show(getContext(), "Choose or load a reference image first.");
                launchImageChooser();
            } else if (analysisReady()) {
                showClassifyAllDialog(this::proceedToInspection);
            }
        });
    }

    /**
     * Whether what is on screen is what would be saved; if not, says what is still to happen.
     * A previewed method's boxes are not saved — they have not been classified — so saving
     * waits until the operator either uses that method or steps back to the one in use.
     */
    private boolean analysisReady() {
        String message = null;
        if (awaitingCount) {
            message = "Enter how many features this part has, then press FIND BEST METHOD"
                    + " (or SKIP).";
        } else if (analysing || pendingResult == null) {
            message = "Still analysing this image — one moment.";
        } else if (shownMethod != null && !shownMethod.equals(inUseMethod())) {
            message = "The photo is showing " + EdgeMethodLab.nameOf(shownMethod) + ". Press USE"
                    + " THIS METHOD to save with it, or step back to "
                    + EdgeMethodLab.nameOf(inUseMethod()) + ".";
        }
        if (message == null) return true;
        Notice.show(getContext(), message);
        return false;
    }


    // ── Reference preview: the same pass an inspection runs ──────────────

    /**
     * A photo has been chosen: the card asks how many features the part should have, on the
     * screen beside the photo the operator counts them from. Nothing is analysed until they
     * answer (FIND BEST METHOD) or decline (SKIP), because the answer decides which edge method
     * the analysis runs with, and an analysis run first would only be thrown away. The count
     * already in the field is kept: a second photo of the same part has the same hardware.
     */
    private void awaitExpectedCount() {
        ++analysisToken;                   // a pass still running for the previous photo is stale
        releasePendingCrops();
        pendingMethod = null;
        pendingAutoPicked = false;
        awaitingCount = true;
        analysing = false;
        shownMethod = null;
        methodOrder.clear();

        binding.cardReferenceAnalysis.setVisibility(View.VISIBLE);
        binding.tvReferenceAnalysisStatus.setText("How many features should this part have?"
                + " Count them on the photo, enter the number and press FIND BEST METHOD: every"
                + " edge method is tried, and the one whose count is closest is used. SKIP starts"
                + " from the photo's own noise and contrast instead.");
        binding.layoutReferenceTypeRows.removeAllViews();
        binding.tvReferenceLocalisationReadiness.setText("");
        hideMethodChoice();
        binding.scrollReferenceAnalysis.scrollTo(0, 0);
    }

    private void hideMethodChoice() {
        binding.layoutMethodSwitcher.setVisibility(View.GONE);
        binding.btnUseMethod.setVisibility(View.GONE);
        binding.layoutMethodAbout.setVisibility(View.GONE);
        binding.tvReferenceStrengthNote.setVisibility(View.GONE);
    }

    /**
     * The drop-down under the switcher: what the method on the photo is good at and the key
     * steps it runs, following the arrows so each method can be read as it is compared.
     */
    private void renderAbout() {
        boolean open = aboutOpen();
        binding.tvMethodAboutHeader.setText((open ? "▾  " : "▸  ") + "ABOUT "
                + EdgeMethodLab.nameOf(shownMethod).toUpperCase(Locale.ROOT));
        binding.tvMethodAbout.setText(MethodBrief.of(shownMethod));
        binding.cardMethodAbout.setVisibility(open ? View.VISIBLE : View.GONE);
        binding.layoutMethodAbout.setVisibility(View.VISIBLE);
    }

    private boolean aboutOpen() {
        return uiPrefs().getBoolean(KEY_ABOUT_OPEN, true);
    }

    private android.content.SharedPreferences uiPrefs() {
        return requireContext().getSharedPreferences(UI_PREFS,
                android.content.Context.MODE_PRIVATE);
    }

    /** FIND BEST METHOD: analyses with the edge method whose count is nearest the operator's. */
    private void findMethodForCount() {
        if (selectedImageExternalUri == null) {
            Notice.show(getContext(), "Choose a reference image first.");
            return;
        }
        int n;
        try {
            n = Integer.parseInt(binding.etExpectedCount.getText().toString().trim());
        } catch (NumberFormatException e) {
            n = 0;
        }
        if (n <= 0) {
            binding.etExpectedCount.setError("Enter how many features: 1 or more");
            return;
        }
        expectedCount = n;
        dismissKeyboard();
        analyseReference(selectedImageExternalUri.toString());
    }

    /** SKIP: analyses from the image-quality starting point, with no count to match. */
    private void skipCount() {
        if (selectedImageExternalUri == null) {
            Notice.show(getContext(), "Choose a reference image first.");
            return;
        }
        expectedCount = 0;
        binding.etExpectedCount.setError(null);
        dismissKeyboard();
        analyseReference(selectedImageExternalUri.toString());
    }

    private void dismissKeyboard() {
        android.view.inputmethod.InputMethodManager imm = requireContext()
                .getSystemService(android.view.inputmethod.InputMethodManager.class);
        if (imm != null) imm.hideSoftInputFromWindow(binding.etExpectedCount.getWindowToken(), 0);
        binding.etExpectedCount.clearFocus();
    }

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
        analyseReference(uriString, null);
    }

    /**
     * @param pinnedMethod the catalogue id of the edge method to run, or null to let the
     *        adaptive stage start from its suggestion. Reference setup is the one place an
     *        operator picks one: the adaptive stage chooses by scoring methods against a
     *        labelled reference, and this is the step that creates it, so there is nothing to
     *        score against yet. The operator can see the photograph, so they choose instead.
     */
    private void analyseReference(String uriString, String pinnedMethod) {
        final int token = ++analysisToken;
        final android.content.Context context = requireContext().getApplicationContext();
        pendingMethod = pinnedMethod;
        pendingAutoPicked = false;
        awaitingCount = false;
        analysing = true;
        final int target = pinnedMethod == null ? expectedCount : 0;
        final java.util.Map<String, java.util.List<android.graphics.RectF>> cachedBoxes =
                uriString.equals(surveyUri) ? methodBoxes : null;

        binding.cardReferenceAnalysis.setVisibility(View.VISIBLE);
        binding.tvReferenceAnalysisStatus.setText(target > 0
                ? "Trying every edge method against your count of " + target + "..."
                : pinnedMethod != null
                        ? "Analysing with " + EdgeMethodLab.nameOf(pinnedMethod) + "..."
                        : "Analysing reference image...");
        binding.layoutReferenceTypeRows.removeAllViews();
        binding.tvReferenceLocalisationReadiness.setText("");
        hideMethodChoice();
        binding.scrollReferenceAnalysis.scrollTo(0, 0);

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

            // What every method would find. Edge stage and geometry filter only, no crops and
            // no encoder. With nothing to score against yet, the operator's own count (when
            // given) is the target, and the operator flips between these boxes on the photo.
            final java.util.Map<String, java.util.List<android.graphics.RectF>> boxes =
                    cachedBoxes != null ? cachedBoxes
                            : HardwareFeatureExtractor.surveyMethods(context, uriString);
            final String counted = AdaptivePreprocessing.closestToCount(
                    AdaptivePreprocessing.counts(boxes), target);
            final String runMethod = counted != null ? counted : pinnedMethod;
            final boolean autoPicked = counted != null;

            HardwareFeatureExtractor.AnalysisResult result =
                    HardwareFeatureExtractor.extractFeatures(context, uriString, library,
                            (crop, index) -> {
                                Bitmap kept = scaleForKeeping(crop);
                                if (kept != null) crops.put(index, kept);
                                return null;
                            }, null, null, runMethod);
            if (result.processedBitmap != null) result.processedBitmap.recycle();

            requireActivity().runOnUiThread(() -> {
                if (!isAdded() || binding == null || token != analysisToken) {
                    for (Bitmap b : crops.values()) b.recycle();
                    return;
                }
                releasePendingCrops();
                analysing = false;
                surveyUri = uriString;
                methodBoxes = boxes;
                pendingResult = result;
                pendingAutoPicked = autoPicked;
                pendingSimilarityThreshold = simThreshold;
                pendingMarginThreshold = marginThreshold;
                pendingCrops.putAll(crops);
                showReferenceAnalysis(uriString, result);
                showMethodChoice(result);
                binding.scrollReferenceAnalysis.scrollTo(0, 0);
            });
        }).start();
    }

    /**
     * Sets up the method switcher after an analysis: the method that ran is in use and on the
     * photo, and the arrows step through every other method's boxes from there.
     *
     * <p>The adaptive stage chooses a method by scoring each one against a labelled reference,
     * but this screen is what creates that reference, so there is nothing to score against.
     * The operator's count, when given, picks the starting method. After that the operator is
     * the one party who can see which boxes are hardware, so they flip between every method's
     * boxes on the photo itself and keep the best. Whatever they settle on is recorded in the
     * layout and is what every later inspection has to beat.
     *
     * <p>Measured case: {@code screws_none.jpg} starts on Hybrid v6, which finds five and
     * misses an out-of-focus bolt; the shipped stack and the light strength find all six.
     */
    private void showMethodChoice(HardwareFeatureExtractor.AnalysisResult result) {
        if (binding == null) return;
        final String ran = inUseMethod();
        methodOrder.clear();
        shownMethod = null;
        if (ran == null || methodBoxes == null || methodBoxes.isEmpty()) {
            // A method pinned in the Edge Lab runs for every photo, here and in inspections,
            // so there is nothing to compare or choose on this screen.
            String lab = result.edgeMethodId;
            if (lab == null || EdgeMethodLab.ADAPTIVE_ID.equals(lab)) return;
            binding.tvReferenceStrengthNote.setText("Edge method: " + EdgeMethodLab.nameOf(lab)
                    + ", set in the Edge Lab, so it runs for every photo and the methods are not"
                    + " compared" + (expectedCount > 0
                    ? "; your count of " + expectedCount + " was not used" : "")
                    + ". Set the Edge Lab back to Adaptive to choose here.");
            binding.tvReferenceStrengthNote.setTextColor(
                    ContextCompat.getColor(requireContext(), R.color.orange_warn));
            binding.tvReferenceStrengthNote.setVisibility(View.VISIBLE);
            return;
        }

        int ranCount = countOf(ran);
        inUseExact = pendingAutoPicked && ranCount == expectedCount;
        StringBuilder sb = new StringBuilder();
        if (pendingAutoPicked) {
            sb.append(inUseExact ? "Matches your count of " + expectedCount + "."
                    : "No method finds exactly " + expectedCount + "; this one is closest.");
        } else if (pendingMethod != null) {
            sb.append("You chose this method.");
            if (expectedCount > 0) sb.append(" Your count is ").append(expectedCount).append('.');
        } else {
            sb.append("No count given, so this is a starting point from the photo's noise and"
                    + " contrast; there is no reference to score methods against yet, because"
                    + " this is where you create it.");
        }
        sb.append(" Use the arrows to compare every method's boxes on the photo, and keep the"
                + " one that boxes each fastener once. The method you save with becomes the one"
                + " every inspection of this part is measured against.");
        inUseNote = sb;

        methodOrder.addAll(AdaptivePreprocessing.byCloseness(
                AdaptivePreprocessing.counts(methodBoxes), expectedCount));
        if (!methodOrder.contains(ran)) methodOrder.add(0, ran);
        shownMethod = ran;
        renderMethod();
    }

    /** The method the analysis on screen ran with: the one this reference would be saved with. */
    private String inUseMethod() {
        return pendingResult == null || pendingResult.preprocessing == null ? null
                : pendingResult.preprocessing.methodId;
    }

    /** How many boxes a method finds here; the analysis' own count where the survey has none. */
    private int countOf(String id) {
        java.util.List<android.graphics.RectF> boxes = methodBoxes == null ? null
                : methodBoxes.get(id);
        if (boxes != null) return boxes.size();
        return pendingResult != null && id.equals(inUseMethod()) ? pendingResult.features.size() : 0;
    }

    /** ◀ ▶: puts the previous or next method's boxes on the photo. Instant — nothing is run. */
    private void stepMethod(int step) {
        if (methodOrder.isEmpty() || shownMethod == null) return;
        int n = methodOrder.size();
        int i = Math.max(0, methodOrder.indexOf(shownMethod));
        shownMethod = methodOrder.get(((i + step) % n + n) % n);
        renderMethod();
    }

    /** The method's name: jump straight to any method, nearest the count first. */
    private void chooseMethodFromList() {
        if (methodOrder.isEmpty()) return;
        final String inUse = inUseMethod();
        String[] labels = new String[methodOrder.size()];
        for (int i = 0; i < labels.length; i++) {
            String id = methodOrder.get(i);
            labels[i] = String.format(Locale.getDefault(), "%s — %d%s", EdgeMethodLab.nameOf(id),
                    countOf(id), id.equals(inUse) ? "  (in use)" : "");
        }
        new androidx.appcompat.app.AlertDialog.Builder(requireContext())
                .setTitle(expectedCount > 0
                        ? "Features each method finds, nearest " + expectedCount + " first"
                        : "Features each method finds")
                .setItems(labels, (d, which) -> {
                    shownMethod = methodOrder.get(which);
                    renderMethod();
                })
                .setNegativeButton("Cancel", null)
                .show();
    }

    /**
     * Shows {@link #shownMethod}: its boxes on the photo, its name and count on the switcher,
     * and what the operator can do about it. The method in use shows the classified analysis;
     * any other shows its bare boxes, which is all that can be had without running it.
     */
    private void renderMethod() {
        if (binding == null || shownMethod == null || pendingResult == null) return;
        final String inUse = inUseMethod();
        final boolean isInUse = shownMethod.equals(inUse);
        if (isInUse) {
            binding.overlayReferencePreview.setFeatures(pendingResult.features);
        } else {
            binding.overlayReferencePreview.setPreviewBoxes(methodBoxes.get(shownMethod),
                    ContextCompat.getColor(requireContext(), R.color.blue_primary));
        }

        binding.tvMethodName.setText(EdgeMethodLab.nameOf(shownMethod));
        binding.tvMethodDetail.setText(String.format(Locale.getDefault(),
                "%s · finds %d · %d of %d  ▾", isInUse ? "IN USE" : "PREVIEW",
                countOf(shownMethod), methodOrder.indexOf(shownMethod) + 1, methodOrder.size()));
        binding.tvMethodDetail.setTextColor(ContextCompat.getColor(requireContext(),
                isInUse ? R.color.green_pass : R.color.orange_warn));
        binding.layoutMethodSwitcher.setVisibility(View.VISIBLE);
        binding.btnUseMethod.setVisibility(isInUse ? View.GONE : View.VISIBLE);
        renderAbout();

        if (isInUse) {
            binding.tvReferenceStrengthNote.setText(inUseNote);
            binding.tvReferenceStrengthNote.setTextColor(ContextCompat.getColor(requireContext(),
                    inUseExact ? R.color.green_pass : R.color.orange_warn));
        } else {
            binding.tvReferenceStrengthNote.setText(String.format(Locale.getDefault(),
                    "Boxes only, not yet classified. In use: %s, which finds %d.%s USE THIS"
                            + " METHOD analyses the photo with this one instead.",
                    EdgeMethodLab.nameOf(inUse), countOf(inUse), expectedCount > 0
                            ? " Your count is " + expectedCount + "." : ""));
            binding.tvReferenceStrengthNote.setTextColor(
                    ContextCompat.getColor(requireContext(), R.color.orange_warn));
        }
        binding.tvReferenceStrengthNote.setVisibility(View.VISIBLE);
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
        binding.overlayReferencePreview.setResultData(uriString, result.features,
                result.backgroundOverlay);

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
            Notice.show(getContext(), "No features detected — nothing to classify.");
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
        com.example.ui.util.SystemBars.keepClear(root);
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
        // Under the adaptive method, which method the reference's boxes came out of. It is the
        // incumbent every inspection of this part scores the other methods against.
        final String preprocessingRung = pendingResult.preprocessing == null ? null
                : pendingResult.preprocessing.label();
        final float simThreshold = pendingSimilarityThreshold;
        final float marginThreshold = pendingMarginThreshold;

        new Thread(() -> {
            FeatureEmbedder embedder = FeatureEmbedder.get(context);
            if (embedder == null) {
                requireActivity().runOnUiThread(() -> Notice.show(getContext(),
                        "Feature encoder unavailable (missing assets/models/feature_encoder.onnx)."));
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

            requireActivity().runOnUiThread(() -> {
                if (dialog.isShowing()) dialog.dismiss();
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
            Notice.show(getContext(), "Missing bundled image: " + filename);
            return;
        }
        try {
            selectedImageExternalUri = Uri.parse("file:///android_asset/chassis_images/" + filename);
            // Preview only — analyseReference() re-loads the asset at its own resolution.
            Bitmap bitmap = AssetBitmaps.decodeSampled(
                    requireContext(), "chassis_images/" + filename, 1080);
            if (bitmap != null) binding.ivReferencePreview.setImageBitmap(bitmap);
            // Plain preview first, which the operator counts the hardware on; the overlay
            // replaces it once the count is in and the classification pass has run.
            binding.ivReferencePreview.setVisibility(View.VISIBLE);
            binding.overlayReferencePreview.setVisibility(View.GONE);
            binding.layoutUploadPlaceholder.setVisibility(View.GONE);
            binding.btnSaveReferenceToGallery.setVisibility(View.VISIBLE);
            binding.btnCaptureReference.setText("SAVE REFERENCE");
            binding.btnCaptureReference.setBackgroundColor(ContextCompat.getColor(requireContext(), R.color.green_pass_header));
            awaitExpectedCount();
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

            if (imageFiles.isEmpty()) {
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
                takeReferencePhoto();
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

    /**
     * Makes a photo in this app's own storage the reference being set up: shown on the
     * screen, then the count is asked for. Decoded sampled and upright for the preview — a
     * camera still is twelve megapixels and usually stored on its side with an EXIF rotation.
     */
    private void useAsReference(Uri photo) {
        selectedImageExternalUri = photo;
        Bitmap preview = HardwareFeatureExtractor.loadBitmap(requireContext(), photo.toString(),
                1080);
        if (preview != null) binding.ivReferencePreview.setImageBitmap(preview);
        binding.ivReferencePreview.setVisibility(View.VISIBLE);
        binding.overlayReferencePreview.setVisibility(View.GONE);
        binding.layoutUploadPlaceholder.setVisibility(View.GONE);
        binding.btnSaveReferenceToGallery.setVisibility(View.VISIBLE);
        binding.btnCaptureReference.setText("SAVE REFERENCE");
        binding.btnCaptureReference.setBackgroundColor(ContextCompat.getColor(requireContext(),
                R.color.green_pass_header));
        awaitExpectedCount();
    }

    // ── Taking the reference photo with the camera ──────────────────────────

    /** TAKE A PHOTO: asks for the camera permission the first time, then opens the camera. */
    private void takeReferencePhoto() {
        if (ContextCompat.checkSelfPermission(requireContext(), android.Manifest.permission.CAMERA)
                == android.content.pm.PackageManager.PERMISSION_GRANTED) {
            launchCamera();
        } else {
            cameraPermissionLauncher.launch(android.Manifest.permission.CAMERA);
        }
    }

    /**
     * Hands the phone's camera app a file in this app's references folder to write into. The
     * app declares the camera permission, so Android only lets it ask for a photo once that
     * permission is granted.
     */
    private void launchCamera() {
        File dir = requireContext().getExternalFilesDir("references");
        if (dir == null) {
            Notice.show(getContext(), "Storage is not available for a new photo.");
            return;
        }
        if (!dir.exists()) dir.mkdirs();
        String stamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault())
                .format(new java.util.Date());
        pendingCapture = new File(dir, "REF_" + PartFiles.safeName(partId) + "_" + stamp + ".jpg");
        Uri target = androidx.core.content.FileProvider.getUriForFile(requireContext(),
                requireContext().getPackageName() + ".fileprovider", pendingCapture);
        try {
            takePictureLauncher.launch(target);
        } catch (android.content.ActivityNotFoundException e) {
            pendingCapture = null;
            Notice.show(getContext(),
                    "No camera app on this device. Choose a photo from the" + " gallery instead.");
        }
    }

    /** The camera app is back: review what it took, or tidy up when nothing was taken. */
    private void onPhotoTaken(boolean saved) {
        File photo = pendingCapture;
        pendingCapture = null;
        if (photo == null) return;
        if (!saved || !photo.exists() || photo.length() == 0) {
            photo.delete();
            return;
        }
        showTakenPhoto(photo);
    }

    /**
     * The photo just taken, full screen, with CONFIRM REFERENCE and TAKE ANOTHER PHOTO. A
     * reference defines what correct means for the part, so it is looked at before it is used.
     */
    private void showTakenPhoto(File photo) {
        android.app.Dialog dialog = new android.app.Dialog(requireContext(),
                android.R.style.Theme_DeviceDefault_NoActionBar_Fullscreen);
        dialog.setCancelable(true);
        // Backing out discards the photo, as TAKE ANOTHER PHOTO would.
        dialog.setOnCancelListener(d -> photo.delete());

        android.widget.LinearLayout root = new android.widget.LinearLayout(requireContext());
        root.setOrientation(android.widget.LinearLayout.VERTICAL);
        root.setBackgroundColor(Color.rgb(20, 20, 25));
        root.setPadding(32, 48, 32, 32);

        android.widget.TextView title = new android.widget.TextView(requireContext());
        title.setText("REVIEW THE REFERENCE PHOTO");
        title.setTextColor(Color.rgb(74, 222, 128));
        title.setTextSize(16f);
        title.setTypeface(null, android.graphics.Typeface.BOLD);
        root.addView(title);

        android.widget.TextView hint = new android.widget.TextView(requireContext());
        hint.setText("Part " + partId + ". The whole part should be in frame, sharp and evenly"
                + " lit: every inspection of this part is compared against this photo.");
        hint.setTextColor(Color.LTGRAY);
        hint.setTextSize(12f);
        hint.setPadding(0, 8, 0, 16);
        root.addView(hint);

        android.widget.ImageView image = new android.widget.ImageView(requireContext());
        image.setLayoutParams(new android.widget.LinearLayout.LayoutParams(
                android.widget.LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f));
        image.setScaleType(android.widget.ImageView.ScaleType.FIT_CENTER);
        image.setBackgroundColor(Color.BLACK);
        Bitmap shown = HardwareFeatureExtractor.loadBitmap(requireContext(),
                Uri.fromFile(photo).toString(), 1600);
        if (shown != null) image.setImageBitmap(shown);
        root.addView(image);

        android.widget.LinearLayout buttons = new android.widget.LinearLayout(requireContext());
        buttons.setOrientation(android.widget.LinearLayout.HORIZONTAL);
        buttons.setPadding(0, 24, 0, 0);

        android.widget.Button retake = new android.widget.Button(requireContext());
        android.widget.LinearLayout.LayoutParams retakeParams =
                new android.widget.LinearLayout.LayoutParams(0,
                        android.widget.LinearLayout.LayoutParams.WRAP_CONTENT, 1f);
        retakeParams.rightMargin = 12;
        retake.setLayoutParams(retakeParams);
        retake.setText("TAKE ANOTHER PHOTO");
        retake.setTextColor(Color.WHITE);
        retake.setBackgroundColor(Color.rgb(40, 50, 70));
        retake.setOnClickListener(v -> {
            photo.delete();
            dialog.dismiss();
            takeReferencePhoto();
        });
        buttons.addView(retake);

        android.widget.Button confirm = new android.widget.Button(requireContext());
        android.widget.LinearLayout.LayoutParams confirmParams =
                new android.widget.LinearLayout.LayoutParams(0,
                        android.widget.LinearLayout.LayoutParams.WRAP_CONTENT, 1f);
        confirmParams.leftMargin = 12;
        confirm.setLayoutParams(confirmParams);
        confirm.setText("CONFIRM REFERENCE");
        confirm.setTextColor(Color.WHITE);
        confirm.setBackgroundColor(ContextCompat.getColor(requireContext(),
                R.color.green_pass_header));
        confirm.setOnClickListener(v -> {
            dialog.dismiss();
            useAsReference(Uri.fromFile(photo));
        });
        buttons.addView(confirm);
        root.addView(buttons);

        com.example.ui.util.SystemBars.keepClear(root);
        dialog.setContentView(root);
        dialog.show();
    }

    private void persistSelectedImage(Uri inputUri) {
        try {
            InputStream inputStream = requireContext().getContentResolver().openInputStream(inputUri);
            if (inputStream == null) {
                Notice.show(getContext(), "Cannot read selected image");
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

            useAsReference(Uri.fromFile(photoFile));
        } catch (Exception e) {
            Log.e(TAG, "Failed to copy reference image", e);
            Notice.show(getContext(), "Error saving image: " + e.getMessage());
        }
    }

    private void saveReferenceToDb(String imageUri) {
        binding.btnCaptureReference.setEnabled(false);
        new Thread(() -> {
            AppDatabase db = AppDatabase.getInstance(requireContext());
            ReferenceProfile profile = new ReferenceProfile(partId, imageUri, System.currentTimeMillis());
            db.referenceProfileDao().insertProfile(profile);

            requireActivity().runOnUiThread(() -> {
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
