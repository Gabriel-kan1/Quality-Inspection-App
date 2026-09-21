package com.example.ml;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import android.graphics.RectF;

import com.example.ui.fragment.ReferenceSetupFragment;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * How a reference's detections become the slots localisation tracks, now that nobody labels
 * them by hand.
 *
 * <p>The judgement is which type a box gets. A fresh part's library rejects most crops as
 * "unknown", and a layout built from accepted types alone would be nearly empty; the nearest
 * prototype is the right call for a photo of a part known to be correct. Lives in this package
 * because {@link ReferenceLibrary.Classification} is constructed here and nowhere else.
 */
@RunWith(RobolectricTestRunner.class)
public class ReferenceSlotTypesTest {

    private static final RectF BOX = new RectF(0.1f, 0.1f, 0.2f, 0.2f);

    private static HardwareFeatureExtractor.ExtractedFeature feature(String type, String closest) {
        ReferenceLibrary.Classification c = closest == null ? null
                : new ReferenceLibrary.Classification(type, closest, 0.5f, "low", null, -1f,
                        ReferenceLibrary.Reason.BELOW_THRESHOLD, Collections.emptyMap());
        return new HardwareFeatureExtractor.ExtractedFeature(type, BOX, 0, type, 0.5f, null, c);
    }

    @Test
    public void theNearestPrototypeNamesTheSlotEvenWhenRejected() {
        // Rejected as unknown, but the nearest prototype was a stud: that is the slot.
        List<String> types = ReferenceSetupFragment.slotTypes(
                Arrays.asList(feature(ReferenceLibrary.UNKNOWN, "stud")), t -> true);
        assertEquals(Arrays.asList("stud"), types);
    }

    @Test
    public void aBoxWithNoPrototypeAtAllMakesNoSlot() {
        // "none" is the sentinel for an empty library; an unknown with nothing nearer is
        // a box the layout cannot name, and an unnamed slot would match anything.
        List<String> types = ReferenceSetupFragment.slotTypes(Arrays.asList(
                feature(ReferenceLibrary.UNKNOWN, "none"),
                feature(ReferenceLibrary.UNKNOWN, null)), t -> true);
        assertTrue(types.isEmpty());
    }

    @Test
    public void onlyCriticalTypesBecomeSlots() {
        // An advisory scratch has no place it belongs, so it is not a landmark.
        List<String> types = ReferenceSetupFragment.slotTypes(Arrays.asList(
                feature("screw", "screw"), feature("scratch", "scratch"), feature("hole", "hole")),
                t -> !"scratch".equals(t));
        assertEquals(Arrays.asList("screw", "hole"), types);
    }

    @Test
    public void anAcceptedTypeWithoutAClassificationStillCounts() {
        List<String> types = ReferenceSetupFragment.slotTypes(
                Arrays.asList(feature("bolt_head", null)), t -> true);
        assertEquals(Arrays.asList("bolt_head"), types);
    }
}
