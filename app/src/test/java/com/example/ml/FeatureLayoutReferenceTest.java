package com.example.ml;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;

import android.content.Context;

import androidx.test.core.app.ApplicationProvider;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;

/**
 * Covers the layout persisted from reference setup — the operator's labels used as the
 * definition of the part rather than as training data.
 *
 * The point of storing it is that an inspection compares against what a human said, not
 * against whatever the classifier makes of the reference photo on the day. That only holds
 * if a layout is refused once the photo it describes has been replaced.
 */
@RunWith(RobolectricTestRunner.class)
public class FeatureLayoutReferenceTest {

    private static final String PART = "PART-A";
    private static final String REF = "file:///android_asset/chassis_images/screws.jpg";

    private static Context ctx() {
        return ApplicationProvider.getApplicationContext();
    }

    private static FeatureLayout labelled(String uri) {
        FeatureLayout layout = new FeatureLayout();
        layout.setReferenceUri(uri);
        layout.addSlot("screw", 0.2f, 0.2f);
        layout.addSlot("screw", 0.8f, 0.2f);
        layout.addSlot("stud", 0.2f, 0.8f);
        layout.addSlot("scratches", 0.5f, 0.5f);
        return layout;
    }

    @Test
    public void roundTripsTheLabelsAndTheirReference() {
        labelled(REF).save(ctx(), PART);

        FeatureLayout loaded = FeatureLayout.loadFor(ctx(), PART, REF);
        assertNotNull(loaded);
        assertEquals(4, loaded.slots().size());
        assertEquals(REF, loaded.referenceUri());
        assertEquals("screw", loaded.slots().get(0).type);
    }

    /** Expected counts come straight from the labels — no re-classification involved. */
    @Test
    public void countsByTypeAreTheExpectedConfiguration() {
        FeatureLayout layout = labelled(REF);
        java.util.Map<String, Integer> counts = layout.countsByType();
        assertEquals(Integer.valueOf(2), counts.get("screw"));
        assertEquals(Integer.valueOf(1), counts.get("stud"));
        assertEquals(Integer.valueOf(1), counts.get("scratches"));
        assertEquals(3, counts.size());
    }

    /**
     * The guard that makes persisting safe: swapping the reference photo must invalidate the
     * labels, so the caller falls back to classifying the new image instead of comparing an
     * inspection against an arrangement that no longer exists.
     */
    @Test
    public void refusesALayoutLabelledAgainstADifferentPhoto() {
        labelled(REF).save(ctx(), PART);

        assertNull(FeatureLayout.loadFor(ctx(), PART,
                "file:///android_asset/chassis_images/screws_flash.jpg"));
        // ...but the file itself is untouched, so the original reference still resolves.
        assertNotNull(FeatureLayout.loadFor(ctx(), PART, REF));
    }

    /** A layout written before reference URIs were recorded cannot be trusted to any photo. */
    @Test
    public void refusesAnUntaggedLegacyLayout() {
        labelled(null).save(ctx(), PART);
        assertNull(FeatureLayout.loadFor(ctx(), PART, REF));
    }

    @Test
    public void missingLayoutIsNotAnError() {
        assertNull(FeatureLayout.loadFor(ctx(), "PART-NEVER-SET-UP", REF));
    }
}
