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

    /**
     * Box sizes and the frame they were measured in survive a save. They are what the size
     * check compares against, so losing either silently turns it off for the part.
     */
    @Test
    public void roundTripsSlotSizesAndTheFrameTheyWereMeasuredIn() {
        FeatureLayout layout = new FeatureLayout();
        layout.setReferenceUri(REF);
        layout.setImageSize(1280, 960);
        layout.addSlot("screw", 0.2f, 0.2f, 0.05f, 0.04f);
        layout.addSlot("stud", 0.8f, 0.8f);
        layout.save(ctx(), PART);

        FeatureLayout loaded = FeatureLayout.loadFor(ctx(), PART, REF);
        assertNotNull(loaded);
        assertEquals(1280, loaded.imageWidth());
        assertEquals(960, loaded.imageHeight());

        FeatureLayout.Slot sized = loaded.slots().get(0);
        assertEquals(0.05f, sized.w, 1e-6f);
        assertEquals(0.04f, sized.h, 1e-6f);
        assertEquals("shorter side in pixels: 0.04 x 960", 38.4f, loaded.shortSidePx(sized), 0.01f);

        FeatureLayout.Slot unsized = loaded.slots().get(1);
        assertEquals(0f, unsized.w, 0f);
        assertEquals(0f, unsized.h, 0f);
        assertEquals(Float.NaN, loaded.shortSidePx(unsized), 0f);
    }

    /** A layout written before sizes existed reads back as positioned but unsized. */
    @Test
    public void aLayoutSavedWithoutSizesLoadsAsUnsized() {
        labelled(REF).save(ctx(), PART);

        FeatureLayout loaded = FeatureLayout.loadFor(ctx(), PART, REF);
        assertNotNull(loaded);
        assertEquals(0, loaded.imageWidth());
        for (FeatureLayout.Slot s : loaded.slots()) {
            assertEquals(0f, s.w, 0f);
            assertEquals(Float.NaN, loaded.shortSidePx(s), 0f);
        }
    }

    /** The edge method a reference was detected with survives a save; its absence reads as null. */
    @Test
    public void roundTripsTheEdgeMethodItWasLabelledWith() {
        FeatureLayout layout = labelled(REF);
        layout.setEdgeMethodId("hybrid-v4");
        layout.save(ctx(), PART);
        FeatureLayout loaded = FeatureLayout.loadFor(ctx(), PART, REF);
        assertNotNull(loaded);
        assertEquals("hybrid-v4", loaded.edgeMethodId());

        labelled(REF).save(ctx(), PART);
        loaded = FeatureLayout.loadFor(ctx(), PART, REF);
        assertNotNull(loaded);
        assertNull("a layout that predates the choice must not claim one", loaded.edgeMethodId());
    }

    /**
     * The sizes inspections have measured at each slot survive a save — they are what the
     * adaptive edge stage learns from, and a history that reset on every launch would leave
     * the expected size pinned to the reference box forever.
     */
    @Test
    public void roundTripsTheLearnedSizeHistoryAndTheRung() {
        FeatureLayout layout = new FeatureLayout();
        layout.setReferenceUri(REF);
        layout.setImageSize(1280, 960);
        layout.setEdgeMethodId("adaptive");
        layout.setPreprocessingRung("strong");
        layout.addSlot("screw", 0.2f, 0.2f, 0.05f, 0.04f);
        layout.addSlot("stud", 0.8f, 0.8f, 0.05f, 0.05f);
        layout.recordSizeRatio(1, 0.55f);
        layout.recordSizeRatio(1, 0.60f);
        layout.save(ctx(), PART);

        FeatureLayout loaded = FeatureLayout.loadFor(ctx(), PART, REF);
        assertNotNull(loaded);
        assertEquals("strong", loaded.preprocessingRung());
        assertEquals(2, loaded.slotById(1).sizeEvidence());
        assertEquals(0.55f, loaded.slotById(1).sizeRatios.get(0), 1e-6f);
        assertEquals(0.60f, loaded.slotById(1).expectedSizeFactor(), 1e-6f);
        assertEquals(0, loaded.slotById(2).sizeEvidence());
        assertEquals(1f, loaded.slotById(2).expectedSizeFactor(), 0f);

        labelled(REF).save(ctx(), PART);
        loaded = FeatureLayout.loadFor(ctx(), PART, REF);
        assertNotNull(loaded);
        assertNull("a layout that predates the adaptive stage names no rung",
                loaded.preprocessingRung());
    }

    /**
     * Filtering to critical slots must keep the persisted ids: a finding at "slot 3" has to
     * mean the same slot in the file the measured size is written back to. Renumbering would
     * record a fastener's size against its neighbour.
     */
    @Test
    public void retainingKeepsIdsSizesAndHistory() {
        FeatureLayout layout = new FeatureLayout();
        layout.setReferenceUri(REF);
        layout.setImageSize(1280, 960);
        layout.setEdgeMethodId("adaptive");
        layout.setPreprocessingRung("medium");
        layout.addSlot("screw", 0.2f, 0.2f, 0.05f, 0.04f);     // id 1
        layout.addSlot("scratches", 0.5f, 0.5f, 0.02f, 0.02f); // id 2
        layout.addSlot("stud", 0.8f, 0.8f, 0.05f, 0.05f);      // id 3
        layout.recordSizeRatio(3, 0.7f);

        FeatureLayout kept = layout.retaining(s -> !"scratches".equals(s.type));
        assertEquals(2, kept.slots().size());
        assertNull(kept.slotById(2));
        FeatureLayout.Slot stud = kept.slotById(3);
        assertNotNull("the third slot keeps its id, not a renumbered 2", stud);
        assertEquals("stud", stud.type);
        assertEquals(0.05f, stud.w, 1e-6f);
        assertEquals(1, stud.sizeEvidence());
        assertEquals(REF, kept.referenceUri());
        assertEquals(1280, kept.imageWidth());
        assertEquals("adaptive", kept.edgeMethodId());
        assertEquals("medium", kept.preprocessingRung());

        // A copy, not a view: recording into it leaves the original untouched.
        kept.recordSizeRatio(3, 0.7f);
        assertEquals(1, layout.slotById(3).sizeEvidence());
    }
}
