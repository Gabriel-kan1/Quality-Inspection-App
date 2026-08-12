package com.example.ml;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;

import org.json.JSONObject;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;
import org.robolectric.annotation.Config;

import java.io.File;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;

/**
 * Pins how much weight a saved library carries when new examples are added to it.
 *
 * A prototype is the running mean of its examples, and a save/load round trip has to
 * preserve how many examples that mean was taken over. It previously did not: every
 * library came back as a single example, so the next correction was averaged in
 * fifty-fifty against the whole accumulated history. That is the difference between a
 * correction nudging a prototype and a correction replacing it.
 *
 * Robolectric is needed only because {@link ReferenceLibrary} touches android.graphics.Color
 * in a static initialiser for its display palette.
 */
@RunWith(RobolectricTestRunner.class)
@Config(sdk = {36})
public class ReferenceLibraryPersistenceTest {

    private static final String TYPE = "stud";

    /** Nine examples all pointing one way, so the prototype's direction is unambiguous. */
    private static final int HISTORY_SIZE = 9;
    private static final float[] HISTORY_DIRECTION = {1f, 0f, 0f};
    /** One later correction pointing somewhere else entirely. */
    private static final float[] CORRECTION = {0f, 1f, 0f};

    private static final float EPSILON = 1e-4f;

    private ReferenceLibrary libraryWithHistory() {
        ReferenceLibrary lib = new ReferenceLibrary();
        for (int i = 0; i < HISTORY_SIZE; i++) {
            lib.addExample(TYPE, HISTORY_DIRECTION.clone());
        }
        return lib;
    }

    private File saveToTempFile(ReferenceLibrary lib) throws Exception {
        File file = File.createTempFile("reference_library", ".json");
        file.deleteOnExit();
        lib.save(file);
        return file;
    }

    @Test
    public void aSavedLibraryRoundTripsToTheSamePrototype() throws Exception {
        ReferenceLibrary reloaded = ReferenceLibrary.load(saveToTempFile(libraryWithHistory()));

        assertNotNull(reloaded);
        float[] proto = reloaded.prototype(TYPE);
        assertNotNull(proto);
        assertEquals(1f, proto[0], EPSILON);
        assertEquals(0f, proto[1], EPSILON);
    }

    @Test
    public void aCorrectionIsWeightedAgainstTheWholeSavedHistory() throws Exception {
        ReferenceLibrary reloaded = ReferenceLibrary.load(saveToTempFile(libraryWithHistory()));
        assertNotNull(reloaded);

        reloaded.addExample(TYPE, CORRECTION.clone());

        // Ten examples now, nine of which point along x: the correction moves the prototype
        // by a tenth, not by a half. Before the example count was persisted this came back
        // as (0.5, 0.5) — one crop outweighing nine.
        float[] proto = reloaded.prototype(TYPE);
        assertNotNull(proto);
        assertEquals("history should still dominate", 0.9f, proto[0], EPSILON);
        assertEquals("one correction is worth one tenth here", 0.1f, proto[1], EPSILON);
    }

    @Test
    public void repeatedSaveAndLoadDoesNotDiluteTheHistory() throws Exception {
        // The drift this guards against is cumulative: every inspection that ends in a
        // correction saves and the next one loads, so a count that resets on each round trip
        // halves the accumulated history once per inspection rather than once ever.
        ReferenceLibrary lib = libraryWithHistory();
        for (int round = 0; round < 3; round++) {
            lib = ReferenceLibrary.load(saveToTempFile(lib));
            assertNotNull(lib);
        }

        lib.addExample(TYPE, CORRECTION.clone());

        float[] proto = lib.prototype(TYPE);
        assertNotNull(proto);
        assertEquals(0.9f, proto[0], EPSILON);
        assertEquals(0.1f, proto[1], EPSILON);
    }

    @Test
    public void aLibraryWrittenBeforeCountsExistedStillLoads() throws Exception {
        // Back-compat: files on existing installs have no "_counts" key. They load as one
        // example each, which is exactly the weighting they were written under — the point
        // is that they load at all rather than being discarded.
        JSONObject root = new JSONObject();
        root.put("_encoder", FeatureEmbedder.ENCODER_ID);
        root.put(TYPE, new org.json.JSONArray(new double[]{1.0, 0.0, 0.0}));

        File file = File.createTempFile("legacy_library", ".json");
        file.deleteOnExit();
        try (FileOutputStream fos = new FileOutputStream(file)) {
            fos.write(root.toString().getBytes(StandardCharsets.UTF_8));
        }

        ReferenceLibrary loaded = ReferenceLibrary.load(file);
        assertNotNull(loaded);

        loaded.addExample(TYPE, CORRECTION.clone());
        float[] proto = loaded.prototype(TYPE);
        assertNotNull(proto);
        assertEquals(0.5f, proto[0], EPSILON);
        assertEquals(0.5f, proto[1], EPSILON);
    }

    @Test
    public void aLibraryFromAnotherEncoderIsStillRejected() throws Exception {
        // The encoder guard has to survive the counts change: prototypes from a different
        // embedding space compare as confident nonsense rather than failing outright.
        JSONObject root = new JSONObject();
        root.put("_encoder", "some-other-encoder-512");
        root.put("_counts", new JSONObject().put(TYPE, 9));
        root.put(TYPE, new org.json.JSONArray(new double[]{1.0, 0.0, 0.0}));

        File file = File.createTempFile("foreign_library", ".json");
        file.deleteOnExit();
        try (FileOutputStream fos = new FileOutputStream(file)) {
            fos.write(root.toString().getBytes(StandardCharsets.UTF_8));
        }

        assertNull(ReferenceLibrary.load(file));
    }
}
