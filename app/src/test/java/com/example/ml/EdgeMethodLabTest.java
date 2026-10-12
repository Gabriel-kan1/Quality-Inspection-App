package com.example.ml;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import android.content.Context;

import androidx.test.core.app.ApplicationProvider;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Covers the part of the edge-method comparison that can be checked without a device.
 *
 * <p>The operators themselves cannot be: every one of them is a native OpenCV call, and the
 * .so files are built for arm64 and x86_64 rather than for the JVM this runs on. So what is
 * pinned here is the catalogue and the reporting — the things that decide whether the screen
 * is a fair comparison rather than whether OpenCV can convolve.
 *
 * <p>That matters more than it sounds. A duplicated name, a method quietly dropped from the
 * list, or a second entry claiming to be the shipped pipeline would all produce a screen that
 * looks entirely normal and answers the question wrongly.
 *
 * <p>Runs under Robolectric only because {@code MethodResult} names {@code android.graphics
 * .Bitmap} in its signature.
 */
@RunWith(RobolectricTestRunner.class)
public class EdgeMethodLabTest {

    private static EdgeMethodLab.MethodResult result(int candidates, int contours,
                                                     double density, long ms) {
        return new EdgeMethodLab.MethodResult("method", "Method", "Summary", null, density,
                contours, candidates, ms, false, false, null);
    }

    @Test
    public void everyMethodIsNamedDescribedAndRunnable() {
        List<EdgeMethodLab.Method> methods = EdgeMethodLab.catalogue();
        assertFalse("the catalogue must not be empty", methods.isEmpty());

        for (EdgeMethodLab.Method method : methods) {
            assertFalse("a method needs a name", method.name.trim().isEmpty());
            assertFalse(method.name + " needs a summary", method.summary.trim().isEmpty());
            // The summary is the only thing on the card explaining what was run, so a
            // placeholder one word long would leave the map uninterpretable.
            assertTrue(method.name + " needs a summary worth reading",
                    method.summary.length() > 40);
            assertTrue(method.name + " needs an operator", method.operator != null);
        }
    }

    @Test
    public void namesAreUnique() {
        Set<String> seen = new HashSet<>();
        for (EdgeMethodLab.Method method : EdgeMethodLab.catalogue()) {
            assertTrue("duplicate method name: " + method.name, seen.add(method.name));
        }
    }

    @Test
    public void exactlyOneEntryIsTheShippedPipeline() {
        int shipped = 0;
        for (EdgeMethodLab.Method method : EdgeMethodLab.catalogue()) {
            if (method.shipped) shipped++;
        }
        // Two badges would say the app runs two pipelines; none would leave every alternative
        // being compared against nothing in particular. The tuned variants are candidates to
        // replace it, so none of them may claim the badge until one is actually adopted.
        assertEquals("exactly one row is the production pipeline", 1, shipped);
    }

    /**
     * The methods read Hybrid v1 to v18 in that order, v1 being the stack the detector shipped
     * with, and the adaptive stage that chooses among them closes the list. The ids are older
     * than the names and must not follow them: layouts and preferences store the ids.
     */
    @Test
    public void theCatalogueRunsFromHybridV1ToV18ThenTheAdaptiveStage() {
        List<EdgeMethodLab.Method> methods = EdgeMethodLab.catalogue();
        assertEquals(19, methods.size());
        assertEquals(EdgeMethodLab.ADAPTIVE_ID, methods.get(18).id);
        for (int i = 0; i < 18; i++) {
            assertEquals("Hybrid v" + (i + 1), methods.get(i).name);
        }
        assertTrue("v1 is the shipped stack", methods.get(0).shipped);
        assertEquals(EdgeMethodLab.SHIPPED_ID, methods.get(0).id);
        assertEquals("hybrid-v2", methods.get(1).id);
        assertEquals(PreprocessingStrength.MEDIUM.id, methods.get(5).id);
        assertEquals(PreprocessingStrength.LIGHT.id, methods.get(6).id);
        assertEquals(PreprocessingStrength.STRONG.id, methods.get(7).id);
        assertEquals("clahe-bilateral-canny", methods.get(8).id);
        assertEquals("canny-otsu", methods.get(9).id);
        assertEquals("unmasked-strict-canny", methods.get(10).id);
    }

    /** Every screen that offers a method says what it is good at and what it runs. */
    @Test
    public void everyMethodSaysWhatItIsGoodAtAndWhatItUses() {
        for (EdgeMethodLab.Method method : EdgeMethodLab.catalogue()) {
            assertFalse(method.name, EdgeMethodLab.goodAtOf(method.id).isEmpty());
            assertFalse(method.name, EdgeMethodLab.usesOf(method.id).isEmpty());
        }
        assertEquals("", EdgeMethodLab.goodAtOf("no-such-method"));
        assertEquals("", EdgeMethodLab.usesOf("no-such-method"));
    }

    // ── The adaptive stage ───────────────────────────────────────────────────

    /**
     * Every rung is a catalogue entry under the id the enum names, so a layout that records
     * a rung, and an operator who pins one in the lab, both refer to something that exists.
     * The middle rung is v6 itself: two cards with identical output would read as two methods.
     */
    @Test
    public void everyRungIsCataloguedAndTheMiddleOneIsV6() {
        for (PreprocessingStrength rung : PreprocessingStrength.values()) {
            EdgeMethodLab.Method m = EdgeMethodLab.byId(rung.id);
            assertNotNull("no catalogue entry for rung " + rung, m);
            assertEquals(rung, PreprocessingStrength.byId(m.id));
            assertEquals(rung, PreprocessingStrength.byLabel(rung.label));
        }
        assertEquals("hybrid-v6", PreprocessingStrength.MEDIUM.id);
        assertTrue(EdgeMethodLab.nameOf(PreprocessingStrength.MEDIUM.id).startsWith("Hybrid v6"));
        assertNotNull(EdgeMethodLab.byId(EdgeMethodLab.ADAPTIVE_ID));
        assertNull(PreprocessingStrength.byId("no-such-rung"));
        assertNull(PreprocessingStrength.byId(null));
    }

    @Test
    public void aNoteTravelsWithTheResultAndIsOptional() {
        EdgeMethodLab.MethodResult plain = result(3, 40, 1.0, 12);
        assertNull(plain.note);
        EdgeMethodLab.MethodResult noted = new EdgeMethodLab.MethodResult("adaptive",
                "Adaptive", "Summary", null, 1.0, 40, 3, 12, false, false, null,
                "MEDIUM preprocessing — noise 3.3, edge contrast 121");
        assertTrue(noted.note.startsWith("MEDIUM"));
        assertFalse(noted.failed());
    }

    @Test
    public void cropsLineSaysHowManyTheInspectionWouldTakeAndHowManyAreShown() {
        List<android.graphics.RectF> boxes = new java.util.ArrayList<>();
        List<android.graphics.Bitmap> crops = new java.util.ArrayList<>();
        for (int i = 0; i < 40; i++) {
            boxes.add(new android.graphics.RectF(0, 0, 0.1f, 0.1f));
            if (i < EdgeMethodLab.MAX_CROPS_SHOWN) crops.add(null);
        }
        EdgeMethodLab.MethodResult many = new EdgeMethodLab.MethodResult("m", "M", "S", null,
                1.0, 50, 40, 10, false, false, null, null, boxes, crops, false);
        String line = many.cropsLine();
        assertTrue(line, line.contains("40 features"));
        assertTrue(line, line.contains("first " + EdgeMethodLab.MAX_CROPS_SHOWN + " shown"));

        EdgeMethodLab.MethodResult none = result(0, 5, 0.1, 3);
        assertTrue(none.cropsLine().contains("nothing"));
        assertFalse(none.picked);
        assertTrue(Double.isNaN(none.referenceScore));
    }

    @Test
    public void verdictWithAReferenceRanksTheMethodsAndNamesTheWinner() {
        java.util.Map<String, Double> scores = new java.util.LinkedHashMap<>();
        scores.put(EdgeMethodLab.SHIPPED_ID, 0.02);
        scores.put(PreprocessingStrength.MEDIUM.id, 0.31);
        scores.put("canny-otsu", Double.NaN);
        String all = String.join("\n", EdgeMethodLab.verdict(new ImageQuality(2.7, 62),
                EdgeMethodLab.ADAPTIVE_ID, "PART-1", null, "6 slots sized",
                scores, PreprocessingStrength.MEDIUM.id, EdgeMethodLab.SHIPPED_ID));
        assertTrue(all, all.contains("PART-1"));
        assertTrue("best first: " + all, all.indexOf("0.020") < all.indexOf("0.310"));
        assertTrue(all, all.contains("(reference's method)"));
        assertTrue(all, all.contains(EdgeMethodLab.nameOf(EdgeMethodLab.SHIPPED_ID)
                + " reproduces the reference better"));
        assertFalse("image statistics are not the reason when a reference exists: " + all,
                all.contains("starting point"));
    }

    @Test
    public void verdictWithoutAReferenceSaysItIsOnlyAStartingPoint() {
        String all = String.join("\n", EdgeMethodLab.verdict(new ImageQuality(3.0, 90),
                EdgeMethodLab.SHIPPED_ID, "P", "it has no labelled reference layout", null,
                new java.util.LinkedHashMap<>(), PreprocessingStrength.MEDIUM.id,
                PreprocessingStrength.MEDIUM.id));
        assertTrue(all, all.contains("starting point"));
        assertTrue(all, all.contains("Cannot score against P"));
        assertTrue("a fixed method is named as not adapting: " + all,
                all.contains("does not adapt"));
    }

    @Test
    public void everyMethodIsNumberedAndDistinct() {
        Set<String> names = new HashSet<>();
        for (EdgeMethodLab.Method method : EdgeMethodLab.catalogue()) {
            if (method.name.startsWith("Hybrid v")) names.add(method.name);
        }
        // Eighteen are kept, each with a reason to be here: the shipped stack, five tuned
        // variants with a study run behind each, v6's light and strong rungs, the two Canny
        // baselines, and the eight that each held a plate through the lighting test photos.
        // Adding a nineteenth without that evidence is the failure this guards against.
        assertEquals("v1 through v18", 18, names.size());
    }

    /**
     * v11 to v18 run on the photo as taken, because the chassis mask painted out hardware in
     * shadow on the lighting test photos; v1 to v10 and the adaptive stage keep it, so a
     * reference labelled with one of them still sees what it was labelled on.
     */
    @Test
    public void onlyHybridV11ToV18SkipTheChassisMask() {
        List<EdgeMethodLab.Method> methods = EdgeMethodLab.catalogue();
        for (int i = 0; i < methods.size(); i++) {
            EdgeMethodLab.Method m = methods.get(i);
            assertEquals(m.name, i >= 10 && i < 18, m.unmasked);
            assertEquals(m.name, m.unmasked, EdgeMethodLab.skipsMask(m.id));
        }
        assertFalse(EdgeMethodLab.skipsMask("no-such-method"));
    }

    @Test
    public void catalogueSizeMatchesWhatARunProduces() {
        // The screen announces the count before starting the work; if these ever disagree it
        // promises a number of cards it does not deliver.
        assertEquals(EdgeMethodLab.catalogue().size(), EdgeMethodLab.catalogueSize());
    }

    @Test
    public void metricsLineLeadsWithTheCountThatMatters() {
        String line = result(14, 318, 3.4149, 86).metricsLine();
        assertTrue("candidates come first, before raw contours: " + line,
                line.indexOf("14 candidates") < line.indexOf("318 contours"));
        assertTrue("edge density is a percentage to 2dp: " + line, line.contains("3.41%"));
        assertTrue("timing is reported: " + line, line.contains("86 ms"));
    }

    @Test
    public void aFailedMethodSaysSoInsteadOfReportingZeros() {
        EdgeMethodLab.MethodResult failed = new EdgeMethodLab.MethodResult("broken",
                "Broken", "Summary", null, 0, 0, 0, 0, false, false, "kernel size must be odd");

        assertTrue(failed.failed());
        // Zeros here would read as "this method found nothing", which is a finding about the
        // photograph. It did not run at all, which is a finding about the method.
        assertFalse("a failure must not be rendered as a count of zero",
                failed.metricsLine().contains("0 candidates"));
        assertTrue("the reason belongs on the card", failed.metricsLine()
                .contains("kernel size must be odd"));
    }

    @Test
    public void aSucceedingMethodIsNotMarkedFailed() {
        assertFalse(result(3, 40, 1.0, 12).failed());
    }

    // ── Choosing the method inspections run ──────────────────────────────────

    private static Context ctx() {
        return ApplicationProvider.getApplicationContext();
    }

    /**
     * Ids are what the choice is persisted by, so they have to be present, distinct, and
     * boring: a renamed id silently reverts every device that chose it to the shipped stack.
     */
    @Test
    public void everyMethodHasAStableUniqueId() {
        Set<String> seen = new HashSet<>();
        for (EdgeMethodLab.Method method : EdgeMethodLab.catalogue()) {
            assertNotNull(method.name + " needs an id", method.id);
            assertTrue(method.name + " has an unsafe id: " + method.id,
                    method.id.matches("[a-z0-9-]+"));
            assertTrue("duplicate method id: " + method.id, seen.add(method.id));
        }
        assertTrue("the shipped id must be in the catalogue", seen.contains(EdgeMethodLab.SHIPPED_ID));
    }

    /**
     * The adaptive stage runs until someone chooses. The shipped stack keeps its badge as
     * the pipeline the detector was tuned with, but it is not what runs: its own per-image
     * adaptation was found firing backwards, and v6 beat it 20/26 to 9/26 against ground truth.
     */
    @Test
    public void theAdaptiveStageRunsUntilSomethingIsChosen() {
        assertEquals(EdgeMethodLab.ADAPTIVE_ID, EdgeMethodLab.DEFAULT_ID);
        assertEquals(EdgeMethodLab.DEFAULT_ID, EdgeMethodLab.selectedId(ctx()));
        assertTrue(EdgeMethodLab.defaultSelected(ctx()));
        assertTrue(EdgeMethodLab.adaptiveSelected(ctx()));
        assertFalse(EdgeMethodLab.shippedSelected(ctx()));
        assertFalse(EdgeMethodLab.selected(ctx()).shipped);
        assertTrue(EdgeMethodLab.selectedName(ctx()).startsWith("Adaptive"));
    }

    @Test
    public void aChoicePersistsAndNamesTheMethod() {
        EdgeMethodLab.select(ctx(), "hybrid-v4");
        assertEquals("hybrid-v4", EdgeMethodLab.selectedId(ctx()));
        assertFalse(EdgeMethodLab.shippedSelected(ctx()));
        assertEquals("hybrid-v4", EdgeMethodLab.selected(ctx()).id);
        assertTrue(EdgeMethodLab.selectedName(ctx()).startsWith("Hybrid v4"));
        assertEquals(EdgeMethodLab.selectedName(ctx()), EdgeMethodLab.nameOf("hybrid-v4"));

        EdgeMethodLab.select(ctx(), EdgeMethodLab.SHIPPED_ID);
        assertTrue(EdgeMethodLab.shippedSelected(ctx()));
        assertFalse(EdgeMethodLab.defaultSelected(ctx()));

        EdgeMethodLab.select(ctx(), EdgeMethodLab.ADAPTIVE_ID);
        assertTrue(EdgeMethodLab.adaptiveSelected(ctx()));
    }

    /**
     * An id this build does not know — from a newer build, or a typo in the preferences file —
     * must fall back to the default, never run something arbitrary and never crash an
     * inspection.
     */
    @Test
    public void anUnknownIdFallsBackToTheDefault() {
        EdgeMethodLab.select(ctx(), "hybrid-v4");
        EdgeMethodLab.select(ctx(), "no-such-method");
        assertEquals("an unknown id must be ignored, not stored",
                "hybrid-v4", EdgeMethodLab.selectedId(ctx()));

        ctx().getSharedPreferences(EdgeMethodLab.PREFS, Context.MODE_PRIVATE).edit()
                .putString(EdgeMethodLab.KEY_SELECTED, "from-a-future-build").apply();
        assertEquals(EdgeMethodLab.DEFAULT_ID, EdgeMethodLab.selectedId(ctx()));
        assertEquals(EdgeMethodLab.DEFAULT_ID, EdgeMethodLab.selected(ctx()).id);
        assertNull(EdgeMethodLab.byId("from-a-future-build"));
        assertEquals("from-a-future-build", EdgeMethodLab.nameOf("from-a-future-build"));

        // A method chosen under an earlier build and since removed stops running too.
        ctx().getSharedPreferences(EdgeMethodLab.PREFS, Context.MODE_PRIVATE).edit()
                .putString(EdgeMethodLab.KEY_SELECTED, "sobel").apply();
        assertEquals(EdgeMethodLab.DEFAULT_ID, EdgeMethodLab.selectedId(ctx()));
    }
}
