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

    @Test
    public void tunedVariantsFollowThePipelineTheyAreProposedAgainst() {
        List<EdgeMethodLab.Method> methods = EdgeMethodLab.catalogue();
        int shippedAt = -1;
        for (int i = 0; i < methods.size(); i++) {
            if (methods.get(i).shipped) shippedAt = i;
        }
        assertTrue("the production pipeline must be in the catalogue", shippedAt >= 0);

        // A v2 shown above the stack it improves on reads as just another operator. The
        // ordering is the argument: bare operators, then what ships, then what could replace it.
        List<String> after = new ArrayList<>();
        for (int i = shippedAt + 1; i < methods.size(); i++) after.add(methods.get(i).name);
        assertFalse("the tuned variants are shown after the shipped pipeline", after.isEmpty());
        for (String name : after) {
            assertTrue("only tuned variants may follow the shipped pipeline, found: " + name,
                    name.startsWith("Hybrid v"));
        }
    }

    @Test
    public void everyTunedVariantIsNumberedAndDistinct() {
        Set<String> variants = new HashSet<>();
        for (EdgeMethodLab.Method method : EdgeMethodLab.catalogue()) {
            if (method.name.startsWith("Hybrid v")) variants.add(method.name);
        }
        // Five are kept, each with a study run behind it: v2/v3/v4 from the threshold and
        // preprocessing search, v5 from the adaptive-threshold experiment, v6 from the search
        // against full ground truth. Adding a sixth without that evidence is the failure this
        // guards against.
        assertEquals("v2 through v6 are the tuned variants", 5, variants.size());
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

    @Test
    public void theShippedStackRunsUntilSomethingIsChosen() {
        assertEquals(EdgeMethodLab.SHIPPED_ID, EdgeMethodLab.selectedId(ctx()));
        assertTrue(EdgeMethodLab.shippedSelected(ctx()));
        assertTrue(EdgeMethodLab.selected(ctx()).shipped);
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
    }

    /**
     * An id this build does not know — from a newer build, or a typo in the preferences file —
     * must fall back to the stack the detector was tuned with, never run something arbitrary
     * and never crash an inspection.
     */
    @Test
    public void anUnknownIdFallsBackToTheShippedStack() {
        EdgeMethodLab.select(ctx(), "hybrid-v4");
        EdgeMethodLab.select(ctx(), "no-such-method");
        assertEquals("an unknown id must be ignored, not stored",
                "hybrid-v4", EdgeMethodLab.selectedId(ctx()));

        ctx().getSharedPreferences(EdgeMethodLab.PREFS, Context.MODE_PRIVATE).edit()
                .putString(EdgeMethodLab.KEY_SELECTED, "from-a-future-build").apply();
        assertEquals(EdgeMethodLab.SHIPPED_ID, EdgeMethodLab.selectedId(ctx()));
        assertTrue(EdgeMethodLab.selected(ctx()).shipped);
        assertNull(EdgeMethodLab.byId("from-a-future-build"));
        assertEquals("from-a-future-build", EdgeMethodLab.nameOf("from-a-future-build"));
    }
}
