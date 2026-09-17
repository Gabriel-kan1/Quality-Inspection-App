package com.example.ui.fragment;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.example.ml.EdgeMethodLab;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;

import java.util.Arrays;
import java.util.List;

/**
 * The sentence shown after a method is picked from the operator's count. It is the only
 * thing telling them whether the app agreed with their number or merely got as close as it
 * could, and the two must not read the same.
 */
@RunWith(RobolectricTestRunner.class)
public class ReferenceSetupPickTextTest {

    private static EdgeMethodLab.CountMatch m(String id, String name, int candidates, int expected) {
        return new EdgeMethodLab.CountMatch(id, name, candidates, Math.abs(candidates - expected));
    }

    @Test
    public void anExactMatchSaysSoAndNamesTheMethod() {
        List<EdgeMethodLab.CountMatch> ranked = EdgeMethodLab.rank(Arrays.asList(
                m("hybrid-v6", "Hybrid v6", 4, 4), m(EdgeMethodLab.SHIPPED_ID, "Shipped", 9, 4)));
        String text = ReferenceSetupFragment.describePick(4, ranked);
        assertTrue(text, text.contains("Chose Hybrid v6"));
        assertTrue(text, text.contains("exactly 4"));
        assertFalse("must not hedge when it hit the number", text.contains("closest"));
        assertTrue("the roll-call names every method's count", text.contains("Shipped 9"));
        assertTrue(text.contains("v6 4"));
    }

    @Test
    public void aNearMissSaysNoMethodHitTheCountAndSuggestsReshooting() {
        // head3.jpg: every variant finds 2 of the 4 heads. The operator typed 4.
        List<EdgeMethodLab.CountMatch> ranked = EdgeMethodLab.rank(Arrays.asList(
                m("hybrid-v6", "Hybrid v6", 2, 4), m(EdgeMethodLab.SHIPPED_ID, "Shipped", 4, 4)));
        // The shipped stack hit 4 here, so it wins outright; make the miss case genuine.
        ranked = EdgeMethodLab.rank(Arrays.asList(
                m("hybrid-v6", "Hybrid v6", 2, 4), m("hybrid-v3", "Hybrid v3", 2, 4)));
        String text = ReferenceSetupFragment.describePick(4, ranked);
        assertTrue(text, text.contains("No method finds exactly 4"));
        assertTrue(text, text.contains("the closest, at 2"));
        assertTrue("points at the likely cause rather than blaming the operator",
                text.contains("re-photograph"));
    }

    @Test
    public void theTextAlwaysSaysWhatInspectionsWillNowRun() {
        List<EdgeMethodLab.CountMatch> ranked = EdgeMethodLab.rank(Arrays.asList(
                m("hybrid-v3", "Hybrid v3 (wide Gaussian, best all-round)", 5, 5)));
        String text = ReferenceSetupFragment.describePick(5, ranked);
        assertTrue(text, text.contains(
                "Inspections of this part now run Hybrid v3 (wide Gaussian, best all-round)."));
    }
}
