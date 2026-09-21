package com.example.ui.fragment;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * The reference screen's declaration, without the screen.
 *
 * <p>What the operator types in the count rows becomes the expected side of every inspection
 * of the part, so the parsing has one job: never turn a blank, a slip of the thumb or a stray
 * character into a stored expectation. And the confirm button's three states are the only
 * thing telling the operator why they cannot confirm yet.
 */
public class ReferenceSetupCountsTest {

    private static Map<String, String> raw(String... kv) {
        Map<String, String> m = new LinkedHashMap<>();
        for (int i = 0; i < kv.length; i += 2) m.put(kv[i], kv[i + 1]);
        return m;
    }

    @Test
    public void countsAreKeptInRowOrderWithBlanksAndZerosDropped() {
        Map<String, Integer> counts = ReferenceSetupFragment.countsFrom(
                raw("hole", "1", "stud", "2", "rivet_head", "", "screw", "1", "bolt_head", "0"));
        assertEquals(Arrays.asList("hole", "stud", "screw"),
                new java.util.ArrayList<>(counts.keySet()));
        assertEquals(Integer.valueOf(2), counts.get("stud"));
        assertFalse("a zero is not a declaration", counts.containsKey("bolt_head"));
        assertEquals(4, ReferenceSetupFragment.total(counts));
    }

    @Test
    public void whitespaceIsTrimmedAndGarbageIsIgnored() {
        Map<String, Integer> counts = ReferenceSetupFragment.countsFrom(
                raw("hole", " 3 ", "stud", "two", "screw", "-1", "bolt_head", "1.5"));
        assertEquals(Integer.valueOf(3), counts.get("hole"));
        assertEquals("only the parsable positive number survives", 1, counts.size());
    }

    @Test
    public void nothingTypedMeansNothingDeclared() {
        assertTrue(ReferenceSetupFragment.countsFrom(raw("hole", "", "stud", "  ")).isEmpty());
        assertEquals(0, ReferenceSetupFragment.total(new LinkedHashMap<>()));
    }

    @Test
    public void theButtonSaysWhatIsStillMissing() {
        assertEquals("CHOOSE IMAGE", ReferenceSetupFragment.confirmLabel(false, 0));
        // A count without a photo still asks for the photo first: it is the step that opens
        // the chooser, and the counts are already on screen.
        assertEquals("CHOOSE IMAGE", ReferenceSetupFragment.confirmLabel(false, 4));
        assertEquals("ENTER FEATURE COUNTS TO CONFIRM",
                ReferenceSetupFragment.confirmLabel(true, 0));
        assertEquals("CONFIRM REFERENCE (4 FEATURES)",
                ReferenceSetupFragment.confirmLabel(true, 4));
        assertEquals("CONFIRM REFERENCE (1 FEATURE)",
                ReferenceSetupFragment.confirmLabel(true, 1));
        // Positions are still being detected: the counts are in, but confirming now would
        // save a reference without them.
        assertEquals("DETECTING POSITIONS…",
                ReferenceSetupFragment.confirmLabel(true, 4, true));
        // Without counts the wait is not the operator's problem yet.
        assertEquals("ENTER FEATURE COUNTS TO CONFIRM",
                ReferenceSetupFragment.confirmLabel(true, 0, true));
    }

    @Test
    public void thePositionsLineSaysWhatLocalisationWillBeAbleToTrack() {
        String full = ReferenceSetupFragment.positionsText(4, 4, false);
        assertTrue(full, full.contains("Detected 4 positions"));
        assertTrue(full, full.contains("Position and size will be checked"));

        String fewer = ReferenceSetupFragment.positionsText(3, 4, false);
        assertTrue(fewer, fewer.contains("3 positions"));
        assertTrue("the shortfall is stated against the declaration", fewer.contains("4 features declared"));
        assertTrue("the counts are not overridden by the detector", fewer.contains("counts still stand"));

        String more = ReferenceSetupFragment.positionsText(6, 4, false);
        assertTrue(more, more.contains("6 positions"));
        assertTrue("PASS/FAIL stays with the declaration", more.contains("declared counts still decide"));

        String none = ReferenceSetupFragment.positionsText(0, 4, false);
        assertTrue(none, none.contains("No feature positions"));
        assertTrue("says what is lost, not that the reference is unusable",
                none.contains("Counts will still be checked"));

        String failed = ReferenceSetupFragment.positionsText(0, 4, true);
        assertTrue(failed, failed.contains("Could not detect"));
    }
}
