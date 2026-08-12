package com.example.ui.fragment;

import static org.junit.Assert.assertEquals;

import com.example.ml.ReferenceLibrary;

import org.junit.Test;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Pins the rule that decides PASS / FAIL / UNVERIFIED.
 *
 * <p>This is the single most consequential decision the app makes and it had no test. The
 * case that motivated these: a part with no reference image used to report "INSPECTION
 * PASSED — All expected features verified successfully", because the screen mirrored the
 * detected counts into the expected ones and then compared them against themselves.
 */
public class ResultsVerdictTest {

    private static Map<String, Integer> counts(Object... pairs) {
        Map<String, Integer> m = new LinkedHashMap<>();
        for (int i = 0; i < pairs.length; i += 2) {
            m.put((String) pairs[i], (Integer) pairs[i + 1]);
        }
        return m;
    }

    private static List<String> types(String... t) {
        return Arrays.asList(t);
    }

    @Test
    public void noReferenceIsUnverifiedRatherThanPassed() {
        // Exactly the old bug: expected mirrors detected, so a count comparison cannot fail.
        Map<String, Integer> detected = counts("screw", 4, "stud", 2);
        assertEquals(ResultsFragment.Verdict.UNVERIFIED,
                ResultsFragment.verdictFor(false, detected, detected, types("screw", "stud")));
    }

    @Test
    public void noReferenceStaysUnverifiedEvenWithNothingDetected() {
        assertEquals(ResultsFragment.Verdict.UNVERIFIED,
                ResultsFragment.verdictFor(false, counts(), counts(), types()));
    }

    @Test
    public void matchingCountsPass() {
        assertEquals(ResultsFragment.Verdict.PASSED,
                ResultsFragment.verdictFor(true, counts("screw", 4, "stud", 2),
                        counts("screw", 4, "stud", 2), types("screw", "stud")));
    }

    @Test
    public void aMissingFeatureFails() {
        assertEquals(ResultsFragment.Verdict.FAILED,
                ResultsFragment.verdictFor(true, counts("screw", 4),
                        counts("screw", 3), types("screw")));
    }

    @Test
    public void anExtraFeatureFails() {
        assertEquals(ResultsFragment.Verdict.FAILED,
                ResultsFragment.verdictFor(true, counts("screw", 4),
                        counts("screw", 5), types("screw")));
    }

    @Test
    public void aTypeMissingEntirelyFromDetectionFails() {
        // Absent from the detected map, not merely lower — count() must read that as zero
        // rather than skipping the type.
        assertEquals(ResultsFragment.Verdict.FAILED,
                ResultsFragment.verdictFor(true, counts("screw", 4, "stud", 2),
                        counts("screw", 4), types("screw", "stud")));
    }

    @Test
    public void unknownDetectionsDoNotDecideTheVerdict() {
        // Unclassified crops are noise on both sides; a part is not failed for having them.
        assertEquals(ResultsFragment.Verdict.PASSED,
                ResultsFragment.verdictFor(true, counts("screw", 4),
                        counts("screw", 4, ReferenceLibrary.UNKNOWN, 7),
                        types("screw", ReferenceLibrary.UNKNOWN)));
    }

    @Test
    public void aRealDiscrepancyStillFailsAlongsideUnknowns() {
        assertEquals(ResultsFragment.Verdict.FAILED,
                ResultsFragment.verdictFor(true, counts("screw", 4),
                        counts("screw", 2, ReferenceLibrary.UNKNOWN, 7),
                        types("screw", ReferenceLibrary.UNKNOWN)));
    }
}
