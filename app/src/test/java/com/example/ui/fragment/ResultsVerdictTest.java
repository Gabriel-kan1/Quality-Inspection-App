package com.example.ui.fragment;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.example.data.entity.ExpectedFeature;
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

    /** The default the app ships: anything unclassified is treated as pass/fail relevant. */
    private static final ResultsFragment.CriticalityRule ALL_CRITICAL = type -> true;

    @Test
    public void noReferenceIsUnverifiedRatherThanPassed() {
        // Exactly the old bug: expected mirrors detected, so a count comparison cannot fail.
        Map<String, Integer> detected = counts("screw", 4, "stud", 2);
        assertEquals(ResultsFragment.Verdict.UNVERIFIED,
                ResultsFragment.verdictFor(false, detected, detected, types("screw", "stud"), ALL_CRITICAL, false));
    }

    @Test
    public void noReferenceStaysUnverifiedEvenWithNothingDetected() {
        assertEquals(ResultsFragment.Verdict.UNVERIFIED,
                ResultsFragment.verdictFor(false, counts(), counts(), types(), ALL_CRITICAL, false));
    }

    @Test
    public void matchingCountsPass() {
        assertEquals(ResultsFragment.Verdict.PASSED,
                ResultsFragment.verdictFor(true, counts("screw", 4, "stud", 2),
                        counts("screw", 4, "stud", 2), types("screw", "stud"), ALL_CRITICAL, false));
    }

    @Test
    public void aMissingFeatureFails() {
        assertEquals(ResultsFragment.Verdict.FAILED,
                ResultsFragment.verdictFor(true, counts("screw", 4),
                        counts("screw", 3), types("screw"), ALL_CRITICAL, false));
    }

    @Test
    public void anExtraFeatureFails() {
        assertEquals(ResultsFragment.Verdict.FAILED,
                ResultsFragment.verdictFor(true, counts("screw", 4),
                        counts("screw", 5), types("screw"), ALL_CRITICAL, false));
    }

    @Test
    public void aTypeMissingEntirelyFromDetectionFails() {
        // Absent from the detected map, not merely lower — count() must read that as zero
        // rather than skipping the type.
        assertEquals(ResultsFragment.Verdict.FAILED,
                ResultsFragment.verdictFor(true, counts("screw", 4, "stud", 2),
                        counts("screw", 4), types("screw", "stud"), ALL_CRITICAL, false));
    }

    @Test
    public void unknownDetectionsDoNotDecideTheVerdict() {
        // Unclassified crops are noise on both sides; a part is not failed for having them.
        assertEquals(ResultsFragment.Verdict.PASSED,
                ResultsFragment.verdictFor(true, counts("screw", 4),
                        counts("screw", 4, ReferenceLibrary.UNKNOWN, 7),
                        types("screw", ReferenceLibrary.UNKNOWN), ALL_CRITICAL, false));
    }

    @Test
    public void aRealDiscrepancyStillFailsAlongsideUnknowns() {
        assertEquals(ResultsFragment.Verdict.FAILED,
                ResultsFragment.verdictFor(true, counts("screw", 4),
                        counts("screw", 2, ReferenceLibrary.UNKNOWN, 7),
                        types("screw", ReferenceLibrary.UNKNOWN), ALL_CRITICAL, false));
    }

    @Test
    public void nonCriticalTypesDoNotFailThePart() {
        // Advisory and ignored types are reported but never drive pass/fail — an operator
        // trending scratches must not have the part failed for an extra one.
        ResultsFragment.CriticalityRule onlyScrews = "screw"::equals;
        assertEquals(ResultsFragment.Verdict.PASSED,
                ResultsFragment.verdictFor(true, counts("screw", 4, "scratch", 1),
                        counts("screw", 4, "scratch", 9),
                        types("screw", "scratch"), onlyScrews, false));
    }

    @Test
    public void aCriticalTypeStillFailsWhenOthersAreAdvisory() {
        ResultsFragment.CriticalityRule onlyScrews = "screw"::equals;
        assertEquals(ResultsFragment.Verdict.FAILED,
                ResultsFragment.verdictFor(true, counts("screw", 4, "scratch", 1),
                        counts("screw", 3, "scratch", 1),
                        types("screw", "scratch"), onlyScrews, false));
    }

    @Test
    public void aPositionalDefectFailsEvenWhenEveryCountMatches() {
        // The case the localisation pass exists for: the right number of fasteners, in the
        // wrong places. Invisible to a count comparison.
        assertEquals(ResultsFragment.Verdict.FAILED,
                ResultsFragment.verdictFor(true, counts("screw", 4), counts("screw", 4),
                        types("screw"), ALL_CRITICAL, true));
    }

    @Test
    public void aPositionalDefectCannotResurrectAnUnverifiedRun() {
        // Without a reference there is no layout to register against, so a positional defect
        // is not something the app could have established either.
        assertEquals(ResultsFragment.Verdict.UNVERIFIED,
                ResultsFragment.verdictFor(false, counts("screw", 4), counts("screw", 4),
                        types("screw"), ALL_CRITICAL, true));
    }

    // ── The declared expected side ────────────────────────────────────────

    /**
     * The operator's declaration is used exactly as stored, in the order it was entered, and a
     * row with nothing expected is not an expectation of zero: a zero would make the verdict
     * fail every part that has that hardware, which is the opposite of what an absent row means.
     */
    @Test
    public void declaredCountsAreReadVerbatimAndNonPositiveRowsAreDropped() {
        Map<String, Integer> declared = ResultsFragment.declaredCounts(Arrays.asList(
                new ExpectedFeature("CHASSIS-8821-B", "hole", 1),
                new ExpectedFeature("CHASSIS-8821-B", "stud", 2),
                new ExpectedFeature("CHASSIS-8821-B", "rivet_head", 0),
                new ExpectedFeature("CHASSIS-8821-B", "screw", 1)));
        assertEquals(Arrays.asList("hole", "stud", "screw"),
                new java.util.ArrayList<>(declared.keySet()));
        assertEquals(Integer.valueOf(2), declared.get("stud"));

        // Against a detection that found everything, the declaration passes the part; one
        // stud short, it fails it — the count rule itself is unchanged by where the numbers
        // came from.
        assertEquals(ResultsFragment.Verdict.PASSED,
                ResultsFragment.verdictFor(true, declared, counts("hole", 1, "stud", 2, "screw", 1),
                        types("hole", "stud", "screw"), ALL_CRITICAL, false));
        assertEquals(ResultsFragment.Verdict.FAILED,
                ResultsFragment.verdictFor(true, declared, counts("hole", 1, "stud", 1, "screw", 1),
                        types("hole", "stud", "screw"), ALL_CRITICAL, false));
    }

    @Test
    public void noDeclarationIsAnEmptyMapNotNull() {
        assertTrue(ResultsFragment.declaredCounts(null).isEmpty());
        assertTrue(ResultsFragment.declaredCounts(new java.util.ArrayList<>()).isEmpty());
    }
}
