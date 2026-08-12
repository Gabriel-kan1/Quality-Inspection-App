package com.example.ml;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import android.graphics.RectF;

import org.junit.Test;
import org.robolectric.RobolectricTestRunner;
import org.junit.runner.RunWith;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Covers the positional check for critical features: that it finds which specific expected
 * feature is missing, notices hardware in the wrong place, and refuses to make a claim when
 * the photo cannot be registered onto the reference.
 *
 * Runs under Robolectric only because {@link RectF} is an Android type; the logic itself has
 * no framework dependency.
 */
@RunWith(RobolectricTestRunner.class)
public class FeatureLocalisationTest {

    private static final Set<String> CRITICAL = new LinkedHashSet<>(
            Arrays.asList("bolt_head", "screw", "stud"));

    /** A small square detection centred on (x, y) in fractional coordinates. */
    private static RectF at(float x, float y) {
        float h = 0.02f;
        return new RectF(x - h, y - h, x + h, y + h);
    }

    /**
     * Four bolt heads roughly at the corners plus a screw near the middle.
     *
     * Deliberately not a perfect square: a 4-fold symmetric arrangement can be aligned onto
     * itself by a quarter turn, so which corner is "the missing one" is genuinely ambiguous
     * there — see {@link #aSymmetricLayoutCannotSayWhichCornerIsMissing}. Real parts are
     * rarely that regular, and the asymmetry here is what makes slot identity meaningful.
     *
     * Slot ids follow insertion order: 1 upper-left, 2 upper-right, 3 lower-left,
     * 4 lower-right, 5 the centre screw.
     */
    private static FeatureLayout referenceLayout() {
        List<RectF> rects = Arrays.asList(
                at(0.15f, 0.15f), at(0.85f, 0.20f), at(0.30f, 0.85f), at(0.85f, 0.80f),
                at(0.45f, 0.50f));
        List<String> types = Arrays.asList(
                "bolt_head", "bolt_head", "bolt_head", "bolt_head", "screw");
        return FeatureLocalisation.layoutFrom(rects, types, CRITICAL);
    }

    /**
     * Detections built from the layout itself, so a test states only what it changes.
     *
     * @param omitSlotIds slot numbers to leave empty
     * @param retypes     slot number to the type actually found there, for swaps and mislabels
     */
    private static Detections shotOf(FeatureLayout layout, Set<Integer> omitSlotIds,
                                     java.util.Map<Integer, String> retypes) {
        Detections d = new Detections();
        for (FeatureLayout.Slot s : layout.slots()) {
            if (omitSlotIds != null && omitSlotIds.contains(s.id)) continue;
            d.rects.add(at(s.x, s.y));
            String type = retypes == null ? null : retypes.get(s.id);
            d.types.add(type == null ? s.type : type);
        }
        return d;
    }

    private static final class Detections {
        final List<RectF> rects = new ArrayList<>();
        final List<String> types = new ArrayList<>();
    }

    private static FeatureLocalisation.SlotFinding findingFor(FeatureLocalisation.Result r, int slotId) {
        for (FeatureLocalisation.SlotFinding f : r.findings) if (f.slotId == slotId) return f;
        return null;
    }

    @Test
    public void zoneNamesTheThirdsOfThePart() {
        assertEquals("upper left", FeatureLocalisation.zone(0.1f, 0.1f));
        assertEquals("upper right", FeatureLocalisation.zone(0.9f, 0.1f));
        assertEquals("lower left", FeatureLocalisation.zone(0.1f, 0.9f));
        assertEquals("lower right", FeatureLocalisation.zone(0.9f, 0.9f));
        assertEquals("centre", FeatureLocalisation.zone(0.5f, 0.5f));
        assertEquals("upper centre", FeatureLocalisation.zone(0.5f, 0.1f));
        assertEquals("middle left", FeatureLocalisation.zone(0.1f, 0.5f));
    }

    @Test
    public void layoutKeepsOnlyTheRegistrationTypes() {
        List<RectF> rects = Arrays.asList(at(0.2f, 0.2f), at(0.5f, 0.5f), at(0.8f, 0.8f));
        List<String> types = Arrays.asList("bolt_head", "scratches", "surface");
        FeatureLayout layout = FeatureLocalisation.layoutFrom(rects, types, CRITICAL);
        assertEquals(1, layout.slots().size());
        assertEquals("bolt_head", layout.slots().get(0).type);
    }

    /**
     * Non-critical types are excluded from the geometry, not merely from the report.
     *
     * A scratch has no place it is supposed to be, so it cannot occupy a slot or be missing
     * from one — offering it as the occupant of a bolt head's position yields findings like
     * "WRONG: Scratch", which is noise wearing the vocabulary of a defect.
     */
    @Test
    public void onlyCriticalTypesBecomeSlots() {
        List<RectF> rects = Arrays.asList(
                at(0.15f, 0.15f), at(0.85f, 0.20f), at(0.30f, 0.85f),
                at(0.60f, 0.45f), at(0.45f, 0.70f));
        List<String> types = Arrays.asList(
                "bolt_head", "bolt_head", "bolt_head", "scratches", "surface");

        FeatureLayout layout = FeatureLocalisation.layoutFrom(rects, types, CRITICAL);
        assertEquals("only the bolt heads become slots", 3, layout.slots().size());
        for (FeatureLayout.Slot s : layout.slots()) {
            assertEquals("bolt_head", s.type);
        }
    }

    /**
     * The cost of critical-only, pinned so it is a decision rather than a surprise: a
     * reference carrying only a few critical features can no longer borrow scratches to reach
     * a registerable size, and reports the check as unavailable instead of guessing.
     */
    @Test
    public void aReferenceWithTooFewCriticalFeaturesRefusesRatherThanBorrowLandmarks() {
        List<RectF> rects = Arrays.asList(
                at(0.15f, 0.15f), at(0.85f, 0.20f), at(0.30f, 0.85f),
                at(0.60f, 0.45f), at(0.45f, 0.70f));
        List<String> types = Arrays.asList(
                "bolt_head", "bolt_head", "bolt_head", "scratches", "scratches");

        FeatureLayout layout = FeatureLocalisation.layoutFrom(rects, types, CRITICAL);
        assertEquals(3, layout.slots().size());

        Detections d = shotOf(layout, null, null);
        FeatureLocalisation.Result result =
                FeatureLocalisation.localise(layout, d.rects, d.types, null);

        assertFalse(result.aligned);
        assertTrue(result.unavailableReason.contains("Reference photo"));
    }

    /**
     * The regression this pins: a small layout must not demand perfection.
     *
     * The inlier floor is an absolute 3, so on a 4-slot layout an uncapped rule would need
     * 3 of 4 — fine — but on a 3-slot layout it needed 3 of 3, meaning one absent fastener
     * made the check vanish rather than report the fastener. The requirement is capped at
     * {@code slots - 1} so there is always somewhere for a defect to be found.
     */
    @Test
    public void aSmallLayoutStillLeavesRoomForOneMissingFeature() {
        assertEquals(3, LayoutMatcher.requiredInliers(4));
        assertEquals(3, LayoutMatcher.requiredInliers(5));
        assertEquals(3, LayoutMatcher.requiredInliers(6));
        assertEquals(5, LayoutMatcher.requiredInliers(10));
        for (int slots = 4; slots <= 30; slots++) {
            assertTrue("a layout of " + slots + " must tolerate at least one missing feature",
                    LayoutMatcher.requiredInliers(slots) <= slots - 1);
        }
    }

    /** Below four slots there is no room for both corroboration and a reportable defect. */
    @Test
    public void layoutsUnderFourSlotsAreRefusedWithAnActionableReason() {
        assertTrue(LayoutMatcher.tooSmallToRegister(3));
        assertFalse(LayoutMatcher.tooSmallToRegister(4));

        List<RectF> rects = Arrays.asList(at(0.2f, 0.2f), at(0.8f, 0.2f), at(0.5f, 0.8f));
        List<String> types = Arrays.asList("bolt_head", "bolt_head", "screw");
        FeatureLayout tiny = FeatureLocalisation.layoutFrom(rects, types, CRITICAL);

        Detections d = shotOf(tiny, null, null);
        FeatureLocalisation.Result result =
                FeatureLocalisation.localise(tiny, d.rects, d.types, null);
        assertFalse(result.aligned);
        assertNotNull(result.unavailableReason);
        assertTrue("the reason must point at the reference photo, not the framing",
                result.unavailableReason.contains("Reference photo"));
    }

    @Test
    public void layoutSkipsUnclassifiedDetections() {
        List<RectF> rects = Arrays.asList(at(0.2f, 0.2f), at(0.8f, 0.8f));
        List<String> types = Arrays.asList("bolt_head", ReferenceLibrary.UNKNOWN);
        FeatureLayout layout = FeatureLocalisation.layoutFrom(rects, types, CRITICAL);
        assertEquals(1, layout.slots().size());
    }

    @Test
    public void everyPositionPresentOnAnIdenticalShot() {
        FeatureLayout layout = referenceLayout();
        Detections d = shotOf(layout, null, null);

        FeatureLocalisation.Result result = FeatureLocalisation.localise(layout, d.rects, d.types, null);
        assertTrue(result.aligned);
        assertEquals(5, result.findings.size());
        assertEquals(5, result.countWithStatus(FeatureLocalisation.Status.PRESENT));
        assertFalse(result.hasPositionalDefect());
    }

    /**
     * The headline capability: not "one bolt head short" but "the lower-right one is the
     * missing one".
     */
    @Test
    public void identifiesWhichSpecificFeatureIsMissing() {
        FeatureLayout layout = referenceLayout();
        Detections d = shotOf(layout, java.util.Collections.singleton(4), null);

        FeatureLocalisation.Result result = FeatureLocalisation.localise(layout, d.rects, d.types, null);
        assertTrue(result.aligned);
        assertTrue(result.hasPositionalDefect());
        assertEquals(1, result.countWithStatus(FeatureLocalisation.Status.MISSING));

        FeatureLocalisation.SlotFinding missing = findingFor(result, 4);
        assertNotNull(missing);
        assertEquals(FeatureLocalisation.Status.MISSING, missing.status);
        assertEquals("bolt_head", missing.expectedType);
        assertEquals("lower right", FeatureLocalisation.zone(missing.x, missing.y));
        assertTrue(missing.position().startsWith("lower right"));
    }

    /**
     * The defect counting cannot see at all: the totals are right — four bolt heads and one
     * screw — but the screw and one bolt head have traded places.
     */
    @Test
    public void catchesSwappedHardwareThatCountsWouldMiss() {
        FeatureLayout layout = referenceLayout();
        java.util.Map<Integer, String> swap = new java.util.LinkedHashMap<>();
        swap.put(4, "screw");      // corner now holds the screw
        swap.put(5, "bolt_head");  // centre now holds a bolt head
        Detections d = shotOf(layout, null, swap);

        FeatureLocalisation.Result result = FeatureLocalisation.localise(layout, d.rects, d.types, null);
        assertTrue(result.aligned);
        assertTrue("a swap must register as a defect", result.hasPositionalDefect());
        assertEquals(2, result.countWithStatus(FeatureLocalisation.Status.WRONG_TYPE));
        assertEquals("screw", findingFor(result, 4).foundType);
        assertEquals("bolt_head", findingFor(result, 5).foundType);
    }

    /**
     * Reported positions must be numbered 1..N with no gaps.
     *
     * The registration layout contains landmarks that are never reported, so numbering the
     * report by raw slot id skipped values — "#1, #3, #4" — which reads as though #2 had gone
     * missing rather than never having been a critical feature.
     */
    @Test
    public void reportedPositionsAreNumberedContiguously() {
        // Bolt heads at slots 1, 3, 5; scratches at 2 and 4 — landmarks only.
        List<RectF> rects = Arrays.asList(
                at(0.15f, 0.15f), at(0.50f, 0.10f), at(0.85f, 0.20f),
                at(0.20f, 0.60f), at(0.80f, 0.80f));
        List<String> types = Arrays.asList(
                "bolt_head", "scratches", "bolt_head", "scratches", "bolt_head");
        Set<String> registration = new LinkedHashSet<>(Arrays.asList("bolt_head", "scratches"));
        FeatureLayout layout = FeatureLocalisation.layoutFrom(rects, types, registration);

        Detections d = shotOf(layout, null, null);
        Set<String> reportOnly = new LinkedHashSet<>(Arrays.asList("bolt_head"));
        FeatureLocalisation.Result result =
                FeatureLocalisation.localise(layout, d.rects, d.types, reportOnly);

        assertTrue(result.aligned);
        assertEquals(3, result.findings.size());
        for (int i = 0; i < result.findings.size(); i++) {
            assertEquals("display numbers must run 1..N",
                    i + 1, result.findings.get(i).displayNumber);
        }
        // The underlying layout ids still have the gaps; only the display side is contiguous.
        assertEquals(1, result.findings.get(0).slotId);
        assertEquals(3, result.findings.get(1).slotId);
        assertEquals(5, result.findings.get(2).slotId);
    }

    /**
     * Reported coordinates are relative to the part, not to the photo frame.
     *
     * The same physical arrangement shot with more background around it must produce the same
     * numbers — that is the whole point of normalising to the feature envelope.
     */
    @Test
    public void positionsAreRelativeToThePartNotTheFrame() {
        // Same four bolt heads, once filling the frame and once shot small in a corner.
        List<String> types = Arrays.asList("bolt_head", "bolt_head", "bolt_head", "bolt_head");
        FeatureLayout wide = FeatureLocalisation.layoutFrom(Arrays.asList(
                at(0.10f, 0.10f), at(0.90f, 0.10f), at(0.10f, 0.90f), at(0.90f, 0.90f)),
                types, CRITICAL);
        FeatureLayout tight = FeatureLocalisation.layoutFrom(Arrays.asList(
                at(0.40f, 0.40f), at(0.60f, 0.40f), at(0.40f, 0.60f), at(0.60f, 0.60f)),
                types, CRITICAL);

        FeatureLocalisation.Result a = FeatureLocalisation.localise(
                wide, shotOf(wide, null, null).rects, shotOf(wide, null, null).types, null);
        FeatureLocalisation.Result b = FeatureLocalisation.localise(
                tight, shotOf(tight, null, null).rects, shotOf(tight, null, null).types, null);

        assertTrue(a.aligned);
        assertTrue(b.aligned);
        for (int i = 0; i < 4; i++) {
            assertEquals("relative x must not depend on framing",
                    a.findings.get(i).relX, b.findings.get(i).relX, 0.001f);
            assertEquals("relative y must not depend on framing",
                    a.findings.get(i).relY, b.findings.get(i).relY, 0.001f);
        }
        // Corners of the envelope map to the corners of the 0-1 range.
        assertEquals(0f, a.findings.get(0).relX, 0.001f);
        assertEquals(0f, a.findings.get(0).relY, 0.001f);
        assertEquals(1f, a.findings.get(3).relX, 0.001f);
        assertEquals(1f, a.findings.get(3).relY, 0.001f);
        assertEquals("upper left", FeatureLocalisation.zone(
                a.findings.get(0).relX, a.findings.get(0).relY));
        assertEquals("lower right", FeatureLocalisation.zone(
                a.findings.get(3).relX, a.findings.get(3).relY));
    }

    /** Frame coordinates are still recorded — only the reported ones are normalised. */
    @Test
    public void frameCoordinatesAreRetainedAlongsideRelativeOnes() {
        FeatureLayout layout = FeatureLocalisation.layoutFrom(Arrays.asList(
                at(0.40f, 0.40f), at(0.60f, 0.40f), at(0.40f, 0.60f), at(0.60f, 0.60f)),
                Arrays.asList("bolt_head", "bolt_head", "bolt_head", "bolt_head"), CRITICAL);
        Detections d = shotOf(layout, null, null);
        FeatureLocalisation.Result r =
                FeatureLocalisation.localise(layout, d.rects, d.types, null);

        assertEquals(0.40f, r.findings.get(0).x, 0.001f);
        assertEquals(0.40f, r.findings.get(0).y, 0.001f);
        assertEquals(0f, r.findings.get(0).relX, 0.001f);
        assertEquals(0f, r.findings.get(0).relY, 0.001f);
    }

    /**
     * The plain substitution case, and the reason positional checking exists: slot #3 is
     * supposed to hold a bolt head, a stud is physically there instead. The report must name
     * that slot, its position, and what was actually found — not call the bolt head missing
     * and the stud unexpected.
     */
    @Test
    public void namesTheSlotWhereTheWrongHardwareIsFitted() {
        FeatureLayout layout = referenceLayout();
        Detections d = shotOf(layout, null,
                java.util.Collections.singletonMap(3, "stud"));

        FeatureLocalisation.Result result =
                FeatureLocalisation.localise(layout, d.rects, d.types, null);

        assertTrue(result.aligned);
        assertEquals("nothing is missing — the position is occupied",
                0, result.countWithStatus(FeatureLocalisation.Status.MISSING));
        assertEquals(1, result.countWithStatus(FeatureLocalisation.Status.WRONG_TYPE));
        assertTrue(result.hasPositionalDefect());

        FeatureLocalisation.SlotFinding wrong = findingFor(result, 3);
        assertEquals(FeatureLocalisation.Status.WRONG_TYPE, wrong.status);
        assertEquals("bolt_head", wrong.expectedType);
        assertEquals("stud", wrong.foundType);
        assertEquals("lower left", FeatureLocalisation.zone(wrong.x, wrong.y));
    }

    /**
     * Substitution must not starve the search.
     *
     * Seeding the transform only from same-type pairs made alignment progressively less
     * likely as substitutions grew — the registration failed exactly as the defect got worse.
     * Here three of five positions hold the wrong hardware and every one must still be named.
     */
    @Test
    public void heavySubstitutionStillRegistersAndIsFullyReported() {
        FeatureLayout layout = referenceLayout();
        java.util.Map<Integer, String> wrong = new java.util.LinkedHashMap<>();
        wrong.put(1, "stud");
        wrong.put(3, "stud");
        wrong.put(4, "screw");
        Detections d = shotOf(layout, null, wrong);

        FeatureLocalisation.Result result =
                FeatureLocalisation.localise(layout, d.rects, d.types, null);

        assertTrue("position alone must be able to carry the registration", result.aligned);
        assertEquals(3, result.countWithStatus(FeatureLocalisation.Status.WRONG_TYPE));
        assertEquals(0, result.countWithStatus(FeatureLocalisation.Status.MISSING));
        assertEquals("stud", findingFor(result, 1).foundType);
        assertEquals("stud", findingFor(result, 3).foundType);
        assertEquals("screw", findingFor(result, 4).foundType);
        assertEquals(FeatureLocalisation.Status.PRESENT, findingFor(result, 2).status);
        assertEquals(FeatureLocalisation.Status.PRESENT, findingFor(result, 5).status);
    }

    /**
     * The extreme: every position holds the wrong hardware.
     *
     * The only same-type pairs left all point at the single bolt head sitting in the screw's
     * slot, so every one of them shares a feature index and the two-point sample is rejected
     * on each draw. The typed pool therefore yields no transform at all — silence on a part
     * where nothing is correct. Position alone has to carry it.
     */
    @Test
    public void aPartWithNothingCorrectIsStillFullyLocated() {
        FeatureLayout layout = referenceLayout();
        java.util.Map<Integer, String> allWrong = new java.util.LinkedHashMap<>();
        allWrong.put(1, "stud");
        allWrong.put(2, "stud");
        allWrong.put(3, "stud");
        allWrong.put(4, "stud");
        allWrong.put(5, "bolt_head");
        Detections d = shotOf(layout, null, allWrong);

        FeatureLocalisation.Result result =
                FeatureLocalisation.localise(layout, d.rects, d.types, null);

        assertTrue("no same-type pair exists, but position still registers", result.aligned);
        assertEquals(5, result.countWithStatus(FeatureLocalisation.Status.WRONG_TYPE));
        assertTrue(result.hasPositionalDefect());
    }

    /**
     * Type agreement must still be preferred when it is available: with a correct shot, the
     * added untyped sampling pass must not pull the answer away from the obvious alignment.
     */
    @Test
    public void untypedSamplingDoesNotDisturbACorrectShot() {
        FeatureLayout layout = referenceLayout();
        Detections d = shotOf(layout, null, null);

        FeatureLocalisation.Result result =
                FeatureLocalisation.localise(layout, d.rects, d.types, null);
        assertTrue(result.aligned);
        assertEquals(5, result.countWithStatus(FeatureLocalisation.Status.PRESENT));
    }

    /**
     * An honest limit, pinned so it cannot be mistaken for a regression later.
     *
     * A perfectly 4-fold symmetric arrangement maps onto itself under a quarter turn, so with
     * one corner absent there are several alignments that fit equally well and no evidence
     * distinguishes them. The check still reports that a position is empty — the count is
     * right and the part still fails — but which corner it names is arbitrary.
     */
    @Test
    public void aSymmetricLayoutCannotSayWhichCornerIsMissing() {
        List<RectF> rects = Arrays.asList(
                at(0.20f, 0.20f), at(0.80f, 0.20f), at(0.20f, 0.80f), at(0.80f, 0.80f),
                at(0.50f, 0.50f));
        List<String> types = Arrays.asList(
                "bolt_head", "bolt_head", "bolt_head", "bolt_head", "screw");
        FeatureLayout square = FeatureLocalisation.layoutFrom(rects, types, CRITICAL);

        Detections d = shotOf(square, java.util.Collections.singleton(4), null);
        FeatureLocalisation.Result result = FeatureLocalisation.localise(square, d.rects, d.types, null);

        assertTrue(result.aligned);
        assertEquals("the count of missing positions is still right",
                1, result.countWithStatus(FeatureLocalisation.Status.MISSING));
        assertTrue(result.hasPositionalDefect());
        // Which slot id is named is deliberately not asserted: it is not determined.
    }

    /** A shifted, rotated and rescaled shot of a good part must still read as all-present. */
    @Test
    public void toleratesAFreehandShiftRotationAndScale() {
        FeatureLayout layout = referenceLayout();
        double angle = Math.toRadians(12);
        double scale = 0.85, dx = 0.06, dy = -0.03;

        List<RectF> rects = new ArrayList<>();
        List<String> types = new ArrayList<>();
        for (FeatureLayout.Slot s : layout.slots()) {
            double x = scale * (s.x * Math.cos(angle) - s.y * Math.sin(angle)) + dx;
            double y = scale * (s.x * Math.sin(angle) + s.y * Math.cos(angle)) + dy;
            rects.add(at((float) x, (float) y));
            types.add(s.type);
        }

        FeatureLocalisation.Result result = FeatureLocalisation.localise(layout, rects, types, null);
        assertTrue(result.aligned);
        assertEquals(5, result.countWithStatus(FeatureLocalisation.Status.PRESENT));
    }

    /**
     * An unclassified detection is allowed to occupy a slot. Reporting a fastener that is
     * physically there as missing hardware, purely because the classifier did not recognise
     * it, is the more misleading of the two available errors.
     */
    @Test
    public void anUnknownDetectionStillFillsItsSlot() {
        FeatureLayout layout = referenceLayout();
        Detections d = shotOf(layout, null,
                java.util.Collections.singletonMap(4, ReferenceLibrary.UNKNOWN));

        FeatureLocalisation.Result result = FeatureLocalisation.localise(layout, d.rects, d.types, null);
        assertTrue(result.aligned);
        assertEquals(0, result.countWithStatus(FeatureLocalisation.Status.MISSING));
    }

    @Test
    public void reportsUnavailableRatherThanEmptyWithoutALayout() {
        FeatureLocalisation.Result result = FeatureLocalisation.localise(null, null, null, null);
        assertFalse(result.aligned);
        assertFalse(result.hasPositionalDefect());
        assertNotNull(result.unavailableReason);
        assertTrue(result.summary().contains("reference"));
    }

    @Test
    public void reportsUnavailableWhenNothingWasDetected() {
        FeatureLocalisation.Result result =
                FeatureLocalisation.localise(referenceLayout(), new ArrayList<>(), new ArrayList<>(), null);
        assertFalse(result.aligned);
        assertFalse(result.hasPositionalDefect());
    }

    /**
     * A photo of something else entirely must produce no positional claim. Reporting every
     * slot as missing would turn an unusable shot into a confident failure verdict.
     */
    @Test
    public void makesNoClaimWhenThePhotoCannotBeRegistered() {
        FeatureLayout layout = referenceLayout();
        List<RectF> rects = Arrays.asList(at(0.10f, 0.10f));
        List<String> types = Arrays.asList("bolt_head");

        FeatureLocalisation.Result result = FeatureLocalisation.localise(layout, rects, types, null);
        assertFalse(result.aligned);
        assertTrue(result.findings.isEmpty());
        assertFalse(result.hasPositionalDefect());
    }

    /**
     * A fastener fitted off its position must be reported as off, not as absent.
     *
     * "The bolt is missing" sends the operator looking for a part that is sitting in front of
     * them a few millimetres out. The nearest otherwise-unaccounted-for detection is offered
     * instead, with the distance, so the two faults can be told apart.
     */
    @Test
    public void anOffsetFastenerIsReportedWithHowFarOffItIs() {
        FeatureLayout layout = referenceLayout();
        // Slots 1, 2, 3, 5 in place; slot 4's fastener sits well outside matching tolerance.
        Detections d = shotOf(layout, java.util.Collections.singleton(4), null);
        FeatureLayout.Slot four = layout.slotById(4);
        d.rects.add(at(four.x - 0.10f, four.y - 0.10f));
        d.types.add("bolt_head");

        FeatureLocalisation.Result result =
                FeatureLocalisation.localise(layout, d.rects, d.types, null);

        assertTrue(result.aligned);
        assertEquals("it is present, just not where it belongs",
                0, result.countWithStatus(FeatureLocalisation.Status.MISSING));
        assertEquals(1, result.countWithStatus(FeatureLocalisation.Status.DISPLACED));

        FeatureLocalisation.SlotFinding off = findingFor(result, 4);
        assertEquals(FeatureLocalisation.Status.DISPLACED, off.status);
        assertEquals("bolt_head", off.foundType);
        assertTrue("an offset must be measured", off.offset > 0f);
        assertTrue(off.offset <= FeatureLocalisation.DISPLACEMENT_LIMIT);
        assertTrue("a displaced fastener is still a defect", result.hasPositionalDefect());
    }

    /** Beyond the limit there is nothing credible to point at, so it stays MISSING. */
    @Test
    public void aFarAwayDetectionIsNotOfferedAsTheMissingOne() {
        FeatureLayout layout = referenceLayout();
        Detections d = shotOf(layout, java.util.Collections.singleton(4), null);
        // Nowhere near slot 4, and nowhere near anything else either.
        d.rects.add(at(0.02f, 0.02f));
        d.types.add("bolt_head");

        FeatureLocalisation.Result result =
                FeatureLocalisation.localise(layout, d.rects, d.types, null);

        assertTrue(result.aligned);
        assertEquals(1, result.countWithStatus(FeatureLocalisation.Status.MISSING));
        assertEquals(0, result.countWithStatus(FeatureLocalisation.Status.DISPLACED));
        assertNull(findingFor(result, 4).foundType);
    }

    /** Two empty positions must not both blame the same stray detection. */
    @Test
    public void oneStrayDetectionExplainsAtMostOnePosition() {
        FeatureLayout layout = referenceLayout();
        Detections d = shotOf(layout,
                new LinkedHashSet<>(Arrays.asList(1, 4)), null);
        FeatureLayout.Slot one = layout.slotById(1);
        d.rects.add(at(one.x + 0.08f, one.y + 0.08f));
        d.types.add("bolt_head");

        FeatureLocalisation.Result result =
                FeatureLocalisation.localise(layout, d.rects, d.types, null);

        assertTrue(result.aligned);
        assertEquals("only one position can claim the stray detection",
                1, result.countWithStatus(FeatureLocalisation.Status.DISPLACED));
        assertEquals(1, result.countWithStatus(FeatureLocalisation.Status.MISSING));
    }

    /**
     * A fully swapped part on a symmetric layout must read as swapped, not as correct.
     *
     * Four fasteners at the corners of a rectangle map onto themselves under a half turn, so
     * "every fastener swapped" and "nothing wrong, photographed upside down" have identical
     * positional support — a genuine tie. Type agreement used to break it, which is the one
     * rule guaranteed to pick the reading that hides the defect: rotating the layout 180°
     * lands every stud slot on a stud and every screw slot on a screw, and the report came
     * back all-present on a part where nothing was in the right place.
     */
    @Test
    public void aSwappedPartOnASymmetricLayoutIsNotReportedAsCorrect() {
        List<RectF> rects = Arrays.asList(
                at(0.30f, 0.70f), at(0.50f, 0.70f), at(0.30f, 0.40f), at(0.50f, 0.40f));
        List<String> layoutTypes = Arrays.asList("stud", "screw", "stud", "screw");
        FeatureLayout layout = FeatureLocalisation.layoutFrom(rects, layoutTypes, CRITICAL);

        // Same four positions, hardware swapped at every one of them.
        List<String> foundTypes = Arrays.asList("screw", "stud", "screw", "stud");

        FeatureLocalisation.Result result =
                FeatureLocalisation.localise(layout, rects, foundTypes, null);

        assertTrue(result.aligned);
        assertEquals("every position holds the wrong hardware",
                4, result.countWithStatus(FeatureLocalisation.Status.WRONG_TYPE));
        assertEquals(0, result.countWithStatus(FeatureLocalisation.Status.PRESENT));
        assertTrue(result.hasPositionalDefect());
    }

    /**
     * A handful of detections must not be able to satisfy a whole layout.
     *
     * Inliers used to be counted as "does any detection land near this slot", so one dense
     * clump could vouch for every slot at once: a transform that collapsed the layout onto a
     * cluster of scratches scored full marks and was accepted, and the one-to-one assignment
     * that follows could then only pair a couple of them — reporting the rest as missing on a
     * part whose counts matched exactly. Counting one-to-one closes that gap.
     */
    @Test
    public void aTinyClusterCannotVouchForEveryPosition() {
        List<RectF> rects = Arrays.asList(
                at(0.10f, 0.10f), at(0.90f, 0.10f), at(0.10f, 0.90f),
                at(0.90f, 0.90f), at(0.50f, 0.30f), at(0.50f, 0.70f));
        List<String> types = java.util.Collections.nCopies(6, "bolt_head");
        FeatureLayout layout = FeatureLocalisation.layoutFrom(rects, types, CRITICAL);
        assertEquals(6, layout.slots().size());

        // Two detections, close together: nothing here can account for six positions.
        List<RectF> found = Arrays.asList(at(0.50f, 0.50f), at(0.54f, 0.52f));
        List<String> foundTypes = Arrays.asList("bolt_head", "bolt_head");

        FeatureLocalisation.Result result =
                FeatureLocalisation.localise(layout, found, foundTypes, null);
        assertFalse("two detections cannot verify six positions", result.aligned);
    }

    /**
     * Two correspondences must not be enough to claim an alignment.
     *
     * A similarity transform has four degrees of freedom and two point pairs supply exactly
     * four constraints, so any two features can be mapped onto any two slots perfectly. Before
     * the inlier floor existed, a barely-related photo aligned on that coincidence and then
     * confidently named three specific slots as missing.
     */
    @Test
    public void twoMatchingFeaturesAreNotEvidenceOfAnAlignment() {
        FeatureLayout layout = referenceLayout();
        // Only slots 3 and 4 filled — 2 of 5, below the half-the-layout floor.
        Detections d = shotOf(layout,
                new LinkedHashSet<>(Arrays.asList(1, 2, 5)), null);
        assertEquals(2, d.rects.size());

        FeatureLocalisation.Result result = FeatureLocalisation.localise(layout, d.rects, d.types, null);
        assertFalse("2 of 5 positions is a coincidence, not a registration", result.aligned);
        assertTrue(result.findings.isEmpty());
    }

    /** Three of five is above the floor, so the two absent positions are named. */
    @Test
    public void aMajorityOfPositionsIsEnoughToRegister() {
        FeatureLayout layout = referenceLayout();
        Detections d = shotOf(layout, new LinkedHashSet<>(Arrays.asList(2, 5)), null);

        FeatureLocalisation.Result result = FeatureLocalisation.localise(layout, d.rects, d.types, null);
        assertTrue(result.aligned);
        assertEquals(2, result.countWithStatus(FeatureLocalisation.Status.MISSING));
        assertEquals(FeatureLocalisation.Status.MISSING, findingFor(result, 2).status);
        assertEquals(FeatureLocalisation.Status.MISSING, findingFor(result, 5).status);
    }

    /** A projection outside the frame is not drawable, so the overlay must not be told it is. */
    @Test
    public void aPositionInsideTheFrameIsDrawable() {
        FeatureLayout layout = referenceLayout();
        Detections d = shotOf(layout, java.util.Collections.singleton(4), null);

        FeatureLocalisation.Result result = FeatureLocalisation.localise(layout, d.rects, d.types, null);
        FeatureLocalisation.SlotFinding missing = findingFor(result, 4);
        assertTrue(missing.hasProjection());
        assertTrue(missing.projectedX >= 0f && missing.projectedX <= 1f);
        assertTrue(missing.projectedY >= 0f && missing.projectedY <= 1f);
    }

    @Test
    public void findingsAreOrderedBySlotNumber() {
        FeatureLayout layout = referenceLayout();
        Detections d = shotOf(layout, java.util.Collections.singleton(3), null);

        FeatureLocalisation.Result result = FeatureLocalisation.localise(layout, d.rects, d.types, null);
        assertTrue(result.aligned);
        assertEquals(5, result.findings.size());
        for (int i = 1; i < result.findings.size(); i++) {
            assertTrue("findings must ascend by slot id",
                    result.findings.get(i).slotId > result.findings.get(i - 1).slotId);
        }
    }
}
