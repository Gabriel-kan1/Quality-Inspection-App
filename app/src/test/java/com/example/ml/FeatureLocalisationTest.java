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
     * The requirement is capped at {@code slots - 1} so there is always somewhere for a defect
     * to be found. Without the cap the floor dominates a small layout and demands perfection:
     * a 3-slot layout needed 3 of 3, meaning one absent fastener made the check vanish rather
     * than report the fastener.
     *
     * The floor itself depends on which transform the layout can afford. Four slots get the
     * 4-DOF similarity, where two correspondences fit exactly and three is the first evidence.
     * Five and above get the 6-DOF affine, where three fit exactly and four is the first
     * evidence — measurably the better model (it cuts the p95 residual on real photo pairs by
     * 38%), at the price of one more position needing to be located.
     */
    @Test
    public void theInlierFloorFollowsTheModelTheLayoutCanAfford() {
        assertEquals("4 slots keep the similarity model", 3, LayoutMatcher.requiredInliers(4));
        assertEquals("5 slots can afford affine, which needs one more", 4,
                LayoutMatcher.requiredInliers(5));
        assertEquals(4, LayoutMatcher.requiredInliers(6));
        // Past 7 slots the half-the-layout rule already exceeds either floor, so the model
        // choice stops affecting the requirement at all.
        assertEquals(4, LayoutMatcher.requiredInliers(7));
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

    /**
     * Two empty positions must not both blame the same stray detection.
     *
     * Six slots rather than five: with two positions empty, four remain to be located, which
     * is what the affine model's floor requires. On a five-slot layout the same scenario now
     * refuses to register at all — see
     * {@link #aFiveSlotLayoutMissingTwoPositionsCannotAffordTheAffineModel}.
     */
    @Test
    public void oneStrayDetectionExplainsAtMostOnePosition() {
        FeatureLayout layout = FeatureLocalisation.layoutFrom(Arrays.asList(
                        at(0.15f, 0.15f), at(0.85f, 0.20f), at(0.30f, 0.85f),
                        at(0.85f, 0.80f), at(0.45f, 0.50f), at(0.55f, 0.15f)),
                Arrays.asList("bolt_head", "bolt_head", "bolt_head", "bolt_head", "screw", "stud"),
                CRITICAL);
        assertEquals(6, layout.slots().size());

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

    /** Four of five is above the floor, so the absent position is named. */
    @Test
    public void aMajorityOfPositionsIsEnoughToRegister() {
        FeatureLayout layout = referenceLayout();
        Detections d = shotOf(layout, java.util.Collections.singleton(2), null);

        FeatureLocalisation.Result result = FeatureLocalisation.localise(layout, d.rects, d.types, null);
        assertTrue(result.aligned);
        assertEquals(1, result.countWithStatus(FeatureLocalisation.Status.MISSING));
        assertEquals(FeatureLocalisation.Status.MISSING, findingFor(result, 2).status);
    }

    /**
     * The price of the affine model, pinned so it stays a decision rather than a surprise.
     *
     * A 5-slot layout is the smallest that can afford a 6-DOF transform, and three
     * correspondences determine one exactly — so four must be located before an alignment is
     * evidence of anything. A part missing two of five therefore reports no position at all,
     * where the 4-DOF similarity would have named them.
     *
     * That is the trade accepted for a model that fits real photo pairs 38% more tightly at
     * p95: the residual a rigid transform leaves is dominated by the anisotropic
     * foreshortening of a tilted camera, which no tolerance can tell apart from a genuinely
     * misplaced fastener. The count comparison still reports the shortfall; only the
     * positional claim is withheld. See {@code python/localisation_tolerance_study.py}.
     */
    @Test
    public void aFiveSlotLayoutMissingTwoPositionsCannotAffordTheAffineModel() {
        FeatureLayout layout = referenceLayout();
        Detections d = shotOf(layout, new LinkedHashSet<>(Arrays.asList(2, 5)), null);
        assertEquals(3, d.rects.size());

        FeatureLocalisation.Result result = FeatureLocalisation.localise(layout, d.rects, d.types, null);
        assertFalse("3 of 5 no longer clears the affine floor", result.aligned);
        assertTrue(result.findings.isEmpty());
    }

    /**
     * A position the photo does not cover is unverifiable, not missing.
     *
     * The operator shooting tighter than the reference is the ordinary case, not an
     * error: the reference for a real part carried six fasteners, four of them in one
     * cluster, and a capture framed on that cluster left the other two off-image. They
     * found nothing nearby, were reported MISSING, and a positional defect fails the
     * part — so a photograph of correct hardware failed it. The frame the operator chose
     * is not evidence about the hardware they did not photograph.
     *
     * The shot here is scaled 1.15x and shifted down, which pushes slot 3 (the lowest)
     * past the bottom edge while the other four stay in frame and carry the alignment.
     */
    @Test
    public void aPositionOutsideThePhotoIsNotReportedMissing() {
        FeatureLayout layout = referenceLayout();
        List<RectF> rects = new ArrayList<>();
        List<String> types = new ArrayList<>();
        for (FeatureLayout.Slot s : layout.slots()) {
            if (s.id == 3) continue;              // off the bottom edge, so never photographed
            rects.add(shot(s, 0, 1.15, 0, 0.05));
            types.add(s.type);
        }

        FeatureLocalisation.Result result =
                FeatureLocalisation.localise(layout, rects, types, null);

        assertTrue("four located slots still register", result.aligned);
        FeatureLocalisation.SlotFinding off = findingFor(result, 3);
        assertNotNull(off);
        assertEquals(FeatureLocalisation.Status.OUT_OF_FRAME, off.status);
        assertFalse("a slot off the edge has no drawable projection", off.hasProjection());
        assertEquals(0, result.countWithStatus(FeatureLocalisation.Status.MISSING));
        assertFalse("framing is not a defect in the part",
                result.hasPositionalDefect());
    }

    /**
     * ...but it is still reported. Dropping the row would turn a false FAIL into a false
     * PASS, which is the worse trade: the operator would read four verified positions and
     * no complaint, with nothing saying the fifth was never looked at.
     */
    @Test
    public void anUnverifiedPositionIsStillDeclared() {
        FeatureLayout layout = referenceLayout();
        List<RectF> rects = new ArrayList<>();
        List<String> types = new ArrayList<>();
        for (FeatureLayout.Slot s : layout.slots()) {
            if (s.id == 3) continue;
            rects.add(shot(s, 0, 1.15, 0, 0.05));
            types.add(s.type);
        }

        FeatureLocalisation.Result result =
                FeatureLocalisation.localise(layout, rects, types, null);

        assertTrue(result.hasUnverifiedPositions());
        assertEquals(1, result.countWithStatus(FeatureLocalisation.Status.OUT_OF_FRAME));
        assertTrue("the summary must admit the check was partial",
                result.summary().contains("outside the photo"));
    }

    /**
     * The regression guard for the above: an empty position the photo <em>does</em> cover is
     * still MISSING and still fails the part. The fix must not have made absence unreportable.
     */
    @Test
    public void anEmptyPositionInsideThePhotoIsStillMissing() {
        FeatureLayout layout = referenceLayout();
        Detections d = shotOf(layout, java.util.Collections.singleton(4), null);

        FeatureLocalisation.Result result =
                FeatureLocalisation.localise(layout, d.rects, d.types, null);

        assertEquals(FeatureLocalisation.Status.MISSING, findingFor(result, 4).status);
        assertEquals(0, result.countWithStatus(FeatureLocalisation.Status.OUT_OF_FRAME));
        assertFalse(result.hasUnverifiedPositions());
        assertTrue(result.hasPositionalDefect());
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

    /** Shifts, turns and rescales a point the way a freehand second photo would. */
    private static RectF shot(FeatureLayout.Slot s, double degrees, double scale,
                              double dx, double dy) {
        double a = Math.toRadians(degrees);
        return at((float) (scale * (s.x * Math.cos(a) - s.y * Math.sin(a)) + dx),
                (float) (scale * (s.x * Math.sin(a) + s.y * Math.cos(a)) + dy));
    }

    /**
     * The same defect must read the same however far back the operator stood.
     *
     * The matching tolerance is a fraction of the part, but it was computed from the layout's
     * spread and then applied to distances measured in the photo — so shooting the part at half
     * its reference size silently doubled the tolerance relative to the part. A fastener a
     * fifth of the way out of position landed inside it and was reported as correctly fitted,
     * which is the false-PASS direction. The tolerance is now scaled by the transform's own
     * scale, so the check means the same thing at every shooting distance.
     */
    @Test
    public void theCheckIsAsStrictWhenThePartIsShotSmaller() {
        FeatureLayout layout = referenceLayout();
        // Slot 4's fastener is fitted well out of position, measured on the part.
        float offX = 0.08f, offY = 0.04f;

        for (double scale : new double[]{1.0, 0.5}) {
            List<RectF> rects = new ArrayList<>();
            List<String> types = new ArrayList<>();
            for (FeatureLayout.Slot s : layout.slots()) {
                if (s.id == 4) continue;
                rects.add(shot(s, 0, scale, 0.1, 0.1));
                types.add(s.type);
            }
            FeatureLayout.Slot four = layout.slotById(4);
            rects.add(at((float) (scale * (four.x + offX) + 0.1),
                    (float) (scale * (four.y + offY) + 0.1)));
            types.add(four.type);

            FeatureLocalisation.Result result =
                    FeatureLocalisation.localise(layout, rects, types, null);

            assertTrue("must still register at scale " + scale, result.aligned);
            assertEquals("the misplaced fastener must not read as fitted at scale " + scale,
                    FeatureLocalisation.Status.DISPLACED, findingFor(result, 4).status);
            assertEquals("and nothing else may be disturbed at scale " + scale,
                    4, result.countWithStatus(FeatureLocalisation.Status.PRESENT));
        }
    }

    /**
     * The alignment must be more accurate than any single detection that produced it.
     *
     * The winning transform used to be the one derived exactly from the two detections that
     * seeded it, so it inherited their measurement error and carried it into every projection —
     * the ghost markers drawn over the photo and the displacement offsets alike. It is now
     * fitted by least squares over the whole agreed correspondence set, which averages that
     * error down instead of enshrining whichever two boxes happened to seed the winner.
     *
     * The jitter here is deterministic and cancels across the set, so this asserts a property
     * of the fit rather than the outcome of one lucky draw.
     */
    @Test
    public void theAlignmentIsFittedOverEveryAgreeingFeatureNotJustTheSeedPair() {
        FeatureLayout layout = referenceLayout();
        double angle = Math.toRadians(8), scale = 0.9, dx = 0.05, dy = 0.03;
        float jitter = 0.008f;
        float[][] nudge = {{jitter, 0}, {-jitter, 0}, {0, jitter}, {0, -jitter}, {0, 0}};

        List<RectF> rects = new ArrayList<>();
        List<String> types = new ArrayList<>();
        List<float[]> truth = new ArrayList<>();
        int i = 0;
        for (FeatureLayout.Slot s : layout.slots()) {
            float tx = (float) (scale * (s.x * Math.cos(angle) - s.y * Math.sin(angle)) + dx);
            float ty = (float) (scale * (s.x * Math.sin(angle) + s.y * Math.cos(angle)) + dy);
            truth.add(new float[]{tx, ty});
            rects.add(at(tx + nudge[i][0], ty + nudge[i][1]));
            types.add(s.type);
            i++;
        }

        FeatureLocalisation.Result result =
                FeatureLocalisation.localise(layout, rects, types, null);
        assertTrue(result.aligned);
        assertEquals(5, result.countWithStatus(FeatureLocalisation.Status.PRESENT));

        double worst = 0;
        for (int k = 0; k < result.findings.size(); k++) {
            FeatureLocalisation.SlotFinding f = result.findings.get(k);
            worst = Math.max(worst, Math.hypot(f.projectedX - truth.get(k)[0],
                    f.projectedY - truth.get(k)[1]));
        }
        assertTrue("the fit must beat the error of the detections it was fitted to, got " + worst,
                worst < jitter);
    }

    /**
     * A layout with many positions must register reliably.
     *
     * Seeds used to be 400 random draws from every slot paired with every detection, which on a
     * layout this size is sparse coverage of a large candidate space — so registration on a
     * perfectly good photo was partly a matter of luck, and the luck depended on a hardcoded
     * random seed. Seeding now enumerates the candidate space, with the distance network
     * discarding the implausible pairings for the cost of one division each.
     */
    @Test
    public void aLayoutWithManyPositionsStillRegisters() {
        List<RectF> rects = new ArrayList<>();
        List<String> types = new ArrayList<>();
        // Twelve fasteners around an irregular ring, so no rotation maps it onto itself.
        for (int k = 0; k < 12; k++) {
            double a = 2 * Math.PI * k / 12;
            double r = 0.30 + 0.06 * Math.sin(3 * a) + 0.03 * k / 12.0;
            rects.add(at((float) (0.5 + r * Math.cos(a)), (float) (0.5 + r * Math.sin(a))));
            types.add(k % 3 == 0 ? "screw" : "bolt_head");
        }
        FeatureLayout layout = FeatureLocalisation.layoutFrom(rects, types, CRITICAL);
        assertEquals(12, layout.slots().size());

        Detections d = shotOf(layout, new LinkedHashSet<>(Arrays.asList(3, 9)), null);
        FeatureLocalisation.Result result =
                FeatureLocalisation.localise(layout, d.rects, d.types, null);

        assertTrue(result.aligned);
        assertEquals(2, result.countWithStatus(FeatureLocalisation.Status.MISSING));
        assertEquals(FeatureLocalisation.Status.MISSING, findingFor(result, 3).status);
        assertEquals(FeatureLocalisation.Status.MISSING, findingFor(result, 9).status);
        assertEquals(10, result.countWithStatus(FeatureLocalisation.Status.PRESENT));
    }

    /**
     * Applies a tilted view of the part: rotate into the tilt axis, foreshorten across it by
     * {@code cos(tilt)}, rotate back, then rotate and translate freehand.
     *
     * This is what an off-axis camera does to a flat part, to first order. It is not a
     * similarity transform at any parameter setting — the pairwise separations no longer share
     * one common factor, which is exactly why growing a correspondence set on separations alone
     * cannot reach it.
     */
    private static float[] tilted(float x, float y, double tiltDeg, double axisDeg,
                                  double rollDeg, double scale, double dx, double dy) {
        double a = Math.toRadians(axisDeg);
        // Into the tilt frame.
        double u = x * Math.cos(a) + y * Math.sin(a);
        double v = -x * Math.sin(a) + y * Math.cos(a);
        v *= Math.cos(Math.toRadians(tiltDeg));
        // Back out of it.
        double px = u * Math.cos(a) - v * Math.sin(a);
        double py = u * Math.sin(a) + v * Math.cos(a);
        double r = Math.toRadians(rollDeg);
        return new float[]{
                (float) (scale * (px * Math.cos(r) - py * Math.sin(r)) + dx),
                (float) (scale * (px * Math.sin(r) + py * Math.cos(r)) + dy)};
    }

    /**
     * A part photographed off-axis must still register.
     *
     * The correspondence set used to be grown on the {@link DistanceNetwork} alone and then
     * fitted once. Separations are invariant to translation, rotation and <em>uniform</em>
     * scale, and to nothing else: a tilt foreshortens them by different factors depending on
     * their direction — measured at median x1.28 and up to x2.80 on real pairs. Growth runs at
     * half of a 12% tolerance, so a moderate tilt stalls it well before the inlier floor.
     *
     * Registration now iterates fit-reassign-refit and lets the affine model propose from the
     * three correspondences that determine it, rather than waiting for the four that make it
     * evidence. Swept against the shipped constants on this layout, the limit moves from
     * 30 degrees of tilt to 66 — where {@code MAX_ANISOTROPY} stops it, as
     * {@link #aTiltTooSteepToModelIsRefusedRatherThanForced} pins.
     *
     * 35 degrees costs 18% across the tilt axis, comfortably past the 6% growth budget, and is
     * where the old code first refused.
     */
    @Test
    public void aPartPhotographedOffAxisStillRegisters() {
        FeatureLayout layout = referenceLayout();
        List<RectF> rects = new ArrayList<>();
        List<String> types = new ArrayList<>();
        for (FeatureLayout.Slot s : layout.slots()) {
            float[] p = tilted(s.x, s.y, 35, 25, 10, 0.95, 0.06, 0.04);
            rects.add(at(p[0], p[1]));
            types.add(s.type);
        }

        FeatureLocalisation.Result result = FeatureLocalisation.localise(layout, rects, types, null);
        assertTrue("an off-axis shot of an intact part must still align", result.aligned);
        assertEquals(5, result.countWithStatus(FeatureLocalisation.Status.PRESENT));
    }

    /** The tilt tolerance must not cost the defect: the same shot with one fastener absent. */
    @Test
    public void anOffAxisShotStillNamesTheMissingPosition() {
        FeatureLayout layout = referenceLayout();
        List<RectF> rects = new ArrayList<>();
        List<String> types = new ArrayList<>();
        for (FeatureLayout.Slot s : layout.slots()) {
            if (s.id == 2) continue;
            float[] p = tilted(s.x, s.y, 30, 25, 10, 0.95, 0.06, 0.04);
            rects.add(at(p[0], p[1]));
            types.add(s.type);
        }

        FeatureLocalisation.Result result = FeatureLocalisation.localise(layout, rects, types, null);
        assertTrue(result.aligned);
        assertEquals(1, result.countWithStatus(FeatureLocalisation.Status.MISSING));
        assertEquals(FeatureLocalisation.Status.MISSING, findingFor(result, 2).status);
    }

    /**
     * Past the tilt an affine can model, the check refuses rather than forcing a fit.
     *
     * The extra freedom that absorbs foreshortening is also freedom to squash the layout onto a
     * line and collect inliers along it, so {@code MAX_ANISOTROPY} caps the stretch at x2.5 —
     * about 66 degrees. Beyond it the honest answer is no positional claim at all, and that is
     * what has to happen: a forced fit would name specific slots, confidently and wrongly.
     *
     * 75 degrees stretches by x3.86, comfortably past the cap.
     */
    @Test
    public void aTiltTooSteepToModelIsRefusedRatherThanForced() {
        FeatureLayout layout = referenceLayout();
        List<RectF> rects = new ArrayList<>();
        List<String> types = new ArrayList<>();
        for (FeatureLayout.Slot s : layout.slots()) {
            float[] p = tilted(s.x, s.y, 75, 25, 10, 0.95, 0.06, 0.04);
            rects.add(at(p[0], p[1]));
            types.add(s.type);
        }

        FeatureLocalisation.Result result = FeatureLocalisation.localise(layout, rects, types, null);
        assertFalse("a tilt past the anisotropy guard must not be forced into a fit",
                result.aligned);
        assertTrue("and no position may be claimed from it", result.findings.isEmpty());
    }

    /**
     * Proposing an affine from three correspondences must not manufacture an alignment.
     *
     * Three points determine an affine exactly, so any three detections can be interpolated by
     * one — which is precisely why the proposal is scored rather than believed. A photo of a
     * different arrangement has to fail on the inlier count regardless of how well some triple
     * inside it can be fitted.
     */
    @Test
    public void threePointsCannotTalkTheirWayIntoAnAlignment() {
        FeatureLayout layout = referenceLayout();
        // Five detections in a wholly different arrangement: a tight line, which no similarity
        // or affine view of the reference produces without collapsing it.
        List<RectF> rects = new ArrayList<>();
        List<String> types = new ArrayList<>();
        for (int k = 0; k < 5; k++) {
            rects.add(at(0.20f + 0.12f * k, 0.50f + 0.01f * k));
            types.add("bolt_head");
        }

        FeatureLocalisation.Result result = FeatureLocalisation.localise(layout, rects, types, null);
        assertFalse("a different arrangement must not register", result.aligned);
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

    // ── Size against the reference ───────────────────────────────────────────

    /**
     * The reference layout with a frame size, so every 0.04-fraction box in {@link #at} comes
     * to 38 px on its shorter side — above {@link FeatureLocalisation#MIN_SIZE_JUDGED_PX}, as a
     * labelled fastener at the working resolution is.
     */
    private static FeatureLayout sizedReferenceLayout() {
        FeatureLayout layout = referenceLayout();
        layout.setImageSize(1280, 960);
        return layout;
    }

    /** A detection centred on (x, y) whose box is {@code factor} times the standard one. */
    private static RectF atScaled(float x, float y, float factor) {
        float h = 0.02f * factor;
        return new RectF(x - h, y - h, x + h, y + h);
    }

    @Test
    public void everyPositionIsSizeMatchedOnAnIdenticalShot() {
        FeatureLayout layout = sizedReferenceLayout();
        Detections d = shotOf(layout, null, null);

        FeatureLocalisation.Result result = FeatureLocalisation.localise(layout, d.rects, d.types, null);
        assertTrue(result.aligned);
        assertEquals(5, result.countSizeJudged());
        assertEquals(0, result.countSizeChanged());
        for (FeatureLocalisation.SlotFinding f : result.findings) {
            assertTrue(f.sizeJudged());
            assertEquals(1f, f.sizeRatio, 0.02f);
            assertFalse(f.sizeChanged());
        }
        assertFalse("nothing to say about size when nothing differs",
                result.summary().contains("size"));
    }

    /**
     * A detection half again as wide and tall as the reference recorded is reported, with the
     * ratio, and named to its slot. It is a warning: the position is still PRESENT and the
     * part does not fail on it, because the check's noise floor is the detector's own jitter.
     */
    @Test
    public void aSwollenDetectionIsReportedAsASizeChangeWithoutFailingThePart() {
        FeatureLayout layout = sizedReferenceLayout();
        Detections d = shotOf(layout, null, null);
        FeatureLayout.Slot two = layout.slotById(2);
        d.rects.set(1, atScaled(two.x, two.y, 2f));

        FeatureLocalisation.Result result = FeatureLocalisation.localise(layout, d.rects, d.types, null);
        assertTrue(result.aligned);
        FeatureLocalisation.SlotFinding f = findingFor(result, 2);
        assertNotNull(f);
        assertEquals(FeatureLocalisation.Status.PRESENT, f.status);
        assertTrue(f.sizeChanged());
        assertEquals(2f, f.sizeRatio, 0.1f);
        assertEquals(100f, f.sizeChangePercent(), 10f);

        assertEquals(1, result.countSizeChanged());
        assertTrue(result.summary().contains("1 differing in size"));
        assertFalse("size is reported, never failed on", result.hasPositionalDefect());
        for (FeatureLocalisation.SlotFinding other : result.findings) {
            if (other.slotId != 2) assertFalse(other.sizeChanged());
        }
    }

    /**
     * The tolerance is symmetric in log space: a box at 0.6x the reference is as far out as one
     * at 1.67x and is flagged, while 0.8x sits inside 1/1.5 and is not. A plain percentage test
     * would call 0.6x a 40% change and let it through.
     */
    @Test
    public void shrinkageIsJudgedOnTheSameScaleAsGrowth() {
        FeatureLayout layout = sizedReferenceLayout();
        Detections d = shotOf(layout, null, null);
        FeatureLayout.Slot one = layout.slotById(1);
        FeatureLayout.Slot three = layout.slotById(3);
        d.rects.set(0, atScaled(one.x, one.y, 0.6f));
        d.rects.set(2, atScaled(three.x, three.y, 0.8f));

        FeatureLocalisation.Result result = FeatureLocalisation.localise(layout, d.rects, d.types, null);
        assertTrue(result.aligned);
        assertTrue("0.6x is beyond 1/1.5", findingFor(result, 1).sizeChanged());
        assertEquals(0.6f, findingFor(result, 1).sizeRatio, 0.05f);
        assertFalse("0.8x is inside 1/1.5", findingFor(result, 3).sizeChanged());
        assertEquals(1, result.countSizeChanged());
    }

    /**
     * A part photographed closer than the reference has every box larger by the same factor.
     * That is framing, not hardware, and it is exactly what the alignment's scale divides
     * out: a raw pixel comparison would flag all five positions here.
     */
    @Test
    public void aCloserShotIsNotASizeChange() {
        FeatureLayout layout = sizedReferenceLayout();
        float zoom = 1.3f;
        List<RectF> rects = new ArrayList<>();
        List<String> types = new ArrayList<>();
        for (FeatureLayout.Slot s : layout.slots()) {
            // Positions spread out about the frame centre by the same factor the boxes grow.
            float x = 0.5f + (s.x - 0.5f) * zoom;
            float y = 0.5f + (s.y - 0.5f) * zoom;
            rects.add(atScaled(x, y, zoom));
            types.add(s.type);
        }

        FeatureLocalisation.Result result = FeatureLocalisation.localise(layout, rects, types, null);
        assertTrue(result.aligned);
        assertEquals(5, result.countSizeJudged());
        assertEquals(0, result.countSizeChanged());
        for (FeatureLocalisation.SlotFinding f : result.findings) {
            assertEquals("scale must be divided out", 1f, f.sizeRatio, 0.03f);
        }
    }

    /**
     * And a genuinely oversized fastener on that closer shot is still caught once the framing
     * is divided out — the correction must not also absorb the defect.
     */
    @Test
    public void aSizeChangeSurvivesTheFramingCorrection() {
        FeatureLayout layout = sizedReferenceLayout();
        float zoom = 0.8f;
        List<RectF> rects = new ArrayList<>();
        List<String> types = new ArrayList<>();
        for (FeatureLayout.Slot s : layout.slots()) {
            float x = 0.5f + (s.x - 0.5f) * zoom;
            float y = 0.5f + (s.y - 0.5f) * zoom;
            rects.add(atScaled(x, y, s.id == 4 ? zoom * 1.8f : zoom));
            types.add(s.type);
        }

        FeatureLocalisation.Result result = FeatureLocalisation.localise(layout, rects, types, null);
        assertTrue(result.aligned);
        assertEquals(1, result.countSizeChanged());
        assertTrue(findingFor(result, 4).sizeChanged());
        assertEquals(1.8f, findingFor(result, 4).sizeRatio, 0.1f);
    }

    /**
     * A layout labelled before box sizes were recorded has none to compare against. Such
     * slots are positioned as before and simply not size-judged — NaN, not 1.0, so an old
     * reference cannot read as "every size verified".
     */
    @Test
    public void aLayoutWithoutRecordedSizesMakesNoSizeClaim() {
        FeatureLayout layout = new FeatureLayout();
        layout.setImageSize(1280, 960);
        for (FeatureLayout.Slot s : referenceLayout().slots()) layout.addSlot(s.type, s.x, s.y);
        Detections d = shotOf(layout, null, null);

        FeatureLocalisation.Result result = FeatureLocalisation.localise(layout, d.rects, d.types, null);
        assertTrue(result.aligned);
        assertEquals(5, result.countWithStatus(FeatureLocalisation.Status.PRESENT));
        assertEquals(0, result.countSizeJudged());
        for (FeatureLocalisation.SlotFinding f : result.findings) {
            assertFalse(f.sizeJudged());
            assertTrue(Float.isNaN(f.sizeRatio));
            assertTrue(Float.isNaN(f.sizeChangePercent()));
        }
    }

    /**
     * Below {@link FeatureLocalisation#MIN_SIZE_JUDGED_PX} the detector's boxes do not repeat
     * between shots, so a slot that small is refused rather than judged. The same fractional
     * boxes are 38 px on a 1280x960 frame and 18 px on a 600x450 one; only the first is
     * judged. A layout that cannot say what frame it was measured in is refused too.
     */
    @Test
    public void slotsTooSmallToRepeatAreNotJudged() {
        FeatureLayout small = referenceLayout();
        small.setImageSize(600, 450);
        Detections d = shotOf(small, null, null);
        FeatureLocalisation.Result result = FeatureLocalisation.localise(small, d.rects, d.types, null);
        assertTrue(result.aligned);
        assertEquals("18 px boxes must not be judged", 0, result.countSizeJudged());

        FeatureLayout unknownFrame = referenceLayout();
        result = FeatureLocalisation.localise(unknownFrame, d.rects, d.types, null);
        assertTrue(result.aligned);
        assertEquals("no frame size, no pixel floor, no claim", 0, result.countSizeJudged());

        result = FeatureLocalisation.localise(sizedReferenceLayout(), d.rects, d.types, null);
        assertEquals(5, result.countSizeJudged());
    }

    /**
     * An empty slot has nothing to size. A displaced one does — the stray detection offered as
     * the fastener is compared like any other, so "fitted off its position" and "fitted off its
     * position and the wrong size" stay distinguishable.
     *
     * Two shots rather than one: a five-slot layout with one position empty and another
     * displaced leaves three inliers, under the affine model's floor of four, so it would not
     * register at all — see {@link #aFiveSlotLayoutMissingTwoPositionsCannotAffordTheAffineModel}.
     */
    @Test
    public void onlyPositionsWithADetectionAreSizeJudged() {
        FeatureLayout layout = sizedReferenceLayout();

        Detections empty = shotOf(layout, java.util.Collections.singleton(3), null);
        FeatureLocalisation.Result result =
                FeatureLocalisation.localise(layout, empty.rects, empty.types, null);
        assertTrue(result.aligned);
        FeatureLocalisation.SlotFinding missing = findingFor(result, 3);
        assertEquals(FeatureLocalisation.Status.MISSING, missing.status);
        assertFalse(missing.sizeJudged());
        assertEquals(4, result.countSizeJudged());

        // Slot 4's fastener sits well outside matching tolerance, and is doubled in size.
        Detections displaced = shotOf(layout, java.util.Collections.singleton(4), null);
        FeatureLayout.Slot four = layout.slotById(4);
        displaced.rects.add(atScaled(four.x - 0.10f, four.y - 0.10f, 2f));
        displaced.types.add("bolt_head");
        result = FeatureLocalisation.localise(layout, displaced.rects, displaced.types, null);
        assertTrue(result.aligned);
        FeatureLocalisation.SlotFinding off = findingFor(result, 4);
        assertEquals(FeatureLocalisation.Status.DISPLACED, off.status);
        assertTrue(off.sizeJudged());
        assertTrue(off.sizeChanged());
        assertEquals(2f, off.sizeRatio, 0.1f);
    }
}
