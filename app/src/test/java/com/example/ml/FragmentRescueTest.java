package com.example.ml;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.opencv.core.Rect;
import org.robolectric.RobolectricTestRunner;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Covers the rule that lets a fastener's own fragments survive the dynamic area floor.
 *
 * <p>The defect it exists to fix: {@code findFeatureBoxes} applies the area filter
 * <em>before</em> {@code mergeNearbyBoxes}, so a feature whose outline Canny broke into
 * pieces has those pieces judged individually against a floor sized for whole fasteners and
 * discarded — even where the merge's own guards would have rejoined them. Measured on
 * {@code screws_none.jpg}: two identical studs boxed 68x68 and 44x49, because the second's
 * pin fragmented into contours of area 366 and 356 against a floor of 468.
 *
 * <p>{@code org.opencv.core.Rect} is an ordinary Java class and {@code assembleBoxes} is
 * arithmetic on it, with no native call anywhere — Robolectric is needed only because the
 * enclosing class's static initialiser logs through {@code android.util.Log} when the OpenCV
 * natives fail to load, which they always do off-device. The stages either side of this one —
 * {@code findContours} and the contour-area measurement — cannot be tested here at all, which
 * is why {@code python/fragment_rescue_study.py} carries the end-to-end numbers.
 */
@RunWith(RobolectricTestRunner.class)
public class FragmentRescueTest {

    private static List<Rect> boxes(Rect... rects) {
        return new ArrayList<>(Arrays.asList(rects));
    }

    private static List<Rect> none() {
        return new ArrayList<>();
    }

    /** The measured case, at its real coordinates and sizes. */
    @Test
    public void theStudWhosePinFellUnderTheFloorIsMadeWholeAgain() {
        Rect collar = new Rect(496, 391, 44, 49);        // contour area 856, above the floor
        Rect pinA = new Rect(519, 421, 44, 40);          // area 366, below it
        Rect pinB = new Rect(519, 418, 23, 22);          // area 356, below it

        List<Rect> out = HardwareFeatureExtractor.assembleBoxes(boxes(collar), boxes(pinA, pinB));

        assertEquals("the stud is still one detection, not three", 1, out.size());
        Rect r = out.get(0);
        // Its twin in the same photograph measures 68x68; before the fix this was 44x49.
        assertEquals(496, r.x);
        assertEquals(391, r.y);
        assertEquals(67, r.width);
        assertEquals(70, r.height);
    }

    /**
     * The property that makes the change safe against the count ground truth: a rescued
     * fragment can grow a detection but can never become one. Without it, every sub-floor
     * speck near nothing in particular would become a box.
     */
    @Test
    public void fragmentsAloneNeverBecomeADetection() {
        List<Rect> specks = boxes(new Rect(10, 10, 6, 6), new Rect(14, 12, 5, 5));
        assertTrue("no anchor, no detection",
                HardwareFeatureExtractor.assembleBoxes(none(), specks).isEmpty());

        // Even a whole chain of them, each touching the next, stays dropped — which is the
        // failure that applying the area floor AFTER the merge would have introduced.
        List<Rect> chain = new ArrayList<>();
        for (int i = 0; i < 12; i++) chain.add(new Rect(100 + i * 8, 100, 7, 7));
        assertTrue(HardwareFeatureExtractor.assembleBoxes(none(), chain).isEmpty());
    }

    /** A fragment out on its own is noise, and is left where the area floor put it. */
    @Test
    public void aDistantFragmentIsNotRescued() {
        Rect feature = new Rect(100, 100, 50, 50);
        Rect far = new Rect(400, 400, 10, 10);

        List<Rect> out = HardwareFeatureExtractor.assembleBoxes(boxes(feature), boxes(far));
        assertEquals(1, out.size());
        assertEquals("the feature is untouched", feature.width, out.get(0).width);
        assertEquals(feature.height, out.get(0).height);
    }

    /**
     * Rescue offers a fragment to the merge; it does not force the merge to take it. The
     * fill-ratio guard still decides, so a fragment that would balloon the bounding box is
     * still refused — the protection that stops a box chaining outward.
     */
    @Test
    public void theMergeStillRefusesAFragmentThatWouldBallonTheBox() {
        Rect feature = new Rect(100, 100, 20, 20);
        // Touching (gap 0) but diagonally placed, so the union is far larger than the content.
        Rect sliver = new Rect(120, 120, 18, 18);

        List<Rect> out = HardwareFeatureExtractor.assembleBoxes(boxes(feature), boxes(sliver));
        assertEquals(1, out.size());
        assertEquals("content 400+324 against a 38x38 union is 0.50, below the 0.7 guard",
                20, out.get(0).width);
    }

    /** With nothing below the floor, the stage behaves exactly as it did before. */
    @Test
    public void withNoFragmentsTheResultIsUnchanged() {
        List<Rect> sized = boxes(new Rect(0, 0, 40, 40), new Rect(200, 200, 30, 30));
        List<Rect> out = HardwareFeatureExtractor.assembleBoxes(sized, none());
        assertEquals(2, out.size());
        List<Rect> alsoOut = HardwareFeatureExtractor.assembleBoxes(sized, null);
        assertEquals(2, alsoOut.size());
    }

    /** Two real features that touch still merge, and the result is still anchored. */
    @Test
    public void twoAnchoredBoxesStillMerge() {
        List<Rect> out = HardwareFeatureExtractor.assembleBoxes(
                boxes(new Rect(0, 0, 40, 40), new Rect(42, 0, 40, 40)), none());
        assertEquals(1, out.size());
        assertEquals(82, out.get(0).width);
    }

    // ── The adjacency predicate the rescue and the merge share ───────────────

    @Test
    public void adjacencyIsTheMergeStagesOwnTest() {
        Rect a = new Rect(0, 0, 10, 10);
        assertTrue("overlapping", HardwareFeatureExtractor.adjacent(a, new Rect(5, 5, 10, 10), 10));
        assertTrue("touching", HardwareFeatureExtractor.adjacent(a, new Rect(10, 0, 10, 10), 10));
        assertTrue("within the threshold", HardwareFeatureExtractor.adjacent(a, new Rect(20, 0, 10, 10), 10));
        assertFalse("beyond it", HardwareFeatureExtractor.adjacent(a, new Rect(21, 0, 10, 10), 10));
        // Both axes have to be within the threshold, not just one.
        assertFalse(HardwareFeatureExtractor.adjacent(a, new Rect(0, 40, 10, 10), 10));
        assertTrue("symmetric", HardwareFeatureExtractor.adjacent(new Rect(20, 0, 10, 10), a, 10));
    }
}
