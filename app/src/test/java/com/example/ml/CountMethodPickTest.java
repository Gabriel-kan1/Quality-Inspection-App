package com.example.ml;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Pins how a method is chosen from the operator's feature count.
 *
 * <p>Running the methods needs native OpenCV and cannot happen here; what can is the part
 * with a judgement in it. Given six candidate counts and a target, which one wins, and what
 * happens on a tie, decides whether "there are four on this part" picks a method that is
 * right elsewhere or one that happened to land on four by accident.
 */
@RunWith(RobolectricTestRunner.class)
public class CountMethodPickTest {

    private static EdgeMethodLab.CountMatch m(String id, int candidates, int expected) {
        return new EdgeMethodLab.CountMatch(id, id, candidates, Math.abs(candidates - expected));
    }

    @Test
    public void theClosestCountWins() {
        // stud1.jpg on the device: shipped 42, v2 4, v3 4, v4 4, v5 4, v6 4. Target 4.
        List<EdgeMethodLab.CountMatch> ranked = EdgeMethodLab.rank(Arrays.asList(
                m(EdgeMethodLab.SHIPPED_ID, 42, 4), m("hybrid-v2", 4, 4), m("hybrid-v3", 4, 4),
                m("hybrid-v4", 4, 4), m("hybrid-v5", 4, 4), m("hybrid-v6", 4, 4)));
        assertEquals(0, ranked.get(0).distance);
        assertEquals("the shipped stack, 38 off, comes last",
                EdgeMethodLab.SHIPPED_ID, ranked.get(ranked.size() - 1).id);
    }

    @Test
    public void tiesGoToTheMethodMostAccurateAgainstGroundTruth() {
        // Five methods all exact: the one that wins must be the one most likely to be right
        // on the photographs this one was not tuned on, not whichever happens to be listed
        // first. v6 was 18/26 exact on the device; v5 was 11/26.
        List<EdgeMethodLab.CountMatch> ranked = EdgeMethodLab.rank(Arrays.asList(
                m("hybrid-v5", 4, 4), m("hybrid-v2", 4, 4), m("hybrid-v4", 4, 4),
                m("hybrid-v3", 4, 4), m("hybrid-v6", 4, 4)));
        assertEquals("hybrid-v6", ranked.get(0).id);
        assertEquals("hybrid-v5", ranked.get(ranked.size() - 1).id);
    }

    @Test
    public void aNearMissBeatsAnExactTieLoserOnlyIfItIsCloser() {
        // three.jpg on the device: shipped 12, v2 16, v3 19, v4 10, v5 20, v6 11. Target 3.
        // Nothing is exact; v4 at 10 is the least wrong and must win even though v6 outranks
        // it on the tie order — the tie order only decides between equal distances.
        List<EdgeMethodLab.CountMatch> ranked = EdgeMethodLab.rank(Arrays.asList(
                m(EdgeMethodLab.SHIPPED_ID, 12, 3), m("hybrid-v2", 16, 3), m("hybrid-v3", 19, 3),
                m("hybrid-v4", 10, 3), m("hybrid-v5", 20, 3), m("hybrid-v6", 11, 3)));
        assertEquals("hybrid-v4", ranked.get(0).id);
        assertEquals(7, ranked.get(0).distance);
        assertEquals("hybrid-v6", ranked.get(1).id);
    }

    @Test
    public void undershootAndOvershootAreJudgedByDistanceAlone() {
        // 2 and 6 are both two away from 4; the tie order decides, not the sign.
        List<EdgeMethodLab.CountMatch> ranked = EdgeMethodLab.rank(Arrays.asList(
                m("hybrid-v2", 6, 4), m("hybrid-v6", 2, 4)));
        assertEquals(2, ranked.get(0).distance);
        assertEquals(2, ranked.get(1).distance);
        assertEquals("hybrid-v6", ranked.get(0).id);
    }

    @Test
    public void theChoosableSetIsTheShippedStackAndTheHybridsOnly() {
        // Scharr found 215 candidates in a photograph of four. A bare operator must never be
        // electable by a lucky count.
        Set<String> ids = new HashSet<>(Arrays.asList(EdgeMethodLab.HYBRID_IDS));
        assertTrue(ids.contains(EdgeMethodLab.SHIPPED_ID));
        for (String id : ids) {
            assertTrue("not a hybrid: " + id,
                    id.equals(EdgeMethodLab.SHIPPED_ID) || id.startsWith("hybrid-v"));
            assertTrue("choosable id must exist in the catalogue: " + id,
                    EdgeMethodLab.byId(id) != null);
        }
        assertEquals("shipped plus v2..v6", 6, ids.size());
    }

    @Test
    public void everyChoosableMethodHasATiePosition() {
        // A method missing from TIE_ORDER would sort last on every tie, silently.
        List<String> order = Arrays.asList(EdgeMethodLab.TIE_ORDER);
        for (String id : EdgeMethodLab.HYBRID_IDS) {
            assertTrue(id + " has no tie position", order.contains(id));
        }
        assertEquals(new HashSet<>(order).size(), order.size());
    }

    @Test
    public void rankingDoesNotMutateItsInput() {
        List<EdgeMethodLab.CountMatch> input = new ArrayList<>(Arrays.asList(
                m("hybrid-v2", 9, 4), m("hybrid-v6", 4, 4)));
        EdgeMethodLab.rank(input);
        assertEquals("hybrid-v2", input.get(0).id);
    }
}
