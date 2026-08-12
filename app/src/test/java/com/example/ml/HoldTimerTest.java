package com.example.ml;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/** Covers the dwell semantics the alignment traffic light depends on. */
public class HoldTimerTest {

    @Test
    public void doesNotPassBeforeTheFullThreeSeconds() {
        HoldTimer timer = new HoldTimer();
        timer.update(true, 0L);
        assertEquals(0.5f, timer.update(true, 1500L), 0.01f);
        assertFalse(timer.isSatisfied());
    }

    @Test
    public void passesOnceThreeSecondsOfAlignmentHaveElapsed() {
        HoldTimer timer = new HoldTimer();
        timer.update(true, 0L);
        assertEquals(1f, timer.update(true, 3000L), 0.001f);
        assertTrue(timer.isSatisfied());
        assertEquals(0L, timer.remainingMs());
    }

    @Test
    public void singleDroppedFrameInsideGraceDoesNotRestartTheCount() {
        HoldTimer timer = new HoldTimer();
        timer.update(true, 0L);
        timer.update(true, 2000L);
        // One bad frame 100ms later, well inside the 300ms grace.
        timer.update(false, 2100L);
        assertEquals(1f, timer.update(true, 3000L), 0.001f);
        assertTrue(timer.isSatisfied());
    }

    @Test
    public void losingAlignmentForLongerThanGraceRestartsTheCount() {
        HoldTimer timer = new HoldTimer();
        timer.update(true, 0L);
        timer.update(true, 2000L);
        assertEquals(0f, timer.update(false, 2400L), 0.001f);
        assertFalse(timer.isSatisfied());

        // The next good frame starts a fresh dwell rather than resuming the old one.
        timer.update(true, 2500L);
        assertEquals(0.5f, timer.update(true, 4000L), 0.01f);
        assertFalse(timer.isSatisfied());
    }

    @Test
    public void timeSpentInsideTheGraceWindowIsNotCreditedTowardTheHold() {
        HoldTimer timer = new HoldTimer();
        timer.update(true, 0L);
        timer.update(true, 2900L);
        // 200ms of dropout must not tip the timer over the line on its own.
        assertFalse(timer.update(false, 3100L) >= 1f);
        assertFalse(timer.isSatisfied());
    }

    @Test
    public void resetClearsASatisfiedTimer() {
        HoldTimer timer = new HoldTimer();
        timer.update(true, 0L);
        timer.update(true, 3000L);
        assertTrue(timer.isSatisfied());

        timer.reset();
        assertFalse(timer.isSatisfied());
        assertEquals(HoldTimer.DEFAULT_HOLD_MS, timer.remainingMs());
        assertEquals(0f, timer.update(true, 3100L), 0.001f);
    }

    @Test
    public void misalignedFramesAloneNeverStartTheCount() {
        HoldTimer timer = new HoldTimer();
        assertEquals(0f, timer.update(false, 0L), 0.001f);
        assertEquals(0f, timer.update(false, 5000L), 0.001f);
        assertFalse(timer.isSatisfied());
    }
}
