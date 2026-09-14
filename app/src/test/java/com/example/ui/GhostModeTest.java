package com.example.ui;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import java.util.HashSet;
import java.util.Set;

/**
 * Pins the ghost button's cycle. The rendering itself is a native edge stage and a colour
 * filter on an ImageView, verified by screenshot; what can go wrong in plain Java is the
 * order, a dead end, or two states that read the same on screen.
 */
public class GhostModeTest {

    @Test
    public void outlineIsTheDefaultAndTheCycleIsOutlineTintedPlainOff() {
        // The outline is the mode that exists because the photograph occluded the live view,
        // so it is where every inspection starts. The two photograph renderings follow, the
        // tinted one first because it is the more legible of the two, and off comes last.
        assertEquals(GhostMode.OUTLINE, GhostMode.values()[0]);
        assertEquals(GhostMode.TINTED, GhostMode.OUTLINE.next());
        assertEquals(GhostMode.PLAIN, GhostMode.TINTED.next());
        assertEquals(GhostMode.OFF, GhostMode.PLAIN.next());
    }

    @Test
    public void theButtonNeverDeadEnds() {
        // A cycle that stopped at OFF would leave no way back to the outline but restarting
        // the screen.
        assertEquals(GhostMode.OUTLINE, GhostMode.OFF.next());
        GhostMode m = GhostMode.OUTLINE;
        for (int i = 0; i < GhostMode.values().length; i++) m = m.next();
        assertEquals("a full lap returns to the start", GhostMode.OUTLINE, m);
    }

    @Test
    public void onlyOffHidesTheOverlay() {
        assertTrue(GhostMode.OUTLINE.showsOverlay());
        assertTrue(GhostMode.TINTED.showsOverlay());
        assertTrue(GhostMode.PLAIN.showsOverlay());
        assertFalse(GhostMode.OFF.showsOverlay());
    }

    @Test
    public void everyModeReadsDifferentlyOnTheButtonAndTheChip() {
        Set<String> buttons = new HashSet<>();
        Set<String> chips = new HashSet<>();
        for (GhostMode m : GhostMode.values()) {
            assertTrue(m + " button label names the mode", m.buttonLabel().contains(m.name()));
            assertTrue(m + " status label names the mode", m.statusLabel().contains(m.name()));
            assertTrue("button label is unique: " + m.buttonLabel(), buttons.add(m.buttonLabel()));
            assertTrue("status label is unique: " + m.statusLabel(), chips.add(m.statusLabel()));
            assertNotEquals("button and chip say different things", m.buttonLabel(), m.statusLabel());
        }
    }
}
