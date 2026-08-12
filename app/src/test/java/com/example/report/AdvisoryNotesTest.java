package com.example.report;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Covers the thresholds that decide whether an advisory type earns a line in the report's
 * warning notes. The interesting cases are all about small histories, which is what a real
 * part on a shop floor actually has.
 */
public class AdvisoryNotesTest {

    private static List<Integer> counts(Integer... values) {
        return Arrays.asList(values);
    }

    @Test
    public void reportsNoBaselineUntilEnoughPriorInspections() {
        AdvisoryNotes.Note note = AdvisoryNotes.forType("Scratches", 9, counts(1, 1));
        assertEquals(AdvisoryNotes.Level.NO_BASELINE, note.level);
        assertTrue(note.text.contains("No baseline yet"));
    }

    @Test
    public void treatsNullHistoryAsNoBaseline() {
        AdvisoryNotes.Note note = AdvisoryNotes.forType("Scratches", 3, null);
        assertEquals(AdvisoryNotes.Level.NO_BASELINE, note.level);
        assertEquals(0, note.historyCount);
    }

    @Test
    public void flagsACountWellAboveTheBaseline() {
        AdvisoryNotes.Note note = AdvisoryNotes.forType("Scratches", 8, counts(2, 1, 2, 3));
        assertEquals(AdvisoryNotes.Level.ELEVATED, note.level);
        assertTrue(note.text.contains("appear more often than previous statistics"));
        assertEquals(2.0, note.baselineMean, 0.001);
    }

    @Test
    public void staysQuietWhenTheCountIsTypical() {
        AdvisoryNotes.Note note = AdvisoryNotes.forType("Scratches", 2, counts(2, 1, 2, 3));
        assertEquals(AdvisoryNotes.Level.NORMAL, note.level);
    }

    /**
     * The reason MIN_ABSOLUTE_GAP exists. A part that shows exactly two scratches every time
     * has zero deviation, so any sigma multiple of it is also zero and *any* excess at all
     * would clear the bar. The floor makes the requirement one whole extra occurrence:
     * matching the baseline is quiet, exceeding it by one is the note.
     */
    @Test
    public void aZeroVarianceBaselineStillNeedsOneWholeExtraOccurrence() {
        List<Integer> rockSteady = counts(2, 2, 2, 2);
        assertEquals(AdvisoryNotes.Level.NORMAL, AdvisoryNotes.forType("Scratches", 2, rockSteady).level);
        assertEquals(AdvisoryNotes.Level.ELEVATED, AdvisoryNotes.forType("Scratches", 3, rockSteady).level);
    }

    /**
     * The same floor applied to a fractional mean: 3 against a baseline of 2.5 is half an
     * occurrence above average and must not be called out, whereas 4 is a full one.
     */
    @Test
    public void aFractionalExcessOverAStableBaselineIsNotASpike() {
        List<Integer> history = counts(2, 3, 2, 3);
        assertEquals(AdvisoryNotes.Level.NORMAL, AdvisoryNotes.forType("Scratches", 3, history).level);
        assertEquals(AdvisoryNotes.Level.ELEVATED, AdvisoryNotes.forType("Scratches", 4, history).level);
    }

    /** A noisy history should absorb a wider swing before anything is called out. */
    @Test
    public void aWideBaselineAbsorbsAWiderSwing() {
        List<Integer> noisy = counts(0, 6, 1, 5, 2);
        assertEquals(AdvisoryNotes.Level.NORMAL, AdvisoryNotes.forType("Rust", 5, noisy).level);
        assertEquals(AdvisoryNotes.Level.ELEVATED, AdvisoryNotes.forType("Rust", 9, noisy).level);
    }

    @Test
    public void baselineIsCappedToTheRecentWindow() {
        // Twelve stable inspections then nothing else: only HISTORY_WINDOW of them count,
        // so historyCount must report the window, not the full list length.
        List<Integer> long12 = Collections.nCopies(12, 3);
        AdvisoryNotes.Note note = AdvisoryNotes.forType("Scratches", 3, long12);
        assertEquals(AdvisoryNotes.HISTORY_WINDOW, note.historyCount);
    }

    @Test
    public void fallsBackToAGenericNameWhenNoneIsSupplied() {
        AdvisoryNotes.Note note = AdvisoryNotes.forType("  ", 1, counts(0, 0, 0));
        assertTrue(note.text.startsWith("This feature"));
    }
}
