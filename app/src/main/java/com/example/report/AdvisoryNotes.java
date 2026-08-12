package com.example.report;

import java.util.List;
import java.util.Locale;

/**
 * Turns an advisory feature type's count into the one-line trend note that appears in a
 * report's warning section — "Scratches appear more often than previous statistics".
 *
 * Advisory types are the cosmetic middle of {@code FeatureTypeRegistry.Severity}: too
 * incidental to count in the quantitative table and fail a part on, too meaningful to
 * ignore. What matters about them is not the absolute number but whether this run is out
 * of line with how the same part usually comes out, so every note is relative to that
 * part's own history.
 *
 * Pure arithmetic over a list of past counts, with no Android or database dependency, so
 * the thresholds can be exercised directly in unit tests.
 */
public final class AdvisoryNotes {

    /**
     * Prior inspections needed before a baseline is stated at all. Two points give a mean
     * with no usable spread — the first slightly-above-average run would read as a trend.
     * Three is the smallest history where the standard deviation below carries information.
     */
    public static final int MIN_HISTORY = 3;

    /**
     * How many past inspections feed the baseline. Bounded so a part inspected for months
     * is compared against how it currently runs rather than against its whole lifetime: a
     * process change would otherwise stay buried under years of older counts.
     */
    public static final int HISTORY_WINDOW = 10;

    /**
     * How far above the mean a count must sit to be called out, in standard deviations.
     * One sigma is deliberately loose. The cost of a false alarm here is a line of text in
     * a notes section; the cost of silence is the failure mode this whole section exists to
     * catch, so the check is tuned to notice rather than to be sure.
     */
    public static final double SIGMA = 1.0;

    /**
     * Floor on that gap, in units. Without it a part that reliably shows exactly two
     * scratches has zero deviation, so zero excess clears the bar and every run at or above
     * the mean reads as a spike. One whole extra occurrence is the smallest excess that
     * means anything when the thing being counted is discrete.
     */
    public static final double MIN_ABSOLUTE_GAP = 1.0;

    /** What a note is telling the operator, so the UI can style it without parsing the text. */
    public enum Level {
        /** Count is above the part's baseline — the case the section exists for. */
        ELEVATED,
        /** Count is within the usual range. Recorded, not alarming. */
        NORMAL,
        /** Not enough prior inspections of this part to say what usual looks like. */
        NO_BASELINE
    }

    public static final class Note {
        public final String featureType;
        public final int detectedCount;
        public final Level level;
        public final String text;
        /** Mean of the prior counts, or -1 when {@link #level} is {@link Level#NO_BASELINE}. */
        public final double baselineMean;
        /** How many prior inspections the baseline was drawn from. */
        public final int historyCount;

        Note(String featureType, int detectedCount, Level level, String text,
             double baselineMean, int historyCount) {
            this.featureType = featureType;
            this.detectedCount = detectedCount;
            this.level = level;
            this.text = text;
            this.baselineMean = baselineMean;
            this.historyCount = historyCount;
        }
    }

    private AdvisoryNotes() {}

    /**
     * Builds the note for one advisory type.
     *
     * @param displayName     type name as the operator sees it, e.g. "Scratches"
     * @param detectedCount   how many were found on this inspection
     * @param historicalCounts detected counts from earlier inspections of the same part,
     *                        newest first; may be null or short
     */
    public static Note forType(String displayName, int detectedCount, List<Integer> historicalCounts) {
        String name = (displayName == null || displayName.trim().isEmpty()) ? "This feature" : displayName.trim();

        int n = historicalCounts == null ? 0 : historicalCounts.size();
        if (n < MIN_HISTORY) {
            return new Note(name, detectedCount, Level.NO_BASELINE,
                    String.format(Locale.getDefault(),
                            "%s: %d detected. No baseline yet — %d of %d prior inspections of this"
                                    + " part recorded, so there is nothing to compare against.",
                            name, detectedCount, n, MIN_HISTORY),
                    -1, n);
        }

        int window = Math.min(n, HISTORY_WINDOW);
        double sum = 0;
        for (int i = 0; i < window; i++) {
            Integer v = historicalCounts.get(i);
            sum += v == null ? 0 : v;
        }
        double mean = sum / window;

        // Sample (n-1) rather than population standard deviation: these counts are a sample
        // of how the part runs, not the whole of it, and the population form understates the
        // spread on the handful of inspections a real baseline is built from.
        double sqDiff = 0;
        for (int i = 0; i < window; i++) {
            Integer v = historicalCounts.get(i);
            double d = (v == null ? 0 : v) - mean;
            sqDiff += d * d;
        }
        double stdDev = window > 1 ? Math.sqrt(sqDiff / (window - 1)) : 0;

        double gap = Math.max(MIN_ABSOLUTE_GAP, SIGMA * stdDev);
        boolean elevated = detectedCount - mean >= gap;

        if (elevated) {
            return new Note(name, detectedCount, Level.ELEVATED,
                    String.format(Locale.getDefault(),
                            "%s appear more often than previous statistics — %d detected against a"
                                    + " baseline of %.1f (±%.1f) across the last %d inspections of this part.",
                            name, detectedCount, mean, stdDev, window),
                    mean, window);
        }

        return new Note(name, detectedCount, Level.NORMAL,
                String.format(Locale.getDefault(),
                        "%s: %d detected, in line with the baseline of %.1f (±%.1f) across the"
                                + " last %d inspections of this part.",
                        name, detectedCount, mean, stdDev, window),
                mean, window);
    }
}
