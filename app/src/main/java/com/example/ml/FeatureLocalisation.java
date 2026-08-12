package com.example.ml;

import android.graphics.RectF;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Turns a {@link LayoutMatcher} alignment into per-position findings for critical features:
 * not just <em>how many</em> bolt heads were found, but which expected one is missing and
 * where on the part it should have been.
 *
 * <h3>Why this is separate from counting</h3>
 * Count comparison cannot see a fastener that is present but in the wrong hole, nor a screw
 * and a stud swapped between positions — both leave the per-type totals identical. Those are
 * the defects a positional check exists to catch, and they are invisible to everything else
 * in the pipeline.
 *
 * <h3>Critical hardware only</h3>
 * Slots, candidate detections and reported findings are all restricted to critical types.
 *
 * Localisation is a claim about hardware that has a place it is supposed to be. An advisory
 * type does not: a scratch has no fixed location, so it cannot be missing from one, and
 * offering it as the occupant of a bolt head's position produces findings like
 * "WRONG: Scratch" — noise wearing the vocabulary of a defect. An ignored type is not a
 * feature at all. Both are excluded from the geometry as well as from the report.
 *
 * The cost is a smaller layout. Registration genuinely is easier with more landmarks, and
 * advisory detections are usable ones, so a reference whose critical hardware is sparse will
 * now report localisation as unavailable where it previously scraped an alignment together
 * from scratches. That is the intended trade: an alignment resting on scratches rests on
 * things that move between shots, and a confident wrong answer is worse than an honest
 * refusal. {@code tooSmallToRegister} is what turns the shortfall into that refusal.
 *
 * The one exception is an unclassified detection, which stays eligible as a candidate — see
 * {@code ResultsFragment.runLocalisation}. "unknown" is not a type the operator graded as
 * uninteresting; it is a crop the classifier could not name, and it may well be the critical
 * fastener a slot is asking about.
 *
 * <h3>Coordinates</h3>
 * Everything here is in the reference photo's fractional (0-1) space. Positions are reported
 * to the operator as a named zone plus percentages, because a raw fraction pair is not
 * something anyone can act on while holding the part.
 *
 * <h3>Known limit: symmetric layouts</h3>
 * Slot identity comes from geometry alone, so an arrangement that maps onto itself under
 * rotation cannot be pinned uniquely. Four bolt heads on a perfect square with one absent has
 * several equally good alignments, and nothing distinguishes them — the check still reports
 * that one position is empty, and the part still fails, but <em>which</em> slot it names is
 * arbitrary. The count and the verdict are sound; the slot number is not, in that case.
 * Breaking the tie would need a datum the part itself carries, such as an asymmetric
 * fiducial or an edge feature included in the layout.
 *
 * Pure geometry with no Android context or database dependency beyond {@link RectF}, so the
 * zone boundaries and status logic can be exercised directly in unit tests.
 */
public final class FeatureLocalisation {

    /**
     * Grid resolution for zone naming. Three bands per axis is what the vocabulary below
     * supports; a finer grid would need names nobody uses about a physical part.
     */
    private static final float LOWER_THIRD = 1f / 3f;
    private static final float UPPER_THIRD = 2f / 3f;

    /** What the positional check concluded about one expected position. */
    public enum Status {
        /** A feature of the expected type was found at this slot. */
        PRESENT,
        /** Nothing was found here, and nothing nearby could plausibly be it. */
        MISSING,
        /**
         * Nothing landed close enough to count as occupying this position, but an otherwise
         * unaccounted-for detection sits near it. Reported with how far off it is, because
         * "the fastener is absent" and "the fastener is fitted a fifth of the way off" are
         * different faults with different causes, and collapsing both into MISSING sends the
         * operator looking for a part that is actually in front of them.
         */
        DISPLACED,
        /** Something is here, but it is not the type this position expects. */
        WRONG_TYPE
    }

    /** One expected position and what was actually found at it. */
    public static final class SlotFinding {
        /**
         * Position in the registration layout. Internal: the layout includes advisory
         * landmarks that are never reported, so these ids have gaps once filtered.
         * {@link #displayNumber} is what the operator sees.
         */
        public final int slotId;
        /**
         * Contiguous 1..N number over the <em>reported</em> positions, in slot order.
         *
         * Separate from {@link #slotId} because the registration layout contains landmarks
         * that are deliberately not reported (a scratch is a fine reference point and a
         * meaningless finding). Numbering the report by raw slot id produced gaps — "#1, #3,
         * #4" — which reads as if #2 had gone missing rather than never having been a
         * critical feature at all.
         */
        public final int displayNumber;
        /** Type key this position expects, e.g. {@code bolt_head}. */
        public final String expectedType;
        /** Type key actually found, or null when nothing was. */
        public final String foundType;
        /**
         * Position within the reference photo's frame, fractional. Used for the projection
         * maths; {@link #relX}/{@link #relY} are what the operator is shown.
         */
        public final float x;
        public final float y;
        /**
         * Position relative to the part itself: 0 at the top-left of the box enclosing every
         * detected feature, 1 at its bottom-right.
         *
         * This is the reported coordinate. Frame-relative values depend on how much empty
         * background the operator happened to include, so the same physical bolt read (30%,
         * 30%) in one reference photo and (12%, 8%) in a tighter crop of the same part.
         * Normalising to the feature envelope makes the numbers describe the part rather than
         * the photograph, and makes them comparable between references.
         */
        public final float relX;
        public final float relY;
        /**
         * The same position mapped into <em>this</em> inspection photo, fractional, or -1
         * when the projection is unavailable. This is what the overlay draws against; the
         * reference coordinates above would land somewhere arbitrary on a freehand shot.
         */
        public final float projectedX;
        public final float projectedY;
        public final Status status;
        /**
         * How far the {@link Status#DISPLACED} detection sits from where it belongs, as a
         * fraction of the part's own extent. -1 for every other status.
         */
        public final float offset;
        /**
         * Index into the inspection's feature list, or -1 when nothing matched. Lets the UI
         * tie a finding back to the box the operator can see and tap.
         */
        public final int featureIndex;

        SlotFinding(int slotId, int displayNumber, String expectedType, String foundType,
                    float x, float y, float relX, float relY,
                    float projectedX, float projectedY, Status status, float offset,
                    int featureIndex) {
            this.slotId = slotId;
            this.displayNumber = displayNumber;
            this.expectedType = expectedType;
            this.foundType = foundType;
            this.x = x;
            this.y = y;
            this.relX = relX;
            this.relY = relY;
            this.projectedX = projectedX;
            this.projectedY = projectedY;
            this.status = status;
            this.offset = offset;
            this.featureIndex = featureIndex;
        }

        /** Copy with a display number assigned, once the reported set is known and sorted. */
        SlotFinding numbered(int n) {
            return new SlotFinding(slotId, n, expectedType, foundType, x, y, relX, relY,
                    projectedX, projectedY, status, offset, featureIndex);
        }

        /**
         * Whether {@link #projectedX}/{@link #projectedY} land inside the inspection photo.
         *
         * Bounded on both ends, not just checked for the -1 sentinel: a slot can project
         * outside the frame when the operator shot the part cropped, and a marker drawn at
         * 140% would be clipped away silently while the finding list still claimed the
         * position was locatable.
         */
        public boolean hasProjection() {
            return projectedX >= 0f && projectedX <= 1f && projectedY >= 0f && projectedY <= 1f;
        }

        /**
         * Human-readable position relative to the part, e.g. {@code "upper right (72%, 18%)"}.
         */
        public String position() {
            return String.format(Locale.US, "%s (%.0f%%, %.0f%%)",
                    zone(relX, relY), relX * 100, relY * 100);
        }
    }

    public static final class Result {
        public final List<SlotFinding> findings = new ArrayList<>();
        /**
         * Whether the inspection could be registered onto the layout at all. False means no
         * positional claim is possible and every finding list is empty — the caller must fall
         * back to count comparison rather than reporting everything as missing.
         */
        public final boolean aligned;
        /** Why alignment was not attempted or did not succeed; null when {@link #aligned}. */
        public final String unavailableReason;

        Result(boolean aligned, String unavailableReason) {
            this.aligned = aligned;
            this.unavailableReason = unavailableReason;
        }

        public int countWithStatus(Status status) {
            int n = 0;
            for (SlotFinding f : findings) if (f.status == status) n++;
            return n;
        }

        /** True when any expected position is unaccounted for or holds the wrong hardware. */
        public boolean hasPositionalDefect() {
            // Displaced counts: hardware fitted off its position is a defect, not a near-pass.
            return countWithStatus(Status.MISSING) > 0
                    || countWithStatus(Status.WRONG_TYPE) > 0
                    || countWithStatus(Status.DISPLACED) > 0;
        }

        public String summary() {
            if (!aligned) {
                return unavailableReason == null ? "Positional check unavailable." : unavailableReason;
            }
            StringBuilder sb = new StringBuilder(String.format(Locale.US,
                    "%d of %d expected positions verified, %d missing, %d wrong type",
                    countWithStatus(Status.PRESENT), findings.size(),
                    countWithStatus(Status.MISSING), countWithStatus(Status.WRONG_TYPE)));
            int displaced = countWithStatus(Status.DISPLACED);
            if (displaced > 0) sb.append(", ").append(displaced).append(" out of position");
            return sb.append('.').toString();
        }
    }

    private FeatureLocalisation() {}

    private static boolean reported(String slotType, java.util.Set<String> reportTypes) {
        return reportTypes == null || reportTypes.contains(slotType);
    }

    /** Builds one finding, with the display number left to be assigned after sorting. */
    private static SlotFinding finding(FeatureLayout.Slot slot, String foundType, float[] env,
                                       LayoutMatcher.Result match, Status status, float offset,
                                       int featureIndex) {
        float[] p = match.project(slot.x, slot.y);
        return new SlotFinding(slot.id, 0, slot.type, foundType, slot.x, slot.y,
                relative(slot.x, env[0], env[2]), relative(slot.y, env[1], env[3]),
                p == null ? -1f : p[0], p == null ? -1f : p[1], status, offset, featureIndex);
    }

    /**
     * Smallest fraction of the frame the feature envelope may span before normalising against
     * it stops being meaningful. Below this the features are effectively coincident, so
     * dividing by the extent would turn sub-pixel jitter into swings across the whole 0-1
     * range.
     */
    private static final float MIN_ENVELOPE_EXTENT = 0.02f;

    /**
     * How far a stray detection may sit from an empty position and still be offered as the
     * thing that belongs there, as a fraction of the part's extent.
     *
     * Half the part is deliberately generous — the point is to tell the operator "it is here,
     * just off" rather than send them hunting for absent hardware. Past that the claim stops
     * being useful: a detection more than half a part away is at least as close to some other
     * position, so naming this one would be a guess dressed up as a measurement.
     */
    public static final float DISPLACEMENT_LIMIT = 0.5f;

    /**
     * Distance from a slot to a detection, in the layout's own frame, as a fraction of the
     * part's extent. Negative when it cannot be computed.
     *
     * The detection is mapped back through the alignment rather than the slot mapped forward,
     * so the answer is in the frame the part is described in. Measuring in photo coordinates
     * would fold how close the operator stood into a number presented as a property of the part.
     */
    private static float offsetFrom(FeatureLayout.Slot slot, RectF rect, float[] env,
                                    LayoutMatcher.Result match) {
        if (rect == null) return -1f;
        float[] back = match.unproject((rect.left + rect.right) / 2f, (rect.top + rect.bottom) / 2f);
        if (back == null) return -1f;
        double extent = Math.hypot(env[2], env[3]);
        if (!(extent > 1e-6)) return -1f;
        return (float) (Math.hypot(back[0] - slot.x, back[1] - slot.y) / extent);
    }

    /**
     * Bounding box of every slot in the layout, as {@code [minX, minY, spanX, spanY]} — the
     * part's own extent, used to convert frame coordinates into part-relative ones.
     *
     * Computed over the whole registration layout rather than only the reported slots, so the
     * origin does not shift when a borderline detection re-classifies and drops out of the
     * report. A degenerate span is returned as 0 and handled by {@link #relative}.
     */
    private static float[] envelopeOf(FeatureLayout layout) {
        float minX = Float.MAX_VALUE, minY = Float.MAX_VALUE;
        float maxX = -Float.MAX_VALUE, maxY = -Float.MAX_VALUE;
        for (FeatureLayout.Slot s : layout.slots()) {
            minX = Math.min(minX, s.x);
            minY = Math.min(minY, s.y);
            maxX = Math.max(maxX, s.x);
            maxY = Math.max(maxY, s.y);
        }
        float spanX = maxX - minX, spanY = maxY - minY;
        return new float[]{minX, minY,
                spanX < MIN_ENVELOPE_EXTENT ? 0f : spanX,
                spanY < MIN_ENVELOPE_EXTENT ? 0f : spanY};
    }

    /**
     * Maps one axis into the part's own 0-1 range. A degenerate span reports 0.5 — the
     * features really are all in one place, so "the middle" is the only honest answer.
     */
    private static float relative(float value, float min, float span) {
        if (span <= 0f) return 0.5f;
        return (value - min) / span;
    }

    /**
     * Names the region of the part a point falls in, on a 3x3 grid.
     *
     * Bands are equal thirds rather than a small "centre" box with wide margins: the grid is
     * describing where to look on a part held in the hand, and an operator reading "centre"
     * for something two-thirds of the way up would be sent to the wrong place.
     */
    public static String zone(float x, float y) {
        String vertical = y < LOWER_THIRD ? "upper" : (y < UPPER_THIRD ? "middle" : "lower");
        String horizontal = x < LOWER_THIRD ? "left" : (x < UPPER_THIRD ? "centre" : "right");
        if ("middle".equals(vertical) && "centre".equals(horizontal)) return "centre";
        return vertical + " " + horizontal;
    }

    /**
     * Builds the expected arrangement from a reference photo's detections, keeping the types
     * named in {@code registrationTypes} — the landmarks alignment may use, not the subset
     * that gets reported.
     *
     * Unclassified detections are always skipped regardless: an "unknown" slot matches any
     * detection of any type, which makes it a free pass in candidate generation rather than
     * evidence of anything.
     *
     * The layout is derived fresh from the reference pass on every inspection rather than
     * persisted. The reference photo and the detection pipeline are both deterministic, so
     * rebuilding gives identical slots, and it removes any possibility of a saved layout
     * outliving the reference image it describes — a staleness failure that would silently
     * report correct hardware as misplaced.
     */
    public static FeatureLayout layoutFrom(List<RectF> referenceRects, List<String> referenceTypes,
                                           java.util.Set<String> registrationTypes) {
        FeatureLayout layout = new FeatureLayout();
        if (referenceRects == null || referenceTypes == null) return layout;
        int n = Math.min(referenceRects.size(), referenceTypes.size());
        for (int i = 0; i < n; i++) {
            String type = referenceTypes.get(i);
            if (type == null || ReferenceLibrary.UNKNOWN.equals(type)) continue;
            if (registrationTypes != null && !registrationTypes.contains(type)) continue;
            RectF r = referenceRects.get(i);
            if (r == null) continue;
            layout.addSlot(type, (r.left + r.right) / 2f, (r.top + r.bottom) / 2f);
        }
        return layout;
    }

    /**
     * Runs the positional check and converts the raw correspondence into findings.
     *
     * @param layout        expected arrangement, from {@link #layoutFrom}
     * @param featureRects  this inspection's detections, in fractional coordinates
     * @param featureTypes  classified type per detection, index-aligned with {@code featureRects}
     * @param reportTypes   types to emit findings for; null reports every slot. Alignment still
     *                      uses the whole layout — this filters the answer, not the evidence.
     */
    public static Result localise(FeatureLayout layout, List<RectF> featureRects,
                                  List<String> featureTypes,
                                  java.util.Set<String> reportTypes) {
        if (layout == null || layout.isEmpty()) {
            return new Result(false, "No reference layout — set a reference photo for this part"
                    + " to enable positional checking.");
        }
        if (LayoutMatcher.tooSmallToRegister(layout.slots().size())) {
            // Naming the count matters: the fix is a better reference photo, and an operator
            // told only "could not align" would keep re-shooting the part instead.
            return new Result(false, String.format(Locale.US,
                    "Reference photo shows only %d recognised feature%s — too few to register a"
                            + " position against. Re-shoot the reference so more hardware is"
                            + " clearly visible.",
                    layout.slots().size(), layout.slots().size() == 1 ? "" : "s"));
        }
        if (featureRects == null || featureRects.isEmpty()) {
            return new Result(false, "No detections to position against the reference layout.");
        }

        LayoutMatcher.Result match = LayoutMatcher.match(layout, featureRects, featureTypes);
        if (!match.aligned) {
            return new Result(false, String.format(Locale.US,
                    "Could not align this photo to the reference layout — %d of the %d expected"
                            + " features were located, and %d are needed to fix a position. No"
                            + " positional claim is made. Re-shoot framed like the reference.",
                    match.inlierCount, match.slotCount, match.requiredInliers));
        }

        Result result = new Result(true, null);
        float[] env = envelopeOf(layout);
        for (LayoutMatcher.Match m : match.matches) {
            FeatureLayout.Slot slot = layout.slotById(m.slotId);
            if (slot == null || !reported(slot.type, reportTypes)) continue;
            String found = m.featureIndex < featureTypes.size() ? featureTypes.get(m.featureIndex) : null;
            result.findings.add(finding(slot, found, env, match,
                    m.typeMismatch ? Status.WRONG_TYPE : Status.PRESENT, -1f, m.featureIndex));
        }
        // Detections that matched no slot, still available to explain an empty position.
        // Claimed at most once, so two empty slots cannot both blame the same stray feature.
        List<Integer> spare = new ArrayList<>(match.unmatchedFeatureIndices);
        for (Integer slotId : match.missingSlotIds) {
            FeatureLayout.Slot slot = layout.slotById(slotId);
            if (slot == null || !reported(slot.type, reportTypes)) continue;

            int nearest = -1;
            float nearestOffset = Float.MAX_VALUE;
            for (Integer f : spare) {
                if (f == null || f < 0 || f >= featureRects.size()) continue;
                float o = offsetFrom(slot, featureRects.get(f), env, match);
                if (o >= 0 && o < nearestOffset) {
                    nearestOffset = o;
                    nearest = f;
                }
            }
            if (nearest >= 0 && nearestOffset <= DISPLACEMENT_LIMIT) {
                spare.remove(Integer.valueOf(nearest));
                String found = nearest < featureTypes.size() ? featureTypes.get(nearest) : null;
                result.findings.add(finding(slot, found, env, match,
                        Status.DISPLACED, nearestOffset, nearest));
            } else {
                result.findings.add(finding(slot, null, env, match, Status.MISSING, -1f, -1));
            }
        }

        // Slot order, not match order: the operator reads this list against a numbered part,
        // and greedy residual assignment above emits matches closest-first. Display numbers are
        // assigned only after sorting, so they run 1..N down the report with no gaps.
        java.util.Collections.sort(result.findings, (a, b) -> Integer.compare(a.slotId, b.slotId));
        for (int i = 0; i < result.findings.size(); i++) {
            result.findings.set(i, result.findings.get(i).numbered(i + 1));
        }
        return result;
    }
}
