package com.example.data.entity;

import androidx.room.Entity;
import androidx.room.PrimaryKey;

/**
 * One logged classification decision — the raw material of the self-learning loop.
 *
 * Every detected crop produces an event, whether or not a human ever looks at it.
 * Events the operator confirms or corrects ({@link #reviewed} = true) become the
 * labelled dataset that {@link com.example.ml.ThresholdTuner} fits per-part
 * thresholds against, and that a future off-device fine-tune would train on.
 */
@Entity(tableName = "ClassificationEvent")
public class ClassificationEvent {

    /** Set on {@link #reviewSource} — how this event came to be reviewed by a human. */
    public static final String SOURCE_UNKNOWN_PROMPT = "unknown_prompt";
    public static final String SOURCE_SPOT_CHECK = "spot_check";
    /**
     * Operator tapped a box on the overlay to correct it, unprompted. Tracked separately
     * because these are self-selected — people tap what looks wrong — so they over-represent
     * errors relative to what the classifier actually produces.
     */
    public static final String SOURCE_OVERLAY_TAP = "overlay_tap";

    @PrimaryKey(autoGenerate = true)
    private long id;
    private String partId;
    /** Inspection this belonged to, or -1 if the operator never saved the report. */
    private long inspectionId = -1;
    private long timestamp;
    /** Path to the persisted crop PNG, or null if it couldn't be written. */
    private String cropPath;

    private String predictedType;
    private float predictedScore;
    private String runnerUpType;
    private float runnerUpScore;
    /** {@link com.example.ml.ReferenceLibrary.Reason} name at the time of prediction. */
    private String reason;

    /** Thresholds in force when this prediction was made, so replays stay honest. */
    private float similarityThreshold;
    private float marginThreshold;

    /**
     * {@link com.example.ml.FeatureEmbedder#ENCODER_ID} at the time of prediction.
     *
     * The scores above are only comparable within one embedding space, so replaying a
     * threshold against an event from a different encoder is meaningless — the two
     * encoders don't even use the same numeric range. ThresholdTuner filters on this
     * rather than fitting across a mixed history. Rows written before this column
     * existed carry {@link #ENCODER_LEGACY_UNTAGGED}.
     */
    private String encoderId;

    /** Marks pre-migration rows, which came from the retired CLIP encoder. */
    public static final String ENCODER_LEGACY_UNTAGGED = "legacy-untagged";

    /** Ground truth once a human weighs in: a type key, or "unknown" if they skipped. */
    private String confirmedType;
    private boolean reviewed;
    private String reviewSource;

    public ClassificationEvent() {
    }

    public long getId() { return id; }
    public void setId(long id) { this.id = id; }

    public String getPartId() { return partId; }
    public void setPartId(String partId) { this.partId = partId; }

    public long getInspectionId() { return inspectionId; }
    public void setInspectionId(long inspectionId) { this.inspectionId = inspectionId; }

    public long getTimestamp() { return timestamp; }
    public void setTimestamp(long timestamp) { this.timestamp = timestamp; }

    public String getCropPath() { return cropPath; }
    public void setCropPath(String cropPath) { this.cropPath = cropPath; }

    public String getPredictedType() { return predictedType; }
    public void setPredictedType(String predictedType) { this.predictedType = predictedType; }

    public float getPredictedScore() { return predictedScore; }
    public void setPredictedScore(float predictedScore) { this.predictedScore = predictedScore; }

    public String getRunnerUpType() { return runnerUpType; }
    public void setRunnerUpType(String runnerUpType) { this.runnerUpType = runnerUpType; }

    public float getRunnerUpScore() { return runnerUpScore; }
    public void setRunnerUpScore(float runnerUpScore) { this.runnerUpScore = runnerUpScore; }

    public String getReason() { return reason; }
    public void setReason(String reason) { this.reason = reason; }

    public float getSimilarityThreshold() { return similarityThreshold; }
    public void setSimilarityThreshold(float similarityThreshold) { this.similarityThreshold = similarityThreshold; }

    public float getMarginThreshold() { return marginThreshold; }
    public void setMarginThreshold(float marginThreshold) { this.marginThreshold = marginThreshold; }

    public String getEncoderId() { return encoderId; }
    public void setEncoderId(String encoderId) { this.encoderId = encoderId; }

    public String getConfirmedType() { return confirmedType; }
    public void setConfirmedType(String confirmedType) { this.confirmedType = confirmedType; }

    public boolean isReviewed() { return reviewed; }
    public void setReviewed(boolean reviewed) { this.reviewed = reviewed; }

    public String getReviewSource() { return reviewSource; }
    public void setReviewSource(String reviewSource) { this.reviewSource = reviewSource; }
}
