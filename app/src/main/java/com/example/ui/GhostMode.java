package com.example.ui;

/**
 * How the reference is shown over the live image on the inspection screen.
 *
 * <p>The ghost began as the reference photograph at 35% opacity. Blending one whole photograph
 * over another is self-defeating: both lose contrast, and tinting the ghost to tell them apart
 * only dims the live image further — which is the one the operator is actually aligning.
 * {@link #OUTLINE} draws the reference's edges alone, so the live view stays untouched and a
 * thin coloured wireframe cannot be mistaken for anything but the reference. The photograph
 * is kept a tap away in both its tinted and its original form.
 *
 * <p>An enum rather than a boolean so the button and the status chip cannot drift apart, and
 * so the order of the cycle is stated in exactly one place.
 */
public enum GhostMode {
    /** The reference's edge map in the accent colour over a fully visible live image. Default. */
    OUTLINE("GHOST: OUTLINE", "GHOST OVERLAY: OUTLINE"),
    /** The reference photograph, tinted the accent colour and translucent. */
    TINTED("GHOST: TINTED", "GHOST OVERLAY: TINTED"),
    /**
     * The reference photograph untouched at its original 35% - the ghost as it was before
     * either of the above existed. Kept because a true-colour reference is the only rendering
     * in which a bright cap, a dark head and a bare hole look different from each other.
     */
    PLAIN("GHOST: PLAIN", "GHOST OVERLAY: PLAIN"),
    /** No overlay. */
    OFF("GHOST: OFF", "GHOST OVERLAY: OFF");

    private final String buttonLabel;
    private final String statusLabel;

    GhostMode(String buttonLabel, String statusLabel) {
        this.buttonLabel = buttonLabel;
        this.statusLabel = statusLabel;
    }

    /** What the toggle button reads while this mode is active. Names the state, not the action. */
    public String buttonLabel() {
        return buttonLabel;
    }

    /** What the status chip at the top of the screen reads while this mode is active. */
    public String statusLabel() {
        return statusLabel;
    }

    /** The mode one tap of the button moves to. Wraps, so the button never dead-ends. */
    public GhostMode next() {
        GhostMode[] all = values();
        return all[(ordinal() + 1) % all.length];
    }

    /** True when something is drawn over the live image. */
    public boolean showsOverlay() {
        return this != OFF;
    }
}
