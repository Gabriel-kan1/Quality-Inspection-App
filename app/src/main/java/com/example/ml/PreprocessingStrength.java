package com.example.ml;

/**
 * The three rungs the adaptive edge stage chooses between — one pipeline (Hybrid v6's, the
 * most accurate against the ground-truth counts) pushed harder or softer on contrast.
 *
 * <h3>Why contrast and not the filter apertures</h3>
 * "Stronger preprocessing for a noisier photo" reads as a bigger bilateral diameter and a
 * wider median, and that is what the first ladder did: bilateral 3→11, median 3→9, Gaussian
 * 3→9. Measured against the ground truth it was wrong in both directions. At the 1280px
 * working size a 7px median erases small hardware outright — the "strong" rung missed 15
 * fasteners on <em>clean</em> photographs and the widest rung 41 — while the lightest
 * apertures produced 175 extras. What actually moved the count under synthetic noise was
 * CLAHE's clip limit: local contrast equalisation amplifies grain in flat regions exactly as
 * readily as it lifts faint edges, and under noise the grain wins. Under blur the same knob
 * works the other way, and the Gaussian aperture matters too. So the rungs share their
 * bilateral and median apertures, cost the same, and differ in how hard they push contrast.
 * See {@code python/adaptive_preprocessing_study.py}, sections 1 and 2.
 *
 * <p>{@link #MEDIUM} <em>is</em> Hybrid v6 — same id, same parameters — so a photograph the
 * estimators cannot fault runs exactly what the ground-truth search found best.
 */
public enum PreprocessingStrength {
    /**
     * Soft photographs: twice the contrast boost and the smallest Gaussian, so what little
     * edge there is survives to Canny. Catastrophic on anything sharp (25 misses and 205
     * extras on the clean set), which is why {@link ImageQuality} refuses to offer it unless
     * the edges really are weak.
     */
    LIGHT("strength-light", "light", 2.0, 3, 1.1),
    /** Hybrid v6 exactly. */
    MEDIUM("hybrid-v6", "medium", 1.0, 5, 1.1),
    /**
     * Noisy photographs: half the contrast boost, and Canny's high bound back at 1.0x Otsu so
     * the edges that do survive are not thinned further. Under synthetic noise of sigma 10
     * it scored 20 against v6's 31 (2 x misses + extras over 26 photographs).
     */
    STRONG("strength-strong", "strong", 0.5, 5, 1.0);

    /** Catalogue id in {@link EdgeMethodLab}; stable, persisted in layouts. */
    public final String id;
    /** One word for banners and logs. */
    public final String label;
    /** CLAHE clip limit — the knob that measured as the lever. */
    final double claheClip;
    /** Gaussian blur aperture, odd. */
    final int gaussK;
    /** Multiplier on the Otsu threshold used as Canny's high bound. */
    final double otsuHighScale;

    PreprocessingStrength(String id, String label, double claheClip, int gaussK,
                          double otsuHighScale) {
        this.id = id;
        this.label = label;
        this.claheClip = claheClip;
        this.gaussK = gaussK;
        this.otsuHighScale = otsuHighScale;
    }

    /** The rung with this catalogue id, or null. */
    public static PreprocessingStrength byId(String id) {
        if (id == null) return null;
        for (PreprocessingStrength s : values()) if (s.id.equals(id)) return s;
        return null;
    }

    /** The rung with this label (as written into a layout file), or null. */
    public static PreprocessingStrength byLabel(String label) {
        if (label == null) return null;
        for (PreprocessingStrength s : values()) if (s.label.equals(label)) return s;
        return null;
    }
}
