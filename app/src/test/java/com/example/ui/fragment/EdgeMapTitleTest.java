package com.example.ui.fragment;

import static org.junit.Assert.assertEquals;

import com.example.ml.EdgeMethodLab;
import com.example.ml.PreprocessingStrength;

import org.junit.Test;

/**
 * The results screen's edge-map card names the preprocessing method that produced the map.
 * It used to read "CANNY EDGES & FILTERING" whatever ran, which under the adaptive stage could
 * be any of a dozen methods.
 */
public class EdgeMapTitleTest {

    @Test
    public void namesTheMethodThatRan() {
        assertEquals("PREPROCESSING: HYBRID V6 (BEST AGAINST GROUND TRUTH)",
                ResultsFragment.edgeMapTitle(PreprocessingStrength.MEDIUM.id));
        assertEquals("PREPROCESSING: " + EdgeMethodLab.nameOf(EdgeMethodLab.SHIPPED_ID)
                        .toUpperCase(java.util.Locale.ROOT),
                ResultsFragment.edgeMapTitle(EdgeMethodLab.SHIPPED_ID));
    }

    @Test
    public void staysGenericWhenNothingRan() {
        assertEquals("EDGE MAP", ResultsFragment.edgeMapTitle(null));
    }
}
