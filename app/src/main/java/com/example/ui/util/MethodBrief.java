package com.example.ui.util;

import android.graphics.Typeface;
import android.text.SpannableStringBuilder;
import android.text.Spanned;
import android.text.style.StyleSpan;

import com.example.ml.EdgeMethodLab;

/**
 * An edge method's "good at" and "uses" lines, as every screen that offers a method shows
 * them: reference setup while the operator flips between methods, the results screen's edge
 * map card, and the Edge Lab's cards.
 */
public final class MethodBrief {

    private MethodBrief() {}

    /** "Good at: …" over "Uses: …", labels in bold; empty for a method that is not catalogued. */
    public static CharSequence of(String methodId) {
        SpannableStringBuilder sb = new SpannableStringBuilder();
        line(sb, "Good at: ", EdgeMethodLab.goodAtOf(methodId));
        line(sb, "Uses: ", EdgeMethodLab.usesOf(methodId));
        return sb;
    }

    private static void line(SpannableStringBuilder sb, String label, String text) {
        if (text.isEmpty()) return;
        if (sb.length() > 0) sb.append('\n');
        int start = sb.length();
        sb.append(label);
        sb.setSpan(new StyleSpan(Typeface.BOLD), start, sb.length(),
                Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
        sb.append(text);
    }
}
