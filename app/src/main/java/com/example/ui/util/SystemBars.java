package com.example.ui.util;

import android.view.View;

import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

/**
 * Keeps a full-screen dialog's content out from under the system bars.
 *
 * <p>The app targets Android 15, where every window is drawn edge to edge: a full-screen
 * dialog's content runs underneath the navigation bar, and on a phone with the three-button
 * bar that hid the buttons along the dialog's bottom edge — the reference picker's OPEN SYSTEM
 * GALLERY and CLOSE among them.
 */
public final class SystemBars {

    private SystemBars() {}

    /** Adds the system bars' (and any display cutout's) insets to the view's own padding. */
    public static void keepClear(View root) {
        final int left = root.getPaddingLeft();
        final int top = root.getPaddingTop();
        final int right = root.getPaddingRight();
        final int bottom = root.getPaddingBottom();
        ViewCompat.setOnApplyWindowInsetsListener(root, (v, insets) -> {
            Insets bars = insets.getInsets(WindowInsetsCompat.Type.systemBars()
                    | WindowInsetsCompat.Type.displayCutout());
            v.setPadding(left + bars.left, top + bars.top, right + bars.right,
                    bottom + bars.bottom);
            return insets;
        });
        ViewCompat.requestApplyInsets(root);
    }
}
