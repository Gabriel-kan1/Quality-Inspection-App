package com.example.ui.util;

import android.app.Activity;
import android.content.Context;
import android.content.ContextWrapper;
import android.util.Log;

import androidx.appcompat.app.AlertDialog;

/**
 * Tells the operator something they need to act on — a failure, a warning, where an export
 * went — in a small dialog with an OK button.
 *
 * <p>Replaces the system toasts the app used to raise along the bottom of the screen. Those
 * sat over the navigation bar and the gallery's buttons, and most only narrated what the
 * screen was already showing (an image loaded, a check running), so those were dropped
 * rather than moved here. What is left needs reading, so it waits to be dismissed.
 */
public final class Notice {

    private static final String TAG = "Notice";

    private Notice() {}

    /**
     * Shows {@code message} over whichever activity {@code context} belongs to. Logged and
     * otherwise dropped when there is no live activity to show it on, as after the operator
     * has left the screen a background save reports from.
     */
    public static void show(Context context, String message) {
        Activity activity = activityOf(context);
        if (activity == null || activity.isFinishing() || activity.isDestroyed()) {
            Log.i(TAG, message);
            return;
        }
        new AlertDialog.Builder(activity)
                .setMessage(message)
                .setPositiveButton("OK", null)
                .show();
    }

    private static Activity activityOf(Context context) {
        while (context instanceof ContextWrapper) {
            if (context instanceof Activity) return (Activity) context;
            context = ((ContextWrapper) context).getBaseContext();
        }
        return null;
    }
}
