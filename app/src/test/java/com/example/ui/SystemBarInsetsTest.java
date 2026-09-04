package com.example.ui;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import android.view.View;

import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.test.core.app.ActivityScenario;

import com.example.MainActivity;
import com.example.R;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;

/**
 * The bottom navigation has to stay clear of the system navigation bar.
 *
 * <p>targetSdk 35 lays the window out edge to edge on Android 15 whether the app asks or not,
 * and the theme's statusBarColor / navigationBarColor are ignored from that release on.
 * BottomNavigationView applies its own bottom inset, so the gap that had to be closed by hand
 * is the back bar against the status bar: measured on a device it sat at [12,0][276,144], with
 * the upper 80px of the BACK button underneath the status bar.
 */
@RunWith(RobolectricTestRunner.class)
public class SystemBarInsetsTest {

    /** A 3-button navigation bar and a status bar, in pixels, as a real device reports them. */
    private static final int STATUS_BAR_PX = 80;
    private static final int NAV_BAR_PX = 144;

    private static WindowInsetsCompat systemBars() {
        return new WindowInsetsCompat.Builder()
                .setInsets(WindowInsetsCompat.Type.systemBars(),
                        Insets.of(0, STATUS_BAR_PX, 0, NAV_BAR_PX))
                .build();
    }

    @Test
    public void keepsTheBackBarClearOfTheStatusBar() {
        try (ActivityScenario<MainActivity> scenario = ActivityScenario.launch(MainActivity.class)) {
            scenario.onActivity(activity -> {
                View root = activity.findViewById(R.id.main);
                View backBar = activity.findViewById(R.id.layout_back_bar);
                ViewCompat.dispatchApplyWindowInsets(root, systemBars());

                assertEquals("the back bar must clear the status bar, or the BACK button's"
                                + " upper half sits underneath it",
                        STATUS_BAR_PX, backBar.getPaddingTop());
            });
        }
    }

    /**
     * The bars are auto-hidden a few seconds after each appearance. Once they are gone the
     * insets are zero and the padding has to go with them, otherwise the app keeps a dead
     * margin above a screen that is mostly camera preview.
     */
    @Test
    public void reclaimsTheSpaceOnceTheBarsAreHidden() {
        try (ActivityScenario<MainActivity> scenario = ActivityScenario.launch(MainActivity.class)) {
            scenario.onActivity(activity -> {
                View root = activity.findViewById(R.id.main);
                View backBar = activity.findViewById(R.id.layout_back_bar);
                ViewCompat.dispatchApplyWindowInsets(root, systemBars());
                assertTrue(backBar.getPaddingTop() > 0);

                ViewCompat.dispatchApplyWindowInsets(root,
                        new WindowInsetsCompat.Builder()
                                .setInsets(WindowInsetsCompat.Type.systemBars(), Insets.NONE)
                                .build());

                assertEquals(0, backBar.getPaddingTop());
            });
        }
    }

    /** Both bottom-navigation destinations must exist in the graph and be reachable by id. */
    @Test
    public void bottomNavigationItemsMatchGraphDestinations() {
        try (ActivityScenario<MainActivity> scenario = ActivityScenario.launch(MainActivity.class)) {
            scenario.onActivity(activity -> {
                com.google.android.material.bottomnavigation.BottomNavigationView nav =
                        activity.findViewById(R.id.bottom_navigation);
                assertEquals(2, nav.getMenu().size());
                assertEquals(R.id.navigation_home, nav.getMenu().getItem(0).getItemId());
                assertEquals(R.id.navigation_history, nav.getMenu().getItem(1).getItemId());
            });
        }
    }
}
