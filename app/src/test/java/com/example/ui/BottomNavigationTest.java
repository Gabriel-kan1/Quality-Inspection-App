package com.example.ui;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import androidx.navigation.NavController;
import androidx.navigation.fragment.NavHostFragment;
import androidx.test.core.app.ActivityScenario;

import com.example.MainActivity;
import com.example.R;
import com.google.android.material.bottomnavigation.BottomNavigationView;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;

/**
 * Selecting Home in the bottom bar from the History screen has to get back to Home.
 *
 * <p>Reported from a device as "the Home button does nothing". NavigationUI swallows an
 * IllegalArgumentException from navigate() and returns false, so a graph that cannot satisfy
 * the request produces no navigation, no crash and no log line — which is indistinguishable
 * from a dead touch target until you take the touch out of the picture, as this does.
 */
@RunWith(RobolectricTestRunner.class)
public class BottomNavigationTest {

    private static NavController navControllerOf(MainActivity activity) {
        NavHostFragment host = (NavHostFragment) activity.getSupportFragmentManager()
                .findFragmentById(R.id.nav_host_fragment);
        assertNotNull("nav host missing", host);
        return host.getNavController();
    }

    @Test
    public void selectingHomeFromHistoryReturnsToHome() {
        try (ActivityScenario<MainActivity> scenario = ActivityScenario.launch(MainActivity.class)) {
            scenario.onActivity(activity -> {
                NavController nav = navControllerOf(activity);
                assertEquals(R.id.navigation_home, nav.getCurrentDestination().getId());

                // Exactly what the Home screen's own button does.
                nav.navigate(R.id.action_home_to_history);
                assertEquals(R.id.navigation_history, nav.getCurrentDestination().getId());

                BottomNavigationView bar = activity.findViewById(R.id.bottom_navigation);
                bar.setSelectedItemId(R.id.navigation_home);

                assertEquals("selecting Home in the bottom bar left us on another destination",
                        R.id.navigation_home, nav.getCurrentDestination().getId());
            });
        }
    }

    /** And the reverse, so the bar is not merely one-way. */
    @Test
    public void selectingHistoryFromHomeOpensHistory() {
        try (ActivityScenario<MainActivity> scenario = ActivityScenario.launch(MainActivity.class)) {
            scenario.onActivity(activity -> {
                NavController nav = navControllerOf(activity);
                BottomNavigationView bar = activity.findViewById(R.id.bottom_navigation);

                bar.setSelectedItemId(R.id.navigation_history);

                assertEquals(R.id.navigation_history, nav.getCurrentDestination().getId());
            });
        }
    }
}
