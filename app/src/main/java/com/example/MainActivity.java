package com.example;

import android.os.Bundle;
import android.util.Log;
import android.os.Handler;
import android.os.Looper;
import android.view.MotionEvent;
import android.view.View;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.core.view.WindowInsetsControllerCompat;
import androidx.navigation.NavController;
import androidx.navigation.NavDestination;
import androidx.navigation.NavOptions;
import androidx.navigation.fragment.NavHostFragment;

import com.example.R;
import com.example.databinding.ActivityMainBinding;

public class MainActivity extends AppCompatActivity {

    /** How long the status and navigation bars stay up after the operator summons them. */
    private static final long SYSTEM_BAR_HIDE_DELAY_MS = 3000L;

    private static final String TAG = "MainActivity";

    private ActivityMainBinding binding;
    private NavController navController;

    /** True while the bar's checked item is being aligned to the current destination. */
    private boolean syncingSelection;

    private final Handler uiHandler = new Handler(Looper.getMainLooper());
    private final Runnable hideSystemBars = this::hideSystemBars;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        
        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        // The phone's own bars steal height from a screen that is mostly camera preview and
        // overlay, so they are dropped a few seconds after every appearance. Swiping from an
        // edge brings them back transiently — they overlay rather than resize the layout —
        // and the timer below sends them away again.
        WindowInsetsControllerCompat controller = getInsetsController();
        controller.setSystemBarsBehavior(
                WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE);

        // targetSdk 35 lays this window out edge to edge on Android 15 whether the app asks or
        // not, and android:statusBarColor / android:navigationBarColor in the theme are ignored
        // from that release on. BottomNavigationView insets itself, so only the back bar needs
        // padding here - doing it at the root instead would double the bottom inset.
        ViewCompat.setOnApplyWindowInsetsListener(binding.getRoot(), (v, insets) -> {
            Insets bars = insets.getInsets(
                    WindowInsetsCompat.Type.systemBars() | WindowInsetsCompat.Type.displayCutout());
            binding.layoutBackBar.setPadding(bars.left, bars.top, bars.right, 0);
            if (insets.isVisible(WindowInsetsCompat.Type.systemBars())) scheduleHideSystemBars();
            return insets;
        });
        scheduleHideSystemBars();

        NavHostFragment navHostFragment = (NavHostFragment) getSupportFragmentManager()
                .findFragmentById(R.id.nav_host_fragment);
                
        if (navHostFragment != null) {
            navController = navHostFragment.getNavController();
            setupBottomNavigation();
            
            // Routes the global back button through the same dispatcher as the system
            // back gesture, so both behave identically on every screen.
            binding.btnGlobalBack.setOnClickListener(v -> getOnBackPressedDispatcher().onBackPressed());

            // Show bottom navigation only on Top-Level screens (Home & History)
            navController.addOnDestinationChangedListener((controller2, destination, arguments) -> {
                int id = destination.getId();
                if (id == R.id.navigation_home || id == R.id.navigation_history) {
                    binding.bottomNavigation.setVisibility(View.VISIBLE);
                    // A view that was GONE gets no insets dispatch when it returns, so
                    // BottomNavigationView comes back without the bottom padding that keeps it
                    // clear of the system navigation bar. Ask for a fresh one.
                    ViewCompat.requestApplyInsets(binding.bottomNavigation);
                    // Reflect where we actually are. Selecting an item navigates, so this is
                    // guarded in the listener below to avoid navigating in a circle.
                    syncingSelection = true;
                    binding.bottomNavigation.setSelectedItemId(id);
                    syncingSelection = false;
                } else {
                    binding.bottomNavigation.setVisibility(View.GONE);
                }

                // Home is the start destination: going back from there would exit the app.
                binding.layoutBackBar.setVisibility(
                        id == R.id.navigation_home ? View.GONE : View.VISIBLE);
            });
        }
    }

    /**
     * Wires the bottom bar by hand rather than through
     * {@code NavigationUI.setupWithNavController}.
     *
     * <p>NavigationUI builds every selection as popUpTo(startDestination, saveState = true)
     * plus restoreState = true - the multi-back-stack feature, which gives each tab its own
     * saved stack. From History that means: pop History and save it under Home's key, then
     * navigate to Home restoring the stack saved under Home's key, which puts History straight
     * back. The net effect is nothing at all, and because navigate() reports it by returning
     * false rather than throwing, the tap produced no navigation, no crash and no log line.
     *
     * <p>It only bites because Home reaches History through a plain action, so the two sit in
     * one stack rather than being independent tabs. This app has two top-level screens and no
     * per-tab history worth preserving, so the fix is to stop asking for machinery it does not
     * use: pop back to the start destination and go, with no state saved or restored.
     */
    private void setupBottomNavigation() {
        binding.bottomNavigation.setOnItemSelectedListener(item -> {
            if (syncingSelection || navController == null) return true;
            int id = item.getItemId();
            NavDestination current = navController.getCurrentDestination();
            if (current != null && current.getId() == id) return true;

            NavOptions options = new NavOptions.Builder()
                    .setLaunchSingleTop(true)
                    .setPopUpTo(navController.getGraph().getStartDestinationId(), false)
                    .build();
            try {
                navController.navigate(id, null, options);
                return true;
            } catch (IllegalArgumentException e) {
                // The destination is not reachable from here. Swallowing this is what hid the
                // original bug, so say so rather than failing silently again.
                Log.w(TAG, "Bottom navigation could not reach destination " + id, e);
                return false;
            }
        });
    }

    private WindowInsetsControllerCompat getInsetsController() {
        return androidx.core.view.WindowCompat.getInsetsController(
                getWindow(), getWindow().getDecorView());
    }

    /** Restarts the countdown, so a fresh appearance always gets its full three seconds. */
    private void scheduleHideSystemBars() {
        uiHandler.removeCallbacks(hideSystemBars);
        uiHandler.postDelayed(hideSystemBars, SYSTEM_BAR_HIDE_DELAY_MS);
    }

    private void hideSystemBars() {
        getInsetsController().hide(WindowInsetsCompat.Type.systemBars());
    }

    @Override
    public boolean dispatchTouchEvent(MotionEvent event) {
        // A swipe that pulls the bars back in usually lands here first; re-arming on any touch
        // also stops the bars vanishing mid-gesture while the operator is reaching for them.
        if (event.getActionMasked() == MotionEvent.ACTION_DOWN) scheduleHideSystemBars();
        return super.dispatchTouchEvent(event);
    }

    @Override
    public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);
        if (hasFocus) scheduleHideSystemBars();
        else uiHandler.removeCallbacks(hideSystemBars);
    }

    @Override
    protected void onDestroy() {
        uiHandler.removeCallbacks(hideSystemBars);
        super.onDestroy();
    }

    @Override
    public boolean onSupportNavigateUp() {
        if (navController != null) {
            return navController.navigateUp() || super.onSupportNavigateUp();
        }
        return super.onSupportNavigateUp();
    }
}
