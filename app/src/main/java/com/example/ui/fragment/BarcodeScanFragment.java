package com.example.ui.fragment;

import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.drawable.GradientDrawable;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.navigation.Navigation;

import com.example.R;
import com.example.data.AppDatabase;
import com.example.data.entity.PartProfile;
import com.example.data.entity.Product;
import com.example.data.entity.ReferenceProfile;
import com.example.databinding.FragmentBarcodeScanBinding;
import com.example.ml.HardwareFeatureExtractor;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import com.example.ui.util.Notice;

public class BarcodeScanFragment extends Fragment {

    /**
     * Part IDs seeded into the registry on a fresh install, so the test shortcuts are there
     * before anything has been saved. Upgrading devices get the same set from
     * {@code MIGRATION_11_12}; this covers the database that was created at v12 directly and
     * therefore never ran it.
     */
    private static final String[] SEED_PART_IDS =
            {"CHASSIS-8821-B", "PART-CHASSIS-A03", "ENGINE-COV-X42"};

    /** Thumbnail edge in px to decode at — small enough that a 4000px reference costs little. */
    private static final int THUMB_PX = 160;

    private FragmentBarcodeScanBinding binding;

    /** One row of the profile list: a registered part, and its reference photo if it has one. */
    private static final class ProfileRow {
        final long id;
        final String partId;
        final String label;
        final String imageUri;
        Bitmap thumb;

        ProfileRow(long id, String partId, String label, String imageUri) {
            this.id = id;
            this.partId = partId;
            this.label = label;
            this.imageUri = imageUri;
        }

        boolean hasReference() {
            return imageUri != null && !ReferenceProfile.PLACEHOLDER_URI.equals(imageUri);
        }

        /** True when the label is just the id, so the row need not print both. */
        boolean labelIsId() {
            return label.equals(partId);
        }
    }

    /** A product and the parts under it, in the order the list draws them. */
    private static final class ProductGroup {
        final long id;
        final String name;
        final List<ProfileRow> parts = new ArrayList<>();

        ProductGroup(long id, String name) {
            this.id = id;
            this.name = name;
        }
    }

    /**
     * Which products are open, by id. Held in the fragment rather than the database: it is
     * where the operator has scrolled to, not a property of the product, and it should not
     * survive to confuse the next person to pick up the tablet.
     */
    private final Set<Long> expandedProductIds = new HashSet<>();

    /** True until the first load has decided what to open by default. */
    private boolean expansionInitialised;

    /**
     * The part the operator has picked, or null before they pick one.
     *
     * <p>This used to be whatever text sat in a free-entry field. Typing an identifier is how a
     * part came to exist at all, which meant a typo created a new part rather than failing —
     * and the resulting near-duplicate ("CHASIS-8821-B") got its own reference photo, its own
     * layout and its own history, silently. Parts are added deliberately now, from a list, so
     * the field had nothing left to do that the list does not do better.
     */
    @Nullable private String selectedPartId;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        binding = FragmentBarcodeScanBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        binding.btnConfirmPart.setOnClickListener(v -> {
            if (selectedPartId == null) {
                Notice.show(getContext(), "Select a part from the list first");
                return;
            }
            checkReferenceAndProceed(selectedPartId);
        });

        // Go straight to setup so an already-configured part can be given a new reference photo.
        binding.btnChangeReference.setOnClickListener(v -> {
            if (selectedPartId == null) {
                Notice.show(getContext(), "Select a part from the list first");
                return;
            }
            Bundle args = new Bundle();
            args.putString("partId", selectedPartId);
            Navigation.findNavController(requireView()).navigate(R.id.action_barcode_to_reference_setup, args);
        });

        binding.btnAddProduct.setOnClickListener(v -> promptForProduct());
        // No loadProfiles() here: onResume follows immediately and does it. Calling it from
        // both started two loader threads at once on every first show, and both raced to seed
        // the empty registry - the loser threw on the unique product name, caught it with an
        // empty list in hand, and could post that empty list over the winner's full one.
    }

    @Override
    public void onResume() {
        super.onResume();
        // Coming back from reference setup, a part may now have a photo it did not have before.
        loadProfiles();
    }

    /**
     * Reads the product registry and decodes reference thumbnails off the main thread.
     *
     * <p>References can be camera-resolution stills, so decoding them at full size for a 56dp
     * row would cost tens of megabytes each — they are sampled down during decode instead.
     *
     * <p>Seeding happens here rather than only in the migration so that a database created at
     * the current version, which never ran that migration, still opens with the presets and a
     * product to hold them.
     */
    /**
     * Bumped on every {@link #loadProfiles} call; a worker that finishes after a newer one has
     * started drops its result rather than painting stale rows over fresh ones.
     */
    private int loadGeneration = 0;

    private void loadProfiles() {
        final int generation = ++loadGeneration;
        new Thread(() -> {
            List<ProductGroup> groups = new ArrayList<>();
            boolean loaded = false;
            try {
                AppDatabase db = AppDatabase.getInstance(requireContext());
                seedIfEmpty(db);

                Map<String, String> imageByPartId = new LinkedHashMap<>();
                List<ReferenceProfile> profiles = db.referenceProfileDao().getAllProfiles();
                if (profiles != null) {
                    for (ReferenceProfile p : profiles) {
                        if (p.getPartType() != null) {
                            imageByPartId.put(p.getPartType(), p.getImageUri());
                        }
                    }
                }

                // A part can have a reference photo without a registry row if it was created by
                // typing an id straight into the field. Adopt it rather than hiding it.
                adoptUnregisteredParts(db, imageByPartId.keySet());

                for (Product product : db.productDao().getAllProducts()) {
                    ProductGroup group = new ProductGroup(product.getId(), product.getName());
                    for (PartProfile part
                            : db.partProfileDao().getPartProfilesForProduct(product.getId())) {
                        group.parts.add(new ProfileRow(part.getId(), part.getPartId(),
                                part.label(), imageByPartId.get(part.getPartId())));
                    }
                    groups.add(group);
                }

                for (ProductGroup g : groups) {
                    for (ProfileRow r : g.parts) {
                        if (r.hasReference()) {
                            r.thumb = HardwareFeatureExtractor.loadBitmap(
                                    requireContext(), r.imageUri, THUMB_PX);
                        }
                    }
                }
                loaded = true;
            } catch (Exception e) {
                android.util.Log.e("BarcodeScanFragment", "Failed to load profiles", e);
            }

            // A load that failed part-way holds a partial list; rendering it would replace a
            // full registry on screen with a fragment of one. Leave whatever is showing.
            if (!loaded) return;
            final List<ProductGroup> finalGroups = groups;
            if (!isAdded()) return;
            requireActivity().runOnUiThread(() -> {
                if (!isAdded() || binding == null) return;
                // Superseded by a newer load: its result is the one that reflects the database.
                if (generation != loadGeneration) return;
                if (!expansionInitialised) {
                    expansionInitialised = true;
                    // Open the first product so the screen never opens as a wall of closed
                    // headers with no indication there is anything inside them.
                    if (!finalGroups.isEmpty()) expandedProductIds.add(finalGroups.get(0).id);
                }
                renderProducts(finalGroups);
            });
        }).start();
    }

    /**
     * Creates the default product and the preset parts when the registry is still empty.
     *
     * <p>Check and insert run in one transaction. SQLite serialises writers, so a second
     * caller blocks until the first commits and then sees a non-empty registry, where two
     * unsynchronised callers would both pass the emptiness check and the second would throw
     * on the product name's unique index.
     */
    private void seedIfEmpty(AppDatabase db) {
        db.runInTransaction(() -> {
            if (!db.productDao().getAllProducts().isEmpty()) return;
            long now = System.currentTimeMillis();
            long productId = db.productDao().insertProduct(new Product(Product.DEFAULT_NAME, now));
            for (String partId : SEED_PART_IDS) {
                db.partProfileDao().insertPartProfile(new PartProfile(productId, partId, null, now));
            }
        });
    }

    /** Files any part that has a reference photo but no registry row under the default product. */
    private void adoptUnregisteredParts(AppDatabase db, Set<String> partIds) {
        for (String partId : partIds) {
            if (partId == null || partId.isEmpty()) continue;
            if (db.partProfileDao().getPartProfileByPartId(partId) != null) continue;
            db.partProfileDao().insertPartProfile(
                    new PartProfile(defaultProductId(db), partId, null,
                            System.currentTimeMillis()));
        }
    }

    private long defaultProductId(AppDatabase db) {
        return db.runInTransaction(() -> {
            Product product = db.productDao().getProductByName(Product.DEFAULT_NAME);
            if (product != null) return product.getId();
            List<Product> all = db.productDao().getAllProducts();
            if (!all.isEmpty()) return all.get(0).getId();
            return db.productDao().insertProduct(
                    new Product(Product.DEFAULT_NAME, System.currentTimeMillis()));
        });
    }

    private void renderProducts(List<ProductGroup> groups) {
        LinearLayout container = binding.layoutProfileRows;
        container.removeAllViews();

        int parts = 0, withReference = 0;
        for (ProductGroup g : groups) {
            parts += g.parts.size();
            for (ProfileRow r : g.parts) if (r.hasReference()) withReference++;
        }
        binding.tvProfilesSubtitle.setText(parts == 0
                ? "No parts yet. Add a product, then add parts to it."
                : groups.size() + (groups.size() == 1 ? " product, " : " products, ")
                        + withReference + " of " + parts + " parts have a reference photo.");

        for (ProductGroup group : groups) {
            container.addView(buildProductHeader(group));
            if (expandedProductIds.contains(group.id)) {
                // The add row sits above the parts, where it reads as "add one of these"
                // rather than as a footnote after a long list.
                container.addView(buildAddPartRow(group));
                for (ProfileRow r : group.parts) container.addView(buildRow(r));
                if (group.parts.isEmpty()) container.addView(buildEmptyNote());
            }
        }

        // A reload rebuilds every row, so the selection has to be painted back on.
        if (selectedPartId != null) highlightSelection();
    }

    /**
     * The collapsible product header.
     *
     * <p>Tapping anywhere on it toggles the group; the pencil renames. Rename is a separate
     * target rather than a long-press because a long-press is invisible — nothing on screen
     * would say the name could be changed.
     */
    private View buildProductHeader(ProductGroup group) {
        LinearLayout header = new LinearLayout(requireContext());
        header.setOrientation(LinearLayout.HORIZONTAL);
        header.setGravity(android.view.Gravity.CENTER_VERTICAL);
        header.setPadding(dp(12), dp(12), dp(8), dp(12));

        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        lp.setMargins(0, 0, 0, dp(8));
        header.setLayoutParams(lp);

        boolean expanded = expandedProductIds.contains(group.id);
        GradientDrawable bg = new GradientDrawable();
        bg.setColor(Color.rgb(32, 44, 62));
        bg.setCornerRadius(dp(12));
        bg.setStroke(dp(1), expanded ? Color.rgb(90, 140, 200) : Color.rgb(70, 78, 92));
        header.setBackground(bg);

        TextView chevron = new TextView(requireContext());
        chevron.setText(expanded ? "▾" : "▸");
        chevron.setTextColor(Color.rgb(150, 190, 240));
        chevron.setTextSize(16f);
        chevron.setWidth(dp(24));
        header.addView(chevron);

        LinearLayout text = new LinearLayout(requireContext());
        text.setOrientation(LinearLayout.VERTICAL);
        text.setLayoutParams(new LinearLayout.LayoutParams(
                0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));

        TextView name = new TextView(requireContext());
        name.setText(group.name);
        name.setTextColor(Color.WHITE);
        name.setTextSize(15f);
        name.setTypeface(android.graphics.Typeface.DEFAULT_BOLD);
        name.setSingleLine(true);
        name.setEllipsize(android.text.TextUtils.TruncateAt.END);
        text.addView(name);

        TextView count = new TextView(requireContext());
        count.setText(group.parts.size() == 1 ? "1 part" : group.parts.size() + " parts");
        count.setTextColor(Color.rgb(150, 160, 175));
        count.setTextSize(11.5f);
        text.addView(count);
        header.addView(text);

        TextView rename = new TextView(requireContext());
        rename.setText("✎");
        rename.setTextColor(Color.rgb(150, 190, 240));
        rename.setTextSize(18f);
        rename.setPadding(dp(12), dp(4), dp(8), dp(4));
        rename.setOnClickListener(v -> promptRenameProduct(group));
        header.addView(rename);

        TextView delete = new TextView(requireContext());
        delete.setText("✕");
        delete.setTextColor(Color.rgb(230, 120, 120));
        delete.setTextSize(16f);
        delete.setPadding(dp(8), dp(4), dp(12), dp(4));
        delete.setOnClickListener(v -> promptDeleteProduct(group));
        header.addView(delete);

        header.setOnClickListener(v -> {
            if (!expandedProductIds.remove(group.id)) expandedProductIds.add(group.id);
            loadProfiles();
        });
        return header;
    }

    /** The "add a part to this product" row that sits at the top of an open group. */
    private View buildAddPartRow(ProductGroup group) {
        TextView add = new TextView(requireContext());
        add.setText("+  Add part profile");
        add.setTextColor(Color.rgb(120, 190, 255));
        add.setTextSize(13.5f);
        add.setTypeface(android.graphics.Typeface.DEFAULT_BOLD);
        add.setPadding(dp(14), dp(12), dp(14), dp(12));

        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        lp.setMargins(dp(16), 0, 0, dp(8));
        add.setLayoutParams(lp);

        GradientDrawable bg = new GradientDrawable();
        bg.setColor(Color.rgb(30, 38, 50));
        bg.setCornerRadius(dp(12));
        // Dashed, so it reads as a slot to fill rather than as another part already there.
        bg.setStroke(dp(1), Color.rgb(90, 130, 180), dp(6), dp(4));
        add.setBackground(bg);

        add.setOnClickListener(v -> promptForPart(group));
        return add;
    }

    private View buildEmptyNote() {
        TextView note = new TextView(requireContext());
        note.setText("No parts in this product yet.");
        note.setTextColor(Color.rgb(140, 140, 152));
        note.setTextSize(12f);
        note.setPadding(dp(16), 0, 0, dp(10));
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        lp.setMarginStart(dp(16));
        note.setLayoutParams(lp);
        return note;
    }

    private View buildRow(ProfileRow row) {
        LinearLayout item = new LinearLayout(requireContext());
        item.setTag(row.partId);
        item.setOrientation(LinearLayout.HORIZONTAL);
        item.setGravity(android.view.Gravity.CENTER_VERTICAL);
        item.setPadding(dp(10), dp(10), dp(14), dp(10));

        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        // Indented, so the nesting is visible without drawing a connector.
        lp.setMargins(dp(16), 0, 0, dp(8));
        item.setLayoutParams(lp);

        GradientDrawable bg = new GradientDrawable();
        bg.setColor(Color.rgb(43, 41, 48));
        bg.setCornerRadius(dp(12));
        bg.setStroke(dp(1), Color.rgb(80, 80, 90));
        item.setBackground(bg);

        ImageView thumb = new ImageView(requireContext());
        int size = dp(56);
        thumb.setLayoutParams(new LinearLayout.LayoutParams(size, size));
        thumb.setScaleType(ImageView.ScaleType.CENTER_CROP);
        thumb.setClipToOutline(true);
        GradientDrawable thumbBg = new GradientDrawable();
        thumbBg.setColor(Color.rgb(24, 24, 28));
        thumbBg.setCornerRadius(dp(8));
        thumb.setBackground(thumbBg);
        if (row.thumb != null) {
            thumb.setImageBitmap(row.thumb);
        } else {
            thumb.setImageResource(android.R.drawable.ic_menu_camera);
            thumb.setColorFilter(Color.rgb(110, 110, 120));
            thumb.setPadding(dp(14), dp(14), dp(14), dp(14));
        }
        item.addView(thumb);

        LinearLayout text = new LinearLayout(requireContext());
        text.setOrientation(LinearLayout.VERTICAL);
        LinearLayout.LayoutParams textLp =
                new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f);
        textLp.setMarginStart(dp(12));
        text.setLayoutParams(textLp);

        TextView name = new TextView(requireContext());
        name.setText(row.label);
        name.setTextColor(Color.WHITE);
        name.setTextSize(15f);
        name.setTypeface(android.graphics.Typeface.DEFAULT_BOLD);
        name.setSingleLine(true);
        name.setEllipsize(android.text.TextUtils.TruncateAt.END);
        text.addView(name);

        TextView status = new TextView(requireContext());
        if (row.hasReference()) {
            status.setText("Reference set — " + fileNameOf(row.imageUri));
            status.setTextColor(Color.rgb(120, 200, 150));
        } else {
            status.setText("No reference photo — setup required");
            status.setTextColor(Color.rgb(235, 170, 90));
        }
        // A renamed part still has to show the id it files under, or the operator cannot tell
        // which record a report belongs to.
        if (!row.labelIsId()) {
            TextView id = new TextView(requireContext());
            id.setText(row.partId);
            id.setTextColor(Color.rgb(140, 140, 152));
            id.setTextSize(11f);
            id.setSingleLine(true);
            id.setEllipsize(android.text.TextUtils.TruncateAt.END);
            text.addView(id);
        }
        status.setTextSize(11.5f);
        status.setSingleLine(true);
        status.setEllipsize(android.text.TextUtils.TruncateAt.END);
        text.addView(status);
        item.addView(text);

        TextView rename = new TextView(requireContext());
        rename.setText("✎");
        rename.setTextColor(Color.rgb(150, 190, 240));
        rename.setTextSize(16f);
        rename.setPadding(dp(10), dp(4), dp(4), dp(4));
        rename.setOnClickListener(v -> promptRenamePart(row));
        item.addView(rename);

        TextView delete = new TextView(requireContext());
        delete.setText("✕");
        delete.setTextColor(Color.rgb(230, 120, 120));
        delete.setTextSize(15f);
        delete.setPadding(dp(6), dp(4), dp(4), dp(4));
        delete.setOnClickListener(v -> promptDeletePart(row));
        item.addView(delete);

        item.setOnClickListener(v -> {
            selectedPartId = row.partId;
            highlightSelection();
        });
        return item;
    }

    /**
     * Repaints the rows so the chosen one stands out.
     *
     * <p>Matched on the tag rather than the visible text: a renamed part shows its label, and
     * two products could hold parts whose labels read the same. The tag is the identifier.
     */
    private void highlightSelection() {
        LinearLayout container = binding.layoutProfileRows;
        for (int i = 0; i < container.getChildCount(); i++) {
            View child = container.getChildAt(i);
            Object tag = child.getTag();
            if (!(tag instanceof String)) continue;
            boolean selected = tag.equals(selectedPartId);
            GradientDrawable bg = (GradientDrawable) child.getBackground();
            bg.setColor(selected ? Color.rgb(26, 70, 128) : Color.rgb(43, 41, 48));
            bg.setStroke(dp(selected ? 2 : 1),
                    selected ? Color.rgb(74, 222, 128) : Color.rgb(80, 80, 90));
        }
    }

    // ── Add, rename and delete ──────────────────────────────────────────────

    /**
     * A single-field prompt. Built by hand rather than from a layout because all four callers
     * want the same thing with different words, and an XML file per prompt would be four files
     * that have to stay in step.
     */
    private void textPrompt(String title, String hint, String initial, boolean upperCase,
                            java.util.function.Consumer<String> onAccept) {
        android.widget.EditText input = new android.widget.EditText(requireContext());
        input.setHint(hint);
        input.setSingleLine(true);
        input.setText(initial == null ? "" : initial);
        input.setSelection(input.getText().length());
        input.setTextColor(Color.WHITE);
        input.setHintTextColor(Color.rgb(150, 150, 160));
        if (upperCase) {
            input.setInputType(android.text.InputType.TYPE_CLASS_TEXT
                    | android.text.InputType.TYPE_TEXT_FLAG_CAP_CHARACTERS);
        }

        FrameLayout frame = new FrameLayout(requireContext());
        frame.setPadding(dp(24), dp(8), dp(24), 0);
        frame.addView(input);

        new androidx.appcompat.app.AlertDialog.Builder(requireContext())
                .setTitle(title)
                .setView(frame)
                .setPositiveButton("Save", (d, w) -> {
                    String value = input.getText().toString().trim();
                    if (value.isEmpty()) {
                        Notice.show(getContext(), "Name cannot be empty");
                        return;
                    }
                    onAccept.accept(value);
                })
                .setNegativeButton("Cancel", null)
                .show();
    }

    private void promptForProduct() {
        textPrompt("New product", "Product name", "", false, name -> runOnDb(db -> {
            if (db.productDao().getProductByName(name) != null) {
                return "A product called \"" + name + "\" already exists.";
            }
            long id = db.productDao().insertProduct(new Product(name, System.currentTimeMillis()));
            // Open it immediately: an operator adding a product is about to put parts in it.
            expandedProductIds.add(id);
            return null;
        }));
    }

    private void promptRenameProduct(ProductGroup group) {
        textPrompt("Rename product", "Product name", group.name, false, name -> runOnDb(db -> {
            if (name.equals(group.name)) return null;
            Product clash = db.productDao().getProductByName(name);
            if (clash != null && clash.getId() != group.id) {
                return "A product called \"" + name + "\" already exists.";
            }
            db.productDao().renameProduct(group.id, name);
            return null;
        }));
    }

    /**
     * Adds a part to a product. This asks for the <em>identifier</em>, not a label: it is the
     * key the reference photo, the learned layout and every filed inspection are stored under,
     * so it is fixed at creation. The pencil on the row edits the label afterwards.
     */
    private void promptForPart(ProductGroup group) {
        textPrompt("Add part to " + group.name, "Part ID (e.g. CHASSIS-8821-B)", "", true,
                partId -> runOnDb(db -> {
                    PartProfile existing = db.partProfileDao().getPartProfileByPartId(partId);
                    if (existing != null) {
                        if (existing.getProductId() == group.id) {
                            return partId + " is already in this product.";
                        }
                        // Registered elsewhere: move it rather than refuse. A part belongs to
                        // one product, and the operator has just said which.
                        db.partProfileDao().movePartProfile(existing.getId(), group.id);
                        return null;
                    }
                    db.partProfileDao().insertPartProfile(new PartProfile(
                            group.id, partId, null, System.currentTimeMillis()));
                    return null;
                }));
    }

    /** Renames the label only — see {@link PartProfile} for why the id is left alone. */
    private void promptRenamePart(ProfileRow row) {
        textPrompt("Rename part", "Display name for " + row.partId, row.label, false,
                label -> runOnDb(db -> {
                    // Setting the label back to the id clears it, so the row returns to the
                    // "no label" state rather than holding a label that merely looks like none.
                    db.partProfileDao().renamePartProfile(
                            row.id, label.equals(row.partId) ? null : label);
                    return null;
                }));
    }

    /**
     * A destructive-action prompt. Separate from {@link #textPrompt} because the thing that
     * matters here is not what you type but what you are about to lose, so the message says it
     * in full rather than relying on the operator inferring it from a title.
     */
    private void confirmPrompt(String title, String message, String confirmLabel,
                               Runnable onConfirm) {
        new androidx.appcompat.app.AlertDialog.Builder(requireContext())
                .setTitle(title)
                .setMessage(message)
                .setPositiveButton(confirmLabel, (d, w) -> onConfirm.run())
                .setNegativeButton("Cancel", null)
                .show();
    }

    /**
     * A confirmation that has to be typed out.
     *
     * <p>Reserved for the one action that destroys several things at once. A Delete button
     * beside a Cancel button is a coin toss for a mis-tap, and the cost of getting this one
     * wrong is a product, every part in it and every reference photo they held. Typing the word
     * cannot happen by accident.
     */
    private void confirmByTyping(String title, String message, String requiredWord,
                                 Runnable onConfirm) {
        LinearLayout column = new LinearLayout(requireContext());
        column.setOrientation(LinearLayout.VERTICAL);
        column.setPadding(dp(24), dp(8), dp(24), 0);

        TextView prompt = new TextView(requireContext());
        prompt.setText("Type " + requiredWord + " to confirm.");
        prompt.setTextColor(Color.rgb(230, 120, 120));
        prompt.setTextSize(13f);
        column.addView(prompt);

        android.widget.EditText input = new android.widget.EditText(requireContext());
        input.setHint(requiredWord);
        input.setSingleLine(true);
        input.setTextColor(Color.WHITE);
        input.setHintTextColor(Color.rgb(150, 150, 160));
        input.setInputType(android.text.InputType.TYPE_CLASS_TEXT
                | android.text.InputType.TYPE_TEXT_FLAG_CAP_CHARACTERS);
        column.addView(input);

        androidx.appcompat.app.AlertDialog dialog =
                new androidx.appcompat.app.AlertDialog.Builder(requireContext())
                        .setTitle(title)
                        .setMessage(message)
                        .setView(column)
                        .setPositiveButton("Delete", null)
                        .setNegativeButton("Cancel", null)
                        .create();
        dialog.show();

        // Wired after show() and left open on a mismatch: the framework dismisses the dialog on
        // any button click, which would drop the operator back to the list with no idea why
        // nothing happened.
        dialog.getButton(androidx.appcompat.app.AlertDialog.BUTTON_POSITIVE)
                .setOnClickListener(v -> {
                    if (!requiredWord.contentEquals(input.getText().toString().trim())) {
                        input.setError("Type " + requiredWord + " exactly");
                        return;
                    }
                    dialog.dismiss();
                    onConfirm.run();
                });
    }

    /**
     * Removes a part from the list.
     *
     * <p>The reference photo goes with it, and has to: the list adopts any part that has a
     * reference but no registry row, so a deleted part with its photo still on file would
     * reappear at the next refresh looking like the delete had failed.
     *
     * <p>Filed inspections are deliberately left alone. They are the record of what was
     * actually checked on a given day, and a part list is not the place from which that can be
     * rewritten.
     */
    private void promptDeletePart(ProfileRow row) {
        String name = row.labelIsId() ? row.partId : row.label + " (" + row.partId + ")";
        confirmPrompt("Delete " + name + "?",
                (row.hasReference()
                        ? "This removes the part from the list and deletes its reference photo."
                        : "This removes the part from the list.")
                        + "\n\nInspection reports already filed against " + row.partId
                        + " are kept.",
                "Delete",
                () -> runOnDb(db -> {
                    db.partProfileDao().deletePartProfile(row.id);
                    db.referenceProfileDao().deleteByPartType(row.partId);
                    if (row.partId.equals(selectedPartId)) selectedPartId = null;
                    return null;
                }));
    }

    /**
     * Removes a product and everything filed under it.
     *
     * <p>The foreign key cascades the registry rows, but the reference photos have to go
     * explicitly: the list adopts any part holding a reference with no registry row, so the
     * parts would reappear under another product on the next refresh — the deletion undone by
     * the very mechanism that keeps typed-in parts visible.
     *
     * <p>Filed inspections are still kept, as they are for a single part. What was inspected on
     * a given day is not something a tidy-up of the part list gets to rewrite.
     */
    private void promptDeleteProduct(ProductGroup group) {
        int count = group.parts.size();
        String message = count == 0
                ? "This product has no parts in it."
                : "This deletes the product, its " + count
                        + (count == 1 ? " part profile" : " part profiles")
                        + ", and their reference photos."
                        + "\n\nInspection reports already filed against those parts are kept.";
        confirmByTyping("Delete product \"" + group.name + "\"?", message, "CONFIRM",
                () -> runOnDb(db -> {
                    if (db.productDao().countProducts() <= 1) {
                        return "This is the only product. Add another before deleting it.";
                    }
                    // Read the parts back rather than trusting the rendered list, which may be
                    // a refresh behind if something else changed the registry.
                    for (PartProfile part
                            : db.partProfileDao().getPartProfilesForProduct(group.id)) {
                        db.referenceProfileDao().deleteByPartType(part.getPartId());
                        if (part.getPartId().equals(selectedPartId)) selectedPartId = null;
                    }
                    db.productDao().deleteProduct(group.id);
                    expandedProductIds.remove(group.id);
                    return null;
                }));
    }

    /**
     * Runs a database edit off the main thread and reloads. The action returns an error message
     * to show, or null when it succeeded.
     */
    private void runOnDb(java.util.function.Function<AppDatabase, String> action) {
        new Thread(() -> {
            String error;
            try {
                error = action.apply(AppDatabase.getInstance(requireContext()));
            } catch (Exception e) {
                android.util.Log.e("BarcodeScanFragment", "Profile edit failed", e);
                error = "Could not save: " + e.getMessage();
            }
            final String finalError = error;
            if (!isAdded()) return;
            requireActivity().runOnUiThread(() -> {
                if (!isAdded() || binding == null) return;
                if (finalError != null) {
                    Notice.show(getContext(), finalError);
                }
                loadProfiles();
            });
        }).start();
    }

    private static String fileNameOf(String uri) {
        if (uri == null) return "";
        int slash = uri.lastIndexOf('/');
        return slash >= 0 ? uri.substring(slash + 1) : uri;
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private void checkReferenceAndProceed(String partId) {
        new Thread(() -> {
            // Check in AppDatabase if ReferenceProfile exists
            AppDatabase db = AppDatabase.getInstance(requireContext());
            ReferenceProfile profile = db.referenceProfileDao().getProfileByPartType(partId);

            requireActivity().runOnUiThread(() -> {
                Bundle args = new Bundle();
                args.putString("partId", partId);
                
                // A placeholder profile means setup was skipped, so there's no image for the
                // alignment gate to work against and inspection would open already blocked.
                // Treat it the same as having no profile at all and ask for a reference.
                boolean hasReference = profile != null && profile.hasRealImage();

                if (hasReference) {
                    Navigation.findNavController(requireView()).navigate(R.id.action_barcode_to_inspection, args);
                } else {
                    Navigation.findNavController(requireView()).navigate(R.id.action_barcode_to_reference_setup, args);
                }
            });
        }).start();
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}
