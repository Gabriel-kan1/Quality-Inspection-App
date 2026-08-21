package com.example.data;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;

import android.content.Context;

import androidx.room.Room;
import androidx.test.core.app.ApplicationProvider;

import com.example.data.entity.PartProfile;
import com.example.data.entity.Product;
import com.example.data.entity.ReferenceProfile;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;


/**
 * Covers what deleting actually removes, and — the part that is easy to get wrong — what it has
 * to remove to make the deletion stick.
 */
@RunWith(RobolectricTestRunner.class)
public class ProductRegistryDeleteTest {

    private AppDatabase db;
    private long productId;

    @Before
    public void setUp() {
        Context context = ApplicationProvider.getApplicationContext();
        db = Room.inMemoryDatabaseBuilder(context, AppDatabase.class)
                .allowMainThreadQueries()
                .build();
        productId = db.productDao().insertProduct(new Product("Chassis", 1L));
    }

    @After
    public void tearDown() {
        db.close();
    }

    private long addPart(String partId) {
        return db.partProfileDao().insertPartProfile(
                new PartProfile(productId, partId, null, 1L));
    }

    @Test
    public void deletingAPartRemovesItFromTheRegistry() {
        long id = addPart("WIDGET-1");

        db.partProfileDao().deletePartProfile(id);

        assertNull(db.partProfileDao().getPartProfileByPartId("WIDGET-1"));
    }

    /**
     * The list adopts any part that has a reference photo but no registry row, so deleting the
     * registry row alone would let the part reappear at the next refresh — looking exactly like
     * a delete button that does not work.
     */
    @Test
    public void deletingAPartAlsoClearsTheReferenceThatWouldResurrectIt() {
        long id = addPart("WIDGET-1");
        db.referenceProfileDao().insertProfile(
                new ReferenceProfile("WIDGET-1", "file:///widget.jpg", 1L));

        db.partProfileDao().deletePartProfile(id);
        db.referenceProfileDao().deleteByPartType("WIDGET-1");

        assertNull(db.partProfileDao().getPartProfileByPartId("WIDGET-1"));
        assertNull("the reference would re-adopt the part on the next load",
                db.referenceProfileDao().getProfileByPartType("WIDGET-1"));
    }

    /** One part's reference going must not disturb another part's. */
    @Test
    public void deletingOnePartLeavesOtherPartsAlone() {
        long id = addPart("WIDGET-1");
        addPart("WIDGET-2");
        db.referenceProfileDao().insertProfile(
                new ReferenceProfile("WIDGET-1", "file:///1.jpg", 1L));
        db.referenceProfileDao().insertProfile(
                new ReferenceProfile("WIDGET-2", "file:///2.jpg", 2L));

        db.partProfileDao().deletePartProfile(id);
        db.referenceProfileDao().deleteByPartType("WIDGET-1");

        assertNotNull(db.partProfileDao().getPartProfileByPartId("WIDGET-2"));
        assertNotNull(db.referenceProfileDao().getProfileByPartType("WIDGET-2"));
    }

    /**
     * Deleting a product takes its parts with it. The foreign key does the registry rows; the
     * reference photos have to be removed explicitly or the parts would be re-adopted into
     * another product on the next load, undoing the delete.
     */
    @Test
    public void deletingAProductDeletesItsPartsAndTheirReferences() {
        addPart("WIDGET-1");
        addPart("WIDGET-2");
        db.referenceProfileDao().insertProfile(
                new ReferenceProfile("WIDGET-1", "file:///1.jpg", 1L));
        db.referenceProfileDao().insertProfile(
                new ReferenceProfile("WIDGET-2", "file:///2.jpg", 2L));
        db.productDao().insertProduct(new Product("Gearbox", 2L));

        for (PartProfile part : db.partProfileDao().getPartProfilesForProduct(productId)) {
            db.referenceProfileDao().deleteByPartType(part.getPartId());
        }
        db.productDao().deleteProduct(productId);

        assertNull(db.productDao().getProductById(productId));
        assertNull(db.partProfileDao().getPartProfileByPartId("WIDGET-1"));
        assertNull(db.partProfileDao().getPartProfileByPartId("WIDGET-2"));
        assertNull("a surviving reference would resurrect the part",
                db.referenceProfileDao().getProfileByPartType("WIDGET-1"));
        assertNull(db.referenceProfileDao().getProfileByPartType("WIDGET-2"));
    }

    /** Another product's parts and references must be untouched by the cascade. */
    @Test
    public void deletingAProductLeavesOtherProductsAlone() {
        addPart("WIDGET-1");
        long other = db.productDao().insertProduct(new Product("Gearbox", 2L));
        db.partProfileDao().insertPartProfile(new PartProfile(other, "GEAR-1", null, 2L));
        db.referenceProfileDao().insertProfile(new ReferenceProfile("GEAR-1", "file:///g.jpg", 2L));

        for (PartProfile part : db.partProfileDao().getPartProfilesForProduct(productId)) {
            db.referenceProfileDao().deleteByPartType(part.getPartId());
        }
        db.productDao().deleteProduct(productId);

        assertNotNull(db.partProfileDao().getPartProfileByPartId("GEAR-1"));
        assertNotNull(db.referenceProfileDao().getProfileByPartType("GEAR-1"));
        assertEquals(1, db.partProfileDao().getPartProfilesForProduct(other).size());
    }

    @Test
    public void countProductsTracksAdditionsAndDeletions() {
        assertEquals(1, db.productDao().countProducts());
        long other = db.productDao().insertProduct(new Product("Gearbox", 2L));
        assertEquals(2, db.productDao().countProducts());
        db.productDao().deleteProduct(other);
        assertEquals(1, db.productDao().countProducts());
    }

    /** A renamed part keeps its id; only the label moves. */
    @Test
    public void renamingAPartChangesOnlyTheLabel() {
        long id = addPart("WIDGET-1");

        db.partProfileDao().renamePartProfile(id, "Front bracket");

        PartProfile part = db.partProfileDao().getPartProfileByPartId("WIDGET-1");
        assertNotNull("the part id must be untouched by a rename", part);
        assertEquals("WIDGET-1", part.getPartId());
        assertEquals("Front bracket", part.label());
    }

    /** Clearing the label falls back to showing the id, rather than showing nothing. */
    @Test
    public void clearingTheLabelFallsBackToThePartId() {
        long id = addPart("WIDGET-1");
        db.partProfileDao().renamePartProfile(id, "Front bracket");

        db.partProfileDao().renamePartProfile(id, null);

        assertEquals("WIDGET-1",
                db.partProfileDao().getPartProfileByPartId("WIDGET-1").label());
    }
}
