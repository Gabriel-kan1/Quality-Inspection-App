package com.example.data.dao;

import androidx.room.Dao;
import androidx.room.Insert;
import androidx.room.Query;

import com.example.data.entity.Product;

import java.util.List;

@Dao
public interface ProductDao {

    /** Oldest first, so the default product stays at the top and the order is stable. */
    @Query("SELECT * FROM Product ORDER BY createdAt ASC, id ASC")
    List<Product> getAllProducts();

    @Query("SELECT * FROM Product WHERE name = :name LIMIT 1")
    Product getProductByName(String name);

    @Query("SELECT * FROM Product WHERE id = :id LIMIT 1")
    Product getProductById(long id);

    /**
     * Fails rather than replaces on a duplicate name. REPLACE would delete the existing row and
     * insert a new one with a fresh id, and the foreign key cascades — so renaming a product
     * onto an existing name would silently take every part profile under it.
     */
    @Insert
    long insertProduct(Product product);

    @Query("UPDATE Product SET name = :name WHERE id = :id")
    void renameProduct(long id, String name);

    /**
     * Removes a product, and by the foreign key on PartProfile every part registered under it.
     *
     * <p>The parts' reference photos are not reached by the cascade and must be deleted by the
     * caller: the part list adopts any part holding a reference with no registry row, so a
     * surviving photo would put the part back under another product on the next load.
     */
    @Query("DELETE FROM Product WHERE id = :id")
    void deleteProduct(long id);

    @Query("SELECT COUNT(*) FROM Product")
    int countProducts();
}
