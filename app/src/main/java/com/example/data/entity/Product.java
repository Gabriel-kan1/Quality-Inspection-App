package com.example.data.entity;

import androidx.room.Entity;
import androidx.room.Index;
import androidx.room.PrimaryKey;

/**
 * A grouping of part profiles — the level above a part, where a part is one thing that gets
 * photographed and inspected.
 *
 * <p>Purely organisational. Nothing in the inspection pipeline reads a product: detection,
 * classification, localisation and the report are all keyed on the part id, and stay that way.
 * This exists because a flat list stops being navigable once a site has more than a handful of
 * parts, and the parts of one product are what an operator actually thinks in terms of.
 *
 * <p>The name is uniquely indexed so two products cannot end up with the same label and leave
 * the operator guessing which one they just filed against.
 */
@Entity(tableName = "Product", indices = {@Index(value = "name", unique = true)})
public class Product {

    /** Holds everything that existed before products did. See {@code MIGRATION_11_12}. */
    public static final String DEFAULT_NAME = "Uncategorised";

    @PrimaryKey(autoGenerate = true)
    private long id;
    private String name;
    private long createdAt;

    public Product() { }

    public Product(String name, long createdAt) {
        this.name = name;
        this.createdAt = createdAt;
    }

    public long getId() { return id; }
    public void setId(long id) { this.id = id; }

    public String getName() { return name; }
    public void setName(String name) { this.name = name; }

    public long getCreatedAt() { return createdAt; }
    public void setCreatedAt(long createdAt) { this.createdAt = createdAt; }
}
