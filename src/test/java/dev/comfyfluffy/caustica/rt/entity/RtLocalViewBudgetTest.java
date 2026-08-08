package dev.comfyfluffy.caustica.rt.entity;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The two-slot admission rule guarding the geometry table. A camera entity that publishes both
 * representations emits TWO table entries from ONE loop iteration, while the table is sized exactly
 * maxEntities() and the loop's own {@code full()} guard is evaluated before the iteration begins. The
 * precheck is what keeps the second write inside the buffer, so its arithmetic is asserted here directly.
 *
 * <p>The surrounding capture path needs Minecraft entities, a render dispatcher, a provider registry and
 * live Vulkan buffers, none of which a unit test can stand up; this covers the admission arithmetic and the
 * resulting write indices, and the rest of P6/P9 stays a review item.
 */
final class RtLocalViewBudgetTest {
    /** Mirrors the precheck in {@code captureEntities}: room for the local view AND the stand-in. */
    private static boolean admitsLocalView(int capacity, int logicalCount) {
        return capacity - logicalCount >= 2;
    }

    @Test
    void admitsBothRepresentationsWithTwoSlotsLeft() {
        assertTrue(admitsLocalView(64, 62));
    }

    @Test
    void refusesTheLocalViewWithOnlyOneSlotLeft() {
        assertFalse(admitsLocalView(64, 63),
                "one free slot must degrade to the world stand-in, not write past the table");
    }

    @Test
    void refusesTheLocalViewWhenAlreadyFull() {
        assertFalse(admitsLocalView(64, 64));
    }

    /**
     * The critical pair from design P6. At capacity minus two the iteration writes the last two indices; at
     * capacity minus one it writes only the final index. Neither may reach {@code capacity}.
     */
    @Test
    void writeIndicesStayInsideTheTableAtBothCriticalPoints() {
        int capacity = 64;

        assertEquals(capacity - 1, highestWriteIndex(capacity, capacity - 2));
        assertEquals(capacity - 1, highestWriteIndex(capacity, capacity - 1));
    }

    @Test
    void writeIndicesStayInsideTheTableAcrossEveryOccupancy() {
        int capacity = 64;
        for (int logicalCount = 0; logicalCount < capacity; logicalCount++) {
            assertTrue(highestWriteIndex(capacity, logicalCount) <= capacity - 1,
                    "occupancy " + logicalCount + " wrote past the geometry table");
        }
    }

    /**
     * Highest geometry-table index a camera-entity iteration writes, given the table occupancy it starts
     * from. writeTableEntry indexes by the pre-increment physical count, so an admitted pair writes
     * {@code logicalCount} and {@code logicalCount + 1}; a refused local view writes only the stand-in.
     */
    private static int highestWriteIndex(int capacity, int logicalCount) {
        return admitsLocalView(capacity, logicalCount) ? logicalCount + 1 : logicalCount;
    }
}
