package dev.comfyfluffy.caustica.rt;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

final class RtCompositeMoonPhaseTest {
    @Test
    void moonNeeTracksMinecraftPhaseOrder() {
        assertEquals(1.0f, RtComposite.moonLitFraction(0.0f));  // full
        assertEquals(0.75f, RtComposite.moonLitFraction(1.0f));
        assertEquals(0.5f, RtComposite.moonLitFraction(2.0f));
        assertEquals(0.25f, RtComposite.moonLitFraction(3.0f));
        assertEquals(0.0f, RtComposite.moonLitFraction(4.0f));  // new
        assertEquals(0.25f, RtComposite.moonLitFraction(5.0f));
        assertEquals(0.5f, RtComposite.moonLitFraction(6.0f));
        assertEquals(0.75f, RtComposite.moonLitFraction(7.0f));
    }

    @Test
    void moonLightUsesTenPercentFixedAndNinetyPercentPhase() {
        assertEquals(1.000f, RtComposite.moonLightScale(0.0f), 1.0e-6f); // full
        assertEquals(0.775f, RtComposite.moonLightScale(1.0f), 1.0e-6f);
        assertEquals(0.550f, RtComposite.moonLightScale(2.0f), 1.0e-6f);
        assertEquals(0.325f, RtComposite.moonLightScale(3.0f), 1.0e-6f);
        assertEquals(0.100f, RtComposite.moonLightScale(4.0f), 1.0e-6f); // new
        assertEquals(0.325f, RtComposite.moonLightScale(5.0f), 1.0e-6f);
        assertEquals(0.550f, RtComposite.moonLightScale(6.0f), 1.0e-6f);
        assertEquals(0.775f, RtComposite.moonLightScale(7.0f), 1.0e-6f);
    }
}
