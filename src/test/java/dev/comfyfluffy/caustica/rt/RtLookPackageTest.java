package dev.comfyfluffy.caustica.rt;

import com.google.gson.JsonParser;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

final class RtLookPackageTest {
    @Test
    void loadsCurrentCalibrationAsDefaultPackage() {
        RtLookPackage look = RtLookPackage.current();

        assertEquals(RtLookPackage.SCHEMA_VERSION, look.schemaVersion());
        assertEquals("default", look.id());
        assertEquals(1, look.packageVersion());
        assertEquals(-15.0f, look.exposure().minEv());
        assertEquals(-2.0f, look.exposure().maxEv());
        assertEquals("-2:-3, 2:-2.0, 8:0.0, 15:1.0", look.exposure().curve());
        assertEquals("/caustica/rt/looks/default/lmt.bin", look.lmtResource());
        assertEquals(128000.0f, look.lighting().sunIlluminanceLux());
        assertEquals(5.0f, look.lighting().moonIlluminanceLux());
        assertEquals(2000.0f, look.lighting().blockEmissionLuminanceCdM2());
        assertEquals(0.2f, look.lighting().nightSkyLuminanceCdM2());
        assertEquals(10.0f, look.lighting().starLuminanceCdM2());
        assertEquals(0.1f, look.lighting().moonPhaseFixedFraction());
        assertEquals(0.9f, look.lighting().moonPhaseFraction());
        assertEquals(1.2f, look.lighting().skySaturation());
    }

    @Test
    void rejectsUnknownSchemaAndInvalidPhysicalRanges() {
        assertThrows(IllegalArgumentException.class, () -> parse("""
                {"schemaVersion":2,"id":"bad","packageVersion":1,
                 "exposure":{"minEv":-15,"maxEv":-2,"curve":"-2:-3,2:-2,8:0,15:1"},
                 "lmt":{"resource":"lmt.bin"},
                 "lighting":{"sunIlluminanceLux":128000,"moonIlluminanceLux":5,
                 "blockEmissionLuminanceCdM2":2000,"nightSkyLuminanceCdM2":0.2,"starLuminanceCdM2":10,
                 "moonPhaseFixedFraction":0.1,"skySaturation":1.2}}
                """));
        assertThrows(IllegalArgumentException.class, () -> parse("""
                {"schemaVersion":1,"id":"bad","packageVersion":1,
                 "exposure":{"minEv":2,"maxEv":-2,"curve":"-2:-3,2:-2,8:0,15:1"},
                 "lmt":{"resource":"lmt.bin"},
                 "lighting":{"sunIlluminanceLux":128000,"moonIlluminanceLux":5,
                 "blockEmissionLuminanceCdM2":2000,"nightSkyLuminanceCdM2":0.2,"starLuminanceCdM2":10,
                 "moonPhaseFixedFraction":0.1,"skySaturation":1.2}}
                """));
    }

    private static RtLookPackage parse(String json) {
        return RtLookPackage.parse(JsonParser.parseString(json).getAsJsonObject(),
                "/caustica/rt/looks/test/look.json");
    }
}
