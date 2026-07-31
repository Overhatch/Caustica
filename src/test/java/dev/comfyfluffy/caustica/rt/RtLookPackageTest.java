package dev.comfyfluffy.caustica.rt;

import com.google.gson.JsonParser;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertThrows;

final class RtLookPackageTest {
    @Test
    void rejectsUnknownSchemaAndInvalidPhysicalRanges() {
        assertThrows(IllegalArgumentException.class, () -> parse("""
                {"schemaVersion":4,"id":"bad","packageVersion":1,
                 "exposure":{"minEv":-15,"maxEv":-2,"curve":"-2:-3,2:-2,8:0,15:1"},
                 "lmt":{"resource":"lmt.bin"},
                 "bloom":{"strength":0.08,"thresholdSceneLinear":1,"softKneeFraction":0.5,"radius":1},
                 "lighting":{"sunIlluminanceLux":128000,"moonIlluminanceLux":5,
                 "blockEmissionLuminanceCdM2":2000,"nightSkyLuminanceCdM2":0.2,"starLuminanceCdM2":10,
                 "twilightFillLuminanceCdM2":120,"twilightShadowSoftnessDegrees":2,
                 "moonPhaseFixedFraction":0.1,"skySaturation":1.2}}
                """));
        assertThrows(IllegalArgumentException.class, () -> parse("""
                {"schemaVersion":3,"id":"bad","packageVersion":1,
                 "exposure":{"minEv":2,"maxEv":-2,"curve":"-2:-3,2:-2,8:0,15:1"},
                 "lmt":{"resource":"lmt.bin"},
                 "bloom":{"strength":0.08,"thresholdSceneLinear":1,"softKneeFraction":0.5,"radius":1},
                 "lighting":{"sunIlluminanceLux":128000,"moonIlluminanceLux":5,
                 "blockEmissionLuminanceCdM2":2000,"nightSkyLuminanceCdM2":0.2,"starLuminanceCdM2":10,
                 "twilightFillLuminanceCdM2":120,"twilightShadowSoftnessDegrees":2,
                 "moonPhaseFixedFraction":0.1,"skySaturation":1.2}}
                """));
    }

    private static RtLookPackage parse(String json) {
        return RtLookPackage.parse(JsonParser.parseString(json).getAsJsonObject(),
                "/caustica/rt/looks/test/look.json");
    }
}
