package dev.comfyfluffy.caustica.rt;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

/**
 * Immutable, versioned calibration package for the scene-to-display pipeline.
 *
 * <p>The default package is deliberately a classpath asset rather than mutable TOML configuration:
 * exposure shaping, its scene-referred LMT, and the photometric light anchors are one authored look
 * and must move together. A future package selector can load another directory with the same schema
 * without reintroducing independent knobs.
 */
public record RtLookPackage(
        int schemaVersion,
        String id,
        int packageVersion,
        Exposure exposure,
        String lmtResource,
        Lighting lighting) {
    public static final int SCHEMA_VERSION = 1;
    public static final String DEFAULT_ID = "default";
    public static final String DEFAULT_JSON = "/caustica/rt/looks/default/look.json";
    private static final RtLookPackage DEFAULT = load(DEFAULT_JSON);

    public static RtLookPackage current() {
        return DEFAULT;
    }

    static RtLookPackage parse(JsonObject root, String jsonResource) {
        int schemaVersion = requiredInt(root, "schemaVersion");
        if (schemaVersion != SCHEMA_VERSION) {
            throw new IllegalArgumentException(jsonResource + ": unsupported look-package schema "
                    + schemaVersion);
        }
        String id = requiredString(root, "id");
        if (id.isBlank()) {
            throw new IllegalArgumentException(jsonResource + ": id must not be blank");
        }
        int packageVersion = requiredInt(root, "packageVersion");
        if (packageVersion < 1) {
            throw new IllegalArgumentException(jsonResource + ": packageVersion must be at least 1");
        }

        JsonObject exposureJson = requiredObject(root, "exposure");
        Exposure exposure = new Exposure(
                requiredFinite(exposureJson, "minEv"),
                requiredFinite(exposureJson, "maxEv"),
                requiredString(exposureJson, "curve"));
        if (exposure.minEv() > exposure.maxEv()) {
            throw new IllegalArgumentException(jsonResource + ": exposure.minEv must be <= maxEv");
        }
        if (exposure.curve().isBlank()) {
            throw new IllegalArgumentException(jsonResource + ": exposure.curve must not be blank");
        }
        validateCurve(exposure.curve(), jsonResource);

        JsonObject lmt = requiredObject(root, "lmt");
        String lmtFile = requiredString(lmt, "resource");
        if (lmtFile.isBlank() || lmtFile.contains("/") || lmtFile.contains("\\")
                || ".".equals(lmtFile) || "..".equals(lmtFile)) {
            throw new IllegalArgumentException(jsonResource + ": lmt.resource must be a local file name");
        }
        int slash = jsonResource.lastIndexOf('/');
        if (slash < 0) {
            throw new IllegalArgumentException("look-package resource must be absolute: " + jsonResource);
        }
        String lmtResource = jsonResource.substring(0, slash + 1) + lmtFile;

        JsonObject lightingJson = requiredObject(root, "lighting");
        Lighting lighting = new Lighting(
                positive(lightingJson, "sunIlluminanceLux", jsonResource),
                positive(lightingJson, "moonIlluminanceLux", jsonResource),
                positive(lightingJson, "blockEmissionLuminanceCdM2", jsonResource),
                nonNegative(lightingJson, "nightSkyLuminanceCdM2", jsonResource),
                nonNegative(lightingJson, "starLuminanceCdM2", jsonResource),
                nonNegative(lightingJson, "moonPhaseFixedFraction", jsonResource),
                nonNegative(lightingJson, "skySaturation", jsonResource));
        if (lighting.moonPhaseFixedFraction() > 1.0f) {
            throw new IllegalArgumentException(jsonResource
                    + ": lighting.moonPhaseFixedFraction must be in [0,1]");
        }
        return new RtLookPackage(schemaVersion, id, packageVersion, exposure, lmtResource, lighting);
    }

    private static RtLookPackage load(String resource) {
        try (InputStream stream = RtLookPackage.class.getResourceAsStream(resource)) {
            if (stream == null) {
                throw new IllegalStateException("missing look package " + resource);
            }
            try (InputStreamReader reader = new InputStreamReader(stream, StandardCharsets.UTF_8)) {
                RtLookPackage value = parse(JsonParser.parseReader(reader).getAsJsonObject(), resource);
                if (RtLookPackage.class.getResource(value.lmtResource()) == null) {
                    throw new IllegalStateException(resource + ": missing LMT " + value.lmtResource());
                }
                return value;
            }
        } catch (IOException | RuntimeException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    private static JsonObject requiredObject(JsonObject object, String name) {
        JsonElement value = object.get(name);
        if (value == null || !value.isJsonObject()) {
            throw new IllegalArgumentException("missing object " + name);
        }
        return value.getAsJsonObject();
    }

    private static String requiredString(JsonObject object, String name) {
        JsonElement value = object.get(name);
        if (value == null || !value.isJsonPrimitive() || !value.getAsJsonPrimitive().isString()) {
            throw new IllegalArgumentException("missing string " + name);
        }
        return value.getAsString();
    }

    private static int requiredInt(JsonObject object, String name) {
        JsonElement value = object.get(name);
        if (value == null || !value.isJsonPrimitive() || !value.getAsJsonPrimitive().isNumber()) {
            throw new IllegalArgumentException("missing integer " + name);
        }
        return value.getAsInt();
    }

    private static float requiredFinite(JsonObject object, String name) {
        JsonElement value = object.get(name);
        if (value == null || !value.isJsonPrimitive() || !value.getAsJsonPrimitive().isNumber()) {
            throw new IllegalArgumentException("missing number " + name);
        }
        float result = value.getAsFloat();
        if (!Float.isFinite(result)) {
            throw new IllegalArgumentException(name + " must be finite");
        }
        return result;
    }

    private static float positive(JsonObject object, String name, String resource) {
        float value = requiredFinite(object, name);
        if (value <= 0.0f) {
            throw new IllegalArgumentException(resource + ": lighting." + name + " must be positive");
        }
        return value;
    }

    private static float nonNegative(JsonObject object, String name, String resource) {
        float value = requiredFinite(object, name);
        if (value < 0.0f) {
            throw new IllegalArgumentException(resource + ": lighting." + name + " must be non-negative");
        }
        return value;
    }

    private static void validateCurve(String spec, String resource) {
        String[] points = spec.split(",");
        if (points.length != 4) {
            throw new IllegalArgumentException(resource
                    + ": exposure.curve must contain exactly four sceneEv:compensationEv points");
        }
        float previousSceneEv = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < points.length; i++) {
            String[] pair = points[i].trim().split(":", -1);
            if (pair.length != 2) {
                throw new IllegalArgumentException(resource + ": exposure.curve point " + (i + 1)
                        + " is not sceneEv:compensationEv");
            }
            float sceneEv;
            float compensationEv;
            try {
                sceneEv = Float.parseFloat(pair[0].trim());
                compensationEv = Float.parseFloat(pair[1].trim());
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException(resource + ": exposure.curve point " + (i + 1)
                        + " contains a non-number", e);
            }
            if (!Float.isFinite(sceneEv) || !Float.isFinite(compensationEv)) {
                throw new IllegalArgumentException(resource + ": exposure.curve point " + (i + 1)
                        + " must be finite");
            }
            if (sceneEv - previousSceneEv < 1.0e-4f) {
                throw new IllegalArgumentException(resource
                        + ": exposure.curve scene EV points must be strictly increasing");
            }
            previousSceneEv = sceneEv;
        }
    }

    public record Exposure(float minEv, float maxEv, String curve) {
    }

    public record Lighting(
            float sunIlluminanceLux,
            float moonIlluminanceLux,
            float blockEmissionLuminanceCdM2,
            float nightSkyLuminanceCdM2,
            float starLuminanceCdM2,
            float moonPhaseFixedFraction,
            float skySaturation) {
        public float moonPhaseFraction() {
            return 1.0f - moonPhaseFixedFraction;
        }
    }
}
