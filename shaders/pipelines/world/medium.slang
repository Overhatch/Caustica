// Participating media: the tint-to-extinction mappings and the depth-3 medium stack the dielectric
// interface pushes and pops. Depends on core only.

// Per-channel Beer–Lambert extinction from a water body's biome tint (carried in the payload's albedo view
// for water hits, or worldPush.waterParams.xyz for the camera's own biome when starting submerged). A blue
// ocean tint (low red) absorbs red fastest → bluer with depth; swamp green-brown shifts the hue. The floor
// keeps even a white-tinted body very slightly absorbing so deep water never reads as clear vacuum.

import world_common;
import world_core;

public static const float WATER_DENSITY = 0.1;
public static const float3 WATER_ABSORB_FLOOR = float3(0.015, 0.010, 0.008);
public float3 waterExtinction(float3 tint) {
    return WATER_ABSORB_FLOOR + WATER_DENSITY * (float3(1.0, 1.0, 1.0) - clamp(tint, 0.0, 1.0));
}

// A non-water dielectric's tint is a filter over one block of travel rather than a biome water colour, so
// it maps to extinction by inverting Beer-Lambert over that reference distance: transmittance t after one
// block means sigma = -ln(t). `transmission` scales that transmittance, so it keeps the sense the
// material's authored factor has everywhere else — 1 passes the tint through unchanged, 0 is opaque.
// Clamped away from zero so a fully saturated texel stays finite instead of producing an infinite sigma.
public static const float VOLUME_TINT_REFERENCE_DISTANCE = 1.0; // blocks
public float3 volumeExtinction(float3 tint, float transmission) {
    float3 blockTransmittance = clamp(clamp(tint, 0.0, 1.0) * clamp(transmission, 0.0, 1.0),
            1.0e-3, 1.0);
    return -log(blockTransmittance) / VOLUME_TINT_REFERENCE_DISTANCE;
}

// ---- Participating medium the path is currently inside.
//
// Refraction needs the RELATIVE index across an interface, so the renderer has to know what it is
// travelling through, not just what it is hitting. `ior` drives Snell/Fresnel; `extinction` drives the
// per-segment Beer-Lambert attenuation.
//
// The stack is depth 3 (current + the two it will return to) held in named fields, NOT an array. A
// dynamically indexed local array lands in scratch memory, and this raygen is already register-bound —
// paying an occupancy hit for nesting the game rarely produces would be a bad trade. Three layers cover
// air->water->glass plus a held dielectric, and because `entering` is re-derived per face from geometry
// rather than toggled, the path re-synchronises at the next crossing instead of staying corrupted.
//
// Air and water own fixed identities; MEDIUM_ID_GENERIC_DIELECTRIC covers every other dielectric.
// Identity comparisons are integer-only — optical parameters never decide what a medium IS.
public static const uint MEDIUM_ID_AIR = 0u;
public static const uint MEDIUM_ID_WATER = 1u;
public static const uint MEDIUM_ID_GENERIC_DIELECTRIC = 2u;

public struct Medium {
    public float ior;
    public float3 extinction;
    public uint mediumId; // canonical 16-bit identity (MEDIUM_ID_*)
};

// Water-specific behaviour — the wave-refraction caustic on submerged receivers, the water miss guard —
// keys off the identity; nothing else about a medium is water-specific.
public bool mediumIsWater(Medium m) {
    return m.mediumId == MEDIUM_ID_WATER;
}

public struct MediumStack {
    public Medium current;
    public Medium parent1;
    public Medium parent2;
};

public Medium airMedium() {
    Medium m;
    m.ior = 1.0;
    m.extinction = float3(0.0, 0.0, 0.0);
    m.mediumId = MEDIUM_ID_AIR;
    return m;
}

public MediumStack makeMediumStack(Medium start) {
    MediumStack s;
    s.current = start;
    s.parent1 = airMedium();
    s.parent2 = airMedium();
    return s;
}

public void mediumPush(inout MediumStack stack, Medium entered) {
    stack.parent2 = stack.parent1;
    stack.parent1 = stack.current;
    stack.current = entered;
}

public void mediumPop(inout MediumStack stack) {
    stack.current = stack.parent1;
    stack.parent1 = stack.parent2;
    stack.parent2 = airMedium();
}

// Water's tint is a biome colour whose absorption is calibrated per block of depth; any other volume
// dielectric's tint is a filter over a reference block of travel. Both end up as per-channel extinction.
public Medium makeDielectricMedium(uint mediumId, float ior, float3 tint, float transmission) {
    Medium m;
    m.ior = ior;
    m.extinction = mediumId == MEDIUM_ID_WATER
            ? waterExtinction(tint) : volumeExtinction(tint, transmission);
    m.mediumId = mediumId;
    return m;
}
