# Versioned look packages

A look package is one authored scene-to-display calibration. It keeps values that must be tuned
together out of independent runtime configuration:

- absolute auto-exposure bounds and the four-point EV100 compensation curve;
- one scene-referred LMT;
- sun and full-moon illuminance in lux;
- the default block-emitter, night-sky, and procedural-star luminances in cd/m²;
- the fixed/phase-dependent moon-light split and sky saturation.

The built-in package is
`src/main/resources/caustica/rt/looks/default/look.json`. `schemaVersion` versions the JSON
contract; `packageVersion` versions the authored calibration. Changing any authored value or
replacing the LMT should increment `packageVersion`.

The runtime loads this package as one immutable unit. There are intentionally no independent
`exposure.min-ev`, `exposure.max-ev`, `exposure.curve`, or `tonemap.look` compatibility settings.
Sun NEE and the visible atmosphere/disc read the same package values, so they cannot drift apart.

## LMT

`lmt.resource` is resolved beside `look.json`. The default `lmt.bin` is the former Resolve Curve
asset imported from `65 Point Cube_9.2026-07-31_12.13.49.cube` (source SHA-256
`e8562e4f95160bb696a86f7975e42d5988f559fa5a3e6e598a28587520b52a8e`).

The `.cube` import contract is a normalized 3D table whose input and output are the renderer's
−12…+12 EV log2 shaper coordinates. To replace the package LMT:

```powershell
uv run python tools\bake_display_lut.py --import-lmt C:\path\to\look.cube
```

The LMT remains scene-referred ACEScg and is applied before the ACES 2.0 SDR/HDR Output
Transform.

## Material emission overrides

Material override schema 2 uses absolute emitting-surface luminance:

```json
{
  "format": 2,
  "match": { "sprite": "minecraft:block/torch" },
  "emission": { "strength_cd_m2": 6000.0 }
}
```

The value replaces the package's block baseline for an existing LabPBR, heuristic, or
state-uniform emission mask. It does not create an emission mask on a non-emissive material.
Format 1 and the old multiplier field `emission.strength` are rejected.
