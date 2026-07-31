# Versioned look packages

A look package is one authored scene-to-display calibration. It keeps values that must be tuned
together out of independent runtime configuration:

- absolute auto-exposure bounds and the four-point EV100 compensation curve;
- one scene-referred LMT;
- scene-referred bloom strength, threshold, soft knee, tent radius, and pyramid depth;
- sun and full-moon illuminance in lux;
- the default block-emitter, night-airglow, and procedural-star luminances in cd/m²;
- the fixed/phase-dependent moon-light split;
- the sky's geometry: noon tilt, NEE angular radii, drawn disc sizes, viewer altitude, ground albedo.

The built-in package is
`src/main/resources/caustica/color/looks/default/look.json`. `schemaVersion` versions the JSON
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

## Bloom

Bloom is extracted from exposed scene-linear ACEScg after DLSS-RR and exposure metering, but before
the LMT and ACES Output Transform. It is a downsample/upsample pyramid (Jimenez, SIGGRAPH 2014):
a thresholded 13-tap Karis-averaged downsample to half resolution, further 13-tap downsamples up the
chain, then 3×3 tent upsamples accumulating each band back onto the level below.

- `strength` scales the reconstructed bloom before the LMT. The compositor divides it by the live
  level count, so the authored value means the same thing whatever depth the resolution supports.
- `thresholdSceneLinear` is the exposed ACEScg luminance where the hard part of extraction begins.
- `softKneeFraction` widens the transition below the threshold as a fraction of that threshold.
- `radius` is the upsample tent radius in **source texels**. It is resolution-independent by
  construction: each level's texel already scales with the frame.
- `levels` is the pyramid depth (1…8), and is what sets how far the glow reaches.

The pyramid replaced a single wide separable Gaussian, which had two visible failures. Its nine taps
per axis were spaced `radius` half-resolution pixels apart, so for a small bright source the output
was nine shifted copies per axis — a visible grid of replicas. And one fixed width has a hard
support, so a source far above the threshold saturated its whole footprint into a flat slab with an
edge instead of a decaying skirt. A pyramid never samples further than a texel or two at its own
level, and the sum of its octaves is a smooth falloff at every intensity.

Bloom is deliberately downstream of exposure metering and temporal reconstruction, so it cannot
make auto exposure chase its own glow or feed blurred detail into DLSS history. The same
scene-referred result goes through both SDR and HDR output transforms.

## Sky

The `sky` section owns everything about the sky's geometry that is not Minecraft world state. These
were `caustica.rt.*` system properties (`sunAngularRadius`, `moonAngularRadius`, `sunNoonSouthDeg`)
and one video-settings slider; they moved here because a package that cannot describe the shape of
its own sky is not a complete look.

- `sunNoonSouthTiltDegrees` tilts the east–west celestial arc toward +Z at its peak.
- `sunAngularRadiusDegrees` / `moonAngularRadiusDegrees` are the half-angles the NEE shadow ray
  samples about the body. They set penumbra softness (and the softness of the planet's shadow on the
  atmosphere) — never the light's level, which comes from the illuminance anchors.
- `sunDiscHalfAngleDegrees` / `moonDiscHalfAngleDegrees` are how large each body is **drawn**,
  matching vanilla's quads (`atan(0.30)` and `atan(0.20)`). Drawn radiance is derived as
  illuminance ÷ drawn solid angle, so enlarging the disc never adds power to the scene.
- `groundAlbedo` is the Lambertian ground the atmosphere sees, which fills below-horizon directions.
- `horizonSoftenDegrees` is the dip over which that ground fades in below the horizon.

The sky-view LUT's viewer altitude is **not** a package constant: it tracks the camera's actual world
height above sea level (`RtComposite.skyPush`, clamped to the modelled [0, 99] km shell), so a
build-limit or space/rocket mod climbing toward the top of the atmosphere sees it actually thin out.

### Horizon softening

The atmosphere's ground is a genuine discontinuity that Minecraft never justifies. A ray an arcminute
above the ground-tangent direction escapes the atmosphere; one an arcminute below terminates on the
surface a few km away. Measured at sea level under a 45° sun that step is **20 EV per degree** of
elevation (51 EV/deg at a 5° sun), and everything below it is a nearly constant Lambertian plate —
a hard grey line across the distant view. Two things make it worse than it is on Earth: the ground is
fictional (it is not the world's terrain, which is what the player would actually see there, and
beyond render distance the rays simply escape), and with the viewer altitude tracking world Y a
player near sea level sits within metres of the shell, where the ground is reached immediately below
the horizon and no in-scatter accumulates to soften it.

So `sky_view.comp` fades the ground in over `horizonSoftenDegrees` instead of switching to it. The
value it fades *from* is the ground-tangent direction marched with **no** ground clip — exactly the
limit the rows above the horizon converge to — so the two sides meet at the same number rather than
merely being blurred together. The interpolation runs in log luminance: the ends can be 5 EV apart,
and a linear blend would spend most of its angular range near the bright end and then fall off a
cliff, trading one hard edge for a slightly softer one. Log space spreads the drop evenly in stops,
which is also how attenuation through haze actually behaves.

At the 15° default the steepest remaining gradient is ~0.2–0.5 EV/deg — comparable to the sky's own
gradient just above the horizon, i.e. a haze falloff rather than an edge. Set it to `0` to restore
the hard physical ground.

See `shaders/world/sky.slang` for the atmosphere model (Hillaire 2020) and `RtSkyLut` for the three
LUTs and the passes that bake them.

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
