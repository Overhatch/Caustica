# Scene Units Plan — physical photometric units + pre-exposure

Status: **U0 + U1 implemented** (2026-07-29, compiles + tests pass, NOT yet GPU-verified); U2–U5
outstanding. Written 2026-07-29 against `bt2020-only`.

> **U0/U1 landed.** Metering and the compensation curve now run on EV100 (`RtSceneUnits`), and
> pre-exposure is plumbed end-to-end behind `exposure.pre-exposure` (default on). Both stages are
> **algebraically exact no-ops** — verified symbolically across scene luminances spanning 6 decades
> and pre-exposure values from 1e−3 to 7.5, rendered output identical in every case, and the curve's
> +3 x-axis shift is exact. **The light constants are still the old arbitrary scale**, so the
> reported EV100 is internally consistent but not yet physically true — do not compare it against
> §1's table until U2 lands.
>
> **Incident, fixed:** the first landing hand-wrote `ExposureResolvePush`'s byte offsets on both
> sides (shader struct + `RtExposurePipeline`'s `ByteBuffer.putFloat` calls) and inserted the two new
> fields at different positions in each — Java appended them at the end, the shader struct had them
> mid-list. That reinterpreted every later field; `preExposure` read as a curve control point
> (`-2.0`), clamped to a tiny epsilon, and `exposure / epsilon` blew every frame to white. Fixed two
> ways: (1) all four display-chain push-constant structs (`ExposureHistPush`, `ExposureResolvePush`,
> `DisplayPush`, `DebugPresentPush` — renamed from the generic `Push` every one of these files used)
> moved to a shared `shaders/display/display_common.slang` module and are now reflected by
> `generateShaderRecords` exactly like `WorldPush`/`WorldPushConstants`, so Java never hand-computes
> an offset again; (2) the resolve shader treats a non-finite/non-positive `preExposure` as 1.0
> (degrades to the pre-U1 pipeline) rather than clamping toward epsilon, so a future layout mismatch
> mis-exposes a frame instead of blowing it to white.

This is the third and most upstream of the three display-chain plans. The pipeline layers as:

```
SCENE_UNITS (this doc) → EXPOSURE_PLAN → DISPLAY_TRANSFORM_PLAN
  what a scene value      how bright       how it becomes pixels
  physically MEANS        to render it
```

Scope: what the number `1.0` in the renderer's scene-linear ACEScg buffer actually means, the
light constants that produce those numbers, and the pre-exposure mechanism that keeps them
representable in `R16G16B16A16_SFLOAT`. Metering and the compensation curve stay in
[EXPOSURE_PLAN.md](EXPOSURE_PLAN.md); the output transform stays in
[DISPLAY_TRANSFORM_PLAN.md](DISPLAY_TRANSFORM_PLAN.md).

## 0. Why

Today the scene scale is arbitrary-but-consistent — roughly "1.0 = a white surface in decent
light" — and every light constant was dialed by eye against it. Measured 2026-07-29:

| Ratio | Renderer | Physical | Error |
|---|---|---|---|
| Sun vs full-moon direct light | 6.7 EV | ~18.6 EV | **12 EV compressed** |
| Torch vs sun disc | 2.3 EV below | ~17.3 EV below | **15 EV hot** |

Two consequences. First, the constants are only meaningful relative to each other, so *every* one
of them has to be re-tuned by eye whenever any other changes — there is no external reference to
check against. Second, and the reason this now matters: with
[EXPOSURE_PLAN.md](EXPOSURE_PLAN.md)'s §S3 compensation curve implemented, the absolute scale is
no longer a free parameter. Under full adaptation a global rescale cancels exactly; under a curve
with slope < 1 it does not — it slides the scene along the curve. Scale and curve are now one
coupled tuning problem, and the only way to break the coupling is to pin the scale to something
external.

This project is also explicitly a place to learn and demonstrate current CG practice, which raises
the value of "principled" above "whatever looks fine": light values checkable against a photometric
table are portable knowledge, fifteen mutually-tuned magic numbers are not.

## 1. The convention

**A scene value is luminance in cd/m² (nits).** `1.0` = 1 cd/m². Radiometric-to-photometric
conversion is not modelled — the renderer is RGB, not spectral, so "luminance" here means the
AP1/D60 Y of the stored ACEScg triple (consistent with `math.slang`'s `luminance()` and the exposure
histogram, which use `ACESCG_LUMA`).

**Metering reports EV100**, the standard photographic scale, so the exposure curve's control points
are directly comparable to published tables:

```
EV100 = log2(L · S / K)  with S = 100, K = 12.5   →   EV100 = log2(8 · L)
```

That is a `+3` constant on `log2(luminance-in-nits)`. It costs one addition and makes the curve
config legible; see §4.

| Scene | lux | L @ 18% albedo | EV100 |
|---|---|---|---|
| Noon, clear | 100,000 | 5,730 | **+15.5** |
| Overcast | 10,000 | 573 | +12.2 |
| Sunset | 1,000 | 57.3 | +8.8 |
| Lit indoor / shade | 200 | 11.5 | +6.5 |
| Torch-lit cave | 15 | 0.86 | +2.8 |
| Full moon | 0.25 | 0.014 | −3.1 |
| Starlight / deep dark | 0.002 | 0.0001 | −10.1 |

These line up with standard photographic references (sunny-16 ≈ EV 15, overcast ≈ EV 12, full moon
≈ EV −3), which is the point: they are checkable.

## 2. Pre-exposure — the enabling mechanism

Physical units span ~26 EV from starlight surfaces to noon surfaces, plus ~17 EV more up to the sun
disc. `fp16` normals cover ~30 EV (6.1e−5 … 65504). Anchoring naively forces a lose-lose choice
between clipping the sun and flushing night to subnormals.

**Pre-exposure removes the choice** — the standard solution in Frostbite / UE (`View.PreExposure`)
/ Unity HDRP. Multiply radiance by an exposure estimate *before* the fp16 write, so stored values
hover near `key` regardless of absolute scene brightness:

```
raygen  writes   stored = L · preExposure          (preExposure = previous frame's exposure)
resolve reports  evScene(EV100) = log2(stored_metered) - log2(preExposure) + 3
display applies  residual = exposure / preExposure
```

Conceptual units stay fully physical; only storage is normalized. Consequences worth stating
explicitly:

- **fp16 precision is best exactly where the signal is**, at both noon and midnight, instead of
  being spent on absolute range nobody looks at.
- **The exposure histogram's `LOG_MIN/LOG_MAX = ±12` stays valid unchanged**, because it meters
  pre-exposed values. Without pre-exposure it would need widening to roughly −24…+14, and the deep
  end would still clip.
- **The display LUT shaper range is likewise unaffected** — it sees `stored · residual = L ·
  exposure`, algebraically identical to today.

**Risk: DLSS-RR temporal stability.** RR's history is at the previous frame's pre-exposure scale.
Two things make this benign: the exposure controller is already temporally smoothed (τ ≈ 0.4–0.8 s),
so frame-to-frame change is well under 1%; and the standard mitigation if it ever does bite is to
quantize `preExposure` to power-of-2 steps so it changes rarely and exactly. Do not pre-quantize
preemptively — measure first.

**Sun-disc clamp still required.** Even pre-exposed, a true 1.6e9 cd/m² disc overflows during
transitions (dark terrain metered while the sun is in frame → large preExposure → overflow). Clamp
at the fp16 write. Visually free: ACES 2.0 renders anything more than a few EV over white as pure
white, so 1e6 and 1.6e9 nits are indistinguishable.

## 3. Light constants

Targets. Every value derives from a published figure, so each is individually checkable rather than
mutually tuned.

| Constant | Today | Physical target | Source |
|---|---|---|---|
| Sun illuminance (noon, clear) | — (implicit) | **100,000 lux** | standard clear-sky noon |
| `sunPeak` (NEE radiance, `RtComposite.skyPush`) | 21.0 | **`100000 / Ω(radius)`** = 2.90e8 @ 0.6° | E = L·Ω |
| `SUN_DISC_RADIANCE` | 24.0 | ~1.6e9, **clamped at write** | solar disc luminance |
| Moon illuminance (full) | — | **0.25 lux** | full-moon ground illuminance |
| `moonPeak` (NEE radiance) | 0.20 | **`0.25 / Ω(radius)`** = 116 @ 1.5° | E = L·Ω |
| `MOON_DISC_RADIANCE` | 0.45 | **~3,000 cd/m²** | sunlit rock, albedo 0.12 |
| `SUN_INTENSITY` (atmosphere) | 22.0 | **~127,000** (solar constant, photometric) | needs in-game calibration, §6 |
| `NIGHT_ZENITH` / `NIGHT_HORIZON` | 0.0008 / 0.003 | **~0.0005 cd/m²** | airglow + starlight |
| `EMISSIVE_STRENGTH` | 5.0 | **~15,000 cd/m²** (torch flame) | wood flame luminance |

**Latent bug found while deriving this:** `sunPeak` is a hardcoded constant
([RtComposite.java](../src/main/java/dev/comfyfluffy/caustica/rt/RtComposite.java), `skyPush`)
while `SUN_ANGULAR_RADIUS` is configurable (0.6° default,
[CausticaConfig.java:549](../src/main/java/dev/comfyfluffy/caustica/CausticaConfig.java)). Since
irradiance `E = L · Ω` and `Ω ∝ sin²(radius)`, widening the sun for softer shadows currently also
*brightens the whole scene* — changing a shadow-softness knob changes exposure. Deriving radiance
from illuminance fixes this by construction, and is a good argument for the change independent of
everything else.

Sanity check on the emissive figure: a torch quad ~0.1 × 0.1 m at 15,000 cd/m² gives intensity
`I = L·A` = 150 cd, so ~17 lux at 3 m — dim-room level, and real torches are ~10–50 cd. The right
order of magnitude, unlike today's value which puts a torch 2.3 EV under the sun.

## 4. Exposure curve, derived

With metering in EV100, the compensation curve's control points can be read against §1's table.
Rendered median (log) = `log2(key) + comp(evScene)`, so `comp` **is** the rendered offset in EV
from the noon reference.

```
exposure.curve = "-10:-3.5, 0:-2.0, 10:-0.6, 15.5:0.0"
```

| Scene | EV100 | rendered, vs noon |
|---|---|---|
| Noon clear | +15.5 | 0.00 |
| Overcast | +12.2 | −0.36 |
| Sunset | +8.8 | −0.76 |
| Lit indoor / shade | +6.5 | −1.09 |
| Torch-lit cave | +2.8 | −1.61 |
| Full moon | −3.1 | −2.47 |
| Starlight / deep dark | −10.1 | −3.50 (floor) |

Effective slope is 0.85 / 0.86 / 0.89 across the three segments — deliberately uniform, so the
curve is "partial adaptation at ~0.86" rather than an arbitrary shape. Note the physical 25.6 EV
scene range compresses to 3.5 EV of rendered difference; that compression is exactly the artistic
decision the curve exists to express, now made explicitly in one place instead of being smeared
across fifteen light constants.

**Exposure clamps must widen substantially.** The multiplier now spans:

| Scene | exposure multiplier | EV |
|---|---|---|
| Noon | 3.14e−5 | **−14.96** |
| Overcast | 2.45e−4 | −12.00 |
| Torch cave | 0.069 | −3.86 |
| Full moon | 2.27 | +1.18 |
| Deep dark | 159 | **+7.31** |

Current `min-ev = -1.5` / `max-ev = 4.0` clamp to a 5.5 EV window and would saturate everywhere.
New defaults: **`min-ev = -18`, `max-ev = +10`** (margin at both ends). Also raise
`Exposure.clampScale`'s `1e-4 … 1e4` bound — `−18 EV` is 3.8e−6, below the current floor.

## 5. Stages

**U0 — EV100 reporting only.** Add the `+3` offset (and the `−log2(preExposure)` term, zero until
U1) to the metered EV in the resolve's `evScene` diagnostic and the debug readout. No visual
change; makes the existing S0 observability speak the same scale the rest of this plan uses.

**U1 — pre-exposure plumbing, no constant changes.** Raygen multiplies by `preExposure`; the
resolve subtracts it back out and writes the residual to the exposure image. **This stage is a
visual no-op** — every product `L · exposure` is algebraically unchanged — which makes it directly
verifiable: the image must look identical, and any difference is a bug. Land it before touching a
single light value.

**U2 — sun / moon / sky constants** (§3), including deriving NEE radiance from illuminance and the
configured angular radius. Widen the exposure clamps (§4) *in the same commit* — with physical
values and the old clamps, exposure saturates and the image is unusable.

**U3 — emissive constants.** `EMISSIVE_STRENGTH` plus an audit of per-material JSON multipliers.
Highest art-facing risk: emitters are the values most likely to have absorbed compensation for
AgX's old 4 EV highlight ceiling.

**U4 — curve retune** against the shipped values, starting from §4's derivation and adjusted by
eye. This is the only stage that is genuinely subjective.

**U5 — validation.** The reference scenes of §1 plus the glowstone/lava scene
[DISPLAY_TRANSFORM_PLAN.md](DISPLAY_TRANSFORM_PLAN.md) §4 already calls for. Confirm measured EV100
in the debug readout matches the table within ~1 EV — that is the whole payoff, and it is a real
pass/fail rather than an aesthetic judgement.

## 6. Open questions

1. **Does `SUN_INTENSITY = 127,000` actually produce ~5,000 cd/m² zenith sky?** The Nishita march
   uses physical Rayleigh/Mie/ozone coefficients, so units *should* work out with irradiance in,
   radiance out — but the implementation may carry baked-in normalizations. Needs one in-game
   calibration pass reading zenith luminance off the debug view; treat the table value as a
   starting point, not a result.
2. **Does the sun-disc clamp interact with MIS?** The visible disc (`world.rmiss`) and the NEE light
   are already decoupled — different angular radii, and a `showCelestial` gate hides the disc from
   diffuse continuations specifically to avoid double-counting
   ([world.rgen.slang:61](../shaders/world/world.rgen.slang)). So clamping the visible disc should
   not perturb the NEE estimator. Worth confirming rather than assuming, since energy conservation
   between the two paths is exactly what that gate is managing.
3. **Should night get an explicit ambient floor?** Physically, moonless starlight is unplayable and
   every shipping game lifts it. Currently the `NIGHT_ZENITH`/`NIGHT_HORIZON` fudge does this
   implicitly. Better to make it an explicit, named gameplay floor than a fudged sky constant — but
   that is a design decision, deferred until U4 shows how dark it actually reads.
4. **Do per-material emissive multipliers need rescaling or re-authoring?** If they were authored as
   ratios against `EMISSIVE_STRENGTH` they rescale for free; if they absorbed AgX-era compensation
   individually, they need a pass. Unknown until U3.
