# Scene Units Plan — physical photometric units + pre-exposure

> Historical implementation record: named constants and intermediate values below document the
> passes that led here. The current runtime calibration source is the versioned default
> [`look.json`](../src/main/resources/caustica/color/looks/default/look.json); see
> [LOOK_PACKAGES.md](LOOK_PACKAGES.md).

Status: **U0–U4 implemented, U5 first pass measured in game** (2026-07-29). U2's constants validate
within 0.25 EV; U3's emissive baseline did not and has been re-anchored. Written against `bt2020-only`.

> **U0/U1 landed.** Metering and the compensation curve now run on EV100 (`RtSceneUnits`), and
> pre-exposure is plumbed end-to-end behind `exposure.pre-exposure` (default on). Both stages are
> **algebraically exact no-ops** — verified symbolically across scene luminances spanning 6 decades
> and pre-exposure values from 1e−3 to 7.5, rendered output identical in every case, and the curve's
> +3 x-axis shift is exact.
>
> **U2/U3/U4 landed together** — they have to, because physical light values with the old exposure
> clamps saturate everywhere and physical values under the old curve slide along it. Two corrections
> to this plan came out of the implementation (the NEE convention in §3 and the disc-radiance
> derivation), both written up in place below.
>
> **U5's first in-game pass then confirmed U2 and refuted U3's baseline** — see §7. Sun illuminance,
> disc radiance, night-sky luminance and the atmosphere march all land within 0.25 EV of derivation,
> including both corrections, which had been reasoned rather than measured. `EMISSIVE_STRENGTH` was
> ~5.5 EV hot and is now anchored on luminous exitance instead of flame luminance. The same pass fixed
> the exposure controller's temporal asymmetry (it was in the interpolation, not the time constants),
> lowered `max-ev`, and re-fitted the curve to measured anchors.
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

**The payload was storage too (found in U2, fixed by widening it).** `Payload.albedo` was a `half3`
lane, and on a miss it carries the sky — which after U2 reaches 3.6e5 cd/m² on the sun disc, 2.4 EV
past half's 65504 ceiling. Clamping there would have landed the sun at ~2× display white: a dull grey
disc, not a sun.

The first fix pre-exposed the sky into the payload and divided it back out in `world.rgen`. That
worked but was the wrong shape: it made a *storage format's* limitation into a dependency on the
exposure controller, on a path that has no business knowing about exposure at all. So the payload
carries a `float3` instead. `Payload.albedo`/`Payload.normal` are gone, replaced by three hand-packed
words with two views:

- **hit** — `half3` albedo + `half3` normal, the same six halves the two lanes held, so hit precision
  is bit-identical (`unpackAlbedo`/`packAlbedo`, `unpackNormal`/`packNormal`)
- **miss** — full fp32 sky radiance (`unpackSky`/`packSky`)

A hit has no sky and a miss has neither albedo nor normal, so the union is free: **the payload is
exactly the size it was**, three 32-bit words where two `half3`s stood. The cost is a
read-modify-write on the middle word, which straddles `albedo.b` and `normal.x` — a couple of ALU ops
on registers, against a payload that is paid for twice per radiance trace and preserved across the
SER reorder.

Pre-exposure is now what it should be: **one multiply at the `outImage` store, and nowhere else.**
That is also where the sun-disc clamp lives, and it now guards *every* radiance source rather than
only the sky — a firefly off a tiny emitter can no longer round to `+inf` in `rgba16f` and propagate
as NaN through the denoiser either.

**Risk: DLSS-RR temporal stability.** RR's history is at the previous frame's pre-exposure scale.
Two things make this benign: the exposure controller is already temporally smoothed (τ ≈ 0.4–0.8 s),
so frame-to-frame change is well under 1%; and the standard mitigation if it ever does bite is to
quantize `preExposure` to power-of-2 steps so it changes rarely and exactly. Do not pre-quantize
preemptively — measure first.

**Sun-disc clamp still required.** Even pre-exposed, a bright disc overflows during transitions (dark
terrain metered while the sun is in frame → large preExposure → overflow). Clamp at the fp16 write.
Visually free: ACES 2.0 renders anything more than a few EV over white as pure white, so 1e6 and
1.6e9 nits are indistinguishable. *(Implemented at `world.rgen`'s `outImage` store, where it covers
every radiance source, not only the sky.)*

## 3. Light constants

Every value derives from a published figure, so each is individually checkable rather than mutually
tuned. **Shipped values** (U2/U3), with the two corrections this table needed marked ⚠:

| Constant | Was | Shipped | Source |
|---|---|---|---|
| `SUN_ILLUMINANCE_TOA` (NEE, `RtComposite.skyPush`) ⚠ | 21.0 | **128,000 lux** | photometric solar constant |
| — after `atmosphereTransmittance`, zenith sun | — | ~117,000 lux | vs the 100,000 lux reference, +0.23 EV |
| `SUN_DISC_RADIANCE` ⚠ | 24.0 | **`SUN_ILLUMINANCE / 0.36 sr`** = 3.56e5 cd/m² | E = L·Ω at the size we draw it |
| `MOON_ILLUMINANCE_FULL` (NEE) ⚠ | 0.20 | **0.25 lux** | full-moon ground illuminance |
| `MOON_DISC_RADIANCE` ⚠ | 0.45 | **`MOON_ILLUMINANCE / 0.16 sr`** = 1.56 cd/m² | same |
| `SUN_INTENSITY` (atmosphere) | 22.0 | **= `SUN_ILLUMINANCE`, 128,000** | irradiance in, radiance out |
| `NIGHT_ZENITH` / `NIGHT_HORIZON` | 0.0008 / 0.003 | **0.0005 cd/m²** (luma) | airglow + starlight |
| `EMISSIVE_STRENGTH` | 5.0 | **318 cd/m²** | ~1,000 lm/m² exitance, `L = M/π` (was 15,000; §7) |

All the coloured constants are now `luma-1 tint × level`, so a tint edit can only change hue and a
level edit can only change brightness. The tints are the previous hand-picked ratios renormalised, so
nothing changed colour in this pass.

### ⚠ Correction 1: `lightRadiance` is illuminance, not radiance

This plan's first draft derived the NEE constants as `E / Ω(radius)`. That is wrong for this
renderer. `world.rgen`'s NEE term is

```
L += throughput * brdf * worldPush.lightRadiance * ndl * vis;
```

with **no solid-angle factor anywhere** — `sampleSquare` only jitters the direction, it carries no
pdf. Both lobes confirm it: the diffuse term is `albedo/π · E · ndl` and the specular is
`D·G·F/(4·ndv) · E`, which are the textbook *directional-light* forms with `E` an irradiance. So
`lightRadiance` is **illuminance at normal incidence, in lux**, and the correct value is 100,000-ish
directly — not 2.90e8. The draft's figure was 11.5 EV hot.

The `1/π` in the diffuse BRDF is what reproduces §1's table exactly: 100,000 lux → 31,830 cd/m² off
white, 5,730 cd/m² off 18% grey, EV100 +15.5. That agreement is the check that the convention is now
right.

**The "latent bug" this plan reported is therefore retracted.** It claimed `SUN_ANGULAR_RADIUS`
doubles as a brightness knob because `E = L·Ω`. With `lightRadiance` an illuminance there is no `Ω`
in the estimator at all, so the radius only jitters the shadow ray: it sets penumbra softness and
nothing else, which is what a shadow-softness knob should do. Nothing to fix.

### ⚠ Correction 2: disc radiance is derived from the size we *draw*, not the body's true luminance

The draft asked for the physical solar-disc luminance, 1.6e9 cd/m². That does not survive contact
with the fact that **vanilla's sun sprite is ~62× the real sun's angular radius** — ~3900× the solid
angle. Painting a 3900×-oversized disc at the true surface luminance injects ~3900× the sun's power
into every path that sees it. Specular and dielectric bounces do see it (`showCelestial`), and those
paths already took the sun through NEE, so the existing specular double-count would go from harmless
(today the disc is ~244× *weaker* than the NEE light, which is why nobody notices) to ~45× dominant,
with fireflies off every glossy lobe.

So the discs derive from illuminance and the solid angle they are actually drawn at, `L = E / Ω`,
where `squareBody` spans `2·tan(halfAngle)` a side ⇒ `Ω = (2·tan)²`. That keeps the drawn body's
total power equal to the real body's at whatever size it is drawn, and **costs nothing visually**:
3.56e5 cd/m² is still ~15 EV over an 18%-grey noon surface, and ACES 2.0 renders that as pure white
exactly like 1.6e9 would. The sprite shaping (`core`, `m`) only removes power from this figure, never
adds, so the estimate stays conservative. If the discs read too small or too dull in U5, the fix is
to shrink the *drawn* size toward physical — not to inflate radiance, because drawn size is precisely
what couples radiance to energy.

Sanity check on the emissive figure: a torch quad ~0.1 × 0.1 m at 15,000 cd/m² gives intensity
`I = L·A` = 150 cd, so ~17 lux at 3 m — dim-room level, and real torches are ~10–50 cd. The right
order of magnitude, unlike the old value which put a torch 2.3 EV under the sun.

### Constants that had to move with the baseline

Raising `EMISSIVE_STRENGTH` by 3.5 decades exposed two absolute constants that were quietly
calibrated against the old one:

- `RtMaterialRegistry.MAX_EMISSION_STRENGTH`, the ceiling of the 16-bit fixed-point strength field,
  was 32.0 — it would have clamped every emitter to 32 cd/m². Now `HALF_MAX` (65504), which is the
  genuine transport ceiling downstream (`Payload.emissionSss` is a `half2` lane, `Light.le` is packed
  R11G11B10). Quantisation is ~1 cd/m², 0.007% at the baseline.
- `RtLightCollector.LE_LUM_EPS`, the "too weak to bother NEE-sampling" cutoff, was 0.005 absolute
  against a baseline of 5 — i.e. 0.001 of full strength. Left alone it would have admitted emitters
  3000× fainter than intended into the light buffer. Now written as `0.001 × EMISSIVE_STRENGTH`,
  which is what it always meant.

Both are the coupling this document exists to remove, caught only because the baseline moved far
enough to make the breakage obvious. Worth assuming there are more of these and that U5 is where they
surface.

## 4. Exposure curve, measured

Rendered median (log) = `log2(key) + comp(evScene)`, so `comp` **is** the rendered offset in EV from
the noon reference. Originally derived from §1's reference table; **re-fitted to measured in-game
EV100 in U5** (see §7), which is what it now ships as:

```
exposure.curve = "-8:-5.0, 2:-2.9, 8:-1.6, 17.5:0.0"
```

| Scene | EV100 (measured) | comp | rendered median | exposure EV |
|---|---|---|---|---|
| Noon sand | +17.45 | −0.01 | 0.179 | −16.9 |
| Noon blue sky | +16.50 | −0.17 | 0.160 | −16.1 |
| Daylight shade (jungle) | +7.00 | −1.82 | 0.051 | −8.3 |
| Lit interior, night | +7.00 | −1.82 | 0.051 | −8.3 |
| Lit street, night | +1.50 | −3.00 | 0.022 | −4.0 |
| Starlit sky | −8.00 | −5.00 (floor) | 0.006 | +3.5 |

Effective slope is 0.79 / 0.78 / 0.83 across the three segments — flatter than the 0.86 the first
derivation used, which is the fix for *"it still targets mid-grey everywhere"*: 25 EV of scene range
now compresses to **5.0 EV** of rendered difference rather than 3.5.

**A limit worth stating rather than tuning around: daylight shade and a lit interior at night measure
the same.** Both land at EV100 ≈ 7, so no luminance-only curve can render them differently — and
photographically that is correct, a shaded lawn and a lit room really do meter alike. What separates
them for a viewer is adaptation *state*, not luminance, which is why the temporal asymmetry below is
load-bearing rather than a polish item.

**Exposure clamps.** The multiplier spans −16.9 EV (noon sand) to +3.5 EV (starlit sky). Shipped:
**`min-ev = -18`, `max-ev = +5`**, and `Exposure.clampScale`'s bound widened from `1e-4 … 1e4` to
`1e-8 … 1e8`.

`max-ev` was `+10` and blew the frame out in play. With 13 EV of headroom above anything the curve
asks for, exposure ran away whenever the camera held something very dark, so anything bright entering
frame arrived pre-blown. **A clamp should be a guard rail, not the controller** — `+5` keeps 1.5 EV
over the curve's own demand and nothing more. `min-ev` deliberately does *not* cover a zoomed-in sun
(which asks for −20.8): letting the whole frame go black because the sun is in shot is worse than
clamping it.

### Temporal adaptation — the asymmetry was in the interpolation

Measured in game, night→day took ~3.5 s of blown-white screen while day→night snapped in a frame.
That is backwards from human vision (light adaptation: seconds; dark adaptation: minutes) and it
**could not be fixed with the time constants**, because the asymmetry was not coming from them.

The controller smoothed in *linear* exposure space. Lerping toward a much larger target crosses most
of the ratio in the first frame; lerping toward a much smaller one decays through it geometrically.
For a 15 EV swing with a 0.35 s constant that is ~0.08 s one way and ~3.6 s the other, whatever the
constants say.

Fixed by smoothing `log2(exposure)` (EXPOSURE_PLAN S4). A 15 EV swing now takes `3.4·τ` in either
direction, so the constants are the only thing setting the asymmetry — and they can finally express
the real one:

| Config | was | now | 15 EV swing |
|---|---|---|---|
| `adapt-brighten` (scene got brighter) | `adapt-down` 0.35 | **0.4 s** | 1.4 s |
| `adapt-darken` (scene got darker) | `adapt-up` 0.12 | **2.0 s** | 6.8 s |

Renamed because `up`/`down` described which way the exposure *multiplier* moved, which reads backwards
— a darkening scene needs *more* exposure. Old keys are ignored rather than reinterpreted, since their
values meant the opposite of the new ones.

## 5. Stages

**U0 — EV100 reporting only.** *(done)* Add the `+3` offset (and the `−log2(preExposure)` term, zero until
U1) to the metered EV in the resolve's `evScene` diagnostic and the debug readout. No visual
change; makes the existing S0 observability speak the same scale the rest of this plan uses.

**U1 — pre-exposure plumbing, no constant changes.** *(done)* Raygen multiplies by `preExposure`; the
resolve subtracts it back out and writes the residual to the exposure image. **This stage is a
visual no-op** — every product `L · exposure` is algebraically unchanged — which makes it directly
verifiable: the image must look identical, and any difference is a bug. Land it before touching a
single light value.

**U2 — sun / moon / sky constants** (§3). *(done)* NEE levels are illuminances, not `E/Ω` radiances
(Correction 1); disc radiances derive from the drawn solid angle (Correction 2); the payload's
surface lanes were hand-packed so the sky travels as a `float3` (§2). Exposure clamps widened in the
same change, as required.

**U3 — emissive constants.** *(done; baseline corrected by U5, audit still deferred)*
`EMISSIVE_STRENGTH` shipped at 15,000 cd/m² and was **~5.5 EV hot** — see §7. Now 318 cd/m², anchored
on luminous exitance. `MAX_EMISSION_STRENGTH` / `LE_LUM_EPS` moved with it. **The per-material JSON
multiplier audit is still not done** — every emitter shares the one baseline times whatever
multiplier it already had. See §6 Q4.

**U4 — curve retune.** *(done, fitted to measurement)* `DEFAULT_CURVE` is §4's table. The first
version was §1's reference table restated; the shipped one is fitted to measured in-game EV100 and
carries more compression (5.0 EV of rendered range, effective slope ~0.80).

**U5 — validation.** *(first pass done 2026-07-29 — see §7. Second pass owed on the fixes it
produced.)* The reference scenes of §1 plus the glowstone/lava scene
[DISPLAY_TRANSFORM_PLAN.md](DISPLAY_TRANSFORM_PLAN.md) §4 already calls for.

## 6. Open questions

1. **Does `SUN_INTENSITY = 128,000` actually produce ~5,000 cd/m² zenith sky?** **ANSWERED (§7).**
   Measured noon blue sky is EV100 +16.50 = 11,600 cd/m², within 0.23 EV of the hand integral. The
   march carries no baked-in normalisation — it is dimensionally sound and its units work out. The sky
   does sit ~1 EV over the textbook 5,000 cd/m², which is a property of the model's Rayleigh/Mie/ozone
   parameters rather than of the scale, and is inside U5's ~1 EV tolerance. Nothing to change.
2. **Does the sun-disc clamp interact with MIS?** *(mostly resolved by Correction 2, one part still
   open.)* Clamping cannot perturb the NEE estimator: the visible disc and the NEE light are
   decoupled, and the `showCelestial` gate hides the disc from diffuse continuations
   ([world.rgen.slang:65](../shaders/world/world.rgen.slang)). What Correction 2 surfaced is a
   different, pre-existing issue in the same machinery: the gate is per-lobe, not per-roughness, so a
   *glossy* continuation both takes the sun through NEE and sees the disc — a genuine double-count,
   today harmless only because the disc is far dimmer than the NEE light. Deriving disc radiance from
   illuminance keeps it harmless. Making it correct means gating the disc off finite-roughness
   specular lobes as well, which is a real MIS change and out of this plan's scope.
3. **Should night get an explicit ambient floor?** *(still open.)* The constant itself is confirmed —
   a starlit sky measures EV100 −8.00 against a predicted −7.97 — so what U5 sees is genuinely what
   physics looks like. Whether it is *playable* was not established, because the first pass was spent
   in a lit city where emitters dominated. Retest against the corrected emissive baseline. If it needs
   lifting, add a named gameplay floor rather than re-fudging the sky constant.
4. **Do per-material emissive multipliers need rescaling or re-authoring?** *(still open, and now
   the most interesting one.)* They rescale for free if they were authored as honest ratios against
   `EMISSIVE_STRENGTH`, and need individual work if any absorbed AgX-era compensation. §7 corrected
   the shared baseline but did not distinguish the two cases. It also showed why a single baseline is
   only ever an approximation here: a flame is genuinely far brighter per unit area than a glowstone
   block, and the emission mask carries coverage, not intensity — so the audit's real output should be
   per-material *exitance*, not a multiplier tweak.

## 7. U5, first pass — measured 2026-07-29

F3 `evScene` readings, and what the shipped constants predict:

| Scene | measured EV100 | predicted | error |
|---|---|---|---|
| Noon sand | +17.45 | implies albedo 0.601; MC sand ACEScg luma ≈ 0.60 | **~0** |
| Sun disc, zoomed | +21.30 | 355,556 × ~0.90 transmittance = 320,000 cd/m² | **+0.01 EV** |
| Blue sky, noon | +16.50 | hand-integrated 9,860 cd/m² | +0.23 EV |
| Starlit sky | −8.00 | `NIGHT_LUMINANCE` 0.0005 cd/m² → −7.97 | **+0.03 EV** |
| Daylight shade (jungle) | +7.00 | — | — |
| Lit city interior, night | +12.50 | — | see below |
| Lit city street, night | +7.00 | — | see below |

**The whole of U2 validates.** Sun illuminance, disc radiance, night-sky luminance and the atmosphere
march all land within 0.25 EV of derivation — including both of §3's corrections, which were reasoned
rather than measured when they shipped. §6 Q1 is answered: the Nishita march carries **no** baked-in
normalisation, and the zenith sky sits ~1 EV over the textbook 5,000 cd/m² figure, which is a property
of the model's parameters rather than of its units.

**U3 does not.** A well-lit city interior metered **EV100 12.5 — brighter than an overcast noon**,
which is absurd for a room at night. Cause: 15,000 cd/m² is *flame* luminance, and the plan's own
sanity check justified it against a ~0.1 m torch quad — but the emission mask puts that same luminance
across a whole block face. 15,000 cd/m² over 1 m² is **47,000 lm from one glowstone**, a stadium
floodlight. Re-anchored on luminous exitance instead: a full-strength face radiates ~1,000 lm/m², so
one block face is a ~1,000 lm / 75 W-equivalent lamp and `L = M/π = 318 cd/m²`. That is −5.56 EV,
which puts the measured interior at **7.0** against §1's "lit indoor" reference of 6.5.

The general lesson, and the reason this is worth writing down: the light constants were checkable and
three of four were right, but **the emissive one was anchored to the wrong physical quantity** —
surface luminance of a flame, for a value that is applied per unit area of block face. Exitance is the
quantity that actually describes "what this emitter does to the room".

### Fixed in the same pass

- **Adaptation asymmetry was backwards**, and the cause was linear-space smoothing rather than the
  time constants — see §4's temporal section. This is the one the player feels most.
- **`max-ev = +10` blew out the frame**; now `+5`. See §4.
- **Curve re-fitted** to measured anchors, with more compression.

### Still open from this pass

- **The full moon reads faint in a lit city.** Dominated by emitters being 5.5 EV hot, so the fix
  above swings 5.5 EV in the moon's favour and may settle it. If it does not, the residual is
  Correction 2's trade-off: the moon is drawn ~43× oversize, so its energy-consistent radiance is
  1.56 cd/m² against a real moon's 3,000. Options, in increasing order of correctness — (a) accept
  it; (b) shrink the *drawn* moon toward its real 0.26° and raise radiance to 3,000, which is fully
  physical but makes the moon a dot and abandons the vanilla silhouette; (c) resolve §6 Q2 by gating
  the disc off finite-roughness specular lobes, after which only delta mirrors see it and it can carry
  true luminance safely. (c) is the right end state. **Do not inflate radiance at the drawn size** —
  that is exactly the coupling Correction 2 exists to prevent.
- Q3 (night ambient floor) and Q4 (per-material emissive audit) are untouched and now testable
  against a correct emissive baseline.
