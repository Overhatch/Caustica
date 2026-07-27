# Exposure Plan — smarter metering and adaptation

Status: **implementation in progress** — S0 state diagnostics and debug views are implemented.
Written 2026-07-27 against `bt2020-only`
(working tree, on top of `5d6bf62`). Scope: the auto-exposure loop that produces the 1x1
`display exposure` image and everything that consumes it. The display transform that consumes
that scalar (AgX / PQ curve shape, and its planned ACES 2.0 replacement) is a separate, downstream
concern — see [DISPLAY_TRANSFORM_PLAN.md](DISPLAY_TRANSFORM_PLAN.md), which also supersedes this
plan's original §S6.

## 0. What exists today

| Piece | File | Role |
|---|---|---|
| Owner / mode switch / config | [RtExposure.java](../src/main/java/dev/comfyfluffy/caustica/rt/pipeline/RtExposure.java) | 1x1 `R32_SFLOAT` image, 256-bin histogram buffer, 64-byte host-visible state buffer |
| Pipelines | [RtExposurePipeline.java](../src/main/java/dev/comfyfluffy/caustica/rt/pipeline/RtExposurePipeline.java) | two compute pipelines (hist, resolve) |
| Metering | [exposure_hist.comp.slang](../shaders/display/exposure_hist.comp.slang) | full-res log2-luminance histogram, shared-memory atomics, one bin per thread |
| Controller | [exposure_resolve.comp.slang](../shaders/display/exposure_resolve.comp.slang) | 1 invocation: percentile trim → key → clamp → exponential smoothing |
| Consumer | [display.comp:131](../shaders/display/display.comp) | one scalar multiply feeding both the SDR AgX path and the PQ HDR path |
| Frame placement | [RtComposite.java:990-1003](../src/main/java/dev/comfyfluffy/caustica/rt/RtComposite.java) | after DLSS-RR, before display mapping |

The current model, stated as math. With `L` = BT.2020 luma of the post-RR image and
`Lw` = trimmed mean of `log2 L` over the 50th–95th percentile window:

```
exposure = clamp(key * 2^(evBias - Lw), 2^minEv, 2^maxEv)          // key = 0.18
prev'    = lerp(prev, exposure, 1 - exp(-dt / tau))                // tau = 0.12 up / 0.35 down
```

Two placement decisions are already correct and should be preserved:

- **Metering post-RR.** DLSS-RR ignores exposure entirely (Integration Guide §3.7), and the
  pre-RR buffer's `log2` average is Monte-Carlo biased by Jensen's inequality, so it drifted
  with spp. Both are recorded at [RtComposite.java:992](../src/main/java/dev/comfyfluffy/caustica/rt/RtComposite.java).
- **Exposure applied at the display seam**, once, to both display transforms.

Also already true and worth knowing: the first-person self is `MASK_SECONDARY`
([RtEntities.java:699](../src/main/java/dev/comfyfluffy/caustica/rt/entity/RtEntities.java)), so
the viewmodel does not pollute primary-visibility metering. The state buffer is host-visible and
permanently mapped, so CPU readback of the controller costs nothing to add.

## 1. Diagnosis

**D1 — "normalize the median to mid-grey" is the wrong model for Minecraft.** The controller has
exactly one goal: drive the trimmed mean of the frame to `key = 0.18`. It therefore removes, by
construction, every luminance difference the renderer works hard to compute — a torch-lit cave,
an overcast dawn and noon in a desert all resolve to the same picture. The evidence that this is
already understood to be wrong is the clamp: `minEv = -1.5 / maxEv = 4.0` is doing the artistic
work, and a clamp that is regularly saturated is a controller that is being overruled. This is
the root cause; D2–D4 are refinements that only matter once this is fixed.

**D2 — no spatial weighting.** Every pixel votes equally
([exposure_hist.comp.slang](../shaders/display/exposure_hist.comp.slang)). Sky is 2–4 EV above any lit
surface, so tilting the camera up past the horizon moves well over half the frame into the top
of the histogram; the 50th–95th percentile window then samples almost nothing but sky and the
terrain crushes. The inverse happens looking down in a cave. The exposure changing because of
where the camera *points* rather than where the player is *looking* is the most visible artifact
of the current system.

**D3 — exposure is coupled to surface colour.** Metering luminance means a white-concrete room
meters ~4x brighter than a black-wool room at identical illumination, and the controller
"corrects" a difference that is not a lighting difference. `gAlbedo` (BT.2020 diffuse albedo,
render res) is already produced for RR — demodulating by it meters *illuminance* instead, which
is the quantity a light meter actually measures.

**D4 — smoothing happens in linear exposure space.** `mix(prev, target, alpha)` on the multiplier
([exposure_resolve.comp:62](../shaders/display/exposure_resolve.comp)) makes the perceived rate of
adaptation depend on absolute level: the same `tau` is a slow crawl at high exposure and a snap at
low exposure. Eye adaptation is logarithmic; the filter should run on EV. The `adaptUp`/`adaptDown`
names are also ambiguous — they refer to the exposure *multiplier* rising, i.e. the scene getting
*darker*, which is the opposite of how a reader will parse them.

**D5 — no transient rejection and no scene-cut reset.** A lightning flash, a creeper explosion or
a nether portal fills the histogram for 2–3 frames and the controller chases it, then chases back:
a visible pump on an event that a human eye would not react to. Conversely `resetAutoHistory()` is
only called at buffer creation ([RtExposure.java:46](../src/main/java/dev/comfyfluffy/caustica/rt/pipeline/RtExposure.java)),
so a dimension change, respawn, teleport or `manual → auto` switch adapts *gradually* from a
completely unrelated exposure — the one case where an instant jump is correct.

**D6 — one exposure value, two display transforms with different anchors.** SDR wants scene
mid-grey at AgX's 0.18; HDR wants `1.0` = paper white with `headroom` above it
([display.comp:112-121](../shaders/display/display.comp)). Sharing one multiplier means an HDR
display is handed the same fully-normalized image as SDR and its headroom is spent on nothing —
the whole point of HDR is that a bright scene is *allowed* to be brighter, i.e. HDR wants *less*
adaptation, not the same amount.

**D7 — zero observability.** There is no way to see the measured EV, the target EV, the applied
EV, or the histogram. Every tuning decision above is currently made by eye, on a moving camera,
against a curve nobody can plot. This is why S0 comes first.

**D8 — robustness nits (contained, not urgent).** A NaN radiance reaches `uint(floor(NaN))`, which
is undefined; in practice `min(..., 255u)` traps it in the top bin and the 95th-percentile cut
discards it, so it is survivable — but it should be an explicit `isfinite` guard rather than luck.
`pc.pixelCount` is the image area, which stops matching the histogram total the moment any pixel
is skipped or weighted (S1/S2 both do that); the resolve should sum the bins itself and drop the
push constant.

## 2. Target model

Replace "median → mid-grey" with **measure, then shape, then filter** — three separable stages,
each independently tunable and testable.

```
                     ┌── measure ──────────┐  ┌── shape ────────────┐  ┌── filter ──────┐
weighted histogram → trimmed mean of log2  → evScene → curve(evScene) → temporal ctrl → evExposure
   (S1, S2, S5)                                    (S3)                     (S4)
```

**Measure (S1, S2, S5).** `evScene = weighted, percentile-trimmed mean of log2(metered)`, where
`metered` is illuminance (radiance / albedo) on surfaces and radiance on sky, and the weight is
`center × sky-cap × validity`.

**Shape (S3).** A piecewise-linear compensation curve over `evScene`, exactly the role of Unreal's
`ExposureCompensationCurve`:

```
evExposure = log2(key) - evScene + comp(evScene) + evBias
```

`comp ≡ 0` reproduces today's behaviour (full adaptation, slope 1). Control points chosen so the
*effective* slope `1 - d comp/d evScene` lands around 0.6–0.8: a scene 4 EV darker still renders
~1 EV darker instead of identical. Defaults to aim for, measured relative to overworld noon:

| Scene | Target render, relative to noon |
|---|---|
| Noon, clear, exterior | 0 EV (reference) |
| Overcast / dawn | −0.5 EV |
| Clear night, moonlit | −1.5 to −2 EV |
| Torch-lit cave | −2 to −2.5 EV, torch cores still not clipped |
| Deep dark, no light source | floor at `minEv`, i.e. genuinely black |

**Filter (S4).** Exponential in EV with a slew limit, a deadband, a median-of-N transient
rejector, and a hard reset input.

## 3. Stages

### S0 — Observability and a repeatable test (do this first)

Nothing below is tunable blind. The state buffer is already host-visible and mapped, so this is
mostly bookkeeping.

- Widen `ExposureState` to a struct: `evScene`, `evTarget`, `evApplied`, `clipLowFrac`,
  `clipHighFrac`, `resetSeq`, plus the S3 history ring. Read it CPU-side one frame late (debug
  only — no fence needed, a stale value is fine for a HUD).
- Debug HUD line behind the existing frame-stats toggle: `EV scene/target/applied`, effective
  slope, and the fraction of pixels clipping at each end.
- Two new `debugView` modes (the plumbing exists —
  [CausticaConfig.java:534](../src/main/java/dev/comfyfluffy/caustica/CausticaConfig.java),
  `writeDebugView` in [guides.slang](../shaders/world/guides.slang)): **false-colour exposure**
  (EV relative to mid-grey, stops-banded) and **metering weight map** (S2's weights as greyscale).
- Optional but cheap: dump the per-frame `evScene/evTarget/evApplied` triple to CSV behind a flag,
  and script a fixed camera path (surface → cave → surface, noon → night, horizon pan). That turns
  "does this feel better" into a diff of two curves. Reuse whatever `profileMinecraft.ps1` does for
  scripted runs.

*Acceptance:* the HUD reads out sane EVs in all four reference scenes; the false-colour view makes
the sky/terrain split obvious.

**Status (2026-07-27): state widening, log line, and the two debug views are done.**
`ExposureState` (std430, `exposure_resolve.comp.slang`) widened from `(previous, initialized)` to
64 bytes: adds `evScene`/`evTarget`/`evApplied` (all EV, i.e. log2), `clipLowFrac`/`clipHighFrac`
(fraction of metered pixels landing in the histogram's extreme bins — the "is the meter's dynamic
range clipping" reading, distinct from whether the EV clamp itself is pinned), and reserves
`resetSeq`/`evHistory[8]` for S4 (declared now, neither read nor written yet, so the buffer layout
doesn't need to change again when S4 lands). No behavior change: the linear-space smoothing math is
untouched, these are read-only diagnostics alongside it. `RtExposure.logDiagnosticsIfDue()` logs
them once/second, gated behind the existing `caustica.rt.frameStats` toggle (reused rather than a
new flag — that's already "I want renderer internals" for this codebase) and flags when `evTarget`
is sitting at the `minEv`/`maxEv` clamp boundary, directly surfacing D1's diagnosis.

**Debug presentation + the two exposure views done (2026-07-27).** `writeDebugView` was removed
from the primary raygen — the hottest, occupancy-bound shader in the renderer — and replaced with
`RtDebugPresentPipeline` / `debug_present.comp.slang`, a small compute pass at the end of the frame.
Crucially, `debugView` is now purely observational and changes no upstream work:

```
primary trace → indirect trace → RR/fallback → exposure meter/resolve → ACES display → debug present
```

This fixes the first attempt's fundamental mistake: it disabled RR/jitter, skipped indirect tracing
and fallback upscale, bypassed exposure/display mapping, froze exposure history, and forced a
full-resolution resource rebuild whenever a debug view was selected. Exposure diagnostics then had
no real post-RR scene to inspect. The final design always renders the ordinary frame first. The debug
pass reads the display-resolution `rrOutput` plus same-frame exposure, nearest-samples the real
render-resolution guide buffers, and only then replaces `displayImage` with literal diagnostic
colors. Those colors never enter the histogram or ACES. Guide motion is scaled from render-pixel to
display-pixel units so its visualization is stable across RR quality modes. Debug output currently
uses the explicit SDR→PQ present fallback in HDR mode; native-PQ debug coloring remains optional
follow-up work.

Modes 8 and 9 are now exposed in the video options:

- **Exposure false colour (8):** BT.2020 luminance from post-RR `rrOutput`, multiplied by the
  same-frame display exposure and ACES mid-grey bias, shown in discrete one-stop bands relative to
  18% grey. Cool colors are below mid-grey, neutral grey is the zero-stop band, and warm colors are
  above it.
- **Metering weight preview (9):** greyscale preview of S2's planned Gaussian centre weight
  (`σ = 0.35`, floor `0.15`) and provisional `0.25` local sky down-weight from reversed-Z depth.
  This is deliberately labelled a preview: the current S0 histogram still gives every pixel one
  vote. S2 must replace the provisional local sky factor with its frame-global sky-cap normalization
  when weighting becomes real metering behavior.

`debugView` was also removed from `WorldPushConstants`; no world shader needs a debug branch now.

Also migrated `exposure_hist.comp`/`exposure_resolve.comp` to Slang (→ `.comp.slang`) while doing
this rewrite, matching `display.comp`'s migration — see
[DISPLAY_TRANSFORM_PLAN.md](DISPLAY_TRANSFORM_PLAN.md) step 4 for the established conventions
(`[shader("compute")]`/`[numthreads]`/`SV_DispatchThreadID`, `StructuredBuffer`/`RWStructuredBuffer`
with an explicit `Std430DataLayout` to guarantee the byte layout matches the CPU-side raw writes).

### S1 — Metering hygiene

Small, self-contained, no behaviour change intended beyond removing noise sources.

- Sum the histogram bins in the resolve for the population total; delete `pc.pixelCount` (D8).
- `isfinite` guard in the histogram (D8).
- Meter at stride 2 in each axis (quarter the invocations). At 4K that is still ~2M samples —
  statistically irrelevant, and it removes a full-res `rgba16f` read from the frame.
- Make the trim window configurable (`low-percentile` / `high-percentile`, default unchanged at
  0.50/0.95) so the S2/S3 tuning can move it without a rebuild.

*Acceptance:* EV trace from S0 is unchanged (within ~0.05 EV) at stride 1 vs stride 2; frame time
for `frame.exposure` drops.

### S2 — Spatial weighting

Weighted histogram: `atomicAdd` a fixed-point weight (`uint(w * 256)`) instead of `1u`.

- **Centre weight** — Gaussian on normalized screen distance, σ ≈ 0.35, floor 0.15 so the periphery
  still contributes. Classic centre-weighted metering; directly addresses D2.
- **Sky cap** — sky is `gDepth ≈ 0` (reversed-Z far, see
  [guides.slang:271](../shaders/world/guides.slang)). Do not exclude it — a bright sky *should*
  stop the ground down somewhat — but cap its total contribution to a configurable fraction
  (default ~0.25) by scaling sky weights by `min(1, cap * total / skyTotal)`. Needs either a
  two-pass histogram or a separate sky bin count; the latter is one extra `atomicAdd`.
- **Validity** — skip pixels with zero weight from the total (already handled by S1's bin sum).

`gDepth` is at render res while metering runs on the display-res post-RR image; fetch it nearest
at `pix * renderRes / displayRes`. Misalignment is a pixel at material boundaries and irrelevant to
a statistical measure.

*Acceptance:* panning across the horizon changes `evScene` by well under half of what it does
today; the weight-map debug view matches expectation.

### S3 — Adaptation curve

The core fix (D1). Push a 4-point piecewise-linear `comp(evScene)` curve (8 floats) and evaluate
it in the resolve.

- Config as control points, e.g. `exposure.curve = "-6:-2.0, -3:-0.8, 0:0.0, 4:0.4"`
  (measured EV → EV compensation), with a `full` preset that sets all zeros to reproduce today.
- Keep `minEv`/`maxEv` as a safety clamp only — with a correct curve they should stop saturating,
  and S0's HUD will show whether they do.
- Retune `key`, min/max and the curve together against the four reference scenes.

*Acceptance:* the four reference scenes land within ~0.3 EV of the §2 table; `minEv`/`maxEv`
saturate only in the intended deep-dark case.

### S4 — Temporal controller

Rewrite the smoothing in EV space (D4, D5):

```
evT   = median(history[N])            // N = 8 frames, rejects lightning/explosion flashes
tau   = (evT < evPrev) ? tauBrighten : tauDarken     // named for what the IMAGE does
step  = (evT - evPrev) * (1 - exp(-dt / tau))
step  = clamp(step, -maxEvPerSec * dt, maxEvPerSec * dt)
ev    = |evT - evPrev| < deadbandEv ? evPrev : evPrev + step
```

- The median ring is 8 floats in the state buffer; the flash is 1–3 frames of 8 and never becomes
  the median. Note this adds ~4 frames of lag to *genuine* steps — the slew limit and `tau` already
  dominate that, so it should not be perceptible, but check it in the S0 trace.
- **Hard reset** (`alpha = 1`, clear the ring) on: dimension change, respawn/teleport
  (large `camDelta`), camera-type change, `manual → auto`, exposure config change, and the first
  frame after a world load. Drive it with a CPU-side `resetSeq` counter in the push constant that
  the shader compares against the stored one — no extra dispatch, no CPU/GPU race.
- Asymmetric defaults with a physical justification: walking out of a cave into noon *should*
  blind briefly (slow darken, ~0.8 s), while the reverse should recover faster than a real eye or
  the game is unplayable (~0.4 s). Current 0.12/0.35 s are both far faster than any eye.

*Acceptance:* a lightning strike moves `evApplied` by <0.1 EV; a portal transition snaps in one
frame; no visible pumping under flickering torchlight.

### S5 — Illuminance metering

Meter `radiance / max(albedo, 0.08)` for non-sky pixels using `gAlbedo`, fetched nearest with the
same render→display scaling as S2's depth (D3).

Risk to handle explicitly: **emissive surfaces break the demodulation** — a torch flame has high
radiance and near-zero albedo, so the ratio explodes. Options, cheapest first: (a) rely on the
95th-percentile trim to discard them (probably sufficient — emitters are a small pixel fraction);
(b) clamp the demodulated value to a max EV above the frame's radiance-based mean; (c) plumb an
emissive bit through a guide buffer and meter those pixels as raw radiance. Start with (a),
validate with the false-colour view, escalate only if it visibly misbehaves in a lava/glowstone
scene. Ship this behind a toggle defaulting to off until it has been validated in play.

*Acceptance:* a room re-skinned white ↔ black shifts `evScene` by <0.3 EV where today it shifts
by ~2 EV; no exposure collapse when facing a wall of glowstone.

### S6 — SDR / HDR anchor (superseded)

D6 originally proposed a separate HDR exposure offset + reduced adaptation slope to compensate for
AgX and the ad-hoc HDR rolloff being different operators with different highlight behavior (one
desaturates, one hue-shifts — see [DISPLAY_TRANSFORM_PLAN.md](DISPLAY_TRANSFORM_PLAN.md) §1 D1).
That plan replaces both with a single peak-luminance-parameterized operator (ACES 2.0, baked to a
LUT), which appearance-matches SDR and HDR by construction. Under that design S6 shrinks to
"select the LUT baked for the configured peak nits" — a startup/config concern, not a per-frame
adaptation-curve split. If HDR should still read deliberately brighter than appearance-matched
SDR, that's a small explicit creative EV offset added later, not a color-science fix. No
per-frame work remains here; kept as a stub so the D6 numbering isn't reused.

**Sequencing note:** do the LUT swap (display-transform plan §7 step 1-3) *before* finalizing this
plan's S3 curve control points — S3 is tuned by eye against whichever operator is live, and tuning
it against AgX now means re-tuning once the LUT lands.

### S7 — Later / speculative, in decreasing confidence

- **Local exposure (bilateral grid).** The one thing global exposure fundamentally cannot fix:
  standing in a cave mouth with noon outside. Build a coarse bilateral grid over log-luminance
  (e.g. 32×32×16), blur, and sample a per-pixel exposure offset clamped to ±1.5 EV. This is a real
  feature with real cost (one downsample + one small 3D blur) and real risk (halos, temporal
  instability). It is the correct next step *only* once S1–S6 are shipped and the remaining
  complaints are all about simultaneous bright/dark content.
- **World-state prior.** The client knows sun elevation, weather, dimension, and sky/block light at
  the camera. A CPU-computed prior EV that metering is only allowed to correct within ±N EV would
  be rock-stable by construction and would make caves dark for a *reason* rather than because the
  histogram happened to be dark. Attractive, but it duplicates the renderer's own lighting model —
  only worth it if S2–S4 leave residual instability.
- **Purkinje shift.** At scotopic levels, desaturate toward blue and lose acuity. Belongs in the
  display transform, not here, but it is the feature that makes a *correctly dark* night look
  intentional rather than broken — so it is the natural companion to S3 and worth prototyping right
  after it.
- **Feed our exposure to DLSS.** Moot for RR (§3.7 — it ignores exposure), so this only matters if
  a plain DLSS-SR path is ever added. Noted so it is not rediscovered.

## 4. Config surface

Additions (all under `[exposure]`, all with a `caustica.rt.exposure.*` system property per the
existing convention):

| Key | Default | Stage |
|---|---|---|
| `low-percentile` / `high-percentile` | 0.50 / 0.95 | S1 |
| `stride` | 2 | S1 |
| `center-weight-sigma` / `center-weight-floor` | 0.35 / 0.15 | S2 |
| `sky-weight-cap` | 0.25 | S2 |
| `curve` (4 control points, or `full`) | see §S3 | S3 |
| `tau-brighten` / `tau-darken` | 0.4 / 0.8 s | S4 |
| `max-ev-per-second` | 1.5 | S4 |
| `deadband-ev` | 0.05 | S4 |
| `median-frames` | 8 | S4 |
| `illuminance-metering` | false → true once validated | S5 |

Renames: `adapt-up`/`adapt-down` → `tau-brighten`/`tau-darken` with inverted sense (D4). These are
user-visible config keys, so either migrate on load or accept the break — the mod is pre-release,
so accepting the break and logging once is fine.

Removals: `pc.pixelCount` from the resolve push constant (S1).

## 5. Cost

The exposure block is two dispatches, one buffer fill and two barriers per frame, currently timed
as `frame.exposure` ([RtFrameStats.java:67](../src/main/java/dev/comfyfluffy/caustica/rt/RtFrameStats.java)).
S1's stride makes the histogram ~4x cheaper; S2/S5 add a nearest fetch of `gDepth`/`gAlbedo` per
sampled pixel; S3/S4 are ~50 extra ALU ops in a single-invocation dispatch, i.e. free. Net expected
change is **negative** (faster than today), which is worth confirming rather than assuming — the
profile in [GPU_PERF_PLAN.md](GPU_PERF_PLAN.md) says this frame is latency-bound, so a removed
full-res read matters more than the op count suggests.

## 6. Open questions

1. **Where does the curve get authored?** Four control points in a TOML string is developer-facing.
   If exposure ever becomes a user-facing quality setting, it wants 2–3 named presets
   (`cinematic` / `neutral` / `full-adaptation`) with the curve hidden behind them.
2. **Does the median rejector fight the slew limiter?** Both add lag; S0's trace will show whether
   `median-frames = 8` is doing anything the limiter was not already doing. Measure before keeping.
3. **Is S5's percentile trim really enough for emitters?** Unknown until it is run against a
   glowstone/lava scene. The fallback (an emissive guide bit) is a shader plumbing change, not a
   redesign, so the risk is bounded.
4. **Weather.** Rain darkens the sky and the sky cap interacts with that; probably fine, but it is
   a reference scene the §S3 table does not currently cover.

## 7. Recommended order

**S0 → S1 → S2 → [display-transform LUT swap] → S3 → S4 → S5 → (S7 as separate plans).**
S6 is superseded (see above) and no longer a step in this sequence.

S3 is the change that actually fixes the complaint, but S0 makes it tunable and S2 removes the
noise source that would otherwise be mistaken for a curve problem. S5 is deliberately last of the
shipping stages: it is the highest-risk metering change and the one most likely to need a shader
plumbing follow-up, and everything before it is valuable without it.
