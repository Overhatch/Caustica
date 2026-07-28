# Display Transform Plan — one output transform, ACES 2.0, baked LUT

Status: **plan only**, nothing implemented. Written 2026-07-27 against `bt2020-only`
(working tree, on top of `5d6bf62`). Companion to [EXPOSURE_PLAN.md](EXPOSURE_PLAN.md) — exposure
decides *how bright*; this decides *how the resulting scene-linear image becomes display pixels*.
The two are sequential (exposure feeds this stage) and mostly independent — this plan does not
change §S0–S5 of the exposure plan, only shrinks its old §S6.

## 0. What exists today

Two unrelated tonemappers, chosen by `hdrEnabled`, both fed the same exposure scalar
([display.comp:131](../shaders/display/display.comp)):

| | SDR path | HDR path |
|---|---|---|
| Gamut step | `gamutMapBt2020ToBt709` — luminance-preserving desaturate into BT.709 | none, stays BT.2020 |
| Operator | AgX: inset matrix → log2 remap over a **fixed 16.5 EV window** (`AGX_MIN_EV`/`AGX_MAX_EV`, [display.comp:41-42](../shaders/display/display.comp)) → degree-6 polynomial sigmoid → outset matrix | per-channel `hi / (headroom-1 + hi)` rolloff above 1.0 ([display.comp:112-121](../shaders/display/display.comp)) |
| Output | BT.709 [0,1], written to `rgba8` → MC's existing gamma-encoded presentation | BT.2020 nits → `pqEncode` → PQ swapchain |

Both are implemented in [display.comp](../shaders/display/display.comp); the UI composite
([hdr_ui_composite.comp](../shaders/display/hdr_ui_composite.comp)) and `sdr_present.comp`
independently place sRGB-authored UI at `paperWhiteNits` in the same BT.2020/PQ target.

## 1. Diagnosis

**D1 — AgX and the HDR rolloff are different operators with different failure modes.** The SDR
path desaturates saturated highlights toward white (AgX's inset does this by design); the HDR
path clips per-channel, so the same lava pool shifts hue (red clips first, then green) instead of
desaturating. Toggling HDR does not extend the SDR image's range — it changes the image's
character. This was flagged as D6 in the exposure plan; it turns out to be a tonemapping problem,
not an exposure-anchor problem.

**D2 — AgX's latitude is fixed and small, so it cannot be "extended" for HDR.** All of AgX's
highlight compression happens inside the 4.03 EV above mid-grey that `AGX_MAX_EV` allows before
the sigmoid ever runs; by the time a value reaches [0,1] the detail above that range is already
gone. Multiplying the sigmoid's output up by `headroom` cannot recover it — it just makes a large,
flat, blown-out region. Confirmed empirically: naively pushing AgX to HDR peak nits in Blender
reproduces exactly this (all-white highlights, less highlight detail than AgX SDR). Re-deriving a
wider AgX means re-fitting the sigmoid against a new EV window and re-tuning toe/mid to match, at
which point it is no longer AgX — not worth doing when a peak-luminance-parameterized operator
already exists (§2).

**D3 — double gamut compression on the SDR path.** `gamutMapBt2020ToBt709` desaturates
out-of-BT.709 chroma toward equal-luminance neutral *before* AgX's inset does a second,
independent desaturation. Not necessarily wrong, but two compressors stacked without either being
aware of the other is very unlikely to be the intended amount of compression.

**D4 — the commented-out `applyLook`** ([display.comp:56-66](../shaders/display/display.comp),
called nowhere) is dead code guarding a real question: base AgX (what's shipped) is intentionally
flat/low-contrast, and the "washed out" impression anyone gets from it is partly that — not a bug,
a missing look layer. Moot if §2 below is adopted, since ACES 2.0 brings its own contrast.

## 2. Target: ACES 2.0 output transform, baked to a LUT

ACES 2.0 (2024) replaced ACES 1.x's per-channel RRT+ODT with a single output transform run
through a color-appearance-model space (Hellwig 2022 JMh), **parameterized on display peak
luminance** — SDR (100 nit) and HDR (1000+ nit) renditions of the same scene are designed to
appearance-match, which is exactly the property AgX+ad-hoc-rolloff lacks (D1). Empirically
(Blender A/B, 2026-07-27): similar character to AgX at SDR peak, more highlight detail retained at
HDR peak, higher default contrast, still desaturates saturated highlights rather than hue-shifting.
This is the plan's chosen replacement for both existing operators.

**Do not implement the CAM math in-shader.** Forward+inverse appearance-model conversion, a
gamut-cusp table, and chroma/gamut compression per pixel is a large amount of transcendental-heavy
code to get bit-accurate against a reference, for a function that is smooth and low-frequency in
its inputs — the textbook case for a LUT.

**Bake, don't derive:** shaper + 3D LUT, produced offline via OCIO from the ACES 2.0 config
(the same config the Blender evaluation used, so what ships matches what was judged), one LUT per
target peak-nits value.

- **Shaper**: log2, scene-linear BT.2020 → [0,1] over the working range (align with the exposure
  histogram's ±12 EV, [exposure_hist.comp](../shaders/display/exposure_hist.comp), so the same
  scalar means the same thing in both places).
- **LUT size**: 65³ (film-standard, normally visually lossless) as the starting point. The risk
  case here is unusual — near-primary saturated emitters (glowstone, lava, nether portals) at
  extreme intensity sit exactly where the gamut compressor is most nonlinear — so validate that
  case specifically (§4) before settling on 65³. 129³ is ~8 MB at `RGBA16_SFLOAT`; trivial to
  afford if 65³ shows banding or hue drift on those emitters.
- **SDR LUT** outputs BT.709 display code values, feeding the existing `rgba8` target and MC's
  gamma-encoded presentation unchanged.
- **HDR LUT** outputs PQ code values directly — the PQ encode is absorbed into the bake, so
  `pqEncode` in [display.comp](../shaders/display/display.comp) goes away.
- Peak nits (`Hdr.PEAK_NITS`) changes effectively never at runtime (display capability, not a
  per-frame quantity): load the LUT matching the configured value at startup/resize, no runtime
  interpolation between LUTs needed.

## 3. What this deletes / changes

From [display.comp](../shaders/display/display.comp): the AgX inset/sigmoid/outset and its
constants, `gamutMapBt2020ToBt709` (ACES 2.0 gamut-compresses toward the *target* display gamut
internally — the hand-rolled BT.2020→BT.709 prepass becomes redundant at best, double-compressing
at worst per D3), `pqEncode`, the HDR per-channel rolloff and `headroom`-based `tonemapHdr`, and
the dead `applyLook`. The pass becomes: apply exposure → shaper → 3D texture fetch → store. Given
the polynomial + two 3×3 matrix multiplies it replaces, likely cheaper than today, not just
simpler — worth confirming against [GPU_PERF_PLAN.md](GPU_PERF_PLAN.md)'s latency-bound framing
once implemented, same as the exposure plan's own cost note.

`paperWhiteNits` **changes meaning**: under ACES 2.0 the peak-luminance parameter (not a
user-set nit level) determines where scene diffuse white lands in the tonemap. The setting has to
survive, but only as the UI-placement value it's independently used for in
[hdr_ui_composite.comp:66](../shaders/display/hdr_ui_composite.comp) and `sdr_present.comp`
(where to put sRGB-authored UI in nits) — it should be renamed/re-scoped in config to make that
clear, and `Hdr.headroom()` goes away since the LUT bake already encodes the peak-nits relationship.

## 4. Infrastructure needed

- **3D image support in `RtContext`.** Currently 2D-only
  ([RtContext.java:350](../src/main/java/dev/comfyfluffy/caustica/rt/RtContext.java) /
  `:440` for the second creation path) — needs `VK_IMAGE_TYPE_3D` + `VK_IMAGE_VIEW_TYPE_3D`,
  trilinear filtering, clamp-to-edge addressing.
- **A combined-image-sampler binding in `RtDisplayPipeline`**, which today is all storage images
  ([RtDisplayPipeline.java:115](../src/main/java/dev/comfyfluffy/caustica/rt/pipeline/RtDisplayPipeline.java)).
  [RtSdrPresentPipeline.java:67](../src/main/java/dev/comfyfluffy/caustica/rt/pipeline/RtSdrPresentPipeline.java)
  already has this exact pattern (combined-image-sampler + `GENERAL` layout) to copy from.
- **LUTs as committed binary resources** (one per supported peak-nits target — likely just SDR/100
  and one HDR default, not a continuum) plus the **offline bake script committed alongside them**,
  not just the baked output, so the LUTs are regenerable when the ACES config or working-space
  bounds change.
- **Validation scene**: a fixed camera path through glowstone/lava/portal — the specific case
  flagged as highest-risk for LUT resolution (§2) and for hue behavior at extreme saturation+intensity.

## 5. Interaction with the exposure plan

- Sits strictly downstream: exposure produces one scalar, this stage consumes it. §S0–S5 of
  [EXPOSURE_PLAN.md](EXPOSURE_PLAN.md) are unaffected by this plan.
- **Replaces exposure plan §S6.** The old §S6 proposed a separate HDR exposure offset + adaptation-
  scale specifically to compensate for AgX-vs-rolloff mismatch (D6/D1 above). With one
  peak-luminance-parameterized operator, SDR and HDR share the same exposure and the same curve —
  §S6 shrinks to "select the LUT matching the configured peak nits," which is a build-time/startup
  concern, not a per-frame one. If HDR should still read deliberately brighter than appearance-
  matched SDR, that becomes a small explicit creative EV offset, not a color-science fix.
  [EXPOSURE_PLAN.md](EXPOSURE_PLAN.md) should be edited to point here.
  Note this is a **prerequisite reordering**, not just a rename: `EXPOSURE_PLAN.md`'s §S3
  (compensation curve) will be *tuned against* whichever operator is live, so if this plan lands
  after exposure §S3 the curve gets tuned against AgX and then invalidated when this plan ships.
  Doing this plan's §2 LUT swap before finalizing exposure §S3's curve control points avoids
  re-tuning twice.
- The Purkinje-shift idea in exposure plan §S7 belongs here once reached — scotopic desaturation
  is a display-transform concern, not a metering one.

## 5a. Mid-grey anchor (measured 2026-07-27)

An operator has an intrinsic mid-grey anchor, and `Exposure.KEY` anchors the metered median in
**scene**-linear terms — so swapping operators at a fixed key silently changes overall brightness:

| scene-linear 0.18 renders at | display code |
|---|---|
| AgX (base, `applyLook` disabled) | 0.497 |
| ACES 2.0 SDR | 0.349 (film convention, 0.18 → ~0.1 linear display) |

That is a **1.01 EV darkening of the whole midtone range**, and because auto-exposure anchors the
*median*, it lands on the bulk of the frame — first in-game A/B read as a large contrast increase
rather than the mild one seen in the Blender evaluation. It is not a LUT or shaper defect: applying
+1.014 EV to the ACES path brings the two operators within ±0.06 code value across −6…+4 EV, leaving
ACES **1.20× steeper at mid-grey**, which *is* the expected "mildly higher contrast".

The temporary +1.014 EV comparison bias was removed with the AgX mode. ACES now receives only the
exposure system's output, so mid-grey placement is tuned in one place: the exposure compensation
curve. Two follow-ups this implies:

- **Which anchor is actually wanted is a creative decision, not a correctness one.** AgX's 0.497 is
  on the bright/milky side; ACES's 0.349 is the film convention. Tune the desired placement through
  the [EXPOSURE_PLAN.md](EXPOSURE_PLAN.md) §S3 compensation curve.
- The AgX baseline being compared against is **base AgX with no look** (`applyLook` is commented out
  at [display.comp](../shaders/display/display.comp)), i.e. the flattest AgX available — flatter than
  Blender's, which ships a base-contrast look. Some of the perceived gap is that, not the operators.

## 5b. Known hue drift on extreme-brightness saturated red/orange (found 2026-07-27)

In-game: lava drifts toward pink/salmon at high brightness under the ACES LUT (both SDR and HDR).
This is a **documented ACES 2.0 characteristic**, not a defect in the bake or the shader wiring —
see the ACEScentral community discussion of the output transform itself: "Red light at high
exposure in ACES 2.0 can produce a fleshy salmon result... red lights might present as too pinkish
or orange in certain situations." ACES 2.0's whole design goal over 1.x is hue preservation via the
Hellwig JMh appearance model (tone-map lightness `J` while keeping hue `h` mostly independent,
fixing 1.x's "reds skew toward yellow"); the salmon cast on saturated red/orange at extreme
brightness is documented as a residual trade-off of that same fix, not something this
implementation introduced.

Checked whether AgX has a comparable failure mode: reconstructed [display.comp](../shaders/display/display.comp)'s
exact AgX math in Python and ran a saturated lava-orange swatch to EV+13. AgX shows no comparable
hue-toward-magenta drift — it desaturates straight to near-white instead (`AGX_INSET` exists
specifically to pull extreme values toward achromatic *before* the tone curve runs). That's the
likely reason it wasn't obviously visible before this LUT swap: AgX's failure mode at extremes is
"washes to white," ACES's is "stays hued but the hue can be slightly wrong" — the latter is a
side effect of ACES rendering highlights *more* faithfully, not less.

No clean fix available from OCIO's built-in config without authoring a custom LMT (out of scope).
Mitigation: retuning exposure (§S3 in [EXPOSURE_PLAN.md](EXPOSURE_PLAN.md)) so lava's brightest
pixels don't sit pinned at the top of the tone-scale directly reduces how often this is visible —
check lava/glowstone specifically once that tuning pass happens, before deciding whether this needs
further attention.

## 6. Open questions

1. **Does ACES 2.0's default SDR (100-nit) rendition read as the desired look for most players?**
   The Blender A/B evaluated HDR-vs-SDR *consistency*; most players will only ever see the SDR
   output, so that image needs its own sign-off, independent of the HDR comparison that motivated
   this plan.
2. **Resolved (2026-07-27), and made fully live the same day.** Turned out not to be a design
   decision — ACES 2.0 itself doesn't parameterize peak luminance continuously. OCIO's builtin
   registry ships REC2020 output transforms as a fixed table: `{500, 1000, 2000, 4000}` nits
   (`ACES-OUTPUT - ACES2065-1_to_CIE-XYZ-D65 - HDR-{n}nit-REC2020_2.0`, composed with `DISPLAY -
   CIE-XYZ-D65_to_REC.2100-PQ` for the final PQ encode — verified bit-exact against the config's
   own named 1000-nit Display/View path before trusting it for the other three). All four baked;
   `CausticaConfig.Rt.Hdr.PEAK_NITS_STEPS` is the shared source of truth (options-menu slider steps
   through it directly; `nearestPeakNitsStep` also snaps a hand-edited config/system-property value
   for anyone bypassing the UI). No interpolation between LUTs — the table itself is the resolution
   ACES 2.0 offers, matching how HDR mastering always targeted a small set of reference peaks
   rather than an arbitrary value.
   **Live, not restart-only:** `RtComposite` checks the nearest step every frame and hot-swaps the
   loaded HDR LUT (destroy + reload + rebind) the frame it changes — same mechanism that made the
   HDR enable/disable toggle itself live, see §5c.

## 5c. Runtime HDR toggle (2026-07-27) — no restart, either direction

Originally `Hdr.ENABLED` was snapshotted at startup (`ENABLED_AT_STARTUP`) because the swapchain's
pixel format is fixed for the life of the `VulkanGpuSurface` object — vanilla's `configure()` (the
resize path) reuses whatever `pickSwapchainSurfaceFormat` decided once at surface construction, so
naively there is no way to switch a live swapchain between SDR and PQ formats without recreating it,
which blaze3d does not expose a path to trigger on demand.

**Current resolution: recreate through the existing resize path.** The HDR option updates
`Hdr.ENABLED` and calls `Minecraft.invalidateSurfaceConfiguration()`. At the next safe render
boundary Minecraft executes the same `GpuSurface.configure()` used by framebuffer resize. The
Vulkan surface mixin re-enumerates advertised formats there, makes vanilla's normally-final
`swapchainImageFormat` selectable, and chooses native SDR (`SRGB_NONLINEAR`) or HDR10/PQ
(`HDR10_ST2084`) to match the toggle.

Capability and current state are deliberately separate:

- `swapchainPqAvailable()` means the surface advertises a PQ pair, so HDR controls remain visible
  while the current swapchain is native SDR.
- `swapchainPqActive()` means the currently configured swapchain is PQ.
- `Hdr.enabled()` requires both the user toggle and an active PQ swapchain.
- `isPqSdrPresentActive()` remains for menus/loading frames while HDR is active (and the short
  toggle-to-reconfigure interval), but is bypassed once HDR-off recreation produces native SDR.

Consequently, `paper-white-nits` no longer changes whole-image brightness with HDR disabled: native
SDR uses Minecraft's ordinary presentation. It still places SDR-authored UI/menu content at an
absolute luminance when that content must be embedded into an active PQ swapchain.

**Consequences elsewhere:**
- `GlxMixin.caustica$preferWaylandForHdr` now always attempts the native Wayland backend on Linux
  (previously gated on `Hdr.enabled()`), since GLFW's platform is chosen before any window exists —
  long before the surface (and so `swapchainPqAvailable()`) exists to gate on, and there is no way
  to switch backends after `_initGlfw` returns. If HDR is ever going to work at all this session,
  Wayland has to already be the active backend by the time the surface is created.
- `RtVideoOptions.runtimeOptions()` omits the HDR toggle/paper-white/peak-nits entries entirely
  (not just disables them) when `swapchainPqAvailable()` is false — this capability is fixed by
  hardware/OS/compositor at surface creation, unlike every other RT setting, so offering controls
  that can never do anything would be actively misleading.
- Deleted `OptionsMixin` (`Options.isRestartRequiredToApplyVideoSettings`) and
  `CausticaConfig.Rt.Hdr.pendingRestart()` — both existed solely to show vanilla's "restart
  required" banner for the old snapshot-at-startup behavior, which no longer exists.
3. **OCIO/ACES tooling dependency.** Baking needs the ACES 2.0 OCIO config and a way to run OCIO
   offline (Python + `PyOpenColorIO`, or `ociobakelut`). This is a one-time dev-environment cost,
   not a runtime dependency, but needs to be set up and the exact bake command recorded here once
   done, so the LUTs are reproducible.
4. **Creative display gamma.** A post-transform `tonemap.gamma` control is now available. It is
   neutral at 1; lower values lift shadows/midtones without moving black or peak white. Both paths
   decode to display-linear light, apply the power to luminance, and scale RGB uniformly to preserve
   chromaticity. SDR then re-encodes sRGB; HDR re-encodes PQ. The uniform scale is gamut-limited
   before individual channels clip, so highly saturated colors retain their channel ratios.

## 7. Recommended order

1. **Done (2026-07-27).** SDR LUT baked and wired behind `tonemap.mode` (default still `agx`).
   Mid-grey anchor mismatch found and fixed — see §5a.
2. **Done (2026-07-27) — validated in play.** No banding at 65³. One hue-drift finding on
   lava at extreme brightness (see §5b) — a documented ACES 2.0 characteristic, not a defect;
   AgX was masking the same underlying limitation by crushing to white instead. Confirmed
   "working correctly" overall; cleared the way for step 4.
3. **Done (2026-07-27).** HDR LUTs baked and wired: `tonemapMode` now switches SDR and HDR together,
   old per-channel rolloff + `pqEncode` still present but only reachable in legacy mode.
   Both LUT fetches consume the same exposed scene value so SDR/HDR stay appearance-matched at one
   exposure. §6.2 (peak-nits variants) resolved same day, see below.
4. **Done (2026-07-27).** Deleted the legacy operator entirely (AgX inset/sigmoid/outset,
   `gamutMapBt2020ToBt709`, the per-channel HDR rolloff, `pqEncode`, the dead `applyLook`) —
   ACES 2.0 is now the only SDR/HDR display transform, no mode switch. `CausticaConfig.Rt.Tonemap`
   lost `MODE`/`acesLut()`. `RtDisplayPipeline.dispatch`'s push constants no longer carry
   `paperWhiteNits`/`headroom`, which were only ever consumed by the deleted rolloff.

   **Also migrated `display.comp` → `display.comp.slang`** (Slang, matching `shaders/world/*`,
   rather than GLSL) while rewriting it — build.gradle already globs `**/*.comp.slang` and strips
   the suffix for the output name, so this needed no build changes; `RtDisplayPipeline`'s loader
   still asks for `display.comp.spv` unmodified. Scoped to this one file for now — the other
   `shaders/display/*.comp` files (`exposure_hist`, `exposure_resolve`, `hdr_ui_composite`,
   `sdr_present`) are still GLSL; `exposure_hist`/`exposure_resolve` migrate as part of
   [EXPOSURE_PLAN.md](EXPOSURE_PLAN.md)'s work, not here.
5. Land exposure plan §S3 (compensation curve) tuning against the new operator, not before.
