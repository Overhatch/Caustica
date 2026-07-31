package dev.comfyfluffy.caustica.rt.pipeline;

import com.mojang.blaze3d.vulkan.VulkanCommandEncoder;
import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.CausticaMod;
import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtDebugLabels;
import dev.comfyfluffy.caustica.rt.RtSceneUnits;
import dev.comfyfluffy.caustica.rt.RtLookPackage;
import dev.comfyfluffy.caustica.rt.accel.RtBuffer;
import dev.comfyfluffy.caustica.rt.accel.RtImage;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkClearColorValue;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkImageSubresourceRange;

import java.util.Objects;

/** Owns the display exposure value shared by the RT compositor's display-mapping passes. */
public final class RtExposure {
    private RtImage image;
    private RtBuffer histogram;
    private RtBuffer state;
    private RtExposurePipeline pipeline;
    private boolean logged;
    private long lastFrameNanos;
    private long lastDiagLogNanos;
    private String cachedCurveSpec;
    private ExposureCurve cachedCurve;
    /** This frame's latched pre-exposure; see {@link #beginFrame()}. */
    private float framePreExposure = 1.0f;

    // ExposureState byte layout (std430, see pipelines/exposure_resolve/bindings.slang) -- must match field-for-
    // field. Population/curve diagnostics append after S4's already-reserved history fields so their
    // existing offsets remain stable.
    private static final int STATE_BYTES = 88;
    private static final long OFF_PREVIOUS = 0L;
    private static final long OFF_INITIALIZED = 4L;
    private static final long OFF_EV_SCENE = 8L;
    private static final long OFF_EV_TARGET = 12L;
    private static final long OFF_EV_APPLIED = 16L;
    private static final long OFF_CLIP_LOW_FRAC = 20L;
    private static final long OFF_CLIP_HIGH_FRAC = 24L;
    private static final long OFF_METERING_SKY_SCALE = 64L;
    private static final long OFF_METERING_SKY_FRAC = 68L;
    private static final long OFF_CURVE_COMPENSATION = 72L;
    private static final long OFF_EFFECTIVE_SLOPE = 76L;
    private static final long OFF_METERING_EMISSIVE_SCALE = 80L;
    private static final long OFF_METERING_EMISSIVE_FRAC = 84L;
    private static final long DIAG_LOG_INTERVAL_NANOS = 1_000_000_000L;

    public RtImage image() {
        return image;
    }

    public boolean ready() {
        return image != null;
    }

    public RtBuffer stateBuffer() {
        return state;
    }

    /** Immutable exposure values attached to a residual-exposed EXR capture. */
    public record CaptureMetadata(
            float preExposure,
            float residualExposure,
            float absoluteExposure,
            String mode,
            float evScene,
            float evTarget,
            float evApplied
    ) {
    }

    /**
     * Snapshot the controller after the capture copy has completed.
     *
     * <p>{@code residualExposure} is read from the same 1x1 GPU image that the display shader samples.
     * The absolute multiplier can therefore be reconstructed exactly as
     * {@code preExposure * residualExposure}, even when auto exposure corrected a stale prediction.
     */
    public CaptureMetadata captureMetadata(float residualExposure) {
        if (!Float.isFinite(residualExposure) || residualExposure <= 0.0f) {
            throw new IllegalArgumentException("Invalid residual exposure " + residualExposure);
        }
        Mode currentMode = mode();
        float pre = preExposure();
        float absolute = pre * residualExposure;
        if (currentMode != Mode.AUTO || state == null || state.mapped == 0L) {
            float ev = manualEv();
            return new CaptureMetadata(pre, residualExposure, absolute, currentMode.configName,
                    Float.NaN, ev, ev);
        }
        state.invalidate();
        return new CaptureMetadata(pre, residualExposure, absolute, currentMode.configName,
                MemoryUtil.memGetFloat(state.mapped + OFF_EV_SCENE),
                MemoryUtil.memGetFloat(state.mapped + OFF_EV_TARGET),
                MemoryUtil.memGetFloat(state.mapped + OFF_EV_APPLIED));
    }

    public void ensureResources(RtContext ctx) {
        if (image == null) {
            image = ctx.createStorageImage(1, 1, VK10.VK_FORMAT_R32_SFLOAT, "display exposure");
        }
        // The final debug pass always binds the state buffer, including in manual mode. Keep this tiny
        // resource permanently available; histogram/pipeline allocation remains auto-only.
        if (state == null) {
            state = ctx.createBuffer(STATE_BYTES, VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true, "exposure state");
            resetAutoHistory();
        }
        if (mode() == Mode.AUTO) {
            if (histogram == null) {
                // Separate ordinary-surface/sky/emissive histograms let resolve enforce both
                // population caps exactly without a second full-image dispatch.
                histogram = ctx.createBuffer(768L * Integer.BYTES,
                        VK10.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK10.VK_BUFFER_USAGE_TRANSFER_DST_BIT, false,
                        "exposure histogram");
            }
            if (pipeline == null) {
                pipeline = RtExposurePipeline.create(ctx);
            }
        }
        logOnce();
    }

    public void record(RtContext ctx, VkCommandBuffer cmd, MemoryStack stack,
                       RtImage traceColor, RtImage guideDepth, RtImage guideAlbedo) {
        if (image == null) {
            throw new IllegalStateException("RT exposure image not created");
        }
        if (mode() == Mode.AUTO) {
            recordAuto(ctx, cmd, stack, traceColor, guideDepth, guideAlbedo);
            return;
        }
        try (RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "exposure manual write")) {
            VkClearColorValue color = VkClearColorValue.calloc(stack);
            // Residual, not absolute: raygen already applied preExposure (which in manual mode IS
            // manualExposureScale, making this exactly 1.0). See preExposure().
            color.float32(0, manualExposureScale() / Math.max(preExposure(), 1.0e-12f));
            VkImageSubresourceRange.Buffer range = VkImageSubresourceRange.calloc(1, stack);
            range.get(0).aspectMask(VK10.VK_IMAGE_ASPECT_COLOR_BIT)
                    .baseMipLevel(0).levelCount(1).baseArrayLayer(0).layerCount(1);
            VK10.vkCmdClearColorImage(cmd, image.image, VK10.VK_IMAGE_LAYOUT_GENERAL, color, range);
        }
    }

    public void destroy() {
        if (pipeline != null) {
            pipeline.destroy();
            pipeline = null;
        }
        if (histogram != null) {
            histogram.destroy();
            histogram = null;
        }
        if (state != null) {
            state.destroy();
            state = null;
        }
        if (image != null) {
            image.destroy();
            image = null;
        }
    }

    // Manual mode's exposure scale, also used as the auto-history seed (resetAutoHistory) so the very
    // first auto-exposure frame starts from the dialed-in EV bias instead of a bare 1.0.
    private float manualExposureScale() {
        return CausticaConfig.Rt.Exposure.clampScale((float) Math.pow(2.0, manualEv()));
    }

    private void recordAuto(RtContext ctx, VkCommandBuffer cmd, MemoryStack stack,
                            RtImage traceColor, RtImage guideDepth, RtImage guideAlbedo) {
        if (pipeline == null || histogram == null || state == null) {
            throw new IllegalStateException("RT auto exposure resources not created");
        }
        pipeline.setResources(traceColor.view, guideDepth.view, guideAlbedo.view,
                histogram, image.view, state);
        try (RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "exposure histogram clear")) {
            VK10.vkCmdFillBuffer(cmd, histogram.handle, 0, histogram.size, 0);
        }
        VulkanCommandEncoder.memoryBarrier(cmd, stack);
        AutoConfig config = autoConfig();
        pipeline.dispatchHistogram(cmd, traceColor.width, traceColor.height, config);
        VulkanCommandEncoder.memoryBarrier(cmd, stack);
        pipeline.dispatchResolve(cmd, config, frameTimeSeconds());
        logDiagnosticsIfDue();
    }

    /**
     * S0 observability (docs/EXPOSURE_PLAN.md): throttled log of the controller's internal EVs, gated
     * behind the frame-stats toggle since that's the existing "I want renderer internals" switch.
     * Reads {@code state.mapped} with no fence — the buffer is host-visible+coherent and this frame's
     * GPU work hasn't executed yet when this runs, so it's last frame's value; fine for a debug log,
     * per the plan's own tolerance for staleness here.
     */
    private void logDiagnosticsIfDue() {
        if (!CausticaConfig.Rt.FrameStats.ENABLED.value() || state == null || state.mapped == 0L) {
            return;
        }
        long now = System.nanoTime();
        if (lastDiagLogNanos != 0L && now - lastDiagLogNanos < DIAG_LOG_INTERVAL_NANOS) {
            return;
        }
        lastDiagLogNanos = now;
        float evScene = MemoryUtil.memGetFloat(state.mapped + OFF_EV_SCENE);
        float evTarget = MemoryUtil.memGetFloat(state.mapped + OFF_EV_TARGET);
        float evApplied = MemoryUtil.memGetFloat(state.mapped + OFF_EV_APPLIED);
        float clipLowFrac = MemoryUtil.memGetFloat(state.mapped + OFF_CLIP_LOW_FRAC);
        float clipHighFrac = MemoryUtil.memGetFloat(state.mapped + OFF_CLIP_HIGH_FRAC);
        float skyScale = MemoryUtil.memGetFloat(state.mapped + OFF_METERING_SKY_SCALE);
        float skyFrac = MemoryUtil.memGetFloat(state.mapped + OFF_METERING_SKY_FRAC);
        float curveCompensation = MemoryUtil.memGetFloat(state.mapped + OFF_CURVE_COMPENSATION);
        float effectiveSlope = MemoryUtil.memGetFloat(state.mapped + OFF_EFFECTIVE_SLOPE);
        float emissiveScale = MemoryUtil.memGetFloat(state.mapped + OFF_METERING_EMISSIVE_SCALE);
        float emissiveFrac = MemoryUtil.memGetFloat(state.mapped + OFF_METERING_EMISSIVE_FRAC);
        AutoConfig cfg = autoConfig();
        boolean pinnedLow = evTarget <= cfg.minEv() + 0.01f;
        boolean pinnedHigh = evTarget >= cfg.maxEv() - 0.01f;
        // evScene is EV100 (docs/SCENE_UNITS_PLAN.md §1); evTarget/evApplied are log2 of the ABSOLUTE
        // exposure multiplier, i.e. pre-exposure already divided back out, so they stay comparable
        // across frames regardless of what preExposure happened to be.
        CausticaMod.LOGGER.info(
                "RT exposure diag: evScene(EV100)={} evTarget={}{} evApplied={} preExposure={} "
                        + "clipLow={}% clipHigh={}% skyScale={} skyWeight={}% emissiveScale={} "
                        + "emissiveWeight={}% curveComp={} effectiveSlope={}",
                fmt(evScene), fmt(evTarget), pinnedLow ? " (at minEv clamp)" : pinnedHigh ? " (at maxEv clamp)" : "",
                fmt(evApplied), fmt(preExposure()), fmt(clipLowFrac * 100.0f), fmt(clipHighFrac * 100.0f),
                fmt(skyScale), fmt(skyFrac * 100.0f), fmt(emissiveScale),
                fmt(emissiveFrac * 100.0f), fmt(curveCompensation), fmt(effectiveSlope));
    }

    private static String fmt(float v) {
        return String.format(java.util.Locale.ROOT, "%.2f", v);
    }

    /**
     * One-line summary for the F3 debug screen ({@code RtExposureDebugEntry}). Unlike
     * {@link #logDiagnosticsIfDue()} this is not throttled and not gated on
     * {@code CausticaConfig.Rt.FrameStats.ENABLED} -- F3 only calls it once the player has enabled
     * that entry, and the game's own render cadence is throttle enough. Returns {@code null} when
     * there is nothing meaningful to show yet (state buffer not created).
     */
    public String debugSummaryLine() {
        if (state == null || state.mapped == 0L) {
            return null;
        }
        if (mode() != Mode.AUTO) {
            return String.format(java.util.Locale.ROOT, "RT exposure: manual %s EV", fmt(manualEv()));
        }
        float evScene = MemoryUtil.memGetFloat(state.mapped + OFF_EV_SCENE);
        float evTarget = MemoryUtil.memGetFloat(state.mapped + OFF_EV_TARGET);
        float evApplied = MemoryUtil.memGetFloat(state.mapped + OFF_EV_APPLIED);
        AutoConfig cfg = autoConfig();
        String clamp = evTarget <= cfg.minEv() + 0.01f ? " (min clamp)"
                : evTarget >= cfg.maxEv() - 0.01f ? " (max clamp)" : "";
        return String.format(java.util.Locale.ROOT, "RT exposure: EV100 %s, applied %s EV%s",
                fmt(evScene), fmt(evApplied), clamp);
    }

    private float frameTimeSeconds() {
        long now = System.nanoTime();
        float dt = lastFrameNanos == 0L ? 1.0f / 60.0f
                : Math.clamp((now - lastFrameNanos) / 1_000_000_000.0f, 1.0f / 240.0f, 0.25f);
        lastFrameNanos = now;
        return dt;
    }

    private void resetAutoHistory() {
        if (state == null || state.mapped == 0L) {
            return;
        }
        // Zero the whole widened struct, not just (previous, initialized): the S0 diagnostic fields
        // and the reserved S4 fields (resetSeq, evHistory) should start clean too, not carry over
        // whatever garbage a fresh VMA allocation happened to contain.
        MemoryUtil.memSet(state.mapped, 0, STATE_BYTES);
        // Under physical units (U2) this seed can be ~15 EV off for an auto-mode daylight scene, since
        // manual-ev defaults to 0. That is a two-frame transient, not a bug: initialized == 0 makes the
        // resolve snap to its computed target rather than smooth toward it, and the frame after that
        // meters against a preExposure derived from it. Deliberately not special-cased -- a seed that
        // guessed at scene brightness would be a second, unowned exposure model.
        MemoryUtil.memPutFloat(state.mapped + OFF_PREVIOUS, manualExposureScale());
        MemoryUtil.memPutInt(state.mapped + OFF_INITIALIZED, 0);
        MemoryUtil.memPutFloat(state.mapped + OFF_METERING_SKY_SCALE, 1.0f);
        MemoryUtil.memPutFloat(state.mapped + OFF_METERING_EMISSIVE_SCALE, 1.0f);
        state.flush(0L, STATE_BYTES);
        lastFrameNanos = 0L;
        lastDiagLogNanos = 0L;
    }

    private void logOnce() {
        if (logged) {
            return;
        }
        logged = true;
        Mode mode = mode();
        AutoConfig autoConfig = autoConfig();
        String exposureText = mode == Mode.AUTO
                ? "auto(key=" + autoConfig.key + ", minEv=" + autoConfig.minEv + ", maxEv=" + autoConfig.maxEv
                + ", adaptDarken=" + autoConfig.adaptDarken + ", adaptBrighten=" + autoConfig.adaptBrighten
                + ", evBias=" + autoConfig.evBias + ", percentiles=" + autoConfig.lowPercentile
                + ".." + autoConfig.highPercentile + ", stride=" + autoConfig.stride
                + ", centerWeight=" + autoConfig.centerWeightSigma + "/" + autoConfig.centerWeightFloor
                + ", skyCap=" + autoConfig.skyWeightCap
                + ", emissiveCap=" + autoConfig.emissiveWeightCap
                + ", curve=" + CausticaConfig.Rt.Exposure.curve() + ")"
                : Float.toString(manualExposureScale());
        CausticaMod.LOGGER.info("RT display exposure: mode={}, exposure={}, "
                        + "tonemap=aces2.0(lookPackage={},gamma={}), DLSS-RR exposure=NGX auto",
                mode.configName, exposureText, RtLookPackage.current().id(),
                CausticaConfig.Rt.Tonemap.GAMMA.value());
    }

    private static Mode mode() {
        return Mode.parse(CausticaConfig.Rt.Exposure.MODE.get());
    }

    private static float manualEv() {
        return CausticaConfig.Rt.Exposure.MANUAL_EV.value();
    }

    private AutoConfig autoConfig() {
        return new AutoConfig(
                CausticaConfig.Rt.Exposure.KEY.value(),
                CausticaConfig.Rt.Exposure.minEv(),
                CausticaConfig.Rt.Exposure.maxEv(),
                CausticaConfig.Rt.Exposure.ADAPT_DARKEN.value(),
                CausticaConfig.Rt.Exposure.ADAPT_BRIGHTEN.value(),
                manualEv(),
                CausticaConfig.Rt.Exposure.LOW_PERCENTILE.value(),
                CausticaConfig.Rt.Exposure.HIGH_PERCENTILE.value(),
                CausticaConfig.Rt.Exposure.STRIDE.value(),
                CausticaConfig.Rt.Exposure.CENTER_WEIGHT_SIGMA.value(),
                CausticaConfig.Rt.Exposure.CENTER_WEIGHT_FLOOR.value(),
                CausticaConfig.Rt.Exposure.SKY_WEIGHT_CAP.value(),
                CausticaConfig.Rt.Exposure.EMISSIVE_WEIGHT_CAP.value(),
                curveConfig(),
                preExposure());
    }

    /**
     * Latches this frame's pre-exposure. MUST be called once per frame before the world push
     * constants are written, and must not be re-latched afterwards.
     *
     * <p>The raygen multiply and the resolve's divide have to use the <em>same</em> value or they
     * stop cancelling and the frame comes out mis-scaled. Both read {@link #preExposure()}, but at
     * different points in CPU time, while the GPU is asynchronously writing {@code previous} with no
     * synchronisation — so reading the mapped buffer at each use site could observe two different
     * values within one frame. Latching once removes that race; the residual absorbs whatever the
     * latched value failed to predict.
     */
    public void beginFrame() {
        framePreExposure = computePreExposure();
    }

    /**
     * The scalar raygen multiplies into scene radiance before the fp16 write (U1,
     * {@code docs/SCENE_UNITS_PLAN.md} §2), so stored values sit near {@code key} at any absolute
     * scene brightness instead of spanning the ~26 EV that physical units require.
     *
     * <p>Correctness does not depend on this being <em>current</em> — the display pass divides by
     * exactly the same latched value, so any pre-exposure cancels algebraically. Staleness only
     * affects how well-centred the stored values are, which is why last frame's readback is fine and
     * no fence is needed. 1.0 disables the mechanism and is exactly the pre-U1 pipeline.
     */
    public float preExposure() {
        return framePreExposure;
    }

    private float computePreExposure() {
        if (!CausticaConfig.Rt.Exposure.PRE_EXPOSURE.value()) {
            return 1.0f;
        }
        // Manual mode has a known fixed absolute exposure, so pre-exposing by it makes the residual
        // exactly 1.0 -- the best-centred choice available, and it needs no readback.
        if (mode() != Mode.AUTO) {
            return manualExposureScale();
        }
        if (state == null || state.mapped == 0L) {
            return 1.0f;
        }
        // Deliberately NOT Exposure.clampScale: its 1e-4 floor is a bound on the artistic exposure
        // multiplier, and physical units (U2) put noon at ~3e-5 absolute, which that floor would
        // truncate -- silently de-centring exactly the case pre-exposure exists to handle. The
        // controller's own minEv/maxEv already bound this value; here we only reject garbage.
        float previous = MemoryUtil.memGetFloat(state.mapped + OFF_PREVIOUS);
        return Float.isFinite(previous) && previous > 0.0f ? previous : 1.0f;
    }

    record AutoConfig(float key, float minEv, float maxEv, float adaptDarken, float adaptBrighten, float evBias,
                      float lowPercentile, float highPercentile, int stride,
                      float centerWeightSigma, float centerWeightFloor, float skyWeightCap,
                      float emissiveWeightCap,
                      ExposureCurve curve, float preExposure) {
        /**
         * Offset taking the resolve's {@code log2(metered stored luminance)} to EV100. The metered
         * buffer holds {@code L * preExposure}, so the pre-exposure has to come back out before the
         * unit convention's offset applies. See {@code docs/SCENE_UNITS_PLAN.md} §1/§2.
         */
        float evOffset() {
            return RtSceneUnits.EV100_OFFSET - (float) (Math.log(Math.max(preExposure, 1.0e-12f)) / Math.log(2.0));
        }
    }

    private ExposureCurve curveConfig() {
        String spec = CausticaConfig.Rt.Exposure.curve();
        if (cachedCurve != null && Objects.equals(cachedCurveSpec, spec)) {
            return cachedCurve;
        }
        ExposureCurve parsed;
        try {
            parsed = parseCurve(spec);
        } catch (IllegalArgumentException e) {
            throw new IllegalStateException("Invalid exposure curve in look package '"
                    + RtLookPackage.current().id() + "': " + spec, e);
        }
        cachedCurveSpec = spec;
        cachedCurve = parsed;
        return parsed;
    }

    static ExposureCurve parseCurve(String spec) {
        if (spec == null) {
            throw new IllegalArgumentException("curve is null");
        }
        if ("full".equalsIgnoreCase(spec.trim())) {
            return new ExposureCurve(-6.0f, 0.0f, -3.0f, 0.0f, 0.0f, 0.0f, 4.0f, 0.0f);
        }
        String[] encodedPoints = spec.split(",");
        if (encodedPoints.length != 4) {
            throw new IllegalArgumentException("expected exactly four scene:compensation points");
        }
        float[] scene = new float[4];
        float[] compensation = new float[4];
        for (int i = 0; i < encodedPoints.length; i++) {
            String[] pair = encodedPoints[i].trim().split(":", -1);
            if (pair.length != 2) {
                throw new IllegalArgumentException("point " + (i + 1) + " is not scene:compensation");
            }
            try {
                scene[i] = Float.parseFloat(pair[0].trim());
                compensation[i] = Float.parseFloat(pair[1].trim());
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("point " + (i + 1) + " contains a non-number", e);
            }
            if (!Float.isFinite(scene[i]) || !Float.isFinite(compensation[i])) {
                throw new IllegalArgumentException("point " + (i + 1) + " is not finite");
            }
        }
        // Four elements: insertion sort avoids a temporary point-object list.
        for (int i = 1; i < 4; i++) {
            float sceneValue = scene[i];
            float compensationValue = compensation[i];
            int j = i - 1;
            while (j >= 0 && scene[j] > sceneValue) {
                scene[j + 1] = scene[j];
                compensation[j + 1] = compensation[j];
                j--;
            }
            scene[j + 1] = sceneValue;
            compensation[j + 1] = compensationValue;
        }
        for (int i = 1; i < 4; i++) {
            if (scene[i] - scene[i - 1] < 1.0e-4f) {
                throw new IllegalArgumentException("scene EV points must be distinct");
            }
        }
        return new ExposureCurve(scene[0], compensation[0], scene[1], compensation[1],
                scene[2], compensation[2], scene[3], compensation[3]);
    }

    record ExposureCurve(float scene0, float compensation0, float scene1, float compensation1,
                         float scene2, float compensation2, float scene3, float compensation3) {
        float compensationAt(float sceneEv) {
            if (sceneEv <= scene0) {
                return compensation0;
            }
            if (sceneEv < scene1) {
                return interpolate(sceneEv, scene0, compensation0, scene1, compensation1);
            }
            if (sceneEv < scene2) {
                return interpolate(sceneEv, scene1, compensation1, scene2, compensation2);
            }
            if (sceneEv < scene3) {
                return interpolate(sceneEv, scene2, compensation2, scene3, compensation3);
            }
            return compensation3;
        }

        float effectiveSlopeAt(float sceneEv) {
            if (sceneEv <= scene0 || sceneEv >= scene3) {
                return 1.0f;
            }
            if (sceneEv < scene1) {
                return 1.0f - slope(scene0, compensation0, scene1, compensation1);
            }
            if (sceneEv < scene2) {
                return 1.0f - slope(scene1, compensation1, scene2, compensation2);
            }
            return 1.0f - slope(scene2, compensation2, scene3, compensation3);
        }

        private static float interpolate(float x, float x0, float y0, float x1, float y1) {
            float t = (x - x0) / (x1 - x0);
            return y0 + t * (y1 - y0);
        }

        private static float slope(float x0, float y0, float x1, float y1) {
            return (y1 - y0) / (x1 - x0);
        }
    }

    private enum Mode {
        MANUAL("manual"),
        AUTO("auto");

        private final String configName;

        Mode(String configName) {
            this.configName = configName;
        }

        static Mode parse(String value) {
            if (value != null) {
                for (Mode mode : values()) {
                    if (mode.configName.equalsIgnoreCase(value)) {
                        return mode;
                    }
                }
            }
            return AUTO;
        }
    }
}
