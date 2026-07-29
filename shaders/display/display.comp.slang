// Maps the display-res scene-linear ACEScg RT image to sRGB SDR and, when enabled, PQ/BT.2020 HDR,
// via baked ACES 2.0 display-transform LUTs (see tools/bake_display_lut.py,
// docs/DISPLAY_TRANSFORM_PLAN.md). Tonemap seam: the path tracer and DLSS-RR use scene-linear
// ACEScg; exposure is applied from the compositor-owned 1x1 image right here, then both outputs are
// a single log2-shaped LUT fetch.

import display_common;

[[vk::binding(0, 0)]] [format("rgba8")]   RWTexture2D<float4> outputImage; // LDR world target image
[[vk::binding(1, 0)]] [format("rgba16f")] RWTexture2D<float4> rtImage;     // scene-linear ACEScg RT radiance
[[vk::binding(2, 0)]] [format("r32f")]    RWTexture2D<float>  exposureImage; // 1x1 linear exposure scale
[[vk::binding(3, 0)]] [format("rgba16f")] RWTexture2D<float4> hdrImage;    // PQ-encoded ([0,1], ST.2084) HDR display
// Baked ACES 2.0 SDR display transform. Domain is exposed scene-linear ACEScg, log2-shaped; range is
// sRGB-OETF-encoded BT.709, matching outputImage exactly, so no further gamut step or encode is needed.
[[vk::binding(4, 0)]] Sampler3D toneLut;
// Baked ACES 2.0 HDR display transform. Same shaper domain as toneLut; range is already ST.2084/PQ-
// encoded BT.2020 (one of a fixed set of mastering-nits targets, selected host-side to match
// Hdr.PEAK_NITS -- see RtComposite), matching hdrImage exactly, so no separate encode is needed either.
[[vk::binding(5, 0)]] Sampler3D hdrToneLut;

[[vk::push_constant]] DisplayPush pc;

// MUST exactly match SHAPER_LO_STOPS/SHAPER_HI_STOPS in tools/bake_display_lut.py -- these are the two
// halves of one shaper function, one run at bake time (Python), one at sample time (here); if they
// drift the LUT fetch reads the wrong shelf of the LUT everywhere, silently.
static const float LUT_SHAPER_LO_STOPS = -12.0;
static const float LUT_SHAPER_HI_STOPS = 12.0;

float3 shaperEncode(float3 linearColor) {
    float3 stops = log2(max(linearColor, float3(1.0e-6)));
    return clamp((stops - LUT_SHAPER_LO_STOPS) / (LUT_SHAPER_HI_STOPS - LUT_SHAPER_LO_STOPS), 0.0, 1.0);
}

// Edge-aligned LUT: texel 0's centre is uvw=0.0 and texel (lutSize-1)'s centre is uvw=1.0, but a
// sampler's normalized [0,1] domain spans texel *centres* at 0.5/N .. 1-0.5/N by default. Remap into
// that convention; RtToneLut's CLAMP_TO_EDGE then holds the boundary texel for any exposed value
// outside the shaper's +/-12 EV range instead of wrapping. Shared by both LUTs -- they're baked at
// the same lutSize (RtComposite asserts this when loading the HDR LUT).
float3 lutTexCoord(float3 uvw) {
    return uvw * ((pc.lutSize - 1.0) / pc.lutSize) + float3(0.5 / pc.lutSize);
}

float3 exposed(float3 sceneAcesCg, float exposure) {
    return max(sceneAcesCg * exposure, float3(0.0));
}

// Artistic gamma after the view/display transform. Apply the curve to display luminance and scale
// RGB uniformly instead of exponentiating each channel: this lifts shadows/midtones without pulling
// chromaticities toward white. The uniform scale is limited at the display boundary so saturated
// colors preserve their channel ratios instead of clipping individual channels.
static const float3 BT709_LUMA = float3(0.2126, 0.7152, 0.0722);
static const float3 BT2020_LUMA = float3(0.2627, 0.6780, 0.0593);

float3 luminanceGamma(float3 linearDisplayColor, float3 lumaCoefficients) {
    float3 color = clamp(linearDisplayColor, 0.0, 1.0);
    float luminance = dot(color, lumaCoefficients);
    if (luminance <= 0.0) {
        return color;
    }

    float adjustedLuminance = pow(luminance, clamp(pc.gamma, 0.1, 5.0));
    float scale = adjustedLuminance / luminance;
    float maxChannel = max(color.r, max(color.g, color.b));
    float gamutLimitedScale = 1.0 / max(maxChannel, 1.0e-8);
    return color * min(scale, gamutLimitedScale);
}

float srgbDecodeChannel(float code) {
    return code <= 0.04045 ? code / 12.92 : pow((code + 0.055) / 1.055, 2.4);
}

float srgbEncodeChannel(float linearValue) {
    return linearValue <= 0.0031308
            ? 12.92 * linearValue
            : 1.055 * pow(linearValue, 1.0 / 2.4) - 0.055;
}

float3 srgbDecode(float3 code) {
    return float3(srgbDecodeChannel(code.r), srgbDecodeChannel(code.g), srgbDecodeChannel(code.b));
}

float3 srgbEncode(float3 linearColor) {
    return float3(
            srgbEncodeChannel(linearColor.r),
            srgbEncodeChannel(linearColor.g),
            srgbEncodeChannel(linearColor.b));
}

float3 displayGammaSdr(float3 displayColor) {
    float3 code = clamp(displayColor, 0.0, 1.0);
    if (abs(pc.gamma - 1.0) < 1.0e-6) {
        return code;
    }
    return clamp(srgbEncode(luminanceGamma(srgbDecode(code), BT709_LUMA)), 0.0, 1.0);
}

static const float PQ_M1 = 0.1593017578125;
static const float PQ_M2 = 78.84375;
static const float PQ_C1 = 0.8359375;
static const float PQ_C2 = 18.8515625;
static const float PQ_C3 = 18.6875;

float3 pqDecode(float3 code) {
    float3 p = pow(clamp(code, 0.0, 1.0), float3(1.0 / PQ_M2));
    float3 y = max((p - PQ_C1) / max(PQ_C2 - PQ_C3 * p, float3(1.0e-6)), float3(0.0));
    return pow(y, float3(1.0 / PQ_M1)) * 10000.0;
}

float3 pqEncode(float3 nits) {
    float3 y = pow(max(nits, float3(0.0)) / 10000.0, float3(PQ_M1));
    return pow((PQ_C1 + PQ_C2 * y) / (1.0 + PQ_C3 * y), float3(PQ_M2));
}

float3 displayGammaHdr(float3 pqColor) {
    if (abs(pc.gamma - 1.0) < 1.0e-6) {
        return clamp(pqColor, 0.0, 1.0);
    }
    float peak = max(pc.hdrPeakNits, 1.0);
    float3 normalizedNits = clamp(pqDecode(pqColor) / peak, 0.0, 1.0);
    float3 adjustedNits = peak * luminanceGamma(normalizedNits, BT2020_LUMA);
    return clamp(pqEncode(adjustedNits), 0.0, 1.0);
}

// Exposure -> log2 shaper -> trilinear LUT fetch. The LUT already bakes in ACES 2.0's gamut mapping to
// BT.709 and the sRGB OETF, so its output goes straight to outputImage. Single mip (RtToneLut), so an
// explicit level-0 SampleLevel is exact, not an approximation.
float3 tonemap(float3 sceneAcesCg, float exposure) {
    float3 uvw = shaperEncode(exposed(sceneAcesCg, exposure));
    return displayGammaSdr(toneLut.SampleLevel(lutTexCoord(uvw), 0.0).rgb);
}

// Same shaper as tonemap() but sampling the HDR LUT, whose range is already PQ-encoded BT.2020 -- no
// separate encode or paperWhite/headroom mapping needed.
float3 tonemapHdr(float3 sceneAcesCg, float exposure) {
    float3 uvw = shaperEncode(exposed(sceneAcesCg, exposure));
    return displayGammaHdr(hdrToneLut.SampleLevel(lutTexCoord(uvw), 0.0).rgb);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
    int2 pix = int2(dispatchId.xy);
    uint w, h;
    outputImage.GetDimensions(w, h);
    if (pix.x >= int(w) || pix.y >= int(h)) {
        return;
    }

    float4 rt = rtImage[pix];
    float exposure = max(exposureImage[int2(0, 0)], 0.0);
    outputImage[pix] = float4(tonemap(rt.rgb, exposure), 1.0);

    if (pc.hdrEnabled != 0) {
        hdrImage[pix] = float4(tonemapHdr(rt.rgb, exposure), 1.0);
    }
}
