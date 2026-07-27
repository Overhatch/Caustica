// Maps the display-res scene-linear BT.2020 RT image to sRGB SDR and, when enabled, PQ/BT.2020 HDR,
// via baked ACES 2.0 display-transform LUTs (see tools/bake_display_lut.py,
// docs/DISPLAY_TRANSFORM_PLAN.md). Tonemap seam: the path tracer and DLSS-RR use scene-linear
// BT.2020; exposure is applied from the compositor-owned 1x1 image right here, then both outputs are
// a single log2-shaped LUT fetch.

[[vk::binding(0, 0)]] [format("rgba8")]   RWTexture2D<float4> outputImage; // LDR world target image
[[vk::binding(1, 0)]] [format("rgba16f")] RWTexture2D<float4> rtImage;     // scene-linear BT.2020 RT radiance
[[vk::binding(2, 0)]] [format("r32f")]    RWTexture2D<float>  exposureImage; // 1x1 linear exposure scale
[[vk::binding(3, 0)]] [format("rgba16f")] RWTexture2D<float4> hdrImage;    // PQ-encoded ([0,1], ST.2084) HDR display
// Baked ACES 2.0 SDR display transform. Domain is exposed scene-linear BT.2020, log2-shaped; range is
// sRGB-OETF-encoded BT.709, matching outputImage exactly, so no further gamut step or encode is needed.
[[vk::binding(4, 0)]] Sampler3D toneLut;
// Baked ACES 2.0 HDR display transform. Same shaper domain as toneLut; range is already ST.2084/PQ-
// encoded BT.2020 (one of a fixed set of mastering-nits targets, selected host-side to match
// Hdr.PEAK_NITS -- see RtComposite), matching hdrImage exactly, so no separate encode is needed either.
[[vk::binding(5, 0)]] Sampler3D hdrToneLut;

struct DisplayPush {
    int hdrEnabled;      // 0 = SDR only, 1 = also write the PQ HDR image
    float lutSize;       // toneLut/hdrToneLut texels per axis; both LUTs share one size (asserted at load)
    float acesExposure;  // linear mid-grey placement bias -- see CausticaConfig.Rt.Tonemap.ACES_EXPOSURE_EV
};
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

// acesExposure is a mid-grey placement bias, tuned together with the exposure system's own key/curve
// (see docs/EXPOSURE_PLAN.md S3) rather than a fixed constant -- it's what actually decides how bright
// scene-linear 0.18 renders, since the LUT itself has no notion of exposure. Shared by SDR and HDR so
// the two stay appearance-matched at one exposure, which is the whole point of a peak-luminance-
// parameterized operator (see docs/DISPLAY_TRANSFORM_PLAN.md).
float3 exposed(float3 scene2020, float exposure) {
    return max(scene2020 * exposure * pc.acesExposure, float3(0.0));
}

// Exposure -> log2 shaper -> trilinear LUT fetch. The LUT already bakes in ACES 2.0's gamut mapping to
// BT.709 and the sRGB OETF, so its output goes straight to outputImage. Single mip (RtToneLut), so an
// explicit level-0 SampleLevel is exact, not an approximation.
float3 tonemap(float3 scene2020, float exposure) {
    float3 uvw = shaperEncode(exposed(scene2020, exposure));
    return clamp(toneLut.SampleLevel(lutTexCoord(uvw), 0.0).rgb, 0.0, 1.0);
}

// Same shaper as tonemap() but sampling the HDR LUT, whose range is already PQ-encoded BT.2020 -- no
// separate encode or paperWhite/headroom mapping needed.
float3 tonemapHdr(float3 scene2020, float exposure) {
    float3 uvw = shaperEncode(exposed(scene2020, exposure));
    return clamp(hdrToneLut.SampleLevel(lutTexCoord(uvw), 0.0).rgb, 0.0, 1.0);
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
