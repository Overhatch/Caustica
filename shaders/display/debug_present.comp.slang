// Computes and presents debugView content as a final inspection pass after ordinary path-traced
// radiance, RR/fallback upscale, exposure metering, and the ACES display transform. Previously this
// coloring lived in
// writeDebugView (shaders/world/guides.slang), called from the primary raygen -- the hottest,
// most register-starved shader in the renderer (~126 live registers already, occupancy-bound per
// docs/GPU_PERF_PLAN.md: "in this kernel, optimizations must REDUCE live state, never add it").
// A debug-only feature has no business costing register pressure there, so it moved here: a small,
// non-performance-critical compute pass that only ever runs while a debug view is active.
//
// Reads the guide buffers directly at their real render resolution and nearest-samples them into the
// display-resolution output. Debug mode deliberately leaves DLSS-RR and its jitter enabled, so this
// pass shows the actual guide inputs used by RR rather than changing the renderer being inspected.

[[vk::binding(0, 0)]] [format("rgba8")]   RWTexture2D<float4> outputImage;
[[vk::binding(1, 0)]] [format("rgba16f")] RWTexture2D<float4> gNormal;     // xyz world normal, w linear roughness
[[vk::binding(2, 0)]] [format("rgba16f")] RWTexture2D<float4> gAlbedo;     // rgb BT.2020 diffuse albedo
[[vk::binding(3, 0)]] [format("r32f")]    RWTexture2D<float>  gDepth;      // HW reversed-Z depth
[[vk::binding(4, 0)]] [format("rg16f")]   RWTexture2D<float2> gMotion;     // screen-space motion (render px)
[[vk::binding(5, 0)]] [format("rgba16f")] RWTexture2D<float4> gSpecAlbedo; // BT.2020 exact-mirror reflected diffuse x reflectance
[[vk::binding(6, 0)]] [format("rg16f")]   RWTexture2D<float2> gSpecMotion; // reflection MVs for RR

struct Push {
    uint debugView; // 1..7; this pass is only ever dispatched for debugView != 0 (see RtComposite)
};
[[vk::push_constant]] Push pc;

// Duplicated from world_common.slang rather than imported: this file lives outside the shaders/world
// module graph, and it's one 3x3 matrix -- not worth a cross-directory import for.
static const float3 BT709_TO_BT2020_R = float3(0.6274039, 0.3292830, 0.0433131);
static const float3 BT709_TO_BT2020_G = float3(0.0690973, 0.9195406, 0.0113612);
static const float3 BT709_TO_BT2020_B = float3(0.0163916, 0.0880132, 0.8955953);

float3 bt709ToBt2020(float3 color) {
    return float3(dot(color, BT709_TO_BT2020_R), dot(color, BT709_TO_BT2020_G), dot(color, BT709_TO_BT2020_B));
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

    uint guideW, guideH;
    gNormal.GetDimensions(guideW, guideH);
    float2 uv = (float2(pix) + 0.5) / float2(w, h);
    int2 guidePix = min(int2(uv * float2(guideW, guideH)), int2(guideW, guideH) - 1);
    float2 motionToDisplay = float2(w, h) / float2(guideW, guideH);

    float4 normalRough = gNormal[guidePix];
    float3 dbg;
    if (pc.debugView == 1u) {
        dbg = bt709ToBt2020(normalRough.xyz * 0.5 + 0.5);
    } else if (pc.debugView == 2u) {
        dbg = gAlbedo[guidePix].rgb;
    } else if (pc.debugView == 3u) {
        float depth = gDepth[guidePix];
        dbg = float3(depth, depth, depth);
    } else if (pc.debugView == 4u) {
        dbg = float3(normalRough.w, normalRough.w, normalRough.w);
    } else if (pc.debugView == 6u) {
        dbg = gSpecAlbedo[guidePix].rgb;
    } else if (pc.debugView == 7u) {
        dbg = bt709ToBt2020(float3(clamp(0.5 + gSpecMotion[guidePix] * motionToDisplay * 0.05, 0.0, 1.0), 0.5));
    } else {
        dbg = bt709ToBt2020(float3(clamp(0.5 + gMotion[guidePix] * motionToDisplay * 0.05, 0.0, 1.0), 0.5));
    }
    outputImage[pix] = float4(clamp(dbg, 0.0, 1.0), 1.0);
}
