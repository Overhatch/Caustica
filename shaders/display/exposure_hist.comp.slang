// Full-res log2-luminance histogram over the post-RR trace color (see RtExposure -- metered here,
// not pre-RR, so the average isn't Jensen-biased by Monte-Carlo noise). See docs/EXPOSURE_PLAN.md.

[[vk::binding(0, 0)]] [format("rgba16f")] RWTexture2D<float4> colorImage;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint, Std430DataLayout> histBins; // 256 bins

static const float LOG_MIN = -12.0;
static const float LOG_MAX = 12.0;
static const float INV_LOG_RANGE = 1.0 / (LOG_MAX - LOG_MIN);
static const float3 BT2020_LUMA = float3(0.2627, 0.6780, 0.0593);

// One 16x16 workgroup == 256 threads == 256 bins, so each thread owns exactly one bin for the
// zero-init and the final flush. Per-pixel atomics land in shared memory (one SM, fast) instead of
// the global buffer (contended across the whole dispatch -- sky pixels alias to the same few bins),
// then only the non-empty bins pay a single global atomicAdd each.
groupshared uint localBins[256];

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID, uint groupIndex : SV_GroupIndex) {
    localBins[groupIndex] = 0u;
    GroupMemoryBarrierWithGroupSync();

    int2 pix = int2(dispatchId.xy);
    uint w, h;
    colorImage.GetDimensions(w, h);
    if (pix.x < int(w) && pix.y < int(h)) {
        float3 rgb = max(colorImage[pix].rgb, float3(0.0));
        // The reconstructed scene image is linear BT.2020; meter it in the transport basis.
        float lum = dot(rgb, BT2020_LUMA);
        float logLum = clamp(log2(max(lum, 1.0e-5)), LOG_MIN, LOG_MAX);
        uint bin = min(uint(floor((logLum - LOG_MIN) * INV_LOG_RANGE * 256.0)), 255u);
        uint unused;
        InterlockedAdd(localBins[bin], 1u, unused);
    }

    GroupMemoryBarrierWithGroupSync();
    uint count = localBins[groupIndex];
    if (count > 0u) {
        uint unused2;
        InterlockedAdd(histBins[groupIndex], count, unused2);
    }
}
