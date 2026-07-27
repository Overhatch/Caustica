// Single-invocation exposure controller: percentile-trimmed mean of log2 luminance -> key -> clamp ->
// exponential smoothing (in LINEAR exposure-scale space -- see docs/EXPOSURE_PLAN.md D4/S4, not yet
// changed to EV-space smoothing). ExposureState is widened past the original (previous, initialized)
// pair with S0's read-only diagnostics (evScene/evTarget/evApplied/clipLowFrac/clipHighFrac), read
// back one frame late on the CPU for the debug log line in RtExposure -- see docs/EXPOSURE_PLAN.md S0.
// resetSeq/evHistory are reserved for S4 (hard reset, median-of-N flash rejection) and are neither
// read nor written here yet.

[[vk::binding(0, 0)]] StructuredBuffer<uint, Std430DataLayout> histBins; // 256 bins, readonly
[[vk::binding(1, 0)]] [format("r32f")] RWTexture2D<float> exposureImage;

struct ExposureState {
    float previous;
    uint initialized;
    float evScene;      // S0: trimmed-mean measured log2 luminance this frame
    float evTarget;      // S0: key-driven target EV, before temporal smoothing
    float evApplied;      // S0: final smoothed EV actually written to exposureImage
    float clipLowFrac;    // S0: fraction of metered pixels landing in the histogram's bottom bin
    float clipHighFrac;    // S0: fraction landing in the top bin
    uint resetSeq;         // reserved for S4, unused here
    float evHistory[8];    // reserved for S4's median-of-N ring, unused here
};
[[vk::binding(2, 0)]] RWStructuredBuffer<ExposureState, Std430DataLayout> stateBuf;

struct Push {
    uint pixelCount;
    float key;
    float minEv;
    float maxEv;
    float adaptUp;
    float adaptDown;
    float frameTimeSeconds;
    float evBias;
};
[[vk::push_constant]] Push pc;

static const float LOG_MIN = -12.0;
static const float LOG_MAX = 12.0;
static const float LOG_STEP = (LOG_MAX - LOG_MIN) / 256.0;

float binCenter(uint bin) {
    return LOG_MIN + (float(bin) + 0.5) * LOG_STEP;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
    uint total = max(pc.pixelCount, 1u);
    uint lowCount = uint(floor(float(total) * 0.50));
    uint highCount = max(lowCount + 1u, uint(ceil(float(total) * 0.95)));

    uint cumulative = 0u;
    float weightedLogLum = 0.0;
    uint weightedCount = 0u;
    for (uint i = 0u; i < 256u; ++i) {
        uint count = histBins[i];
        uint start = cumulative;
        uint end = cumulative + count;
        uint takeStart = max(start, lowCount);
        uint takeEnd = min(end, highCount);
        if (takeEnd > takeStart) {
            uint take = takeEnd - takeStart;
            weightedLogLum += binCenter(i) * float(take);
            weightedCount += take;
        }
        cumulative = end;
    }

    float avgLogLum = weightedCount > 0u ? weightedLogLum / float(weightedCount) : 0.0;
    float target = pc.key * exp2(pc.evBias - avgLogLum);
    target = clamp(target, exp2(pc.minEv), exp2(pc.maxEv));

    bool initialized = stateBuf[0].initialized != 0u;
    float prev = initialized ? max(stateBuf[0].previous, 1.0e-4) : target;
    float rate = target > prev ? pc.adaptUp : pc.adaptDown;
    float alpha = initialized ? 1.0 - exp(-pc.frameTimeSeconds / max(rate, 1.0e-4)) : 1.0;
    float exposure = lerp(prev, target, clamp(alpha, 0.0, 1.0));

    stateBuf[0].previous = exposure;
    stateBuf[0].initialized = 1u;
    stateBuf[0].evScene = avgLogLum;
    stateBuf[0].evTarget = log2(max(target, 1.0e-6));
    stateBuf[0].evApplied = log2(max(exposure, 1.0e-6));
    stateBuf[0].clipLowFrac = float(histBins[0]) / float(total);
    stateBuf[0].clipHighFrac = float(histBins[255]) / float(total);
    exposureImage[int2(0, 0)] = exposure;
}
