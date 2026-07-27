// Single-invocation exposure controller: percentile-trimmed mean of log2 luminance -> key -> clamp ->
// exponential smoothing (in LINEAR exposure-scale space -- see docs/EXPOSURE_PLAN.md D4/S4, not yet
// changed to EV-space smoothing). ExposureState is widened past the original (previous, initialized)
// pair with S0's read-only diagnostics (evScene/evTarget/evApplied/clipLowFrac/clipHighFrac), read
// back one frame late on the CPU for the debug log line in RtExposure -- see docs/EXPOSURE_PLAN.md S0.
// resetSeq/evHistory are reserved for S4 (hard reset, median-of-N flash rejection) and are neither
// read nor written here yet.

[[vk::binding(0, 0)]] StructuredBuffer<uint, Std430DataLayout> histBins; // 256 surface + 256 sky bins
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
    float meteringSkyScale; // S2: frame-global multiplier applied to the separate sky histogram
    float meteringSkyFrac;  // S2: final weighted sky fraction after the cap
    float curveCompensation; // S3: compensation EV evaluated at evScene
    float effectiveSlope;    // S3: 1 - d(compensation)/d(evScene)
};
[[vk::binding(2, 0)]] RWStructuredBuffer<ExposureState, Std430DataLayout> stateBuf;

struct Push {
    float key;
    float minEv;
    float maxEv;
    float adaptUp;
    float adaptDown;
    float frameTimeSeconds;
    float evBias;
    float lowPercentile;
    float highPercentile;
    float skyWeightCap;
    float curveScene0;
    float curveCompensation0;
    float curveScene1;
    float curveCompensation1;
    float curveScene2;
    float curveCompensation2;
    float curveScene3;
    float curveCompensation3;
};
[[vk::push_constant]] Push pc;

static const float LOG_MIN = -12.0;
static const float LOG_MAX = 12.0;
static const float LOG_STEP = (LOG_MAX - LOG_MIN) / 256.0;

float binCenter(uint bin) {
    return LOG_MIN + (float(bin) + 0.5) * LOG_STEP;
}

float curveSegment(float x, float x0, float y0, float x1, float y1) {
    return lerp(y0, y1, (x - x0) / (x1 - x0));
}

float curveCompensation(float sceneEv) {
    if (sceneEv <= pc.curveScene0) return pc.curveCompensation0;
    if (sceneEv < pc.curveScene1) {
        return curveSegment(sceneEv, pc.curveScene0, pc.curveCompensation0,
                pc.curveScene1, pc.curveCompensation1);
    }
    if (sceneEv < pc.curveScene2) {
        return curveSegment(sceneEv, pc.curveScene1, pc.curveCompensation1,
                pc.curveScene2, pc.curveCompensation2);
    }
    if (sceneEv < pc.curveScene3) {
        return curveSegment(sceneEv, pc.curveScene2, pc.curveCompensation2,
                pc.curveScene3, pc.curveCompensation3);
    }
    return pc.curveCompensation3;
}

float curveEffectiveSlope(float sceneEv) {
    if (sceneEv <= pc.curveScene0 || sceneEv >= pc.curveScene3) return 1.0;
    float compensationSlope;
    if (sceneEv < pc.curveScene1) {
        compensationSlope = (pc.curveCompensation1 - pc.curveCompensation0)
                / (pc.curveScene1 - pc.curveScene0);
    } else if (sceneEv < pc.curveScene2) {
        compensationSlope = (pc.curveCompensation2 - pc.curveCompensation1)
                / (pc.curveScene2 - pc.curveScene1);
    } else {
        compensationSlope = (pc.curveCompensation3 - pc.curveCompensation2)
                / (pc.curveScene3 - pc.curveScene2);
    }
    return 1.0 - compensationSlope;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
    float surfacePopulation = 0.0;
    float skyPopulation = 0.0;
    for (uint i = 0u; i < 256u; ++i) {
        surfacePopulation += float(histBins[i]);
        skyPopulation += float(histBins[256u + i]);
    }
    float skyCap = clamp(pc.skyWeightCap, 0.0, 1.0);
    float skyScale = 1.0;
    if (surfacePopulation > 0.0 && skyPopulation > 0.0 && skyCap < 1.0) {
        skyScale = skyCap <= 0.0 ? 0.0
                : min(1.0, (skyCap * surfacePopulation)
                        / max((1.0 - skyCap) * skyPopulation, 1.0e-6));
    }
    float population = surfacePopulation + skyPopulation * skyScale;
    float total = max(population, 1.0);
    float lowPercentile = clamp(min(pc.lowPercentile, pc.highPercentile), 0.0, 1.0);
    float highPercentile = clamp(max(pc.lowPercentile, pc.highPercentile), 0.0, 1.0);
    float lowCount = min(floor(total * lowPercentile), total - 1.0);
    float highCount = clamp(ceil(total * highPercentile), lowCount + 1.0, total);

    float cumulative = 0.0;
    float weightedLogLum = 0.0;
    float weightedCount = 0.0;
    for (uint i = 0u; i < 256u; ++i) {
        float count = float(histBins[i]) + float(histBins[256u + i]) * skyScale;
        float start = cumulative;
        float end = cumulative + count;
        float takeStart = max(start, lowCount);
        float takeEnd = min(end, highCount);
        if (takeEnd > takeStart) {
            float take = takeEnd - takeStart;
            weightedLogLum += binCenter(i) * take;
            weightedCount += take;
        }
        cumulative = end;
    }

    float avgLogLum = weightedCount > 0.0 ? weightedLogLum / weightedCount : 0.0;
    float compensation = curveCompensation(avgLogLum);
    float effectiveSlope = curveEffectiveSlope(avgLogLum);
    float target = pc.key * exp2(pc.evBias - avgLogLum + compensation);
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
    stateBuf[0].clipLowFrac = (float(histBins[0]) + float(histBins[256]) * skyScale) / total;
    stateBuf[0].clipHighFrac = (float(histBins[255]) + float(histBins[511]) * skyScale) / total;
    stateBuf[0].meteringSkyScale = skyScale;
    stateBuf[0].meteringSkyFrac = skyPopulation * skyScale / total;
    stateBuf[0].curveCompensation = compensation;
    stateBuf[0].effectiveSlope = effectiveSlope;
    exposureImage[int2(0, 0)] = exposure;
}
