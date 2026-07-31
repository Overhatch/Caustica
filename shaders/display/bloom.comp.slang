// Scene-referred bloom prefilter and one separable Gaussian blur. The working images are half
// display resolution: extraction averages a 2x2 footprint from the exposed ACEScg DLSS output.
// Half resolution and filtered fractional taps avoid exposing a coarse downsample lattice.
import display_common;

[[vk::binding(0, 0)]] [format("rgba16f")] RWTexture2D<float4> rtImage;
[[vk::binding(1, 0)]] [format("r32f")]    RWTexture2D<float>  exposureImage;
[[vk::binding(2, 0)]] [format("rgba16f")] RWTexture2D<float4> bloomA;
[[vk::binding(3, 0)]] [format("rgba16f")] RWTexture2D<float4> bloomB;
[[vk::binding(4, 0)]] Sampler2D bloomAFiltered;
[[vk::binding(5, 0)]] Sampler2D bloomBFiltered;

[[vk::push_constant]] BloomPush pc;

static const float3 ACESCG_LUMA = float3(0.2722287168, 0.6740817658, 0.0536895174);
static const float HALF_MAX = 65504.0;

int2 clampedPixel(int2 pixel, uint width, uint height) {
    return clamp(pixel, int2(0, 0), int2(int(width) - 1, int(height) - 1));
}

float3 prefilter(float3 color) {
    color = clamp(color, float3(0.0), float3(HALF_MAX));
    float luminance = dot(color, ACESCG_LUMA);
    float knee = max(pc.softKnee, 1.0e-6);
    float soft = clamp(luminance - pc.threshold + knee, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee);
    float contribution = max(luminance - pc.threshold, soft);
    return color * (contribution / max(luminance, 1.0e-6));
}

float2 bloomUv(float2 pixel, uint width, uint height) {
    return (pixel + 0.5) / float2(width, height);
}

float3 sampleA(float2 pixel, uint width, uint height) {
    return bloomAFiltered.SampleLevel(bloomUv(pixel, width, height), 0.0).rgb;
}

float3 sampleB(float2 pixel, uint width, uint height) {
    return bloomBFiltered.SampleLevel(bloomUv(pixel, width, height), 0.0).rgb;
}

float3 blurA(int2 pixel, float2 axis, uint width, uint height) {
    float offset = max(pc.radius, 0.25);
    float2 center = float2(pixel);
    float3 result = sampleA(center, width, height) * 0.2270270270;
    result += (sampleA(center + axis * offset, width, height)
             + sampleA(center - axis * offset, width, height)) * 0.1945945946;
    result += (sampleA(center + axis * (2.0 * offset), width, height)
             + sampleA(center - axis * (2.0 * offset), width, height)) * 0.1216216216;
    result += (sampleA(center + axis * (3.0 * offset), width, height)
             + sampleA(center - axis * (3.0 * offset), width, height)) * 0.0540540541;
    result += (sampleA(center + axis * (4.0 * offset), width, height)
             + sampleA(center - axis * (4.0 * offset), width, height)) * 0.0162162162;
    return result;
}

float3 blurB(int2 pixel, float2 axis, uint width, uint height) {
    float offset = max(pc.radius, 0.25);
    float2 center = float2(pixel);
    float3 result = sampleB(center, width, height) * 0.2270270270;
    result += (sampleB(center + axis * offset, width, height)
             + sampleB(center - axis * offset, width, height)) * 0.1945945946;
    result += (sampleB(center + axis * (2.0 * offset), width, height)
             + sampleB(center - axis * (2.0 * offset), width, height)) * 0.1216216216;
    result += (sampleB(center + axis * (3.0 * offset), width, height)
             + sampleB(center - axis * (3.0 * offset), width, height)) * 0.0540540541;
    result += (sampleB(center + axis * (4.0 * offset), width, height)
             + sampleB(center - axis * (4.0 * offset), width, height)) * 0.0162162162;
    return result;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchId : SV_DispatchThreadID) {
    int2 pixel = int2(dispatchId.xy);
    uint width, height;
    bloomA.GetDimensions(width, height);
    if (pixel.x >= int(width) || pixel.y >= int(height)) {
        return;
    }

    if (pc.mode == 0) {
        uint sourceWidth, sourceHeight;
        rtImage.GetDimensions(sourceWidth, sourceHeight);
        float exposure = max(exposureImage[int2(0, 0)], 0.0);
        float3 sum = float3(0.0);
        int2 sourceBase = pixel * 2;
        for (int y = 0; y < 2; y++) {
            for (int x = 0; x < 2; x++) {
                int2 sourcePixel = clampedPixel(sourceBase + int2(x, y), sourceWidth, sourceHeight);
                sum += prefilter(max(rtImage[sourcePixel].rgb * exposure, float3(0.0)));
            }
        }
        bloomA[pixel] = float4(sum * 0.25, 1.0);
        return;
    }

    if (pc.mode == 1) {
        bloomB[pixel] = float4(blurA(pixel, float2(1.0, 0.0), width, height), 1.0);
    } else {
        bloomA[pixel] = float4(blurB(pixel, float2(0.0, 1.0), width, height), 1.0);
    }
}
