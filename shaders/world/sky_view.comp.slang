// Sky-view LUT bake, once per frame. Renders the whole sky dome — atmosphere in-scatter with real
// multiple scattering — into a 192x108 latlong-ish table per celestial body, in the light-relative
// parameterisation from sky.slang. world.rmiss then answers any miss with two texture fetches instead of
// re-marching the atmosphere per ray, which previously cost 16x8 exp-heavy steps on EVERY GI-bounce miss,
// not just on primary rays.
//
// One slice per body (sun rows first, then moon), each marched with its own direction and illuminance.
// Adding two independently correct slices is what removes the old sun-to-moon crossfade: there is no
// moment where the sky is a blend of two different lighting solutions, so there is no seam to see.
//
// Reads the same WorldPush BDA slot the trace does, so the LUT and the frame it shades can never disagree
// about where the sun is.
import world_common;
import sky;

[[vk::binding(0, 0)]] [format("rgba16f")] RWTexture2D<float4> transmittanceImage;
[[vk::binding(1, 0)]] [format("rgba16f")] RWTexture2D<float4> multiScatterImage;
[[vk::binding(2, 0)]] [format("rgba16f")] RWTexture2D<float4> skyViewImage;
[[vk::binding(3, 0)]] Sampler2D transmittanceLut;
[[vk::binding(4, 0)]] Sampler2D multiScatterLut;

[[vk::push_constant]] PushAddr pcAddr;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= uint(SKY_VIEW_LUT_W) || tid.y >= uint(SKY_VIEW_LUT_TOTAL_H)) {
        return;
    }
    WorldPush push = ConstPtr<WorldPush>(pcAddr.worldPushAddr)[0];
    SkyState sky = skyState(push);

    int body = int(tid.y) / SKY_VIEW_LUT_H;
    int row = int(tid.y) - body * SKY_VIEW_LUT_H;
    bool isSun = body == SKY_VIEW_BODY_SUN;
    float3 worldLightDir = isSun ? sky.sunDir : sky.moonDir;
    float3 illuminance = float3(isSun ? sky.sunIlluminance : sky.moonIlluminance);
    float angularRadius = isSun ? sky.sunAngularRadius : sky.moonAngularRadius;

    // Sampling applies the half-texel correction; the bake must invert exactly that, so it works in the
    // same sub-UV space rather than in the parameterisation's raw [0,1].
    float2 uv = (float2(float(tid.x), float(row)) + 0.5)
              / float2(float(SKY_VIEW_LUT_W), float(SKY_VIEW_LUT_H));
    float viewZenithCos;
    float lightViewCos;
    skyViewLutParams(uv, sky.viewerRadiusKm, viewZenithCos, lightViewCos);

    float3 viewDir;
    float3 lightDir;
    skyViewLocalDirections(worldLightDir.y, viewZenithCos, lightViewCos, viewDir, lightDir);

    float3 origin = float3(0.0, sky.viewerRadiusKm, 0.0);
    ScatteringResult result = integrateScatteredLuminance(
            origin, viewDir, lightDir, illuminance, angularRadius,
            SKY_VIEW_STEPS, true, true,
            transmittanceLut, multiScatterLut, sky.groundAlbedo);

    skyViewImage[tid.xy] = float4(max(result.luminance, float3(0.0)), 1.0);
}
