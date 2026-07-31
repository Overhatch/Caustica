public struct ExposureState {
    public float previous;
    public uint initialized;
    public float evScene;
    public float evTarget;
    public float evApplied;
    public float clipLowFrac;
    public float clipHighFrac;
    public uint resetSeq;
    public float evHistory[8];
    public float meteringSkyScale;
    public float meteringSkyFrac;
    public float curveCompensation;
    public float effectiveSlope;
    public float meteringEmissiveScale;
    public float meteringEmissiveFrac;
};

[[vk::binding(0, 0)]] [format("rgba8")] public RWTexture2D<float4> outputImage;
[[vk::binding(1, 0)]] [format("rgba16f")] public RWTexture2D<float4> gNormal;
[[vk::binding(2, 0)]] [format("rgba16f")] public RWTexture2D<float4> gAlbedo;
[[vk::binding(3, 0)]] [format("r32f")] public RWTexture2D<float> gDepth;
[[vk::binding(4, 0)]] [format("rg16f")] public RWTexture2D<float2> gMotion;
[[vk::binding(5, 0)]] [format("rgba16f")] public RWTexture2D<float4> gSpecAlbedo;
[[vk::binding(6, 0)]] [format("rg16f")] public RWTexture2D<float2> gSpecMotion;
[[vk::binding(7, 0)]] [format("rgba16f")] public RWTexture2D<float4> sceneImage;
[[vk::binding(8, 0)]] [format("r32f")] public RWTexture2D<float> exposureImage;
[[vk::binding(9, 0)]] public StructuredBuffer<ExposureState, Std430DataLayout> exposureState;
