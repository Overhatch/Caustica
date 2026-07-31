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

[[vk::binding(0, 0)]] public StructuredBuffer<uint, Std430DataLayout> histBins;
[[vk::binding(1, 0)]] [format("r32f")] public RWTexture2D<float> exposureImage;
[[vk::binding(2, 0)]] public RWStructuredBuffer<ExposureState, Std430DataLayout> stateBuf;
