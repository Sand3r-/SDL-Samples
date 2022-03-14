struct VSIn
{
    uint VertexId : SV_VertexID;
};

struct VertexShaderOutput
{
	float4 Color    : COLOR;
    float4 Position : SV_Position;
};

static const float2 trinagleData[3] = {
    float2(0.0f, 0.5f),
    float2(0.5f, -0.5f),
    float2(-0.5f, -0.5f)
};

VertexShaderOutput main(VSIn IN)
{
    VertexShaderOutput OUT;

    OUT.Position = float4(trinagleData[IN.VertexId], 0.0, 1.0f);
    OUT.Color = float4(1.0f, 0.0f, 0.0f, 1.0f);

    return OUT;
}