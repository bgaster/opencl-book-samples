Texture2D txDiffuse;
SamplerState samLinear
{
    Filter = MIN_MAG_MIP_LINEAR;
    AddressU = Wrap;
    AddressV = Wrap;
};

struct VS_INPUT
{
    float4 Pos : POSITION;
    float2 Tex : TEXCOORD;
};

struct VS_Simple_INPUT
{
    float4 Pos : POSITION;
};

struct PS_INPUT
{
    float4 Pos : SV_POSITION;
    float2 Tex : TEXCOORD0;
};

struct PS_Simple_INPUT
{
    float4 Pos : SV_POSITION;
};

//
// Vertex Shader
//
PS_INPUT VS( VS_INPUT input )
{
    PS_INPUT output = (PS_INPUT)0;
    output.Pos = input.Pos;
    output.Tex = input.Tex;
        
    return output;
}

PS_Simple_INPUT VS_simple( VS_Simple_INPUT input ) 
{
    PS_Simple_INPUT output = (PS_Simple_INPUT)0;
    output.Pos = input.Pos;
    
    return output;
}

//
// Pixel Shader
//
float4 PS( PS_INPUT input) : SV_Target
{
    return txDiffuse.Sample( samLinear, input.Tex );
}

float4 PS_simple( PS_Simple_INPUT input ) : SV_Target
{
    return float4(1,1,1,1);
}


technique10 Render
{
    pass P0
    {
        SetVertexShader( CompileShader( vs_4_0, VS() ) );
        SetGeometryShader( NULL );
        SetPixelShader( CompileShader( ps_4_0, PS() ) );
    }
    pass P1
    {
	SetVertexShader( CompileShader( vs_4_0, VS_simple() ) );
	SetGeometryShader( NULL );
	SetPixelShader( CompileShader( ps_4_0, PS_simple() ) );
    }
}
