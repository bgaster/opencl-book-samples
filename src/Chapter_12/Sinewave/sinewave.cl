//
// Copyright (c) 2009 Advanced Micro Devices, Inc. All rights reserved.

///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in sine wave pattern
//! @param data  data in global memory
///////////////////////////////////////////////////////////////////////////////
__kernel
void simpleGL(
    __global float4 * pos,
    unsigned int width,
    unsigned int height,
    float time)
{
    unsigned int x = get_group_id(0) * get_local_size(0) + get_local_id(0);
    unsigned int y = get_group_id(1) * get_local_size(1) + get_local_id(1);

    // calculate uv coordinates
    float u = x / (float) width;
    float v = y / (float) height;
    u = u*2.0f - 1.0f;
    v = v*2.0f - 1.0f;

    // calculate simple sine wave pattern
    float freq = 4.0f;
    float w = sin(u*freq + time) * cos(v*freq + time) * 0.5f;

    // write output vertex
    pos[y*width+x] = (float4)(u, w, v, 1.0f);
}
