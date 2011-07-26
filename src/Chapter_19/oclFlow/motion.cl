/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */
// This function calculates the endpoint (where the point moves) given
// an initial position (p) and a motion vector (v).  The buffer p then
// holds 2 ordered pairs [ (x,y),(x,y) ] in its 4 float elements 
// corresponding to the start and end points.  Later, this buffer
// can be emitted as a vertex array for use by OpenGL to draw the motion
// lines for visualization. Every 10th is
// used to avoid the visualization being too cluttered. 
// Overly large estimated motions (>20) are culled. 
__kernel void motion( 
    __global float4 *p,
    __global float2 *v,
    int w, int h )
{
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    if( ix < w && iy < h ) {
		float4 startp =  (float4)( ix, iy, ix, iy);
        float2 motion = v[iy*w + ix] ;
        float4 endp = (float4)( 
            startp.x,
            startp.y,
            startp.x + motion.x , 
            startp.y + motion.y );
        if( ix % 10 == 0 && iy % 10 == 0 && fabs(motion.x) < 20 && fabs(motion.y) < 20) 
        p[iy*w + ix ] = (float4)endp;
        else 
        p[iy*w + ix ] = (float4)(0,0,0,0);
    }
}
