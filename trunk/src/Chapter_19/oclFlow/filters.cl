/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
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
// Simple filtering routines using textures in OpenCL. 
// downfilter_x_g and downfilter_y_g are smoothing filters
// 1D x/y direction (use both as a separable 2D implementation).
// Similarly filter_3x1_g and filter_1x3_g are 1D filters
// with configurable weights usable as 2 passes for a 
// separable 3x3 filter.  
// The filter_G function generates the "G" matrix used in optical flow. 

// launched over downsampled area
// first pass sampling from larger level, so x2 the coordinates
__kernel void downfilter_x_g( 
    __read_only image2d_t src,
    __global uchar *dst, int dst_w, int dst_h )
{

    sampler_t srcSampler = CLK_NORMALIZED_COORDS_FALSE | 
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST ;

    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    float x0 = read_imageui( src, srcSampler, (int2)(ix-2, iy ) ).x/16.0f;
    float x1 = read_imageui( src, srcSampler, (int2)(ix-1, iy ) ).x/4.0f;
    float x2 = (3*read_imageui( src, srcSampler, (int2)(ix, iy )).x)/8.0f;
    float x3 = read_imageui( src, srcSampler, (int2)(ix+1, iy ) ).x/4.0f;
    float x4 = read_imageui( src, srcSampler, (int2)(ix+2, iy ) ).x/16.0f;

    int output = round( x0 + x1 + x2 + x3 + x4 );

    if( ix < dst_w && iy < dst_h ) {
        dst[iy*dst_w + ix ] = (uchar)output;  // uncoalesced when writing to memory object
    }
}

// Simultaneously does a Y smoothing filter and downsampling (i.e. only does filter at 
// downsampled points.  Writes to the next smaller pyramid level whose max dimensions are
// given by dst_w/dst_h
__kernel void downfilter_y_g(
    __read_only image2d_t src,
    __global uchar *dst, int dst_w, int dst_h )
{
    sampler_t srcSampler = CLK_NORMALIZED_COORDS_FALSE | 
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST ;

    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    float x0 = read_imageui( src, srcSampler, (int2)(2*ix, 2*iy -2 ) ).x/16.0f;
    float x1 = read_imageui( src, srcSampler, (int2)(2*ix, 2*iy -1 ) ).x/4.0f;
    float x2 = (3*read_imageui( src, srcSampler, (int2)(2*ix, 2*iy ) ).x)/8.0f;
    float x3 = read_imageui( src, srcSampler, (int2)(2*ix, 2*iy +1) ).x/4.0f;
    float x4 = read_imageui( src, srcSampler, (int2)(2*ix, 2*iy +2) ).x/16.0f;

    int output = round(x0 + x1 + x2 + x3 + x4);

    if( ix < dst_w-2 && iy < dst_h-2 ) {
        dst[iy*dst_w + ix ] = (uchar)output;
    }
 
}

// signed int16 output
//mac: send in int wieghts, cannot use vector type?
__kernel void filter_3x1_g( 
    __read_only image2d_t src, 
    __global short *dst,int dst_w, int dst_h, int W0, int W1, int W2
)
{
    sampler_t srcSampler = CLK_NORMALIZED_COORDS_FALSE | 
                            CLK_ADDRESS_CLAMP_TO_EDGE | 
                            CLK_FILTER_NEAREST ;
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    float x0 = read_imagei( src, srcSampler, (int2)(ix-1, iy)).x * W0;
    float x1 = read_imagei( src, srcSampler, (int2)(ix, iy  )).x * W1;
    float x2 = read_imagei( src, srcSampler, (int2)(ix+1, iy)).x * W2;

    int output = round( x0 + x1 + x2 ); 

    if( ix < dst_w && iy < dst_h ) {
        dst[iy*dst_w + ix ] = (short)output;
    }

}


// signed int16 output
__kernel void filter_1x3_g( __read_only image2d_t src, __global short *dst, int dst_w, int dst_h, int W0, int W1, int W2 )
{
    sampler_t srcSampler = CLK_NORMALIZED_COORDS_FALSE | 
                            CLK_ADDRESS_CLAMP_TO_EDGE | 
                            CLK_FILTER_NEAREST ;
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

	if( ix > 0 && iy > 0 && ix < dst_w-1 && iy < dst_h-1 ) {
		float x0 = read_imagei( src, srcSampler, (float2)(ix, iy-1)).x * W0;
		float x1 = read_imagei( src, srcSampler, (float2)(ix, iy  )).x * W1;
		float x2 = read_imagei( src, srcSampler, (float2)(ix, iy+1)).x * W2;

		int output = round( x0 + x1 + x2 );
		dst[iy*dst_w + ix ] = (short)output;
	}
}



#define FRAD 4
// G is int4 output
// This kernel generates the "G" matrix (2x2 covariance matrix on the derivatives)
// Each thread does one pixel, sampling its neighbourhood of +/- FRAD radius 
// and generates the G matrix entries.
__kernel void filter_G( __read_only image2d_t Ix, 
                        __read_only image2d_t Iy,
                        __global int4 *G, int dst_w, int dst_h )
{
   sampler_t srcSampler = CLK_NORMALIZED_COORDS_FALSE | 
                           CLK_ADDRESS_CLAMP_TO_EDGE | 
                           CLK_FILTER_NEAREST ;
   const int idx = get_global_id(0);
   const int idy = get_global_id(1); 

   int Ix2 = 0;
   int IxIy = 0;
   int Iy2 = 0;
   for( int j=-FRAD ; j <= FRAD; j++ ) {
    for( int i=-FRAD ; i<= FRAD ; i++ ) { 
      int ix = read_imagei( Ix, srcSampler, (int2)(idx+i, idy+j) ).x;
      int iy = read_imagei( Iy, srcSampler, (int2)(idx+i, idy+j) ).x;

      Ix2 += ix*ix;
      Iy2 += iy*iy;
      IxIy += ix*iy;
        
    }
   }
   int4 G2x2 = (int4)( Ix2, IxIy, IxIy, Iy2 );
   if( idx < dst_w && idy < dst_h ) {
        G[ idy * dst_w + idx ] = G2x2;
   }
   
}

__kernel void convertToRGBAFloat( __read_only image2d_t src, __global float4 *dst, int w, int h)
{
    sampler_t srcSampler = CLK_NORMALIZED_COORDS_FALSE | 
                            CLK_ADDRESS_CLAMP_TO_EDGE | 
                            CLK_FILTER_NEAREST ;
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);

    if( ix < w && iy < h ) {
        uint4 pix = read_imageui( src, srcSampler, (float2)(ix, iy));
        float4 fpix;
        fpix.x = pix.x;
        fpix.y = pix.y;
        fpix.z = pix.z;
        fpix.w = pix.w;
        dst[iy*w + ix ] = fpix;
    }
}
