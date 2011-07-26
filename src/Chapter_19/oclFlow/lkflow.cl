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
// Peform the Lucas-Kanade iterations for pyramidal optical flow.
// Each thread performs operations for a single pixel, and 
// iterates multiple times.  This function is launched for each
// pyramid level and each pyarmid level beyond the first uses
// the guess from the previous pyramid level (guess_in) and produces
// a guess for the next pyramid level (guess_out) if there is another level. 
// Note: This heavily uses textures and floating point interpolation when
//  subsampling on  the "J" image. 
// Variables:
// I: first image of the pair
// J: second image of the pair
// Ix, Iy: x & y derivatives of the I image
// G: Each element of G contains the elements of the 2x2 matrix G calculated
//    at each point (in filters.cl::filter_G())
// guess_in, guess_out: the previous level's guess and this level's guess to
//    be used by the next level
// w,h: image dimensions
// use_guess: flag whether or not to use the guess_in to seed motion, 
//   (set false for example at the top pyramid level where no previous
//    guess exists).

#define FRAD 4
#define eps 0.0000001f;
#define LOCAL_X 16
#define LOCAL_Y 8

// at 16x8, GTX460
// 20.5 ms all smem
// 31.5 ms 2 smem, 1 tx
// 41.5 ms 1 smem, 2 tx
// 55.8 ms 3 tx

// GTX580, same as above
// 24.8 ms 0 smem 3 tx
// 17.6 ms 1 smem 2 tx
// 13.5 ms 2 smem 1 tx
// 9.2 ms 3 tx

__kernel void lkflow( 
    __read_only image2d_t I,
    __read_only image2d_t Ix,
    __read_only image2d_t Iy,
    __read_only image2d_t G,
    __read_only image2d_t J_float,
    __global float2 *guess_in,
    int guess_in_w,
    __global float2 *guess_out,
    int guess_out_w,
    int guess_out_h,
    int use_guess )
{
	// declare some shared memory
	__local int smem[2*FRAD + LOCAL_Y][2*FRAD + LOCAL_X] ;
	__local int smemIy[2*FRAD + LOCAL_Y][2*FRAD + LOCAL_X] ;
	__local int smemIx[2*FRAD + LOCAL_Y][2*FRAD + LOCAL_X] ;

	// Create sampler objects.  One is for nearest neighbour, the other fo
	// bilinear interpolation
    sampler_t bilinSampler = CLK_NORMALIZED_COORDS_FALSE |
                           CLK_ADDRESS_CLAMP_TO_EDGE |
                           CLK_FILTER_LINEAR ;
    sampler_t nnSampler = CLK_NORMALIZED_COORDS_FALSE |
                           CLK_ADDRESS_CLAMP_TO_EDGE |
                           CLK_FILTER_NEAREST ;

	// Image indices. Note for the texture, we offset by 0.5 to use the centre
	// of the texel. 
    int2 iIidx = { get_global_id(0), get_global_id(1)};
    float2 Iidx = { get_global_id(0)+0.5, get_global_id(1)+0.5 };

	// load some data into local memory because it will be re-used frequently
	// load upper left region of smem
	int2 tIdx = { get_local_id(0), get_local_id(1) };
	smem[ tIdx.y ][ tIdx.x ] = read_imageui( I, nnSampler, Iidx+(float2)(-FRAD,-FRAD) ).x;
	smemIy[ tIdx.y ][ tIdx.x ] = read_imageui( Iy, nnSampler, Iidx+(float2)(-FRAD,-FRAD) ).x;
	smemIx[ tIdx.y ][ tIdx.x ] = read_imageui( Ix, nnSampler, Iidx+(float2)(-FRAD,-FRAD) ).x;

	// upper right
	if( tIdx.x < 2*FRAD ) { 
		smem[ tIdx.y ][ tIdx.x + LOCAL_X ] = read_imageui( I, nnSampler, Iidx+(float2)(LOCAL_X - FRAD,-FRAD) ).x;
		smemIy[ tIdx.y ][ tIdx.x + LOCAL_X ] = read_imageui( Iy, nnSampler, Iidx+(float2)(LOCAL_X - FRAD,-FRAD) ).x;
		smemIx[ tIdx.y ][ tIdx.x + LOCAL_X ] = read_imageui( Ix, nnSampler, Iidx+(float2)(LOCAL_X - FRAD,-FRAD) ).x;

	}
	// lower left
	if( tIdx.y < 2*FRAD ) {
		smem[ tIdx.y + LOCAL_Y ][ tIdx.x ] = read_imageui( I, nnSampler, Iidx+(float2)(-FRAD, LOCAL_Y-FRAD) ).x;
		smemIy[ tIdx.y + LOCAL_Y ][ tIdx.x ] = read_imageui( Iy, nnSampler, Iidx+(float2)(-FRAD, LOCAL_Y-FRAD) ).x;
		smemIx[ tIdx.y + LOCAL_Y ][ tIdx.x ] = read_imageui( Ix, nnSampler, Iidx+(float2)(-FRAD, LOCAL_Y-FRAD) ).x;

	}
	// lower right
	if( tIdx.x < 2*FRAD && tIdx.y < 2*FRAD ) {
		smem[ tIdx.y + LOCAL_Y ][ tIdx.x + LOCAL_X ] = read_imageui( I, nnSampler, Iidx+(float2)(LOCAL_X - FRAD, LOCAL_Y - FRAD) ).x;
		smemIy[ tIdx.y + LOCAL_Y ][ tIdx.x + LOCAL_X ] = read_imageui( Iy, nnSampler, Iidx+(float2)(LOCAL_X - FRAD, LOCAL_Y - FRAD) ).x;
		smemIx[ tIdx.y + LOCAL_Y ][ tIdx.x + LOCAL_X ] = read_imageui( Ix, nnSampler, Iidx+(float2)(LOCAL_X - FRAD, LOCAL_Y - FRAD) ).x;

	}
	barrier(CLK_LOCAL_MEM_FENCE);
    float2 g = {0,0}; 

	// Previous pyramid levels provide input guess.  Use if available.
    if( use_guess != 0 ) {
        //lookup in higher level, div by two to find position because its smaller
        int gin_x = iIidx.x/2;
        int gin_y = iIidx.y/2;
        float2 g_in = guess_in[gin_y * guess_in_w + gin_x ];
		// multiply the motion by two because we are in a larger level. 
        g.x = g_in.x*2;
        g.y = g_in.y*2;
    }
    float2 v = {0,0};
    
	// invert G, 2x2 matrix , use float since int32 will overflow quickly
	int4 Gmat = read_imagei( G, nnSampler, iIidx );
    float det_G = (float)Gmat.s0 * (float)Gmat.s3 - (float)Gmat.s1 * (float)Gmat.s2 ;
	// avoid possible 0 in denominator
	if( det_G == 0.0f ) det_G = eps;
    float4 Ginv = { Gmat.s3/det_G, -Gmat.s1/det_G, -Gmat.s2/det_G, Gmat.s0/det_G };

	// for large motions we can approximate them faster by applying gain to the motion
    float gain = 4.f;
    for( int k=0 ; k <8 ; k++ ) {
        float2 Jidx = { Iidx.x + g.x + v.x, Iidx.y + g.y + v.y };
        float2 b = {0,0};
        float2 n = {0,0};

        // calculate the mismatch vector
        for( int j=-FRAD ; j <= FRAD ; j++ ) {
            for( int i=-FRAD ; i<= FRAD ; i++ ) {
				// this should use shared memory instead...
                //int Isample = read_imageui( I, nnSampler, Iidx+(float2)(i,j) ).x;
				int Isample = smem[tIdx.y + FRAD +j][tIdx.x + FRAD+ i];

                float Jsample = read_imagef( J_float,bilinSampler, Jidx+(float2)(i,j) ).x;
                float dIk = (float)Isample - Jsample;
				int ix,iy;
				ix = smemIx[tIdx.y + FRAD +j][tIdx.x + FRAD+ i];
				iy = smemIy[tIdx.y + FRAD +j][tIdx.x + FRAD+ i];

                //ix = read_imagei( Ix, nnSampler, Iidx+(float2)(i,j)  ).x; 
				//iy = read_imagei( Iy, nnSampler, Iidx+(float2)(i,j)  ).x; 

                b += (float2)( dIk*ix*gain, dIk*iy*gain );
            }
        }

        // Optical flow (Lucas-Kanade).  
        //  Solve n = G^-1 * b
        //compute n (update), mult Ginv matrix by vector b
        n = (float2)(Ginv.s0*b.s0 + Ginv.s1*b.s1,  Ginv.s2*b.s0 + Ginv.s3*b.s1);

		// if the determinant is not plausible, suppress motion at this pixel
        if( fabs(det_G)<1000) n = (float2)(0,0);
		// break if no motion
		// on test images this changes from 74 ms if no break, 55 if break, on minicooper, k=8, FRAD=4, gain=4
        if( length(n) < 0.004  ) break;

        // guess for next iteration: v_new = v_current + n
        v = v + n;
    }
    int2 outCoords = { get_global_id(0), get_global_id(1) }; 

    if( Iidx.x < guess_out_w && Iidx.y < guess_out_h ) {
        guess_out[ outCoords.y * guess_out_w + outCoords.x ] = (float2)(v.x + g.x, v.y + g.y);
    }
    
}
