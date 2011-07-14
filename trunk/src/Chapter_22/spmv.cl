/*************************************************************************/
/*                                                                       */
/* Licensed Materials - Property of IBM                                  */
/*                                                                       */
/*                                                                       */
/* (C) Copyright IBM Corp. 2010                                          */
/* All Rights Reserved                                                   */
/*                                                                       */
/* US Government Users Restricted Rights - Use, duplication or           */
/* disclosure restricted by GSA ADP Schedule Contract with IBM Corp.     */
/*                                                                       */
/*************************************************************************/
/* ================================================================================================== */
/*                                                                                                    */
/* The "matbuffer" buffer contains the tiled matrix.                                                  */
/* The data in this tiled matrix is organized into "rows of tiles" which are called "slabs."          */
/* The first section of this buffer contains three words of "header information" for each slab of the */
/* matrix.  Each instantiation of the kernel is indexed by get_global_id(0) which matches the         */
/* number of the slab.  The three header words for each slab are:                                     */
/*    "offset": the offset into the buffer where the slab's data begins                               */
/*    "outindex": the index into the output vector where this slab's output is to begin               */
/*    "outspan": the number of elements of the output vector which this slab is responsible for       */
/*                                                                                                    */
/* The actual data in the slab is organized into 16-element "packets", of length 128 bytes.           */
/* (see the definition of the "packet" struct below)                                                  */
/* Each packet contains four "control words" used by the kernels, 16 two-byte indices into            */
/* the input array, and sixteen floating point values from the matrix.                                */
/*                                                                                                    */
/* The four "control words" are:                                                                      */
/*    0: base offset into the input vector for this packet                                            */
/*    1: base offset into the input vector for a FUTURE packet (useful for double buffering)          */
/*    2: the number of packets remaining in this slab                                                 */
/*    3: the offset (relative to the second header word) into the output vector for this packet       */
/*                                                                                                    */
/* These four words are followed by four words of pad, reserved for future use.                       */
/* Next come 16 short integers, containing offsets into the input vector.                             */
/* Next come 16 floating point values, containing the actual matrix data.                             */
/*                                                                                                    */
/* Specific output offsets for each value are not needed, because the packets are created in          */
/* a special format: each value is intended to update the output vector element subsequent to that    */
/* of the previous value.  So, if a packet is targeted to location 512 of the output vector, then     */
/* the sixteen values in the packet will be updating locations 512 through 527 of the output vector   */
/* respectively, and in order.  This adds to the complexity of the code which creates these packets   */
/* but results in significant performance payoffs, when performing the multiplications.               */
/*                                                                                                    */
/* It's frequently the case that there is empty data in these packets, because of this construction.  */
/* This data is carefully set up so that when we are dealing with local buffers, the garbage          */
/* calculations go into an area which never gets written back to main memory.  In the case where      */
/* global memory is accessed directly, the "matrix data" in the empty values is set to zero, so that  */
/* regardless of what the input is, the output is unaffected.                                         */
/*                                                                                                    */
/* For additional information with regards to this implementation and its data format, please         */
/* read the white paper titled "Tiled and Packetized SpMV using OpenCL" publish in the "OpenCL Lounge"*/
/* developerWorks group. See https://www.ibm.com/developerworks/mydeveloperworks/groups               */
/* ================================================================================================== */

/* These two structures are defined both in spmv.c and spmv.cl (using different variable types). */
/* If you change something here, change it in the other file as well. */
typedef struct _slab_header {
   uint offset;
   uint outindex;
   uint outspan;
} slab_header;

typedef struct _packet {
   uint seg_input_offset;
   uint future_seg_input_offset;
   uint npackets_remaining;
   uint seg_output_offset;
   uint pad1;
   uint pad2;
   uint pad3;
   uint pad4;
   ushort input_offset_short[16];
   union {
      float8 matdataV8[2];
      float matdata[16];
   } uf;
} packet;

/* ================================================================================================================= */
/* Kernel using basic load/store mechanisms and local vars. This version is optimized for the GPU and CPU devices    */
/* ================================================================================================================= */

__kernel void tiled_spmv_kernel_LS(__global float *input,         /* pointer to input memory object in global memory */
                                   __global float *output,        /* pointer to output memory object in global memory */
                                   __global uint *matbuffer,      /* pointer to tiled matrix memory object in global memory */
                                   __private uint column_span,    /* size of fixed chunks of the input vector */
                                   __private uint slabspace,      /* size of the variable chunk of output vector to be computed */
                                   __private uint team_size,      /* size of each "team" of local work units */
                                   __private uint num_header_packets,
                                   __local float *outputspace)    /* local buffer to hold computed output, to be written out at the end */
{
   uint i, gunit, lunit, start, span, npackets, teamnum, n_teams, outindex, outspan; 
   __global slab_header *headptr;
   __global float *work_input;
   __global packet *gsegptr;      /* This is a "global pointer."  Compare to variable in other kernel called "lsegptr." */
   __global packet *gsegptr_stop; /* Computed to hold the address of the end of the work for this work unit.            */
   __global float *outptr;
   __local float *outptr16;

   /* The local workgroup is interpreted as a set of "teams," each consisting of 1 or 16 work units. */
   /* This construction is frequently very useful on the GPU device.                                 */

   headptr = ((__global slab_header *) matbuffer) + get_global_id(1);
   outspan = headptr->outspan;
   outindex = headptr->outindex;
   n_teams = get_local_size(0)/team_size;  /* number of teams */
   gunit = get_local_id(0);                 /* unique identifier for this work unit in this work group */
   teamnum = gunit/team_size;               /* which team is this? */
   start = get_global_id(0);               /* where in the packets is my "first word"? */
   span = get_global_size(0);              /* what stride shall I use when clearing or transmitting output buffers? */

   /* Zero out the output buffer */
   /* Each team has its own separate output buffer.  At the end, these are accumulated. */
   for (i = start; i < slabspace; i += span) {
#ifdef DOUBLE
      outputspace[i] = 0.0;     
#else
      outputspace[i] = 0.0f;     
#endif
   }
   barrier(CLK_LOCAL_MEM_FENCE);

   gsegptr = &(((__global packet *) matbuffer)[headptr->offset]); /* Pointer to the start of the packets. */
   outptr = &output[outindex];                                    /* Pointer to pertinent area of output vector. */

   /* We have two clauses here.  The first is optimized for the GPU device, and the second is */
   /* optimized for the CPU device.  The distinction is that in the GPU device the memory     */
   /* access pattern for each work unit is strided, and in the CPU device, the memory access  */
   /* pattern is contiguous. Virtually all processing happens in the selected clause below.   */

   if (team_size == 16) {
      lunit = gunit % team_size; /* Which work unit within the team am I?    */
      __global uint *first_team_offset;
      first_team_offset = (__global uint *) gsegptr;
      int temp_offset, temp_packetcount;
      temp_offset = first_team_offset[teamnum] / 65536;
      temp_packetcount = first_team_offset[teamnum] % 65536;
      gsegptr += num_header_packets + temp_offset;
      for (i=0; i<temp_packetcount; ++i) {
         outptr16 = &outputspace[gsegptr->seg_output_offset];
         work_input = &input[gsegptr->seg_input_offset];
         outptr16[lunit] += gsegptr->uf.matdata[lunit] * work_input[gsegptr->input_offset_short[lunit]];
         ++gsegptr;
      }
   }
   else { /* team_size is 1, and this work unit needs to do all 16 elements in the packet */
      gsegptr += num_header_packets; /* skip over team_offset data */
      npackets = gsegptr->npackets_remaining;                      /* Number of packets to be processed. */
      int stopdex  = ((teamnum + 1) * npackets) / n_teams;
      int startdex = ((teamnum    ) * npackets) / n_teams;
      gsegptr_stop = &gsegptr[stopdex];
      gsegptr = &gsegptr[startdex];
      while (gsegptr < gsegptr_stop) {
         outptr16 = &outputspace[gsegptr->seg_output_offset];
         work_input = &input[gsegptr->seg_input_offset];
         for (lunit=0; lunit<16; ++lunit) {
            outptr16[lunit] += gsegptr->uf.matdata[lunit] * work_input[gsegptr->input_offset_short[lunit]];
         }
         ++gsegptr;
      }
   }
   barrier(CLK_LOCAL_MEM_FENCE);

   /* Now that processing is done, it's time to write out the final results for this slab. */

   for (i=start; i<outspan; i+=span) {
      outptr[i] = outputspace[i];
   }
}

/* ================================================================================================== */
/* Kernel using "async_work_group_copy".  This version is optimized for the ACCELERATOR device        */
/* ================================================================================================== */

/* =========================================================== */
/* Grab a pile of input vector data into local storage         */
/* "_inputspace_index" is the offset into the local space.     */
/* "_input_offset" is the offset into the global input vector. */
/* =========================================================== */

#define GET_INPUT(_inputspace_index, _input_offset) {                                                                 \
   eventI[_inputspace_index] = async_work_group_copy((__local float8 *) &inputspace[column_span * _inputspace_index], \
                                                     (const __global float8 *) &input[_input_offset],                 \
                                                     (size_t) (column_span>>3),                                       \
                                                     (event_t) 0);                                                    \
}


/* ========================================================= */
/* Grab a pile of matrix packets into local storage          */
/* "_lsegspace_index" specifies the correct offset.  .       */
/* ========================================================= */

#define GET_PACKET(_lsegspace_index) {                                                               \
   eventS[lsegspace_tag] = async_work_group_copy((__local uchar16 *) &lsegspace[(_lsegspace_index)], \
                                                 (const __global uchar16 *) &gsegptr[gseg_index],    \
                                                 (size_t) ((sizeof(packet)/16)*(segcachesize/2)),    \
                                                 (event_t) 0);                                       \
   gseg_index += (segcachesize/2);                                                                   \
}

/* ========================================================= */
/*                                                           */
/* For a given packet of matrix data, residing in LOCAL      */
/* memory, do the following:                                 */
/*    Snap a pointer to the beginning of the packet.         */
/*    If it's time, grab a new batch of input data.          */
/*    Snap pointers to the output and matrix float data.     */
/*    Spend 16 lines performing the scalar computations.     */
/*    Perform two 8-way SIMD FMA operations.                 */
/*    Update the index to the next packet.                   */
/*                                                           */
/* ========================================================= */

#define PROCESS_LOCAL_PACKET {                                                   \
   float8 inV[2];                                                                \
   lsegptr = (__local struct _packet *) &lsegspace[lsegspace_index];             \
   if (lsegptr->seg_input_offset != curr_input_offset) {                         \
       curr_input_offset = lsegptr->seg_input_offset;                            \
       next_input_offset = lsegptr->future_seg_input_offset;                     \
       GET_INPUT(inputspace_index, next_input_offset)                            \
       inputspace_index = 1 - inputspace_index;                                  \
       wait_group_events(1, &eventI[inputspace_index]);                          \
   }                                                                             \
   work_input = &inputspace[column_span * inputspace_index];                     \
   outputspaceV8 = (__local float8 *) &outputspace[lsegptr->seg_output_offset];  \
   inV[0].s0 = work_input[lsegptr->input_offset_short[ 0]];                      \
   inV[0].s1 = work_input[lsegptr->input_offset_short[ 1]];                      \
   inV[0].s2 = work_input[lsegptr->input_offset_short[ 2]];                      \
   inV[0].s3 = work_input[lsegptr->input_offset_short[ 3]];                      \
   inV[0].s4 = work_input[lsegptr->input_offset_short[ 4]];                      \
   inV[0].s5 = work_input[lsegptr->input_offset_short[ 5]];                      \
   inV[0].s6 = work_input[lsegptr->input_offset_short[ 6]];                      \
   inV[0].s7 = work_input[lsegptr->input_offset_short[ 7]];                      \
   inV[1].s0 = work_input[lsegptr->input_offset_short[ 8]];                      \
   inV[1].s1 = work_input[lsegptr->input_offset_short[ 9]];                      \
   inV[1].s2 = work_input[lsegptr->input_offset_short[10]];                      \
   inV[1].s3 = work_input[lsegptr->input_offset_short[11]];                      \
   inV[1].s4 = work_input[lsegptr->input_offset_short[12]];                      \
   inV[1].s5 = work_input[lsegptr->input_offset_short[13]];                      \
   inV[1].s6 = work_input[lsegptr->input_offset_short[14]];                      \
   inV[1].s7 = work_input[lsegptr->input_offset_short[15]];                      \
   outputspaceV8[0] = fma(lsegptr->uf.matdataV8[0], inV[0], outputspaceV8[0]);   \
   outputspaceV8[1] = fma(lsegptr->uf.matdataV8[1], inV[1], outputspaceV8[1]);   \
   ++lsegspace_index;                                                            \
}

__kernel __attribute__ ((reqd_work_group_size(1, 1, 1)))
   void tiled_spmv_kernel_AWGC(__global float *input,         /* pointer to input memory object in global memory */
                               __global float *output,        /* pointer to output memory object in global memory */
                               __global uint *matbuffer,      /* pointer to tiled matrix memory object in global memory */
                               __private uint column_span,    /* size of fixed chunks of the input vector */
                               __private uint slabspace,      /* size of the variable chunk of output vector to be computed */
                               __private uint segcachesize,   /* number of tiled matrix packets which will fit in "outputspace" */
                               __private uint num_header_packets,
                               __local float *inputspace,     /* local buffer to hold staged input vector data */
                               __local float *outputspace,    /* local buffer to hold computed output, to be written out at the end */
                               __local packet *lsegspace)     /* local buffer to hold staged tiled matrix packet data */
{
   __global slab_header *headptr;
   __local float *work_input;
   __local float8 *outputspaceV8;
   int i, tempmax;
   event_t eventS[2], eventI[2], eventO;

   int temp = segcachesize/2;
   int half_scs = 0;
   while (temp) {
      ++half_scs;
      temp >>= 1;
   }

   __global packet *gsegptr; /* This is a global packet pointer, indexing into the global memory object containing the tiled matrix */
   __local  packet *lsegptr; /* This is a local packet pointer, indexing into the local storage for our tiled matrix */

   headptr = ((__global slab_header *) matbuffer) + get_global_id(0);
   gsegptr = &(((__global packet *) matbuffer)[headptr->offset]); 
   gsegptr += num_header_packets;  /* skip over team_offset data    */
   lsegptr = &lsegspace[0];

   int gseg_index = 0;        /* index into global memory of the packets for this slab */
   int inputspace_index = 0;  /* offset index into the local space for the input */
   int lsegspace_index = 0;   /* offset index into the local space for the tiled matrix */
   int lsegspace_tag = 0;     /* tag used to manage events regarding local packets */
   GET_PACKET(0)
   wait_group_events(1, &eventS[0]);
   uint npackets = lsegptr->npackets_remaining;  /* how many packets are to be processed in this slab? */
   if (npackets == 0) return;
   GET_PACKET(segcachesize/2)
   tempmax = (segcachesize < npackets) ? segcachesize : npackets;
   for (i=0; i<slabspace; ++i) {
      outputspace[i] = 0.0f; /* zero out the output buffer */
   }

   uint curr_input_offset = lsegptr->seg_input_offset;
   uint next_input_offset = lsegptr->future_seg_input_offset;
   GET_INPUT(0, curr_input_offset)       /* Load the first two parcels */
   GET_INPUT(1, next_input_offset)       /* of local input vector data */
   wait_group_events(1, &eventI[inputspace_index]);  /* and wait on the first one. */

   /* this first loop handles the bulk of the work with minimal if-tests, segcachesize/2 packets at a time */
   while (npackets > tempmax) {
      for (i=0; i<segcachesize/2; ++i) {
         PROCESS_LOCAL_PACKET
      }
      /* load next batch of packets, using double buffering */
      lsegspace_index &= (segcachesize-1);
      lsegspace_tag = (lsegspace_index == (segcachesize/2)) ? 1 : 0;
      npackets -= segcachesize/2;
      GET_PACKET((segcachesize/2)-lsegspace_index);
      wait_group_events(1, &eventS[lsegspace_tag]);
   }
 
   /* this second loop handles the remaining packets, one packet at a time */
   while (npackets) {
      PROCESS_LOCAL_PACKET
      lsegspace_index &= (segcachesize-1);
      lsegspace_tag = (lsegspace_index == (segcachesize/2)) ? 1 : 0;
      --npackets;
      if ((lsegspace_index & ((segcachesize/2)-1)) == 0) {
         if (npackets > segcachesize/2) {
            GET_PACKET((segcachesize/2)-lsegspace_index);
         }
         if (npackets > 0) {
            wait_group_events(1, &eventS[lsegspace_tag]);
         }
      }
   }

   /* Now that processing is done, it's time to write out the final results for this slab. */

   eventO = async_work_group_copy((__global float *) &output[headptr->outindex], (__const local float *) outputspace, (size_t) (headptr->outspan), (event_t) 0);
   wait_group_events(1, &eventO);
   wait_group_events(1, &eventI[1-inputspace_index]);
   wait_group_events(2, eventS);
}
