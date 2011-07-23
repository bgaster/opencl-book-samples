//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <getopt.h>
#include <libgen.h>
#include <unistd.h>
#include <sys/stat.h>
#include <CL/opencl.h>

#define KERNEL_DEFAULT 0
#define KERNEL_LS      1    /* The "load/store" kernel. */
#define KERNEL_AWGC    2    /* The "async work group copy" kernel. */

#define MAX_WGSZ 1024       /* This constant should be a multiple of 512 */
#define CPU_WGSZ 1          /* Work group size when running on a CPU (or an ACCELERATOR). */

/* ============================================================================ */
/* Macro to check success of each memory allocation.                            */
/* ============================================================================ */

#define MEMORY_ALLOC_CHECK(_addr, _len, _addrstr) {                                                             \
   posix_memalign((void **) &(_addr), preferred_alignment, _len);                                               \
   if ((_addr) == NULL) {                                                                                       \
      printf("Failed allocation of %lld bytes for %s\n", (unsigned long long) (unsigned int) (_len), _addrstr); \
      exit (EXIT_FAILURE);                                                                                      \
   }                                                                                                            \
}

/* ============================================================================ */
/* Macro to check success of OpenCL calls.                                      */
/* ============================================================================ */

#define CHECK_RESULT(_string) {                    \
   if (rc != CL_SUCCESS) {                         \
      printf("%s failed. rc = %d\n", _string, rc); \
      exit(EXIT_FAILURE);                          \
   }                                               \
}

/* ============================================================================ */
/* Global variables and declarations related to the matrix creation routine.    */
/* ============================================================================ */

typedef struct _slab_header {
   cl_uint offset;                   /* Offset into tiled matrix structure to beginning of this slab, in units of packets */
   cl_uint outindex;                 /* Offset into the output vector where this slab's "output region" starts */
   cl_uint outspan;                  /* Number of elements in this slab's "output region" */
} slab_header;

typedef struct _packet {
   cl_uint seg_input_offset;         /* identifies which section of the input vector this packet addresses */
   cl_uint future_seg_input_offset;  /* identifies the next input vector section, so pre-loading can be done */
   cl_uint npackets_remaining;       /* number of packets remaining in this slab */
   cl_uint seg_output_offset;        /* within this slab's "output region" which set of 16 outputs is this packet addressing? */
   cl_uint pad1;
   cl_uint pad2;
   cl_uint pad3;
   cl_uint pad4;
   cl_ushort input_offset_short[16]; /* which input values from the identified section of input vector is this packet using? */
   float matdata[16];                /* the sixteen floating point matrix values encoded into this packet */
} packet;

/* ============================================================================ */
/* Communication structure between tiled matrix algorithm code and OpenCL code. */
/* ============================================================================ */

typedef struct _matrix_gen_struct {
   slab_header **matrix_header;
   packet **seg_workspace;
   unsigned int *num_header_packets;
   unsigned int **row_index_array;
   unsigned int **x_index_array;
   float **data_array;
   unsigned int *nx_pad;
   unsigned int *nyround;
   unsigned int **slab_startrow; 
   unsigned int *nx;
   unsigned int *ny;
   unsigned int *non_zero;
   char *file_name;
   unsigned int preferred_alignment;
   unsigned int *max_compute_units;
   unsigned int kernel_type;
   unsigned int *column_span;
   unsigned int local_mem_size;
   unsigned int *segcachesize;
   unsigned int *max_slabheight;
   cl_device_type device_type;
   int *gpu_wgsz;
   size_t kernel_wg_size;
   unsigned int *nslabs_round;
   unsigned int *memsize;
} matrix_gen_struct;

/* ============================================================================ */
/* template for the function call to the code which builds the tiled matrix.    */
/* ============================================================================ */

int matrix_gen(matrix_gen_struct *);
