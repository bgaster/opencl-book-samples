//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//


#include "spmv.h"

/* ===================================================================== */
/* Procedure to print command and command line argument usage.           */
/* ===================================================================== */

void usage()
{
   printf("\n");
   printf("Usage: spmv -f <matrixfile> [device_type] [kernel_type] [options]\n");
   printf("\n");
   printf("Note: <matrixfile> should include the relative path from this executable.\n");
   printf("\n");
   printf(" Device Type:\n");
   printf("\n");
   printf("  -c, --cpu          Use CPU device for kernel computations.\n");
   printf("  -g, --gpu          Use GPU device for kernel computations.\n");
   printf("  -a, --accel        Use ACCELERATOR device for kernel computations.\n");
   printf("\n");
   printf(" Kernel Type (default is -A for ACCELERATOR device, -L otherwise):\n");
   printf("\n");
   printf("  -L, --ls           Use 'load-store' kernel to solve problem.\n");
   printf("  -A, --awgc         Use 'async-work-group-copy' kernel to solve problem.\n");
   printf("\n");
   printf(" Options (all options default to 'not selected'):\n");
   printf("\n");
   printf("  -l, --lwgsize [n]  Specify local work group size for GPU use (coerced to power of 2).\n");
   printf("\n");
   printf("  -h, --help         Print this usage message.\n");
   printf("\n");
}

/* ===================================================================== */
/* Structures to help us determine available platforms and devices.      */
/* ===================================================================== */

typedef struct {
   cl_device_id id;
   cl_device_type type;
   cl_command_queue ComQ;
   char *name;
} device_struct;

typedef struct {
   char *name;
   cl_platform_id id;
   unsigned int num_devices;
   device_struct *device;
   cl_context context;
   cl_program program;
   cl_kernel kernel;
} platform_struct;

/* ================================================================================================== */
/* Load_program_source.                                                                              */
/* Read in the kernel source from an external file.                                                   */
/* ================================================================================================== */

static char *load_program_source(const char *filename)
{
  struct stat statbuf;

  FILE *fh = fopen(filename, "r");
  if (fh == 0) {
    fprintf(stderr, "Couldn't open %s\n", filename);
    return NULL;
  }

  stat(filename, &statbuf);
  char *source = (char *) malloc(statbuf.st_size + 1);
  if (source == NULL) {
    fprintf(stderr, "malloc failed\n");
    return NULL;
  }

  fread(source, statbuf.st_size, 1, fh);
  source[statbuf.st_size] = '\0';

  return source;
}

/* ================================================================================================== */
/* Main.                                                                                              */
/* ================================================================================================== */

int main(int argc, char *argv[]) {

   /* Variables used to manage the OpenCL environment. */
   cl_int rc;
   size_t return_size[1];
   unsigned int column_span = 0;
   static cl_device_type device_type = CL_DEVICE_TYPE_DEFAULT;
   static cl_uint kernel_type = KERNEL_DEFAULT;
   static int gpu_wgsz = MAX_WGSZ;

   /* The external file containing the matrix data in Matrix Market format */
   static char *file_name;
   
   /* These variables deal with the source file for the kernel, and the names of the kernels contained therein. */
   char kernel_source_file[8] = "spmv.cl";
   char kernel_name_LS[21]   = "tiled_spmv_kernel_LS";
   char kernel_name_AWGC[23] = "tiled_spmv_kernel_AWGC";
   char kernel_name[32];
   
   /* Basic "size of problem" variables. */
   unsigned int nx; /* Number of elements in the X direction (length of the "input" vector. */
   unsigned int ny; /* Number of elements in the Y direction (length of the "answer" vector. */
   unsigned int non_zero; /* Number of non_zero elements in the matrix. */
   unsigned int nx_pad, nyround; /* Rounded versions of nx and ny. */
   
   /* Variables used to hold user-specified overrides and intermediate control values derived from them. */
   unsigned int *slab_startrow = NULL;
   
   unsigned int segcachesize;
   unsigned int max_slabheight; /* Maximum matrix chunksize. */
   unsigned int i, j, pdex = 0, ddex = 0;
   size_t param_value_size_ret;

   /* ================================================================================== */
   /* Read in command line arguments.                                                    */
   /* ================================================================================== */

   int opt;
   int option_index;

   struct option long_options[] = {
      {"help", no_argument, NULL, 'h'},
      {"accel", no_argument, NULL, 'a'},
      {"cpu", no_argument, NULL, 'c'},
      {"gpu", no_argument, NULL, 'g'},
      {"ls", no_argument, NULL, 'L'},   
      {"awgc", no_argument, NULL, 'A'},   
      {"verify", no_argument, NULL, 'v'},
      {"lwgsize", required_argument, NULL, 'l'},
      {"filename", required_argument, NULL, 'f'},
      {NULL, 0, NULL, 0}
   };
   char *name;

   /* ================================================================================== */
   /* Change current working directory to that of the invocation path so that spmv can   */
   /* be run from any current working directory.                                         */
   /* ================================================================================== */

   name = basename(argv[0]);
   (void)chdir(dirname(argv[0]));

   while (1) {
      opt = getopt_long(argc, argv, "hacgLAl:f:", long_options, &option_index);

      if (opt == -1) break;

      switch (opt) {

      /* -h, --help */
      case 'h': usage(); exit(EXIT_SUCCESS);

      /* -a, --accel */
      case 'a': device_type = CL_DEVICE_TYPE_ACCELERATOR; break;

      /* -c, --cpu */
      case 'c': device_type = CL_DEVICE_TYPE_CPU; break;

      /* -g, --gpu */
      case 'g': device_type = CL_DEVICE_TYPE_GPU; break;

      /* -L, --ls */
      case 'L': kernel_type = KERNEL_LS; break;

      /* -A, --awgc */
      case 'A': kernel_type = KERNEL_AWGC; break;

      /* -l, --lwgsize */
      case 'l': gpu_wgsz = atoi(optarg); break;

      /* -f, --filename */
      case 'f':
         posix_memalign((void **) &file_name, 128, 1+strlen(optarg));
         strcpy(file_name, optarg);
         break;

      case '?':
         printf("Try '%s --help' for more information.\n", name);
         exit(EXIT_FAILURE);
      }
   }

   if (optind != argc) {
      printf("%s: unrecognized option '%s'.\n", name, argv[optind]);
      printf("Try '%s --help' for more information.\n", name);
      exit(EXIT_FAILURE);
   }

   /* ================================================================================== */
   /* Start up OpenCL.                                                                   */
   /* ================================================================================== */

   cl_uint preferred_alignment = 16; // used by "MEMORY_ALLOC_CHECK" macro   
   cl_uint num_platforms;
   rc = clGetPlatformIDs(0, (cl_platform_id *) NULL, &num_platforms);
   CHECK_RESULT("clGetPlatformIDs(num_platforms)")

   platform_struct *platform;
   MEMORY_ALLOC_CHECK(platform, num_platforms * sizeof(platform_struct), "platform");

   cl_mem *buffer;
   MEMORY_ALLOC_CHECK(buffer, num_platforms * sizeof(cl_mem), "buffer");

   cl_platform_id *temp_platform_id_array;
   MEMORY_ALLOC_CHECK(temp_platform_id_array, num_platforms * sizeof(cl_platform_id), "temp_platform_id_array");
   rc = clGetPlatformIDs(num_platforms, temp_platform_id_array, (cl_uint *) NULL);
   CHECK_RESULT("clGetPlatform IDs(Platform IDs)")
   for (i=0; i<num_platforms; ++i) {
      platform[i].id = temp_platform_id_array[i];
   }
   free(temp_platform_id_array);

   printf("[START RUN]\n");
   printf("command line: "); 
   for (i=0; i<(unsigned int) argc; ++i) {
      printf("%s ", argv[i]);
   }
   printf("\n");
   //printf("num_platforms = %d\n\n", num_platforms);

   for (i=0; i<num_platforms; ++i) {
      rc = clGetPlatformInfo(platform[i].id, CL_PLATFORM_NAME, (size_t) 0, NULL, (size_t *) &param_value_size_ret);
      CHECK_RESULT("clGetPlatformInfo(size of platform name)")
      MEMORY_ALLOC_CHECK(platform[i].name, param_value_size_ret, "platform name");
      rc = clGetPlatformInfo(platform[i].id, CL_PLATFORM_NAME, param_value_size_ret, platform[i].name, (size_t *) NULL);
      CHECK_RESULT("clGetPlatformInfo(platform name)")

      rc = clGetDeviceIDs(platform[i].id, CL_DEVICE_TYPE_ALL, 0, NULL, (cl_uint *) &(platform[i].num_devices));
      CHECK_RESULT("clGetDeviceIDs(number of devices)")

      MEMORY_ALLOC_CHECK(platform[i].device, platform[i].num_devices * sizeof(device_struct), "device structure");

      cl_device_id *tmpdevices;
      MEMORY_ALLOC_CHECK(tmpdevices, platform[i].num_devices * sizeof(cl_device_id), "tmpdevices");
      rc = clGetDeviceIDs(platform[i].id, CL_DEVICE_TYPE_ALL, platform[i].num_devices, tmpdevices, NULL);
      CHECK_RESULT("clGetDeviceIDs(list of device IDs)")
      for (j=0; j<platform[i].num_devices; ++j) {
         platform[i].device[j].id = tmpdevices[j];
         rc = clGetDeviceInfo(platform[i].device[j].id, CL_DEVICE_TYPE, sizeof(cl_device_type), &platform[i].device[j].type, NULL);
         CHECK_RESULT("clGetDeviceInfo(device type)")
      }
      free(tmpdevices);
   }

   /* ================================================================================== */
   /* Choose the best device to use, if one is not explicitly called for.                */
   /* If a device is specified, ensure that device is present on this hardware.          */
   /* ================================================================================== */

   if (device_type == CL_DEVICE_TYPE_DEFAULT) {
      int accel_found = 0;
      for (i=0; i<num_platforms; ++i) {
         for (j=0; j<platform[i].num_devices; ++j) {
            if (platform[i].device[j].type == CL_DEVICE_TYPE_ACCELERATOR) {
               accel_found = 1;
               pdex = i;
               ddex = j;
            }
         }
      }
      if (!accel_found) {
         int gpu_found = 0; 
         for (i=0; i<num_platforms; ++i) {
            for (j=0; j<platform[i].num_devices; ++j) {
               if ((gpu_found == 0) && (platform[i].device[j].type == CL_DEVICE_TYPE_GPU)) {
                  gpu_found = 1;
                  pdex = i;
                  ddex = j;
               }
            }
         }
         if (!gpu_found) {
            int cpu_found = 0; 
            for (i=0; i<num_platforms; ++i) {
               for (j=0; j<platform[i].num_devices; ++j) {
                  if (platform[i].device[j].type == CL_DEVICE_TYPE_CPU) {
                     cpu_found = 1;
                     pdex = i;
                     ddex = j;
                  }
               }
            }
            if (!cpu_found) {
               fprintf(stderr, "no devices of any kind were found on this system.  Leaving...\n"); 
               fflush(stderr);
               exit(EXIT_FAILURE);
            }
         }
      }
   }
   else {
      int device_found = 0;
      for (i=0; i<num_platforms; ++i) for (j=0; j<platform[i].num_devices; ++j) {
         if (platform[i].device[j].type == device_type) {
            device_found = 1;
            pdex = i;
            ddex = j;
         }
      }
      if (device_found == 0) {
         fprintf(stderr, "no devices of the requested type were found on this system.  Leaving...\n"); 
         fflush(stderr);
         exit(EXIT_FAILURE);
      }
   }

   /* ================================================================================== */
   /* Choose the best kernel to use, if one is not explicitly called for.                */
   /* ================================================================================== */

   if (kernel_type == KERNEL_DEFAULT) {
      kernel_type = (platform[pdex].device[ddex].type == CL_DEVICE_TYPE_ACCELERATOR) ? KERNEL_AWGC : KERNEL_LS;
   }

   /* ================================================================================== */
   /* Create a context.                                                                  */
   /* ================================================================================== */

   cl_context_properties properties[3];
   properties[0] = CL_CONTEXT_PLATFORM;
   properties[1] = (const cl_context_properties) platform[pdex].id;
   properties[2] = 0;
   platform[pdex].context = clCreateContext((const cl_context_properties *) properties, 1, &(platform[pdex].device[ddex].id), NULL, NULL, &rc);
   CHECK_RESULT("clCreateContext")

   /* ================================================================================== */
   /* Build the kernel, create the Command Queue, and print kernel/device info.          */
   /* ================================================================================== */

   switch (kernel_type) {
      case KERNEL_LS:
      strcpy(kernel_name, kernel_name_LS);
      break;
      case KERNEL_AWGC: 
      strcpy(kernel_name, kernel_name_AWGC);
      break;
   }

   char *kernel_source;
   kernel_source = load_program_source(kernel_source_file);
   if (kernel_source == NULL) {
      fprintf(stderr, "Error: Failed to load compute program from file!\n");
      exit(EXIT_FAILURE);
   }

   platform[pdex].program = clCreateProgramWithSource(platform[pdex].context, 1, (const char **) &kernel_source, NULL, &rc);
   CHECK_RESULT("clCreateProgramWithSource")
   free(kernel_source);

   rc = clBuildProgram(platform[pdex].program, 1, &(platform[pdex].device[ddex].id), "", NULL, NULL);
   CHECK_RESULT("clBuildProgram")

   platform[pdex].kernel = clCreateKernel(platform[pdex].program, kernel_name, &rc);
   CHECK_RESULT("clCreateKernel")

   platform[pdex].device[ddex].ComQ = clCreateCommandQueue(platform[pdex].context, platform[pdex].device[ddex].id, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &rc);
   CHECK_RESULT("clCreateCommandQueue")

   rc = clGetDeviceInfo(platform[pdex].device[ddex].id, CL_DEVICE_NAME, (size_t) 0, NULL, (size_t *) &param_value_size_ret);
   CHECK_RESULT("clGetDeviceInfo(size of CL_DEVICE_NAME)")
   MEMORY_ALLOC_CHECK(platform[pdex].device[ddex].name, param_value_size_ret, "device name");
   rc = clGetDeviceInfo(platform[pdex].device[ddex].id, CL_DEVICE_NAME, (size_t) param_value_size_ret, platform[pdex].device[ddex].name, (size_t *) NULL);
   CHECK_RESULT("clGetDeviceInfo(CL_DEVICE_NAME)")

   printf("We'll run kernel %s on device %s\n", ((kernel_type == KERNEL_LS) ? "kernel_ls" : "kernel_awgc"), platform[pdex].device[ddex].name); 

   /* ================================================================================== */
   /* Determine device alignment, and whether "out-of-order" processing is supported.    */
   /* ================================================================================== */

   rc = clGetDeviceInfo(platform[pdex].device[ddex].id, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(cl_uint), &preferred_alignment, NULL);
   CHECK_RESULT("clGetDeviceInfo(CL_DEVICE_MEM_BASE_ADDR_ALIGN)")
   if (preferred_alignment > 1024) preferred_alignment = 1024;
   preferred_alignment /= 8;  /* Convert from units of bits to units of bytes. */

   cl_command_queue_properties command_queue_properties;
   clGetDeviceInfo (platform[pdex].device[ddex].id, CL_DEVICE_QUEUE_PROPERTIES, sizeof(cl_command_queue_properties), &command_queue_properties, NULL); 
   command_queue_properties &= CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;

   /* ================================================================================== */
   /* Determine local memory size and maximum compute units.                             */
   /* ================================================================================== */

   size_t kernel_wg_size;
   rc = clGetKernelWorkGroupInfo (platform[pdex].kernel, platform[pdex].device[ddex].id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), (void *) &kernel_wg_size, return_size);
   CHECK_RESULT("clGetKernelWorkGroupInfo(CL_KERNEL_WORK_GROUP_SIZE)")

   cl_ulong total_local_mem;
   rc = clGetDeviceInfo (platform[pdex].device[ddex].id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof (cl_ulong), (void *) &total_local_mem, NULL);
   CHECK_RESULT("clGetDeviceInfo(CL_DEVICE_LOCAL_MEM_SIZE)")

   cl_ulong used_local_mem;
   rc = clGetKernelWorkGroupInfo (platform[pdex].kernel, platform[pdex].device[ddex].id, CL_KERNEL_LOCAL_MEM_SIZE, sizeof (cl_ulong), &used_local_mem, NULL);
   CHECK_RESULT("clGetKernelWorkGroupInfo(CL_KERNEL_LOCAL_MEM_SIZE)")

   cl_ulong local_mem_size;
   local_mem_size = total_local_mem - used_local_mem;

   cl_uint max_compute_units;
   clGetDeviceInfo (platform[pdex].device[ddex].id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &max_compute_units, NULL); 

   /* ================================================================================== */
   /* Set up parameter structure and call the function that builds the tiled matrix.     */
   /* ================================================================================== */

   matrix_gen_struct mgs;
   unsigned int nslabs_round, memsize;
   packet *seg_workspace;
   slab_header *matrix_header;
   unsigned int num_header_packets;
   unsigned int *row_index_array = NULL;
   unsigned int *x_index_array = NULL;
   float *data_array = NULL;

   mgs.matrix_header = &matrix_header;
   mgs.seg_workspace = &seg_workspace;
   mgs.num_header_packets = &num_header_packets;
   mgs.row_index_array = &row_index_array;
   mgs.x_index_array = &x_index_array;
   mgs.data_array = &data_array;
   mgs.nx_pad = &nx_pad;
   mgs.nyround = &nyround;
   mgs.slab_startrow = &slab_startrow;
   mgs.nx = &nx;
   mgs.ny = &ny;
   mgs.non_zero = &non_zero;
   mgs.file_name = (char *) file_name;
   mgs.preferred_alignment = preferred_alignment;
   mgs.max_compute_units = &max_compute_units;
   mgs.kernel_type = kernel_type;
   mgs.column_span = &column_span;
   mgs.local_mem_size = (unsigned int) local_mem_size;
   mgs.segcachesize = &segcachesize;
   mgs.max_slabheight = &max_slabheight;
   mgs.device_type = platform[pdex].device[ddex].type,
   mgs.gpu_wgsz = &gpu_wgsz,
   mgs.kernel_wg_size = kernel_wg_size;
   mgs.nslabs_round = &nslabs_round;
   mgs.memsize = &memsize;

   rc = matrix_gen(&mgs);

   /* =============================================================================================== */
   /* Compute the local and global work group sizes.                                                  */
   /* =============================================================================================== */

   unsigned int ndims;
   unsigned int team_size;

   size_t global_work_size[3];
   size_t local_work_size[3];
   if (kernel_type == KERNEL_AWGC) {
      ndims = 1;
      global_work_size[0] = nslabs_round;
      local_work_size[0] = 1;
   }
   else {
      ndims = 2;
      team_size = (platform[pdex].device[ddex].type == CL_DEVICE_TYPE_GPU) ? 16 : 1;
      global_work_size[1] = nslabs_round;
      local_work_size[1] = 1;
      global_work_size[0] = local_work_size[0] = (platform[pdex].device[ddex].type == CL_DEVICE_TYPE_GPU) ? gpu_wgsz : CPU_WGSZ;
      int max_aggregate_local_work_group_size = 0;
      int aggregate_local_work_group_size = 1;
      for (i=0; i<ndims; ++i) {
         aggregate_local_work_group_size *= local_work_size[i];
      }
      max_aggregate_local_work_group_size = aggregate_local_work_group_size;
      if (max_aggregate_local_work_group_size > (int) kernel_wg_size) {
         while (max_aggregate_local_work_group_size > (int) kernel_wg_size) {
            local_work_size[0] /= 2;
            gpu_wgsz /= 2;
            max_aggregate_local_work_group_size /= 2;
         }
         printf("coercing work group size to fit within hardware limits.  New size is %d\n", gpu_wgsz);
      }
   }

   /* =============================================================================================== */
   /* Our Tiled format is now complete, but still in "working storage".  We cannot allocate its       */
   /* buffer in OpenCL until we know how big it is, and now, we finally know how big it is.  So, we   */
   /* create the Input and Output arrays, and the final array to hold the Tiled Format of the Matrix. */
   /* =============================================================================================== */

   /* Arrays to hold input and output data, and the finished tiled matrix data. */
   float *input_array, *output_array, *output_array_verify;
   unsigned int *tilebuffer;
   
   MEMORY_ALLOC_CHECK(output_array_verify, (nyround * sizeof(float)), "output_array_verify") 
   if (output_array_verify == NULL) {
      fprintf(stderr, "insufficient memory to perform this workload.\n"); fflush(stderr);
      exit(EXIT_FAILURE);
   }

   cl_mem input_buffer;
   cl_mem matrix_buffer;
   cl_mem output_buffer;
   unsigned int input_buffer_size;
   unsigned int matrix_buffer_size;
   /* Create the input and matrix buffer memory objects. */
   input_buffer_size = (nx_pad * sizeof(float));
   input_buffer = clCreateBuffer(platform[pdex].context, CL_MEM_ALLOC_HOST_PTR, input_buffer_size, NULL, &rc);
   CHECK_RESULT("clCreateBuffer(input_buffer)")

   matrix_buffer_size = memsize;
   matrix_buffer = clCreateBuffer(platform[pdex].context, CL_MEM_ALLOC_HOST_PTR, matrix_buffer_size, NULL, &rc);
   CHECK_RESULT("clCreateBuffer(matrix_buffer)")

   cl_event events[2];

   unsigned int output_buffer_size;
   output_buffer_size = (slab_startrow[nslabs_round] - slab_startrow[0]) * sizeof(float);
   output_buffer = clCreateBuffer(platform[pdex].context, CL_MEM_ALLOC_HOST_PTR, output_buffer_size, NULL, &rc);
   CHECK_RESULT("clCreateBuffer(output_buffer)")

   /* =============================================================================================== */
   /* Map these buffers to allocate pointers into these buffers that we can use to load them.         */
   /* =============================================================================================== */

   input_array =       (float *) clEnqueueMapBuffer(platform[pdex].device[ddex].ComQ, 
                                                       input_buffer, 
                                                       CL_TRUE, 
                                                       CL_MAP_WRITE, 
                                                       0, 
                                                       (size_t) input_buffer_size, 
                                                       0, 
                                                       NULL, 
                                                       NULL, 
                                                       &rc);
   CHECK_RESULT("clEnqueueMapBuffer(input_array)")

   tilebuffer = (unsigned int *) clEnqueueMapBuffer(platform[pdex].device[ddex].ComQ, 
                                                       matrix_buffer, 
                                                       CL_TRUE, 
                                                       CL_MAP_WRITE, 
                                                       0, 
                                                       (size_t) matrix_buffer_size, 
                                                       0, 
                                                       NULL, 
                                                       NULL, 
                                                       &rc);
   CHECK_RESULT("clEnqueueMapBuffer(tilebuffer)")

   output_array =     (float *) clEnqueueMapBuffer(platform[pdex].device[ddex].ComQ, 
                                                      output_buffer, 
                                                      CL_TRUE, 
                                                      CL_MAP_WRITE, 
                                                      0, 
                                                      (size_t) output_buffer_size, 
                                                      0, 
                                                      NULL, 
                                                      NULL, 
                                                      &rc);
   CHECK_RESULT("clEnqueueMapBuffer(output_array)")

   /* =============================================================================================== */
   /* Copy the tiled matrix into the memory buffer, and then unmap it.                                */
   /* =============================================================================================== */

   memcpy(tilebuffer, seg_workspace, sizeof(packet) * (matrix_header[nslabs_round].offset));
   rc = clEnqueueUnmapMemObject(platform[pdex].device[ddex].ComQ, matrix_buffer, tilebuffer, 0, NULL, &events[0]);
   CHECK_RESULT("clEnqueueUnmapMemObject(tilebuffer)")
   clWaitForEvents(1, events);

   /* Load random data into the input array.                                         */
   /* The user can substitute initialization of real data at this point in the code. */
   for (i=0; i<nx; ++i) {
      float rval;
      rval = ((float) (rand() & 0x7fff)) * 0.001f - 15.0f;
      input_array[i] = rval;
   }

   /* Zero out the output array.                                                             */
   /* Note that this is only needed because some matrices are singular and have whole rows   */
   /* that are all zero, which is detected, and no work is done on those rows, so that they  */
   /* will never get written by the kernel, so to be safe, we zero it all out here, as well. */

   memset((void *) output_array, 0, output_buffer_size);

   /* =============================================================================================== */
   /* Unmap the input and output memory buffers, to prepare for kernel execution.                     */
   /* =============================================================================================== */

   rc = clEnqueueUnmapMemObject(platform[pdex].device[ddex].ComQ, input_buffer, input_array,   0, NULL, &events[0]);
   CHECK_RESULT("clEnqueueUnmapMemObject(input_array)")
   rc = clEnqueueUnmapMemObject(platform[pdex].device[ddex].ComQ, output_buffer, output_array, 0, NULL, &events[1]);
   CHECK_RESULT("clEnqueueUnmapMemObject(output_array)")
   clWaitForEvents(2, events);

   /* =============================================================================================== */
   /* Execution: Multiplication of the input array times the Tiled Format of the Matrix.              */
   /* =============================================================================================== */

   /* Run once to verifying correct answer, and computing a baseline number of repetitions for later performance measurements. */

   rc = clSetKernelArg(platform[pdex].kernel, 0, sizeof(cl_mem), (const void *) &input_buffer);
   CHECK_RESULT("clSetKernelArg(0)")
   rc = clSetKernelArg(platform[pdex].kernel, 1, sizeof(cl_mem), (const void *) &output_buffer);
   CHECK_RESULT("clSetKernelArg(1)")
   rc = clSetKernelArg(platform[pdex].kernel, 2, sizeof(cl_mem), (const void *) &matrix_buffer);
   CHECK_RESULT("clSetKernelArg(2)")
   rc = clSetKernelArg(platform[pdex].kernel, 3, sizeof(cl_uint), &column_span);
   CHECK_RESULT("clSetKernelArg(3)")
   rc = clSetKernelArg(platform[pdex].kernel, 4, sizeof(cl_uint), &max_slabheight);
   CHECK_RESULT("clSetKernelArg(4)")

   if (kernel_type == KERNEL_LS) {
      rc = clSetKernelArg(platform[pdex].kernel, 5, sizeof(cl_uint), &team_size);
      CHECK_RESULT("clSetKernelArg(5)")
      rc = clSetKernelArg(platform[pdex].kernel, 6, sizeof(cl_uint), &num_header_packets);
      CHECK_RESULT("clSetKernelArg(6)")
      rc = clSetKernelArg(platform[pdex].kernel, 7, (size_t) (max_slabheight * sizeof(float)), (void *) NULL);
      CHECK_RESULT("clSetKernelArg(7)")
   }
   else {
      rc = clSetKernelArg(platform[pdex].kernel, 5, sizeof(cl_uint), &segcachesize);
      CHECK_RESULT("clSetKernelArg(5)")
      rc = clSetKernelArg(platform[pdex].kernel, 6, sizeof(cl_uint), &num_header_packets);
      CHECK_RESULT("clSetKernelArg(6)")
      rc = clSetKernelArg(platform[pdex].kernel, 7, (size_t) (2 * column_span * sizeof(float)), (void *) NULL);
      CHECK_RESULT("clSetKernelArg(7)")
      rc = clSetKernelArg(platform[pdex].kernel, 8, (size_t) (max_slabheight * sizeof(float)), (void *) NULL);
      CHECK_RESULT("clSetKernelArg(8)")
      rc = clSetKernelArg(platform[pdex].kernel, 9, (size_t) (segcachesize * sizeof(packet)), (void *) NULL);
      CHECK_RESULT("clSetKernelArg(9)")
   }

   rc = clEnqueueNDRangeKernel(platform[pdex].device[ddex].ComQ, platform[pdex].kernel, ndims, NULL, global_work_size, local_work_size, 0, NULL, &events[0]);
   CHECK_RESULT("clEnqueueNDRangeKernel")

   clWaitForEvents(1, events);

   output_array = (float *) clEnqueueMapBuffer(platform[pdex].device[ddex].ComQ, 
                                                  output_buffer, 
                                                  CL_TRUE, 
                                                  (CL_MAP_READ|CL_MAP_WRITE), 
                                                  0, 
                                                  (size_t) output_buffer_size, 
                                                  0, 
                                                  NULL, 
                                                  NULL, 
                                                  &rc);
   CHECK_RESULT("clEnqueueMapBuffer(output_array)")

   input_array   = (float *) clEnqueueMapBuffer(platform[pdex].device[ddex].ComQ, 
                                                   input_buffer, 
                                                   CL_TRUE, 
                                                   (CL_MAP_READ|CL_MAP_WRITE), 
                                                   0, 
                                                   (size_t) input_buffer_size, 
                                                   0, 
                                                   NULL, 
                                                   NULL, 
                                                   &rc);
   CHECK_RESULT("clEnqueueMapBuffer(input_array)")

   /* =============================================================== */
   /* Data Verification.                                              */
   /* =============================================================== */

   rc = 0;
   /* Run the trivial (reference) spmv calculation, using the data previously loaded into CSR format. */
   for (i=0; i<ny; ++i) {
      float t = 0;
      unsigned int lb = row_index_array[i];
      unsigned int ub = row_index_array[i+1];
      for (j=lb; j<ub; ++j) {
         t += data_array[j] * input_array[x_index_array[j]];
      }
      output_array_verify[i] = t;
   }

   /* Compare results of kernel computations against trivial calculation results. */
   double sum;
   double diffsum;
   sum = 0.0;
   diffsum = 0.0;
   for (i=0; i<ny; ++i) {
      float a, b;
      double abs_a, delta;
      a = output_array_verify[i];
      b = output_array[i];
      abs_a = ((double) a);
      delta = (((double) a) - ((double) b));
      abs_a = (abs_a < 0.0) ? -abs_a : abs_a;
      delta = (delta < 0.0) ? -delta : delta;
      sum += abs_a;
      diffsum += delta;
   }
   printf("avg error = %le, ", diffsum / sum);
   if (diffsum / sum > 0.0001) {
      rc = -1;
   }

   printf("(matrix %s)\n", file_name);
   int retval = rc;

   /* ================= */
   /* Shut Down OpenCL. */
   /* ================= */

   rc = clEnqueueUnmapMemObject(platform[pdex].device[ddex].ComQ, input_buffer, input_array, 0, NULL, NULL);
   CHECK_RESULT("clEnqueueUnmapMemObject(input)")
   rc = clEnqueueUnmapMemObject(platform[pdex].device[ddex].ComQ, output_buffer, output_array, 0, NULL, NULL);
   CHECK_RESULT("clEnqueueUnmapMemObject(output)")

   rc = clFinish(platform[pdex].device[ddex].ComQ);
   CHECK_RESULT("clFinish")

   rc = clReleaseEvent(events[0]);
   CHECK_RESULT("clReleaseEvent(0)")
   rc = clReleaseEvent(events[1]);
   CHECK_RESULT("clReleaseEvent(1)")
   rc = clReleaseMemObject(input_buffer);
   CHECK_RESULT("clReleaseMemObject(input)")
   rc = clReleaseMemObject(matrix_buffer);
   CHECK_RESULT("clReleaseMemObject(matrix)")
   rc = clReleaseMemObject(output_buffer);
   CHECK_RESULT("clReleaseMemObject(output)")
   rc = clReleaseCommandQueue(platform[pdex].device[ddex].ComQ);
   CHECK_RESULT("clReleaseCommandQueue")
   rc = clReleaseKernel(platform[pdex].kernel);
   CHECK_RESULT("clReleaseKernel")
   rc = clReleaseProgram(platform[pdex].program);
   CHECK_RESULT("clReleaseProgram")
   rc = clReleaseContext(platform[pdex].context);
   CHECK_RESULT("clReleaseContext")

   /* ============================= */
   /* Free up all allocated memory. */
   /* ============================= */

   free(data_array);
   free(x_index_array);
   free(row_index_array);
   free(slab_startrow);
   free(seg_workspace);
   free(output_array_verify);
   free(platform[pdex].device[ddex].name);
   for (i=0; i<num_platforms; ++i) free(platform[i].device);
   for (i=0; i<num_platforms; ++i) free(platform[i].name);
   free(buffer);
   free(platform);

   return retval;
}
