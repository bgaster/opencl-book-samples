#include "spmv.h"

/* ================================================================================= */
/* Here is the routine which does the algorithm work in the host-based code.         */
/* ================================================================================= */

int matrix_gen(matrix_gen_struct *mgs) {
   unsigned int data_present, symmetric, preferred_alignment, preferred_alignment_by_elements;
   FILE *inputMTX;
   unsigned int i, j;

   preferred_alignment = mgs->preferred_alignment;
   preferred_alignment_by_elements = preferred_alignment / sizeof(float);
   if (preferred_alignment_by_elements < 16) preferred_alignment_by_elements = 16;

   /* =============================================================== */
   /* Open Matrix File and read first lines of data.                  */
   /* =============================================================== */

   inputMTX = fopen((mgs->file_name), "r");
   if (inputMTX == NULL) {
      printf("Error opening maxtrix file %s\n", (mgs->file_name));
      exit(EXIT_FAILURE);
   }
   else {
      char tmp[20], pattern_flag[20], symmetric_flag[20];
      if (5 != fscanf(inputMTX, "%19s %19s %19s %19s %19s\n", tmp, tmp, tmp, pattern_flag, symmetric_flag)) {
         fprintf(stderr, "error reading matrix market format header line\n");
         exit(EXIT_FAILURE);
      }
      data_present = strcmp(pattern_flag, "pattern");
      symmetric = strcmp(symmetric_flag, "general");
      fscanf(inputMTX, "%d %d %d\n", (mgs->nx), (mgs->ny), (mgs->non_zero));
   }

   /* =============================================================== */
   /* Create working storage for initial processing of matrix.        */
   /* =============================================================== */

   unsigned int *count_array;
   float **line_data_array;
   unsigned int **line_x_index_array;

   float *raw_data;
   unsigned int *raw_ix;
   unsigned int *raw_iy;

   MEMORY_ALLOC_CHECK(raw_ix, (*(mgs->non_zero) * sizeof (int)), "raw_ix") 
   MEMORY_ALLOC_CHECK(raw_iy, (*(mgs->non_zero) * sizeof (int)), "raw_iy") 
   MEMORY_ALLOC_CHECK(raw_data, (*(mgs->non_zero) * sizeof (float)), "raw_data") 
   MEMORY_ALLOC_CHECK(line_data_array, (*(mgs->ny) * sizeof (float *)), "line_data_array") 
   MEMORY_ALLOC_CHECK(line_x_index_array, (*(mgs->ny) * sizeof (int *)), "line_x_index_array") 
   MEMORY_ALLOC_CHECK(count_array, (*(mgs->ny) * sizeof (int)), "count_array") 
   for (i=0; i<*(mgs->ny); ++i) {
      count_array[i] = 0;
   }

   /* =============================================================== */
   /* Read in the raw data from the matrix file.                      */
   /* Check for anomalous data, and handle symmetric matrices.        */
   /* =============================================================== */

   unsigned int curry, actual_non_zero;
   curry = actual_non_zero = 0;
   unsigned int explicit_zero_count = 0;
   for (i=0; i<*(mgs->non_zero); ++i) {
      unsigned int ix, iy;
      float data; 
      fscanf(inputMTX, "%d %d\n", &ix, &iy);
      if (i == 0) {
         curry = iy-1;
      }
      if (data_present) {
         double double_data;
         fscanf(inputMTX, "%lf\n", &double_data);
         data = (float) double_data;
      }
      else data = ((float) (rand() & 0x7fff)) * 0.001f - 15.0f;
      if (data_present && data == 0.0) {
         ++explicit_zero_count;
      }
      else {
         --ix;
         --iy;
         raw_ix[actual_non_zero] = ix;
         raw_iy[actual_non_zero] = iy;
         raw_data[actual_non_zero] = data;
         ++actual_non_zero;
         ++count_array[iy];
         if (symmetric && (ix != iy)) {
            ++count_array[ix];
         }
         if (iy != curry) {
            if (iy != curry+1) {
               printf("gap in the input (non-invertible matrix): i = %d, iy = %d, curry = %d\n", actual_non_zero, iy, curry);
            }
            curry = iy;
         }
      }
   }
   if (explicit_zero_count) {
      printf("explicit_zero_count = %d\n", explicit_zero_count);
   }
   *(mgs->non_zero) = actual_non_zero;

   /* =============================================================== */
   /* Create working storage for each row's data.                     */
   /* =============================================================== */

   for (i=0; i<*(mgs->ny); ++i) {
      MEMORY_ALLOC_CHECK(line_data_array[i], (count_array[i] * sizeof (float)), "line_data_array[i]") 
      MEMORY_ALLOC_CHECK(line_x_index_array[i], (count_array[i] * sizeof (int)), "line_x_index_array[i]") 
      count_array[i] = 0;
   }

   /* Fill in each row (special handling for symmetric matrices). */
   for (i=0; i<*(mgs->non_zero); ++i) {
      line_data_array[raw_iy[i]][count_array[raw_iy[i]]] = raw_data[i];
      line_x_index_array[raw_iy[i]][count_array[raw_iy[i]]] = raw_ix[i];
      ++count_array[raw_iy[i]];
      if (symmetric && (raw_ix[i] != raw_iy[i])) {
         line_data_array[raw_ix[i]][count_array[raw_ix[i]]] = raw_data[i];
         line_x_index_array[raw_ix[i]][count_array[raw_ix[i]]] = raw_iy[i];
         ++count_array[raw_ix[i]];
      }
   }

   /* The non_zero is now recalculated, as it will be larger if the matrix was symmetric. */
   *(mgs->non_zero) = 0;
   for (i=0; i<*(mgs->ny); ++i) {
      *(mgs->non_zero) += count_array[i];
   }
   double density = ((double) *(mgs->non_zero)) / ((double) *(mgs->nx) * (double) *(mgs->ny));
   printf("nx = %d, ny = %d, non_zero = %d, density = %f\n", *(mgs->nx), *(mgs->ny), *(mgs->non_zero), density);

   *(mgs->nyround) = (*(mgs->ny) + (preferred_alignment_by_elements - 1)) & (~(preferred_alignment_by_elements - 1));

   if (*(mgs->nyround) < preferred_alignment_by_elements) *(mgs->nyround) = preferred_alignment_by_elements;

   /* now that we know the size, we can prevent excessive segmentation of small matrices */
   unsigned int min_compute_units = (*(mgs->nyround) + preferred_alignment_by_elements - 1) / preferred_alignment_by_elements;
   if (*(mgs->max_compute_units) > min_compute_units) *(mgs->max_compute_units) = min_compute_units;

   /* Release no-longer-needed arrays. */
   free(raw_ix);
   free(raw_iy);
   free(raw_data);

   /* =============================================================== */
   /* Create and load the actual CSR arrays.                          */
   /* =============================================================== */

   MEMORY_ALLOC_CHECK(*(mgs->data_array), (*(mgs->non_zero) * sizeof (float)), "data_array") 

   MEMORY_ALLOC_CHECK(*(mgs->x_index_array), ((*(mgs->non_zero)+1) * sizeof (int)), "x_index_array") 

   MEMORY_ALLOC_CHECK(*(mgs->row_index_array), ((*(mgs->nyround)+1) * sizeof (int)), "row_index_array") 

   unsigned int index = 0;

   for (i=0; i<*(mgs->ny); ++i) {
      (*(mgs->row_index_array))[i] = index;
      for (j=0; j<count_array[i]; ++j) {
         (*(mgs->data_array))[index] = line_data_array[i][j];
         (*(mgs->x_index_array))[index] = line_x_index_array[i][j];
         ++index;
      }
   }
   for (i=*(mgs->ny); i<=*(mgs->nyround); ++i) {
      (*(mgs->row_index_array))[i] = *(mgs->non_zero);
   }

   for (i=0; i<*(mgs->ny); ++i) {
      if (count_array[i]) {
         free(line_data_array[i]);
         free(line_x_index_array[i]);
      }
   }
   free(line_data_array);
   free(line_x_index_array);
   free(count_array);

   /* ============================================================================= */
   /* Now that we have the CSR format of the matrix (in "row_index_array",          */
   /* "x_index_array", and "data_array", we begin to compute the best size and      */
   /* shape for the tiles of the final Tiled format of the matrix.                  */
   /* ============================================================================= */

   unsigned int nslabs_base, target_workpacket, candidate_row, target_value, slabsize;
   unsigned int slab_threshhold;

   /* Decide how big the tiles should be, in the X direction.   */
   /* This decision is driven by three factors:                 */
   /* (1) the tile width should not exceed the matrix width     */
   /* (2) the tile width should not exceed 65536 (16 bit index) */
   /* (3) the tile width should not overwhelm local memory.     */
   /* The variable "column_span" holds this tile width.         */

   if ((mgs->kernel_type) == KERNEL_AWGC) {
      *(mgs->column_span) = (mgs->local_mem_size) / 64;
   }
   if ((mgs->kernel_type) == KERNEL_LS) {
      *(mgs->column_span) = 65536;
   }
   if (*(mgs->column_span) > *(mgs->nx)) {
      *(mgs->column_span) = *(mgs->nx);
   }
   if (*(mgs->column_span) > 65536) {
      *(mgs->column_span) = 65536;
   }
   while (*(mgs->column_span) & (*(mgs->column_span)-1)) {
      ++*(mgs->column_span); /* Raise up to a power of 2. */
   }
   *(mgs->nx_pad)  = (*(mgs->nx) + (*(mgs->column_span) - 1)) & (~(*(mgs->column_span) - 1));

   /* Decide how big the tiles should be, in the Y direction, based on local memory considerations. */
   /* While "column_span" is fixed for all tiles, the tile size in the Y direction can vary.        */
   /* Each "slab" of data should be thought as a horizontal row of tiles. The variable              */
   /* "slab_threshhold" holds the largest height that will be permitted for any such slab.          */

   nslabs_base = *(mgs->max_compute_units);
   unsigned int nslabs = 0;

   if ((mgs->kernel_type) == KERNEL_AWGC) {
      slab_threshhold = ((7 * ((mgs->local_mem_size)/sizeof(float))) / 16) - 1;
      slab_threshhold &= ~(preferred_alignment_by_elements - 1);
      unsigned int expected_nslabs = *(mgs->nyround) / slab_threshhold;
      if (expected_nslabs < nslabs_base) {
         expected_nslabs = nslabs_base;
      }
      target_workpacket = *(mgs->non_zero) / expected_nslabs;
      /* Decide how big the local cache for packet data should be, based on local memory considerations. */
      /* (Typically we will read in 16 or 32 packets at a time.)                                          */
      *(mgs->segcachesize) = (mgs->local_mem_size) / 8192;
      while (*(mgs->segcachesize) & (*(mgs->segcachesize)-1)) {
         ++(*(mgs->segcachesize)); /* raise up to a power of 2 */
      }

      /* Scan matrix data to find best split of data for each contiguous group of rows ("slabs"). */
      candidate_row = 0;
      target_value = target_workpacket;
      slabsize = 0;
      while (candidate_row < *(mgs->nyround)) {
         while ((*(mgs->row_index_array))[candidate_row] < target_value && (slabsize+preferred_alignment_by_elements) < slab_threshhold && candidate_row < *(mgs->nyround)) {
            candidate_row += preferred_alignment_by_elements;
            slabsize += preferred_alignment_by_elements;
         }
         ++nslabs;
         slabsize = 0;
         target_value = (*(mgs->row_index_array))[candidate_row] + target_workpacket;
      }
   
      /* Allocate an array to hold row index of beginning of each of these "slabs". */
      MEMORY_ALLOC_CHECK(*(mgs->slab_startrow), ((nslabs + 1) * sizeof (unsigned int)), "slab_startrow") 
      (*(mgs->slab_startrow))[0] = 0;
      (*(mgs->slab_startrow))[nslabs] = *(mgs->nyround);
      candidate_row = 0;
      target_value = target_workpacket;
      slabsize = 0;
      nslabs = 0;

      /* Scan matrix data to implement previously computed split of data for each contiguous group of rows. */
      while (candidate_row < *(mgs->nyround)) {
         while ((*(mgs->row_index_array))[candidate_row] < target_value && slabsize < slab_threshhold && candidate_row < *(mgs->nyround)) {
            candidate_row += preferred_alignment_by_elements;
            slabsize += preferred_alignment_by_elements;
         }
         ++nslabs;
         slabsize = 0;
         (*(mgs->slab_startrow))[nslabs] = candidate_row;
         target_value = (*(mgs->row_index_array))[candidate_row] + target_workpacket;
      }
   
      *(mgs->max_slabheight) = 0;
      for (i=0; i<nslabs; ++i) {
         if ((*(mgs->slab_startrow))[i+1] - (*(mgs->slab_startrow))[i] > *(mgs->max_slabheight)) {
            *(mgs->max_slabheight) = (*(mgs->slab_startrow))[i+1] - (*(mgs->slab_startrow))[i];
         }
      }
   }
   else {
      if ((mgs->device_type) == CL_DEVICE_TYPE_GPU) {
         if (*(mgs->gpu_wgsz) > MAX_WGSZ) {
            printf("coercing gpu work group size to MAX WORK GROUP SIZE, which is %d\n", MAX_WGSZ);
            *(mgs->gpu_wgsz) = MAX_WGSZ;
         }
         if (*(mgs->gpu_wgsz) < 16) {
            printf("coercing gpu work group size to MIN WORK GROUP SIZE, which is 16\n");
            *(mgs->gpu_wgsz) = 16;
         }
         if (*(mgs->gpu_wgsz) & (*(mgs->gpu_wgsz)-1)) {
            while (*(mgs->gpu_wgsz) & (*(mgs->gpu_wgsz)-1)) --(*(mgs->gpu_wgsz));
            printf("coersing gpu work group size to next lower power of 2, which is %d\n", *(mgs->gpu_wgsz));
         }
         if (*(mgs->gpu_wgsz) > (int) (mgs->kernel_wg_size)) {
            while (*(mgs->gpu_wgsz) > (int) (mgs->kernel_wg_size)) {
               *(mgs->gpu_wgsz) /= 2;
            }
            printf("coercing gpu work group size to fit within hardware limits.  New size is %d\n", *(mgs->gpu_wgsz));
         }
   
         nslabs = (*(mgs->nyround) + *(mgs->gpu_wgsz) - 1) / *(mgs->gpu_wgsz);
         while (nslabs < *(mgs->max_compute_units)) {
            *(mgs->gpu_wgsz) /= 2;
            nslabs = (*(mgs->nyround) + *(mgs->gpu_wgsz) - 1) / *(mgs->gpu_wgsz);
         }
         MEMORY_ALLOC_CHECK(*(mgs->slab_startrow), ((nslabs + 1) * sizeof (unsigned int)), "(mgs->slab_startrow)") 
         for (i=0; i<nslabs; ++i) {
            (*(mgs->slab_startrow))[i] = *(mgs->gpu_wgsz) * i;
         }
         (*(mgs->slab_startrow))[nslabs] = *(mgs->nyround);
         *(mgs->max_slabheight) = *(mgs->gpu_wgsz);
      }
      else {
         nslabs = *(mgs->max_compute_units);
         while (*(mgs->nyround) / nslabs >= ((mgs->local_mem_size)/sizeof(float))) nslabs *= 2;
         MEMORY_ALLOC_CHECK((*(mgs->slab_startrow)), ((nslabs + 1) * sizeof (unsigned int)), "(mgs->slab_startrow)") 
         for (i=0; i<=nslabs; ++i) {
            (*(mgs->slab_startrow))[i] = (((*(mgs->nyround)/preferred_alignment_by_elements) * i) / nslabs) * preferred_alignment_by_elements;
         }
         *(mgs->max_slabheight) = 0;
         for (i=0; i<nslabs; ++i) {
            unsigned int temp = (*(mgs->slab_startrow))[i+1] - (*(mgs->slab_startrow))[i];
            if (*(mgs->max_slabheight) < temp) *(mgs->max_slabheight) = temp;
         }
      }
   }

   /* ============================================================================= */
   /* Now that we have computed the size and shape for our tiles, we can allocate   */
   /* space for working storage to hold the data in an intermediate format, as we   */
   /* move towards the final Tiled format.                                          */
   /* ============================================================================= */

   unsigned int biggest_slab = 0;
   unsigned int smallest_slab = 0x7fffffff;
   unsigned int totpackets = 0;
   unsigned int totslabs = 0;

   unsigned int *row_start, *row_curr;
   MEMORY_ALLOC_CHECK(row_start, (4*(*(mgs->max_slabheight)+1)), "row_start") 
   MEMORY_ALLOC_CHECK(row_curr, (4*(*(mgs->max_slabheight))), "row_curr") 
   unsigned int realdata = 0;
   unsigned int totaldata = 0;
   unsigned int current_slab;
   current_slab = 0;

   packet *slab_ptr;

   /* =============================================================== */
   /* Now we create the Tiled Format of the matrix.                   */
   /* The "seg_workspace" array holds the starting point for each     */
   /* device's header data, and linear list of packets.              */
   /* =============================================================== */

   /* each header packet holds information for 512 threads */
   *(mgs->num_header_packets) = (((mgs->kernel_type) == KERNEL_AWGC) || ((mgs->device_type) != CL_DEVICE_TYPE_GPU)) ? 0 : (MAX_WGSZ+511)/512;
   
   /* This large loop does the bulk of the hard work to load the data into the packets. */
   int seg_index;
   unsigned int k;
   /* The size of the array is admittedly derived using heuristics, but has been found satisfactory   */
   /* for all matrices that have been run through this program during development.                    */
   unsigned int temp_count;
   temp_count = (*(mgs->non_zero) > 16) ? *(mgs->non_zero) : 16;
   MEMORY_ALLOC_CHECK(*(mgs->seg_workspace), (temp_count/2) * sizeof(packet), "*seg_workspace") 
   for (i = 0; i < temp_count>>1; ++i) {
      for (j=0; j<16; ++j) { /* Pre-load input and output indices with flag saying "no data here". */
         (*(mgs->seg_workspace))[i].input_offset_short[j] = (cl_ushort) 0;
         (*(mgs->seg_workspace))[i].matdata[j] = 0.0f;
      }
   }
   /* The entire matrix is split across the multiple devices, and as such, */
   /* We need to know, for each device, where do the slabs start and stop. */
   *(mgs->nslabs_round) = nslabs;
   /* The variable "memsize" is a count of how much of the "seg_workspace" array has been loaded with data. */
   *(mgs->memsize) = 3 * 4 * ((*(mgs->nslabs_round))+1);
   *(mgs->memsize) += sizeof(packet);
   *(mgs->memsize) /= sizeof(packet);
   *(mgs->memsize) *= sizeof(packet);

   /* The majority of the tiled matrix format is composed of packets, but the first bytes are header information. */
   /* Use this temporary "matrix_header" variable to load that data. */
   *(mgs->matrix_header) = (slab_header *) (*(mgs->seg_workspace));
   slab_ptr = &(*(mgs->seg_workspace))[(*(mgs->memsize))/sizeof(packet)];
   seg_index = 0;
   int acctg_maxcount = 0;
   float acctg_avgcount = 0.0f;
   for (i=0; i<*(mgs->nslabs_round); ++i) {
      uint nteams = *(mgs->gpu_wgsz)/16;
      /* Load the header data into "seg_workspace" via the "matrix_header" proxy variable. */
      (*(mgs->matrix_header))[current_slab].offset = (*(mgs->memsize)) / sizeof(packet);
      (*(mgs->matrix_header))[current_slab].outindex = (*(mgs->slab_startrow))[i]-(*(mgs->slab_startrow))[0];
      (*(mgs->matrix_header))[current_slab].outspan = (*(mgs->slab_startrow))[i+1]-(*(mgs->slab_startrow))[i];
      if ((*(mgs->row_index_array))[(*(mgs->slab_startrow))[i]] == (*(mgs->row_index_array))[(*(mgs->slab_startrow))[i+1]]) {
         /* set up structure of two packets to record "no work to do in this slab" */
         unsigned int jloop;
         /* if we're using header packets, then use them.  Otherwise just zero out one packet */
         jloop = (*(mgs->num_header_packets) > 0) ? *(mgs->num_header_packets) : 1;
         for (j=0; j<jloop; ++j) {
            int *foo;
            foo = (int *) &slab_ptr[seg_index];
            for (k=0; k<sizeof(packet)/sizeof(int); ++k) {
               foo[k] = 0;
            }
            ++seg_index;
            (*(mgs->memsize)) += sizeof(packet);
         }
      }
      else {
         /* Here we start actually loading packet data. */
         for (j=0; j<=(*(mgs->slab_startrow))[i+1]-(*(mgs->slab_startrow))[i]; ++j) {
            row_start[j] = (*(mgs->row_index_array))[(*(mgs->slab_startrow))[i]+j];
         }
         if (((mgs->device_type) != CL_DEVICE_TYPE_GPU) || ((mgs->kernel_type) == KERNEL_AWGC)) {
            for (j=0; j<*(mgs->nx_pad); j+= *(mgs->column_span)) {
               unsigned int kk;
               for (k=0; k<(*(mgs->slab_startrow))[i+1] - (*(mgs->slab_startrow))[i]; k+= 16) {
                  unsigned int count[16];
                  for (kk = 0; kk<16; ++kk) {
                     count[kk] = 0;
                     row_curr[k+kk] = row_start[k+kk];
                     while (((*(mgs->x_index_array))[row_curr[k+kk]] < (j+*(mgs->column_span))) && row_curr[k+kk] < (*(mgs->row_index_array))[(*(mgs->slab_startrow))[i] + k + kk +1]) {
                        ++row_curr[k+kk];
                        ++count[kk];
                     }
                  }
                  unsigned int maxcount = 0;
                  for (kk=0; kk<16; ++kk) {
                     if (count[kk] > maxcount) {
                        maxcount = count[kk];
                     }
                  }
                  unsigned int sum = 0;
                  for (kk=0; kk<16; ++kk) {
                     sum += count[kk];
                  }
                  realdata += sum;
                  totaldata += 16 * maxcount;
                  unsigned int countdex;
                  for (countdex = 0; countdex < maxcount; ++countdex) {
                     slab_ptr[seg_index].seg_input_offset = j;
                     slab_ptr[seg_index].seg_output_offset = k;
                     for (kk=0; kk<16; ++kk) {
                        if (countdex < count[kk]) {
                           slab_ptr[seg_index].input_offset_short[kk] = 
                               (unsigned short) ((*(mgs->x_index_array))[row_start[k+kk]+countdex] & (*(mgs->column_span)-1));
                           slab_ptr[seg_index].matdata[kk] = (*(mgs->data_array))[row_start[k+kk]+countdex];
                        }
                     }
                     ++seg_index;
                     *(mgs->memsize) += sizeof(packet);
                  }
                  for (kk = 0; kk<16; ++kk) {
                     row_start[k+kk] = row_curr[k+kk];
                  }
               }
            }
         }
         else {
            int *first_team_offset;
            first_team_offset = (int *) &slab_ptr[seg_index];
            for (j=0; j<*(mgs->num_header_packets); ++ j) {
               ++seg_index;
               *(mgs->memsize) += sizeof(packet);
            }
            int packet_offset = 0;
            for (k=0; k<(*(mgs->slab_startrow))[i+1] - (*(mgs->slab_startrow))[i]; k+= 16) {
               int packet_count = 0;
               for (j=0; j<*(mgs->nx_pad); j+= *(mgs->column_span)) {
                  unsigned int kk;
                  unsigned int count[16];
                  for (kk = 0; kk<16; ++kk) {
                     count[kk] = 0;
                     row_curr[k+kk] = row_start[k+kk];
                     while (((*(mgs->x_index_array))[row_curr[k+kk]] < (j+*(mgs->column_span))) && row_curr[k+kk] < (*(mgs->row_index_array))[(*(mgs->slab_startrow))[i] + k + kk +1]) {
                        ++row_curr[k+kk];
                        ++count[kk];
                     }
                  }
                  unsigned int maxcount = 0;
                  for (kk=0; kk<16; ++kk) {
                     if (count[kk] > maxcount) {
                        maxcount = count[kk];
                     }
                  }
                  unsigned int sum = 0;
                  for (kk=0; kk<16; ++kk) {
                     sum += count[kk];
                  }
                  realdata += sum;
                  totaldata += 16 * maxcount;
                  unsigned int countdex;
                  for (countdex = 0; countdex < maxcount; ++countdex) {
                     slab_ptr[seg_index].seg_input_offset = j;
                     slab_ptr[seg_index].seg_output_offset = k;
                     for (kk=0; kk<16; ++kk) {
                        if (countdex < count[kk]) {
                           slab_ptr[seg_index].input_offset_short[kk] = 
                               (unsigned short) ((*(mgs->x_index_array))[row_start[k+kk]+countdex] & (*(mgs->column_span)-1));
                           slab_ptr[seg_index].matdata[kk] = (*(mgs->data_array))[row_start[k+kk]+countdex];
                        }
                     }
                     ++seg_index;
                     ++packet_count;
                     *(mgs->memsize) += sizeof(packet);
                  }
                  for (kk = 0; kk<16; ++kk) {
                     row_start[k+kk] = row_curr[k+kk];
                  }
               }
               if ((packet_offset > 65535) || (packet_count > 65535)) {
                  printf("eek!\n");
                  return(-1);
               }
               first_team_offset[k>>4] = packet_offset * 65536 + packet_count;
               packet_offset += packet_count;
            }
            for (k = (*(mgs->slab_startrow))[i+1] - (*(mgs->slab_startrow))[i]; k < 16*nteams; k+= 16) {
               first_team_offset[k>>4] = 0;
            }
            int tempmaxcount = 0;
            int tempavgcount = 0;
            for (k=0; k<nteams; ++k) {
               int tempcount;
               tempcount = first_team_offset[k] % 65536;
               tempavgcount += tempcount;
               if (tempcount > tempmaxcount) tempmaxcount = tempcount;
            }
            acctg_avgcount += ((float) tempavgcount) / nteams;
            acctg_maxcount += tempmaxcount;
         }
         /* With on exception, all actual packet data is now loaded into the "seg_workspace" array, for this device. */
      }
      ++current_slab;
   }

   free(row_start);
   free(row_curr);

   (*(mgs->matrix_header))[current_slab].offset = (*(mgs->memsize))/sizeof(packet);
   (*(mgs->matrix_header))[current_slab].outindex = (*(mgs->slab_startrow))[*(mgs->nslabs_round)]-(*(mgs->slab_startrow))[0];
   (*(mgs->matrix_header))[current_slab].outspan = 0;

   /* This loop records some statistics, and sets one final value into the packets. */
   for (i=0; i<*(mgs->nslabs_round); ++i) {
      unsigned int npackets = (*(mgs->matrix_header))[i+1].offset - (*(mgs->matrix_header))[i].offset;
      if (npackets < smallest_slab) {
         smallest_slab = npackets;
      }
      if (npackets > biggest_slab) {
         biggest_slab = npackets;
      }
      totpackets += npackets;
      ++totslabs;
      /* load the first "real" packet of data with the count of how many such packets there are in this slab */
      (*(mgs->seg_workspace))[(*(mgs->matrix_header))[i].offset+(*(mgs->num_header_packets))].npackets_remaining = npackets-(*(mgs->num_header_packets));
   }
   (*(mgs->memsize)) += 32*sizeof(packet); /* Add room for reading past end of data, so we don't abnormally terminate. */

   for (i=0; i<*(mgs->nslabs_round); ++i) {
      /* For each row of tiles, we now start at the end, and work backward, to load one last datum into each packet. */
      slab_ptr = &(*(mgs->seg_workspace))[(*(mgs->matrix_header))[i].offset];
      seg_index = (*(mgs->matrix_header))[(i+1)].offset - (*(mgs->matrix_header))[i].offset;
      --seg_index; // back up into set of packets
      unsigned int curr_input_offset, next_input_offset;
      curr_input_offset = slab_ptr[seg_index].seg_input_offset;
      next_input_offset = 0;
      while (seg_index >= (int) (*(mgs->num_header_packets))) {
         if (slab_ptr[seg_index].seg_input_offset < curr_input_offset) {
            next_input_offset = curr_input_offset;
            curr_input_offset = slab_ptr[seg_index].seg_input_offset;
         }
         /* Here is the "exception".  We now load the "input offset for a future tile" into the data, for the benefit of the double-buffered AWGC kernel. */
         slab_ptr[seg_index].future_seg_input_offset = next_input_offset; 
         --seg_index;
      }
   }

   return 0;
}
