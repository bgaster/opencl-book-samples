#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

//
// sum partial histogram results into final histogram bins
//
// num_groups is the number of work-groups used to compute partial histograms.
// partial_histogram is an array of num_groups * (257 * 3 * 32-bits/entry) entries
// we store 257 Red bins, followed by 257 Green bins and then the 257 Blue bins.
//
// final summed results are returned in histogram.
//
kernel
void histogram_sum_partial_results_fp(global uint *partial_histogram, int num_groups, global uint *histogram)
{
    int     tid = (int)get_global_id(0);
    int     group_id = (int)get_group_id(0);
    int     group_indx;
    int     n = num_groups;
    uint    tmp_histogram, tmp_histogram_first;

    int     first_workitem_not_in_first_group = ((get_local_id(0) == 0) && group_id);
    
    tid += group_id;
    int     tid_first = tid - 1;
    if (first_workitem_not_in_first_group)
        tmp_histogram_first = partial_histogram[tid_first];
    
    tmp_histogram = partial_histogram[tid];
    
    group_indx = 257*3;
    while (--n > 0)
    {
        if (first_workitem_not_in_first_group)
            tmp_histogram_first += partial_histogram[tid_first];
            
        tmp_histogram += partial_histogram[group_indx+tid];
        group_indx += 257*3;
    }
    
    if (first_workitem_not_in_first_group)
        histogram[tid_first] = tmp_histogram_first;
    histogram[tid] = tmp_histogram;
}


//
// this kernel takes a RGBA 32-bit or 16-bit FP / channel input image and produces a partial histogram.
// the kernel is executed over multiple work-groups.  for each work-group a partial histogram is generated
// partial_histogram is an array of num_groups * (257 * 3 * 32-bits/entry) entries
// we store 257 Red bins, followed by 257 Green bins and then the 257 Blue bins.
//
kernel
void histogram_image_rgba_fp(image2d_t img, int num_pixels_per_workitem, global uint *histogram)
{
    int     local_size = (int)get_local_size(0) * (int)get_local_size(1);
    int     image_width = get_image_width(img);
    int     image_height = get_image_height(img);
    int     group_indx = mad24(get_group_id(1), get_num_groups(0), get_group_id(0)) * 257 * 3;
    int     x = get_global_id(0);
    int     y = get_global_id(1);
    
    local uint  tmp_histogram[257 * 3];
        
    int     tid = mad24(get_local_id(1), get_local_size(0), get_local_id(0));
    int     j = 257 * 3;
    int     indx = 0;
    
    // clear the local buffer that will generate the partial histogram
    do
    {
        if (tid < j)
            tmp_histogram[indx+tid] = 0;

        j -= local_size;
        indx += local_size;
    } while (j > 0);
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int     i, idx;
    for (i=0, idx=x; i<num_pixels_per_workitem; i++, idx+=get_global_size(0))
    {
        if ((idx < image_width) && (y < image_height))
        {                
            float4 clr = read_imagef(img, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, (float2)(idx, y));
    
            ushort   indx;
            indx = convert_ushort_sat(min(clr.x, 1.0f) * 256.0f);
            atom_inc(&tmp_histogram[indx]);

            indx = convert_ushort_sat(min(clr.y, 1.0f) * 256.0f);
            atom_inc(&tmp_histogram[257+indx]);

            indx = convert_ushort_sat(min(clr.z, 1.0f) * 256.0f);
            atom_inc(&tmp_histogram[514+indx]);
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    // copy the partial histogram to appropriate location in histogram given by group_indx
    if (local_size >= (257 * 3))
    {
        if (tid < (257 * 3))
            histogram[group_indx + tid] = tmp_histogram[tid];
    }
    else
    {
        j = 257 * 3;
        indx = 0;
        do 
        {
            if (tid < j)
                histogram[group_indx + indx + tid] = tmp_histogram[indx + tid];
                
            j -= local_size;
            indx += local_size;
        } while (j > 0);
    }
}


/***************************************************************************************************************/

//
// sum partial histogram results into final histogram bins
//
// num_groups is the number of work-groups used to compute partial histograms.
// partial_histogram is an array of num_groups * (256 * 3 * 32-bits/entry) entries
// we store 256 Red bins, followed by 256 Green bins and then the 256 Blue bins.
//
// final summed results are returned in histogram.
//
kernel
void histogram_sum_partial_results_unorm8(global uint *partial_histogram, int num_groups, global uint *histogram)
{
    int     tid = (int)get_global_id(0);
    int     group_indx;
    int     n = num_groups;
    uint    tmp_histogram;

    tmp_histogram = partial_histogram[tid];
    
    group_indx = 256*3;
    while (--n > 0)
    {
        tmp_histogram += partial_histogram[group_indx + tid];
        group_indx += 256*3;
    }
    
    histogram[tid] = tmp_histogram;
}

//
// this kernel takes a RGBA 8-bit / channel input image and produces a partial histogram.
// the kernel is executed over multiple work-groups.  for each work-group a partial histogram is generated
// partial_histogram is an array of num_groups * (256 * 3 * 32-bits/entry) entries
// we store 256 Red bins, followed by 256 Green bins and then the 256 Blue bins.
//
kernel
void histogram_image_rgba_unorm8(image2d_t img, int num_pixels_per_workitem, global uint *histogram)
{
    int     local_size = (int)get_local_size(0) * (int)get_local_size(1);
    int     image_width = get_image_width(img);
    int     image_height = get_image_height(img);
    int     group_indx = mad24(get_group_id(1), get_num_groups(0), get_group_id(0)) * 256 * 3;
    int     x = get_global_id(0);
    int     y = get_global_id(1);
    
    local uint  tmp_histogram[256 * 3];
        
    int     tid = mad24(get_local_id(1), get_local_size(0), get_local_id(0));
    int     j = 256 * 3;
    int     indx = 0;
    
    // clear the local buffer that will generate the partial histogram
    do
    {
        if (tid < j)
            tmp_histogram[indx+tid] = 0;

        j -= local_size;
        indx += local_size;
    } while (j > 0);
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int     i, idx;
    for (i=0, idx=x; i<num_pixels_per_workitem; i++, idx+=get_global_size(0))
    {
        if ((idx < image_width) && (y < image_height))
        {                
            float4 clr = read_imagef(img, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST, (float2)(idx, y));
    
            uchar   indx_x, indx_y, indx_z;
            indx_x = convert_uchar_sat(clr.x * 255.0f);
            indx_y = convert_uchar_sat(clr.y * 255.0f);
            indx_z = convert_uchar_sat(clr.z * 255.0f);
            atom_inc(&tmp_histogram[indx_x]);
            atom_inc(&tmp_histogram[256+(uint)indx_y]);
            atom_inc(&tmp_histogram[512+(uint)indx_z]);
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    // copy the partial histogram to appropriate location in histogram given by group_indx
    if (local_size >= (256 * 3))
    {
        if (tid < (256 * 3))
            histogram[group_indx + tid] = tmp_histogram[tid];
    }
    else
    {
        j = 256 * 3;
        indx = 0;
        do 
        {
            if (tid < j)
                histogram[group_indx + indx + tid] = tmp_histogram[indx + tid];
                
            j -= local_size;
            indx += local_size;
        } while (j > 0);
    }
}




