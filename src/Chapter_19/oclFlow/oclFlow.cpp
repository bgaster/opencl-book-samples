/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
#include <CL/cl.h>
#include <stdio.h>
#include <shrUtils.h>
#include <oclUtils.h>
#include <iostream>
#include <GL/gl.h>
#ifdef LINUX
#include <GL/glx.h>
#endif

#ifdef MAC
#define SINGLE_CHANNEL_TYPE CL_R
#define GL_SHARING_EXTENSION "cl_APPLE_gl_sharing"
#else
#define SINGLE_CHANNEL_TYPE CL_INTENSITY
#endif 


bool writeImages=false;

static cl_mem cl_imagePacked, cl_imageFull, cl_imageOrig;
static cl_command_queue command_queue;
static cl_context context;
static cl_device_id cdDevice;
char device_string[1024];

void checkErr( cl_int err,int line, const char *n,  bool verbosity=false ) {
  if( err != CL_SUCCESS ) {
	  std::cerr << n << "\r\t\t\t\t\t\tline:" << line<<" "<<oclErrorString(err) << std::endl;
      assert(0);
  } 
  else if( n != NULL ) {
      if( verbosity) std::cerr << n << "\r\t\t\t\t\t\t" << "OK" <<std::endl;

  }
}

double elapsedTimeInSeconds( cl_event event )
{
    cl_int err;
    cl_ulong start = 0;
    cl_ulong end = 0;
    size_t retSz;
    double t = 0;

    err = clGetEventProfilingInfo( event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, &retSz);
    checkErr( err, __LINE__, "clGetEventProfilingInfo" );

    err = clGetEventProfilingInfo( event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, &retSz );
    checkErr( err, __LINE__, "clGetEventProfilingInfo" );

    // end & start are in nanoseconds
    t = ((double)end-(double)start)*(1.0e-9);

    return t;
}

// Helper to get next up value for integer division
static inline size_t DivUp(size_t dividend, size_t divisor)
{
    return (dividend % divisor == 0) ? (dividend / divisor) : (dividend / divisor + 1);
}


int opencl_init(int devId) {

    // Get OpenCL platform ID for NVIDIA if avaiable, otherwise default
    shrLog("OpenCL SW Info:\n\n");
    char cBuffer[1024];
    cl_platform_id clSelectedPlatformID = NULL; 
    cl_int ciErrNum = oclGetPlatformID (&clSelectedPlatformID);
    oclCheckError(ciErrNum, CL_SUCCESS);

    // Get OpenCL platform name and version
    ciErrNum = clGetPlatformInfo (clSelectedPlatformID, CL_PLATFORM_NAME, sizeof(cBuffer), cBuffer, NULL);
    if (ciErrNum == CL_SUCCESS)
    {
        shrLog(" CL_PLATFORM_NAME: \t%s\n", cBuffer);
    } 
    else
    {
        shrLog(" Error %i in clGetPlatformInfo Call !!!\n\n", ciErrNum);
    }

    ciErrNum = clGetPlatformInfo (clSelectedPlatformID, CL_PLATFORM_VERSION, sizeof(cBuffer), cBuffer, NULL);
    if (ciErrNum == CL_SUCCESS)
    {
        shrLog(" CL_PLATFORM_VERSION: \t%s\n", cBuffer);
    } 
    else
    {
        shrLog(" Error %i in clGetPlatformInfo Call !!!\n\n", ciErrNum);
    }

    // Log OpenCL SDK Revision # 
    shrLog(" OpenCL SDK Revision: \t%s\n\n\n", OCL_SDKREVISION);

    // Get and log OpenCL device info 
    cl_uint ciDeviceCount;
    cl_device_id *devices;
    shrLog("OpenCL Device Info:\n\n");
    ciErrNum = clGetDeviceIDs (clSelectedPlatformID, CL_DEVICE_TYPE_ALL, 0, NULL, &ciDeviceCount);

    // check for 0 devices found or errors... 
    if (ciDeviceCount == 0)
    {
        shrLog(" No devices found supporting OpenCL (return code %i)\n\n", ciErrNum);
    } 
    else if (ciErrNum != CL_SUCCESS)
    {
        shrLog(" Error %i in clGetDeviceIDs call !!!\n\n", ciErrNum);
    }
    else
	{
		// Get and log the OpenCL device ID's
		char cTemp[2];
		sprintf(cTemp, "%u", ciDeviceCount);
		if ((devices = (cl_device_id*)malloc(sizeof(cl_device_id) * ciDeviceCount)) == NULL)
		{
			shrLog(" Failed to allocate memory for devices !!!\n\n");
		}
		ciErrNum = clGetDeviceIDs (clSelectedPlatformID, CL_DEVICE_TYPE_ALL, ciDeviceCount, devices, &ciDeviceCount);
		if (ciErrNum == CL_SUCCESS)
		{
			//Create a context for the devices
			cl_context_properties props[] = { 
#ifdef _WIN32
				CL_CONTEXT_PLATFORM, (cl_context_properties)clSelectedPlatformID, 
				CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(), 
				CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(), 
#else
				CL_CONTEXT_PLATFORM, (cl_context_properties)clSelectedPlatformID, 
				CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(), 
				CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(), 
#endif
				0}; 

			if( ciDeviceCount > 1 ) {
				ciDeviceCount = 1;
				shrLog("Note: Multiple device found, but creating context for first device only.\n");
			}
            shrLog("Creating context on device %d\n", devId );
			context = clCreateContext(props, ciDeviceCount, &devices[devId], NULL, NULL, &ciErrNum);
			if (ciErrNum != CL_SUCCESS)
			{
				shrLog("Error %i in clCreateContext call !!!\n\n", ciErrNum);
			}
		}
	}

	cdDevice = devices[devId];

  // Create a command-queue on the GPU device
  // enable profiling by sending the CL_QUEUE_PROFILING_ENABLE flag
  command_queue = clCreateCommandQueue(context, cdDevice, CL_QUEUE_PROFILING_ENABLE, &ciErrNum);
  checkErr(ciErrNum, __LINE__, "clCreateCommandQueue");

  cl_bool img_support = false;
  ciErrNum = clGetDeviceInfo(cdDevice, CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool), &img_support, NULL);
    checkErr(ciErrNum, __LINE__, "clGetDeviceInfo");
    if( img_support ) {
        printf("CL_DEVICE_IMAGE_SUPPORT Found.\n");
    } else {    
        printf("CL_DEVICE_IMAGE_SUPPORT **Missing**\n");
    }   
	size_t n;
	ciErrNum = clGetDeviceInfo( cdDevice, CL_DEVICE_NAME, 1024, device_string, &n );

    size_t retsz;
    cl_device_type dtype;
    ciErrNum = clGetDeviceInfo(cdDevice, CL_DEVICE_TYPE, sizeof(cl_device_type),&dtype, &retsz);
    checkErr(ciErrNum, __LINE__,"clGetDeviceInfo");
    assert( dtype == CL_DEVICE_TYPE_GPU );
    printf("type is GPU\n");


  return 0;
}


// build a program from a file, uses global device, and given context
cl_program buildProgramFromFile(cl_context clctx, const char *source_path) 
{
  cl_int err;
  // Buffer to hold source for compilation 
  size_t program_length;
  //char source_path[] = "lkflow.cl";
  const char *source = oclLoadProgSource(source_path, "", &program_length );
  if( source == NULL ) {
    fprintf(stderr, "Missing source file %s?\n", source_path);
  }

  // Create OpenCL program with source code
  cl_program program = clCreateProgramWithSource(clctx, 1, &source, NULL, &err);
  checkErr(err, __LINE__,"clCreateProgramWithSource");

  // Build the program (OpenCL JIT compilation)
  std::cerr<<"Calling clBuildProgram..."<<std::endl;
  err = clBuildProgram(program, 0, NULL, "-cl-nv-verbose -cl-mad-enable -cl-fast-relaxed-math", NULL, NULL);
  std::cerr<<"OK"<<std::endl;

  cl_build_status build_status;
  err = clGetProgramBuildInfo(program, cdDevice,
    CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &build_status, NULL);
  checkErr(err, __LINE__, "clGetProgramBuildInfo");

	char *build_log;
	size_t ret_val_size;
	err = clGetProgramBuildInfo(program, cdDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
	checkErr(err,__LINE__, "clGetProgramBuildInfo");
	build_log = (char *)malloc(ret_val_size+1);
	err = clGetProgramBuildInfo(program, cdDevice, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
	checkErr(err, __LINE__,"clGetProgramBuildInfo");

	// to be carefully, terminate with \0
	// there's no information in the reference whether the string is 0 terminated or not
	build_log[ret_val_size] = '\0';

	fprintf(stderr, "%s\n", build_log );


  return program;
}
 
////////////////////////////////////////////////////////////////////////////////
// Functions to write pyramids to images or save them to floating point files
// that can be read in octave for verification
////////////////////////////////////////////////////////////////////////////////
template<cl_channel_order co, cl_channel_type dt>
struct ocl_image {
    cl_mem image_mem;
    unsigned int w;
    unsigned int h;
    cl_image_format image_format;
} ;

struct ocl_buffer { 
    cl_mem mem;
    unsigned int w;
    unsigned int h;
    cl_image_format image_format; // the type of image data this temp buffer reflects
};


void save_image( ocl_image<CL_RGBA, CL_UNSIGNED_INT8> img, cl_command_queue cmdq, const char *fname ) 
{
    unsigned char *h_img_ub  = (unsigned char *)malloc( (img.w*4) * img.h ) ;
    size_t origin[3] = {0,0,0};
    size_t region[3] = {img.w,img.h,1};
    
    cl_int err = clEnqueueReadImage( cmdq, img.image_mem, CL_TRUE,  
        origin, region, img.w*4, 0, h_img_ub, NULL, 0, NULL );
    checkErr( err,__LINE__, "save_image_ub: clEnqeueuReadImage"); 
    shrSavePGMub( fname, h_img_ub, img.w*4, img.h );
    free( h_img_ub );
}


void save_octave( ocl_image<CL_RGBA, CL_UNSIGNED_INT8> img, cl_command_queue cmdq, const char *fname )
{
     unsigned char *h_img_ub  = (unsigned char *)malloc( (img.w*4) * img.h ) ;
    size_t origin[3] = {0,0,0};
    size_t region[3] = {img.w,img.h,1};
    
    cl_int err = clEnqueueReadImage( cmdq, img.image_mem, CL_TRUE,  
        origin, region, img.w*4, 0, h_img_ub, NULL, 0, NULL );
    checkErr( err, __LINE__,"save_octave: clEnqeueuReadImage");  
    FILE *fd = fopen(fname, "w");
    if( fd!=NULL ) {
        for( unsigned int j=0 ; j<img.h ; j++ ) {
            for( unsigned int i=0 ; i<img.w*4 ; i++ ) {
                fprintf(fd, "%d ", h_img_ub[j*img.w*4+i] );
            }
            fprintf(fd, "\n");
        }
        fclose(fd);
        std::cerr<<"Wrote:\r\t\t\t\t\t"<<fname<<std::endl;
    }
    free( h_img_ub );
}

void save_image( ocl_image<SINGLE_CHANNEL_TYPE, CL_UNSIGNED_INT8> img, cl_command_queue cmdq, const char *fname ) 
{
    unsigned char *h_img_ub  = (unsigned char *)malloc( (img.w) * img.h ) ;
    size_t origin[3] = {0,0,0};
    size_t region[3] = {img.w,img.h,1};
    
    cl_int err = clEnqueueReadImage( cmdq, img.image_mem, CL_TRUE,  
        origin, region, img.w, 0, h_img_ub, NULL, 0, NULL );
    checkErr( err,__LINE__, "save_image_ub: clEnqeueuReadImage"); 
    shrSavePGMub( fname, h_img_ub, img.w, img.h );
    free( h_img_ub );
}

void save_image( ocl_buffer buf, cl_command_queue cmdq, const char *fname )
{
    unsigned char *h_buf_ub  = (unsigned char *)malloc( buf.w * buf.h ) ;
    cl_int err = clEnqueueReadBuffer( cmdq, buf.mem, CL_TRUE,  
        0, buf.w*buf.h, h_buf_ub, 0, NULL, NULL );
    checkErr( err, __LINE__, "save_image_ub: clEnqeueuReadBuffer"); 
    shrSavePGMub( fname, h_buf_ub, buf.w, buf.h );
    printf("hbuf: %d %d %d %d %d\n", h_buf_ub[0], h_buf_ub[1], h_buf_ub[2], h_buf_ub[3], h_buf_ub[4] );
    free( h_buf_ub );
    std::cerr<<"Wrotebuffer:\r\t\t\t\t\t"<<fname<<std::endl;
}

void save_octave( ocl_image<SINGLE_CHANNEL_TYPE, CL_UNSIGNED_INT8> img, cl_command_queue cmdq, const char *fname )
{
     unsigned char *h_img_ub  = (unsigned char *)malloc( (img.w) * img.h ) ;
    size_t origin[3] = {0,0,0};
    size_t region[3] = {img.w,img.h,1};
    
    cl_int err = clEnqueueReadImage( cmdq, img.image_mem, CL_TRUE,  
        origin, region, img.w, 0, h_img_ub, NULL, 0, NULL );
    checkErr( err,  __LINE__, "save_octave: clEnqeueuReadImage");  
    FILE *fd = fopen(fname, "w");
    if( fd!=NULL ) {
        for( unsigned int j=0 ; j<img.h ; j++ ) {
            for( unsigned int i=0 ; i<img.w ; i++ ) {
                fprintf(fd, "%d ", h_img_ub[j*img.w+i] );
            }
            fprintf(fd, "\n");
        }
        fclose(fd);
        std::cerr<<"Wrote:\r\t\t\t\t\t"<<fname<<std::endl;
    }
    free( h_img_ub );
}


void save_image( ocl_image<SINGLE_CHANNEL_TYPE, CL_SIGNED_INT16> img, cl_command_queue cmdq, const char *fname ) 
{
    cl_int err = CL_SUCCESS;
    unsigned char *h_img_ub  = (unsigned char *)malloc( (img.w) * img.h ) ;
    cl_short *h_img_short  = (cl_short *)malloc( (img.w) * img.h * sizeof( cl_short)  ) ;

    size_t origin[3] = {0,0,0};
    size_t region[3] = {img.w,img.h,1};
    std::cout<<fname<<" : reading stride: "<<img.w<<"->"<<img.w*sizeof(cl_short)<<std::endl;
    
    err = clEnqueueReadImage( cmdq, img.image_mem, CL_TRUE,  
        origin, region, img.w*sizeof(cl_short), 0, h_img_short, NULL, 0, NULL );

    for( unsigned int i=0 ; i<img.w*img.h ; i++ ) {
        int val = abs(h_img_short[i]);
        h_img_ub[i] = (unsigned char)val>>3;
    }


    checkErr( err, __LINE__, "save_image_int16: clEnqeueuReadImage"); 
    shrSavePGMub( fname, h_img_ub, img.w, img.h );
    free( h_img_ub );
    free( h_img_short );
    std::cerr<<"Done saving:\r\t\t\t\t\t\t "<<fname<<std::endl;
}

void save_octave( ocl_image<SINGLE_CHANNEL_TYPE, CL_SIGNED_INT16> img, cl_command_queue cmdq, const char *fname ) 
{
    cl_int err = CL_SUCCESS;
    unsigned char *h_img_ub  = (unsigned char *)malloc( (img.w) * img.h ) ;
    cl_short *h_img_short  = (cl_short *)malloc( (img.w) * img.h * sizeof( cl_short) *2 ) ;

    size_t origin[3] = {0,0,0};
    size_t region[3] = {img.w,img.h,1};
    std::cout<<fname<<" : reading stride: "<<img.w<<"->"<<img.w*sizeof(cl_short)<<std::endl;
    
    err = clEnqueueReadImage( cmdq, img.image_mem, CL_TRUE,  
        origin, region, img.w*sizeof(cl_short), 0, h_img_short, NULL, 0, NULL );

    FILE *fd = fopen(fname, "w");
    if( fd != NULL ) {
         for( unsigned int j=0 ; j<img.h ; j++ ) {
            for( unsigned int i=0 ; i<img.w ; i++ ) {
                fprintf(fd, "%d ", h_img_short[j*img.w+i] );
            }
            fprintf(fd, "\n");
        }
        fclose(fd);
        std::cerr<<"Wrote:\r\t\t\t\t\t"<<fname<<std::endl;
    }
    free( h_img_ub );
    free( h_img_short );
}

void save_octave( ocl_image<CL_RGBA, CL_SIGNED_INT32> img, cl_command_queue cmdq, const char *fname ) 
{
    cl_int err = CL_SUCCESS;
    cl_int *h_img_int  = (cl_int *)malloc( (img.w) * img.h * sizeof( cl_int) *4 ) ;

    size_t origin[3] = {0,0,0};
    size_t region[3] = {img.w,img.h,1};
    std::cout<<fname<<" : reading stride: "<<img.w<<"->"<<img.w*sizeof(cl_int)*4<<std::endl;
    
    err = clEnqueueReadImage( cmdq, img.image_mem, CL_TRUE,  
        origin, region, img.w*sizeof(cl_int)*4, 0, h_img_int, NULL, 0, NULL );
    checkErr( err,  __LINE__, "save_octave: clEnqeueuReadImage"); 
    FILE *fd = fopen(fname, "w");
    if( fd != NULL ) {
         for( unsigned int j=0 ; j<img.h ; j++ ) {
            for( unsigned int i=0 ; i<img.w*4 ; i++ ) {
                fprintf(fd, "%d ", h_img_int[j*img.w*4+i] );
            }
            fprintf(fd, "\n");
        }
        fclose(fd);
        std::cerr<<"Wrote:\r\t\t\t\t\t"<<fname<<std::endl;
    }
    free( h_img_int );
}

void save_octave( ocl_image<CL_RGBA, CL_FLOAT> img, cl_command_queue cmdq, const char *fname ) 
{
    cl_int err = CL_SUCCESS;
    cl_float *h_img_float  = (cl_float *)malloc( (img.w) * img.h * sizeof( cl_float) *4 ) ;

    size_t origin[3] = {0,0,0};
    size_t region[3] = {img.w,img.h,1};
    std::cout<<fname<<" : reading stride: "<<img.w<<"->"<<img.w*sizeof(cl_float)*4<<std::endl;
    
    err = clEnqueueReadImage( cmdq, img.image_mem, CL_TRUE,  
        origin, region, img.w*sizeof(cl_float)*4, 0, h_img_float, NULL, 0, NULL );
    checkErr( err,__LINE__, "save_octave: clEnqeueuReadImage"); 
    FILE *fd = fopen(fname, "w");
    if( fd != NULL ) {
         for( unsigned int j=0 ; j<img.h ; j++ ) {
            for( unsigned int i=0 ; i<img.w*4 ; i++ ) {
                fprintf(fd, "%f ", h_img_float[j*img.w*4+i] );
            }
            fprintf(fd, "\n");
        }
        fclose(fd);
        std::cerr<<"Wrote:\r\t\t\t\t\t"<<fname<<std::endl;
    }
    free( h_img_float );
}

void save_octave( ocl_image<CL_RG, CL_FLOAT> img, cl_command_queue cmdq, const char *fname ) 
{
    cl_int err = CL_SUCCESS;
    cl_float2       *h_img_float2  = (cl_float2 *)malloc( (img.w) * img.h * sizeof( cl_float2) ) ;

    size_t origin[3] = {0,0,0};
    size_t region[3] = {img.w,img.h,1};

    std::cout<<fname<<" : reading stride: "<<img.w<<"->"<<img.w*sizeof(cl_float2)<<std::endl;
    
    err = clEnqueueReadImage( cmdq, img.image_mem, CL_TRUE,  
        origin, region, img.w*sizeof(cl_float2), 0, h_img_float2, NULL, 0, NULL );
    checkErr( err, __LINE__, "save_octave: clEnqeueuReadImage"); 

    FILE *fd = fopen(fname, "w");
    if( fd != NULL ) {
         for( unsigned int j=0 ; j<img.h ; j++ ) {
            for( unsigned int i=0 ; i<img.w ; i++ ) {
                fprintf(fd, "%f ", h_img_float2[j*img.w+i].s[0]);
                fprintf(fd, "%f ", h_img_float2[j*img.w+i].s[1]);
            }
            fprintf(fd, "\n");
        }
        fclose(fd);
        std::cerr<<"Wrote:\r\t\t\t\t\t"<<fname<<std::endl;
    }

    free( h_img_float2 );

}

void save_octave_float2( ocl_buffer img, cl_command_queue cmdq, const char *fname ) 
{
    cl_int err = CL_SUCCESS;
    cl_float2       *h_img_float2  = (cl_float2 *)malloc( (img.w) * img.h * sizeof( cl_float2) ) ;

    size_t origin[3] = {0,0,0};
    size_t region[3] = {img.w,img.h,1};

    std::cout<<fname<<" : reading stride: "<<img.w<<"->"<<img.w*sizeof(cl_float2)<<std::endl;
    
    err = clEnqueueReadBuffer( cmdq, img.mem, CL_TRUE,  
        0, img.w*img.h*sizeof(cl_float2), h_img_float2, 0, NULL, NULL );
    checkErr( err, __LINE__, "save_octave: clEnqeueuReadBuffer Float2"); 

    FILE *fd = fopen(fname, "w");
    if( fd != NULL ) {
         for( unsigned int j=0 ; j<img.h ; j++ ) {
            for( unsigned int i=0 ; i<img.w ; i++ ) {
                fprintf(fd, "%f ", h_img_float2[j*img.w+i].s[0] );
                fprintf(fd, "%f ", h_img_float2[j*img.w+i].s[1] );
            }
            fprintf(fd, "\n");
        }
        fclose(fd);
        std::cerr<<"Wrote:\r\t\t\t\t\t"<<fname<<std::endl;
    }

    free( h_img_float2 );
}

void query_float2_buffer( ocl_buffer img, cl_command_queue cmdq, int i, int j ) 
{
    cl_int err = CL_SUCCESS;
    cl_float2       *h_img_float2  = (cl_float2 *)malloc( (img.w) * img.h * sizeof( cl_float2) ) ;

    size_t origin[3] = {0,0,0};
    size_t region[3] = {img.w,img.h,1};

    err = clEnqueueReadBuffer( cmdq, img.mem, CL_TRUE,  
        0, img.w*img.h*sizeof(cl_float2), h_img_float2, 0, NULL, NULL );
    checkErr( err, __LINE__, "query_float2_buffer: clEnqeueuReadBuffer Float2"); 

    printf("(%d, %d) motion: ", i,j );
    printf( "%f ", h_img_float2[j*img.w+i].s[0] );
    printf( "%f ", h_img_float2[j*img.w+i].s[1] );
    printf( "\n");

    free( h_img_float2 );
}

void query_float_buffer( ocl_image<CL_RGBA, CL_SIGNED_INT32> img, cl_command_queue cmdq, int i, int j ) 
{
    cl_int err = CL_SUCCESS;
    cl_int       *h_img_int  = (cl_int *)malloc( (img.w) * img.h * sizeof( cl_int) *4 ) ;

    size_t origin[3] = {0,0,0};
    size_t region[3] = {img.w,img.h,1};

   err = clEnqueueReadImage( cmdq, img.image_mem, CL_TRUE,  
        origin, region, img.w*sizeof(cl_int)*4, 0, h_img_int, NULL, 0, NULL );
    checkErr( err, __LINE__, "query_float2_buffer: clEnqeueuReadBuffer Float2"); 

    printf("(%d) : ", i,j );
    printf( "%d ", h_img_int[j*(img.w*4)+i*4] );;
    printf( "\n");

    free( h_img_int );
}

void query_img( ocl_image<SINGLE_CHANNEL_TYPE, CL_SIGNED_INT16> img, cl_command_queue cmdq, int i, int j ) 
{
    cl_int err = CL_SUCCESS;
    cl_short       *h_img_int  = (cl_short *)malloc( (img.w) * img.h * sizeof( cl_short)  ) ;

    size_t origin[3] = {0,0,0};
    size_t region[3] = {img.w,img.h,1};

   err = clEnqueueReadImage( cmdq, img.image_mem, CL_TRUE,  
        origin, region, img.w*sizeof(cl_short ), 0, h_img_int, NULL, 0, NULL );
    checkErr( err, __LINE__, "query_float2_buffer: clEnqeueuReadBuffer Float2"); 

    printf("(%d) : ", i,j );
    printf( "%d ", h_img_int[j*(img.w)+i] );;
    printf( "\n");

    free( h_img_int );
}
void query_img8( ocl_image<SINGLE_CHANNEL_TYPE, CL_UNSIGNED_INT8> img, cl_command_queue cmdq, int i, int j, char *p ) 
{
    cl_int err = CL_SUCCESS;
    unsigned char       *h_img_int  = (unsigned char *)malloc( (img.w) * img.h  ) ;

    size_t origin[3] = {0,0,0};
    size_t region[3] = {img.w,img.h,1};

   err = clEnqueueReadImage( cmdq, img.image_mem, CL_TRUE,  
        origin, region, img.w, 0, h_img_int, NULL, 0, NULL );
    checkErr( err, __LINE__, "query_float2_buffer: clEnqeueuReadBuffer Float2"); 

    printf("%s (%d) : ", p,i,j );
    printf( "%d ", h_img_int[j*(img.w)+i] );;
    printf( "\n");

    free( h_img_int );
}

////////////////////////////////////////////////////////////////////////////////
// Pyramid handling functions
////////////////////////////////////////////////////////////////////////////////
template<int lvls, cl_channel_order channel_order, cl_channel_type data_type>
class ocl_pyramid {
    public:
        ocl_image<channel_order, data_type> imgLvl[lvls];
        ocl_buffer scratchBuf;
        ocl_image<channel_order, data_type> scratchImg;

        ocl_pyramid(cl_context &, cl_command_queue &);
        cl_int init(int w, int h, const char *name );
        cl_int fill(ocl_image<SINGLE_CHANNEL_TYPE, CL_UNSIGNED_INT8> , cl_kernel downfilter_x, cl_kernel downfilter_y);
        cl_int pyrFill(ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_UNSIGNED_INT8>, cl_kernel, cl_kernel, cl_int4, cl_int4);
        cl_int convFill( ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_UNSIGNED_INT8>, cl_kernel );
        cl_int G_Fill(
            ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_SIGNED_INT16> &,
            ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_SIGNED_INT16> &,
            cl_kernel  );
        cl_int flowFill(
            ocl_pyramid<3, SINGLE_CHANNEL_TYPE,      CL_UNSIGNED_INT8> &I,
            ocl_pyramid<3, SINGLE_CHANNEL_TYPE,      CL_UNSIGNED_INT8> &J,
            ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_SIGNED_INT16> &Ix,
            ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_SIGNED_INT16> &Iy,
            ocl_pyramid<3, CL_RGBA,      CL_SIGNED_INT32> &G,
            cl_kernel );

        void printInfo();
        void printTimer(int i, bool b=false);

        cl_context ctx;
        cl_command_queue cmdq;
        char pname[256];
        cl_event gpu_timer;
};

template<int lvls, cl_channel_order channel_order, cl_channel_type data_type>
ocl_pyramid<lvls,channel_order,data_type>::ocl_pyramid(cl_context &context_in, cl_command_queue &command_queue_in)
{
    ctx = context_in;
    cmdq = command_queue_in;
}

template<int lvls, cl_channel_order channel_order, cl_channel_type data_type>
void ocl_pyramid<lvls,channel_order,data_type>::printInfo( ){
}


template<int lvls, cl_channel_order channel_order, cl_channel_type data_type>
void ocl_pyramid<lvls,channel_order,data_type>::printTimer(int i, bool showT){
        if( showT ) {
        clWaitForEvents(1, &gpu_timer );
        printf("\t\t\t\t\t%s, L%d Kernel Time: %f [ms]\n", pname, i, elapsedTimeInSeconds(gpu_timer)*1000.0f );
        }
}

// initialize memory for the image pyramid
// name is an optional pyramid "name" for saving images as
template<int lvls, cl_channel_order channel_order, cl_channel_type data_type>
cl_int ocl_pyramid<lvls,channel_order,data_type>::init(int w, int h, const char *name = NULL )
{
    cl_int err;
    cl_mem_flags memflag;
#ifdef MAC
    memflag = CL_MEM_READ_ONLY;
#else
    memflag = CL_MEM_READ_WRITE;
#endif

    // store ubytes as RGBA packed
    for( int i=0 ; i<lvls ; i++ ) {
        imgLvl[i].w = w>>i;
        imgLvl[i].h = h>>i;
        imgLvl[i].image_format.image_channel_data_type = data_type;
        imgLvl[i].image_format.image_channel_order = channel_order;
        imgLvl[i].image_mem = clCreateImage2D( ctx, memflag,
                &imgLvl[i].image_format, imgLvl[i].w, imgLvl[i].h, 0, NULL, &err );
        if( err != CL_SUCCESS ) return err;
    }
    // initialize a scratch buffer
    // on Mac, it is a linear buffer used between texture passes
    // on LInux it is a R/W texture, presumably faster
    scratchImg.w = imgLvl[0].w;
    scratchImg.h = imgLvl[0].h;
    scratchImg.image_format.image_channel_data_type = data_type;
    scratchImg.image_format.image_channel_order = channel_order;
    scratchImg.image_mem = clCreateImage2D( ctx, memflag,
        &scratchImg.image_format, scratchImg.w, scratchImg.h, 0, NULL, &err );
    checkErr(err, __LINE__, "creating Scratchg Image (buf/tex)");


    // just an absurd thing to workaround lack of texture writes
    int sz;
    if( data_type == CL_UNSIGNED_INT8 ) {
        sz = sizeof(cl_uchar);
    } else if( data_type == CL_SIGNED_INT16 ) {
        sz = sizeof(cl_ushort);
    } else if( data_type == CL_SIGNED_INT32 && channel_order == CL_RGBA ) {
        sz = sizeof(cl_int) * 4 ;
    } else if( data_type == CL_FLOAT && channel_order == CL_RGBA ) {
        sz = sizeof(cl_float) * 4 ;
    } else {
        assert(false);
    }
        
    scratchBuf.w = imgLvl[0].w;
    scratchBuf.h = imgLvl[0].h;
    scratchBuf.image_format.image_channel_data_type = data_type;
    scratchBuf.image_format.image_channel_order = channel_order;

    //printf("Scratch buffer is %d x %d x %ld \n", scratchBuf.w, scratchBuf.h, sz );
    int size = scratchBuf.w * scratchBuf.h* sz;
    scratchBuf.mem = clCreateBuffer( ctx, CL_MEM_READ_WRITE, size, NULL, &err );
    checkErr(err, __LINE__, "creating Scratch Buffer");

    if( name != NULL ) {
        strcpy( pname, name);
    }else {
        strcpy( pname, "unknown");
    }    
    return err;

}

// fill in the image data for the downfilter pyramid given a base image
template<int lvls, cl_channel_order channel_order, cl_channel_type data_type>
cl_int ocl_pyramid<lvls,channel_order,data_type>::fill(
	ocl_image<SINGLE_CHANNEL_TYPE, CL_UNSIGNED_INT8> img, 
	cl_kernel downfilter_kernel_x, 
	cl_kernel downfilter_kernel_y )
{
    static cl_int err = CL_SUCCESS;
    static const size_t src_origin[3] = {0, 0, 0};
    static const size_t dst_origin[3] = {0, 0, 0};
    static const size_t region[3] = {img.w, img.h, 1};
    static size_t global_work_size[2];
    static size_t local_work_size[2];
	static int i;

    // copy level 0 image (full size)
    err = clEnqueueCopyImage( cmdq, img.image_mem, imgLvl[0].image_mem,
        src_origin, dst_origin, region, 0, NULL, NULL );
    checkErr(err,__LINE__, "oclPyramid::fill::clEnqueuCopyImage");

    static char fname[256];
	if( writeImages ) {
		sprintf(fname, "%s-L%d.pgm", pname, 0 );
		save_image( imgLvl[0], cmdq, fname );

		sprintf(fname, "%s-L%d.oct", pname, 0 );
		 save_octave( imgLvl[0], cmdq, fname );
	}
     
    for(  i=1 ; i<lvls ; i++ ) {
        local_work_size[0] = 32;
        local_work_size[1] = 4;
        global_work_size[0] = local_work_size[0] * DivUp( imgLvl[i-1].w, local_work_size[0] ) ;
        global_work_size[1] = local_work_size[1] * DivUp( imgLvl[i-1].h, local_work_size[1] ) ;

        int argCnt = 0;
        clSetKernelArg( downfilter_kernel_x, argCnt++, sizeof(cl_mem), &imgLvl[i-1].image_mem );
        clSetKernelArg( downfilter_kernel_x, argCnt++, sizeof(cl_mem), &scratchBuf.mem );
        clSetKernelArg( downfilter_kernel_x, argCnt++, sizeof(cl_int), &imgLvl[i-1].w );
        clSetKernelArg( downfilter_kernel_x, argCnt++, sizeof(cl_int), &imgLvl[i-1].h );
        //printf("%d %d\n", imgLvl[i-1].w, imgLvl[i-1].h );
        err = clEnqueueNDRangeKernel( cmdq, downfilter_kernel_x, 2, 0, 
            global_work_size, local_work_size, 0, NULL, NULL);
        checkErr(err,__LINE__,  "downfilterx");
        printTimer(i);
		
        // perform copy from buffer to clImage
		// when image writes are not available
        {
			size_t origin[3] = {0,0,0};
			size_t region[3] = {imgLvl[i-1].w, imgLvl[i-1].h, 1};
			err = clEnqueueCopyBufferToImage( cmdq, scratchBuf.mem, scratchImg.image_mem, 0, origin, region, 0, NULL, NULL );
			checkErr(err,__LINE__,  "clCopyBufferToImage");
        }

        global_work_size[0] = local_work_size[0] * DivUp( imgLvl[i].w, local_work_size[0] ) ;
        global_work_size[1] = local_work_size[1] * DivUp( imgLvl[i].h, local_work_size[1] ) ;
        argCnt = 0;

        // send the imgLvl[i] texture holding teh first pass as input
        // send the scratch buffer as output
        clSetKernelArg( downfilter_kernel_y, argCnt++, sizeof(cl_mem), &scratchImg.image_mem );
        clSetKernelArg( downfilter_kernel_y, argCnt++, sizeof(cl_mem), &scratchBuf.mem );
        clSetKernelArg( downfilter_kernel_y, argCnt++, sizeof(cl_int), &imgLvl[i].w );
        clSetKernelArg( downfilter_kernel_y, argCnt++, sizeof(cl_int), &imgLvl[i].h );

        err = clEnqueueNDRangeKernel( cmdq, downfilter_kernel_y, 2, 0, 
            global_work_size, local_work_size, 0, NULL, NULL);
        checkErr(err, __LINE__, "downfiltery");
        printTimer(i);

        // perform copy from buffer to clImage
		// when image writes are not available
        { ///XX all wrong indexing!
			size_t origin[3] = {0,0,0};
			size_t region[3] = {imgLvl[i].w, imgLvl[i].h, 1};
			err = clEnqueueCopyBufferToImage( cmdq, scratchBuf.mem, imgLvl[i].image_mem, 0, origin, region, 0, NULL, NULL );
			checkErr(err,__LINE__,  "clCopyBufferToImage");
        }

        char fname[256];
		if(writeImages) {
			sprintf(fname, "%s-L%d.pgm", pname, i );
			save_image( imgLvl[i], cmdq, fname );

			sprintf(fname, "%s-L%d.oct", pname, i );
			save_octave( imgLvl[i], cmdq, fname );
		}


    }
	return err;
}

// given a pyramid, create a pyramid that holds the scharr filtered version of each level
template<int lvls, cl_channel_order channel_order, cl_channel_type data_type>
cl_int ocl_pyramid<lvls,channel_order,data_type>::pyrFill( 
    ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_UNSIGNED_INT8> pyr, 
    cl_kernel kernel_x, 
    cl_kernel kernel_y,
    cl_int4   Wx,
    cl_int4   Wy )
{
    cl_int err = CL_SUCCESS;
    size_t global_work_size[2];
    size_t local_work_size[2];

    for( int i=0; i<lvls ; i++ ) {

        int argCnt = 0;

        local_work_size[0] = 32;
        local_work_size[1] = 4;

        global_work_size[0] = local_work_size[0] * DivUp( imgLvl[i].w, local_work_size[0] ) ;
        global_work_size[1] = local_work_size[1] * DivUp( imgLvl[i].h, local_work_size[1] ) ;

        clSetKernelArg( kernel_x, argCnt++, sizeof(cl_mem), &pyr.imgLvl[i].image_mem );
        clSetKernelArg( kernel_x, argCnt++, sizeof(cl_mem), &scratchBuf.mem );
        clSetKernelArg( kernel_x, argCnt++, sizeof(cl_int), &pyr.imgLvl[i].w );
        clSetKernelArg( kernel_x, argCnt++, sizeof(cl_int), &pyr.imgLvl[i].h );
		clSetKernelArg( kernel_x, argCnt++, sizeof(cl_int), &Wx.s[0] );
		clSetKernelArg( kernel_x, argCnt++, sizeof(cl_int), &Wx.s[1] );
		clSetKernelArg( kernel_x, argCnt++, sizeof(cl_int), &Wx.s[2] );
        err = clEnqueueNDRangeKernel( cmdq, kernel_x, 2, 0, 
            global_work_size, local_work_size, 0, NULL, &gpu_timer);
        checkErr(err, __LINE__, "enq");
        printTimer(i);
        //printf("Wx = %d %d %d %d\n", Wx[0], Wx[1], Wx[2], Wx[3] );

        // perform copy from buffer to clImage
		// when image writes are not available
        {
        size_t origin[3] = {0,0,0};
        size_t region[3] = {pyr.imgLvl[i].w, pyr.imgLvl[i].h, 1};
        err = clEnqueueCopyBufferToImage( cmdq, scratchBuf.mem, scratchImg.image_mem, 0, origin, region, 0, NULL, NULL );
        checkErr(err, __LINE__, "clCopyBufferToImage");
    
        }

        argCnt=0;
        clSetKernelArg( kernel_y, argCnt++, sizeof(cl_mem), &scratchImg.image_mem );
        clSetKernelArg( kernel_y, argCnt++, sizeof(cl_mem), &scratchBuf.mem );
        clSetKernelArg( kernel_y, argCnt++, sizeof(cl_int), &pyr.imgLvl[i].w );
        clSetKernelArg( kernel_y, argCnt++, sizeof(cl_int), &pyr.imgLvl[i].h );
        clSetKernelArg( kernel_y, argCnt++, sizeof(cl_int), &Wy.s[0] );
        clSetKernelArg( kernel_y, argCnt++, sizeof(cl_int), &Wy.s[1] );
        clSetKernelArg( kernel_y, argCnt++, sizeof(cl_int), &Wy.s[2] );
        err = clEnqueueNDRangeKernel( cmdq, kernel_y, 2, 0, 
            global_work_size, local_work_size, 0, NULL, &gpu_timer);

        checkErr(err, __LINE__, "enq");
        printTimer(i);

        // perform copy from buffer to clImage
		// when image writes are not available
        { 
			size_t origin[3] = {0,0,0};
			size_t region[3] = {pyr.imgLvl[i].w, pyr.imgLvl[i].h, 1};
			err = clEnqueueCopyBufferToImage( cmdq, scratchBuf.mem, imgLvl[i].image_mem, 0, origin, region, 0, NULL, NULL );
			checkErr(err,__LINE__,  "clCopyBufferToImage");
        }


        char fname[256];
        sprintf(fname, "%s-L%d.pgm", pname, i );
        if(writeImages) save_image( imgLvl[i],cmdq, fname ) ;

        sprintf(fname, "%s-L%d.oct", pname, i );
        if(writeImages) save_octave( imgLvl[i], cmdq, fname );
        
    }
    return err;
}

// given a pyramid, create a pyramid that holds greyscale image, make a floating point CL_RGBA for interpolation
template<int lvls, cl_channel_order channel_order, cl_channel_type data_type>
cl_int ocl_pyramid<lvls,channel_order,data_type>::convFill( 
    ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_UNSIGNED_INT8> pyr, 
    cl_kernel convert_kernel )
{
    cl_int err = CL_SUCCESS;
    size_t global_work_size[2];
    size_t local_work_size[2];

    for( int i=0; i<lvls ; i++ ) {

        int argCnt = 0;

        local_work_size[0] = 32;
        local_work_size[1] = 4;

        global_work_size[0] = local_work_size[0] * DivUp( imgLvl[i].w, local_work_size[0] ) ;
        global_work_size[1] = local_work_size[1] * DivUp( imgLvl[i].h, local_work_size[1] ) ;

        clSetKernelArg( convert_kernel, argCnt++, sizeof(cl_mem), &pyr.imgLvl[i].image_mem );
        clSetKernelArg( convert_kernel, argCnt++, sizeof(cl_mem), &scratchBuf.mem );
        clSetKernelArg( convert_kernel, argCnt++, sizeof(cl_int), &pyr.imgLvl[i].w );
        clSetKernelArg( convert_kernel, argCnt++, sizeof(cl_int), &pyr.imgLvl[i].h );
        err = clEnqueueNDRangeKernel( cmdq, convert_kernel, 2, 0, 
            global_work_size, local_work_size, 0, NULL, &gpu_timer);
        checkErr(err, __LINE__, "enq");
        printTimer(i);
        //printf("Wx = %d %d %d %d\n", Wx[0], Wx[1], Wx[2], Wx[3] );

        // perform copy from buffer to clImage, use the     
        { 
        size_t origin[3] = {0,0,0};
        size_t region[3] = {pyr.imgLvl[i].w, pyr.imgLvl[i].h, 1};
        err = clEnqueueCopyBufferToImage( cmdq, scratchBuf.mem, imgLvl[i].image_mem, 0, origin, region, 0, NULL, NULL );
        checkErr(err, __LINE__, "clCopyBufferToImage");
        }

        //char fname[256];
        //sprintf(fname, "%s-L%d.oct", pname, i );
        //save_octave( imgLvl[i], cmdq, fname );
        
    }
    return err;
}



template<int lvls, cl_channel_order channel_order, cl_channel_type data_type>
cl_int ocl_pyramid<lvls,channel_order,data_type>::G_Fill( 
                                                            ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_SIGNED_INT16> &Ix, 
                                                            ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_SIGNED_INT16> &Iy, 
                                                            cl_kernel kernel_G )
{
    cl_int err = CL_SUCCESS;
    size_t global_work_size[2];
    size_t local_work_size[2];

    for( int i=0; i<lvls ; i++ ) {

        int argCnt = 0;

        // launch enough workgroups to cover 1/4 result image
        local_work_size[0] = 32;
        local_work_size[1] = 4;

        global_work_size[0] = local_work_size[0] * DivUp( imgLvl[i].w, local_work_size[0] ) ;
        global_work_size[1] = local_work_size[1] * DivUp( imgLvl[i].h, local_work_size[1] ) ;

        clSetKernelArg( kernel_G, argCnt++, sizeof(cl_mem), &Ix.imgLvl[i].image_mem );
        clSetKernelArg( kernel_G, argCnt++, sizeof(cl_mem), &Iy.imgLvl[i].image_mem );
        clSetKernelArg( kernel_G, argCnt++, sizeof(cl_mem), &scratchBuf.mem );
        clSetKernelArg( kernel_G, argCnt++, sizeof(cl_mem), &Iy.imgLvl[i].w );
        clSetKernelArg( kernel_G, argCnt++, sizeof(cl_mem), &Iy.imgLvl[i].h );
   
        err = clEnqueueNDRangeKernel( cmdq, kernel_G, 2, 0, 
            global_work_size, local_work_size, 0, NULL, &gpu_timer );
        checkErr(err, __LINE__, "enq");
        printTimer(i);

        // perform copy from buffer to clImage, use the     
        { 
        size_t origin[3] = {0,0,0};
        size_t region[3] = {imgLvl[i].w, imgLvl[i].h, 1};
        err = clEnqueueCopyBufferToImage( cmdq, scratchBuf.mem, imgLvl[i].image_mem, 0, origin, region, 0, NULL, NULL );
        checkErr(err, __LINE__, "clCopyBufferToImage");
        }



        char fname[256];
        sprintf(fname, "%s-L%d.oct", pname, i );
        if( writeImages) save_octave( imgLvl[i],cmdq, fname ) ;

        
    }
    return err;
}

template<int lvls, cl_channel_order channel_order, cl_channel_type data_type>
cl_int ocl_pyramid<lvls,channel_order,data_type>::flowFill( 
    ocl_pyramid<3, SINGLE_CHANNEL_TYPE,      CL_UNSIGNED_INT8> &I,
    ocl_pyramid<3, SINGLE_CHANNEL_TYPE,      CL_UNSIGNED_INT8> &J,
    ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_SIGNED_INT16> &Ix,
    ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_SIGNED_INT16> &Iy,
    ocl_pyramid<3, CL_RGBA,      CL_SIGNED_INT32> &G,
    cl_kernel lkflow_kernel
)
{
    cl_int err = CL_SUCCESS;
    size_t global_work_size[2];
    size_t local_work_size[2];

    // beginning at the top level work down the base (largest)
    for( int i=lvls-1; i>=0 ; i-- ) {
        int argCnt = 0;
        int use_guess = 0;
        if( i <lvls-1 ) use_guess = 1;

        local_work_size[0] = 16;
        local_work_size[1] = 8;

        global_work_size[0] = local_work_size[0] * DivUp( imgLvl[i].w, local_work_size[0] ) ;
        global_work_size[1] = local_work_size[1] * DivUp( imgLvl[i].h, local_work_size[1] ) ;

        clSetKernelArg( lkflow_kernel, argCnt++, sizeof(cl_mem), &I.imgLvl[i].image_mem );
        clSetKernelArg( lkflow_kernel, argCnt++, sizeof(cl_mem), &J.imgLvl[i].image_mem );
        clSetKernelArg( lkflow_kernel, argCnt++, sizeof(cl_mem), &Ix.imgLvl[i].image_mem );
        clSetKernelArg( lkflow_kernel, argCnt++, sizeof(cl_mem), &Iy.imgLvl[i].image_mem );
        clSetKernelArg( lkflow_kernel, argCnt++, sizeof(cl_mem), &G.imgLvl[i].image_mem );
        if( use_guess  ) {  // send previous level guesses if available
            clSetKernelArg( lkflow_kernel, argCnt++, sizeof(cl_mem), &imgLvl[i+1].image_mem );
        } else { // if no previous level just send irrelevant pointer
            clSetKernelArg( lkflow_kernel, argCnt++, sizeof(cl_mem), &scratchImg.image_mem);
        }
        clSetKernelArg( lkflow_kernel, argCnt++, sizeof(cl_mem), &imgLvl[i].image_mem );
        clSetKernelArg( lkflow_kernel, argCnt++, sizeof(cl_int), &use_guess );

        
        err = clEnqueueNDRangeKernel( cmdq, lkflow_kernel, 2, 0, 
            global_work_size, local_work_size, 0, NULL, &gpu_timer );
        checkErr( err, __LINE__, "lkflowKernel", false);
        printTimer(i);


        
        char fname[256];
        sprintf( fname, "%s-L%d.oct", pname, i );
        if(writeImages) save_octave( imgLvl[i], cmdq, fname );

    }
    return err;
}

float calc_flow( 
    ocl_pyramid<3, SINGLE_CHANNEL_TYPE,      CL_UNSIGNED_INT8> &I,
    ocl_pyramid<3, SINGLE_CHANNEL_TYPE,      CL_UNSIGNED_INT8> &J,
    ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_SIGNED_INT16> &Ix,
    ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_SIGNED_INT16> &Iy,
    ocl_pyramid<3, CL_RGBA,      CL_SIGNED_INT32> &G,
    ocl_pyramid<3, CL_RGBA,      CL_FLOAT> &J_float,
    ocl_buffer flowLvl[3],
    cl_kernel lkflow_kernel,
    cl_command_queue cmdq
)
{
    int lvls = 3;
    cl_int err = CL_SUCCESS;
    size_t global_work_size[2];
    size_t local_work_size[2];
    float t_flow = 0.0f;
    
    static cl_event flow_timer;

    // beginning at the top level work down the base (largest)
    for( int i=lvls-1; i>=0 ; i-- ) {
        int argCnt = 0;
        int use_guess = 0;
        if( i <lvls-1 ) use_guess = 1;

        local_work_size[0] =16;
        local_work_size[1] = 8;

        global_work_size[0] = local_work_size[0] * DivUp( flowLvl[i].w, local_work_size[0] ) ;
        global_work_size[1] = local_work_size[1] * DivUp( flowLvl[i].h, local_work_size[1] ) ;

        clSetKernelArg( lkflow_kernel, argCnt++, sizeof(cl_mem), &I.imgLvl[i].image_mem );
        clSetKernelArg( lkflow_kernel, argCnt++, sizeof(cl_mem), &Ix.imgLvl[i].image_mem );
        clSetKernelArg( lkflow_kernel, argCnt++, sizeof(cl_mem), &Iy.imgLvl[i].image_mem );
        clSetKernelArg( lkflow_kernel, argCnt++, sizeof(cl_mem), &G.imgLvl[i].image_mem );
        clSetKernelArg( lkflow_kernel, argCnt++, sizeof(cl_mem), &J_float.imgLvl[i].image_mem );

        if( use_guess  ) {  // send previous level guesses if available
            clSetKernelArg( lkflow_kernel, argCnt++, sizeof(cl_mem), &flowLvl[i+1].mem );
            clSetKernelArg( lkflow_kernel, argCnt++, sizeof(cl_mem), &flowLvl[i+1].w );
        } else { // if no previous level just send irrelevant pointer
            clSetKernelArg( lkflow_kernel, argCnt++, sizeof(cl_mem), &flowLvl[0].mem );
            clSetKernelArg( lkflow_kernel, argCnt++, sizeof(cl_mem), &flowLvl[0].w );
        }

        clSetKernelArg( lkflow_kernel, argCnt++, sizeof(cl_mem), &flowLvl[i].mem );
        clSetKernelArg( lkflow_kernel, argCnt++, sizeof(cl_mem), &flowLvl[i].w );
        clSetKernelArg( lkflow_kernel, argCnt++, sizeof(cl_mem), &flowLvl[i].h );
        clSetKernelArg( lkflow_kernel, argCnt++, sizeof(cl_int), &use_guess );

        err = clEnqueueNDRangeKernel( cmdq, lkflow_kernel, 2, 0, 
            global_work_size, local_work_size, 0, NULL, &flow_timer );
        checkErr( err, __LINE__, "lkflowKernel");
        
        clWaitForEvents(1, &flow_timer );
		// printf("\t\t\t\t\tcalcflow, GPU L%d Kernel Time: %f [ms]\n", i, elapsedTimeInSeconds(gpu_timer)*1000.0f );
        t_flow += (float)elapsedTimeInSeconds(flow_timer)*1000.0f; 
        char fname[256];
		if(writeImages) {
			sprintf( fname, "results/flow-L%d.oct", i );
			 save_octave_float2( flowLvl[i], cmdq, fname );
		}

    }
    return t_flow;
}

////////////////////////////////////////////////////////////////////////////////
// Image file handling
////////////////////////////////////////////////////////////////////////////////
ocl_image<SINGLE_CHANNEL_TYPE, CL_UNSIGNED_INT8> ocl_init_image( cl_context context, unsigned char *image_grey_ub, int w, int h, cl_int &err )
{
    cl_mem_flags memflag;
#ifdef MAC
    memflag = CL_MEM_READ_ONLY;
#else
    memflag = CL_MEM_READ_WRITE;
#endif

    ocl_image<SINGLE_CHANNEL_TYPE, CL_UNSIGNED_INT8> img; 
    img.image_format.image_channel_order  =  SINGLE_CHANNEL_TYPE;
    img.image_format.image_channel_data_type  = CL_UNSIGNED_INT8;
    img.w = w;
    img.h = h;
   
    img.image_mem = clCreateImage2D( context, memflag,  &img.image_format, img.w, img.h, 0, NULL,  &err);
    checkErr(err, __LINE__, "ocl_load_image::clCreateImage2D");

    if( image_grey_ub != NULL ) {
            size_t origin[3] = {0};
            size_t region[3] = {img.w, img.h, 1};
            err = clEnqueueWriteImage( command_queue, img.image_mem, CL_TRUE, origin, region, img.w, 0, image_grey_ub, 0, NULL, NULL );
            checkErr(err,__LINE__,  "ocl_load_image::clEnqueuWriteImage");
    }
    return img;
}

void ocl_set_image( ocl_image<SINGLE_CHANNEL_TYPE, CL_UNSIGNED_INT8> img, cl_context context, unsigned char *image_grey_ub, cl_int &err )
{
    if( image_grey_ub != NULL ) {
            size_t origin[3] = {0};
            size_t region[3] = {img.w, img.h, 1};
            err = clEnqueueWriteImage( command_queue, img.image_mem, CL_TRUE, origin, region, img.w, 0, image_grey_ub, 0, NULL,  NULL);
            checkErr(err, __LINE__, "ocl_load_image::clEnqueuWriteImage");
    }
    return;
}


ocl_image<SINGLE_CHANNEL_TYPE, CL_UNSIGNED_INT8> ocl_load_image( cl_context context, const char *fname, cl_int &err  )
{


    cl_mem_flags memflag;
#ifdef MAC
    memflag = CL_MEM_READ_ONLY;
#else
    memflag = CL_MEM_READ_WRITE;
#endif

    ocl_image<SINGLE_CHANNEL_TYPE, CL_UNSIGNED_INT8> img; 
    unsigned char *image_ub = NULL;

    err = shrLoadPGMub( fname, (unsigned char **)&image_ub, &img.w, &img.h );
    std::cerr<<"loading "<<fname<<std::endl;
    oclCheckErrorEX( err, shrTRUE, NULL );


    img.image_format.image_channel_order  =  SINGLE_CHANNEL_TYPE;
    img.image_format.image_channel_data_type  = CL_UNSIGNED_INT8;
   
    img.image_mem = clCreateImage2D( context, memflag,  &img.image_format,
        img.w, img.h, 0, NULL,  &err);
    checkErr(err, __LINE__,"ocl_load_image::clCreateImage2D");

    size_t origin[3] = {0};
    size_t region[3] = {img.w, img.h, 1};
    clEnqueueWriteImage( command_queue, img.image_mem, CL_TRUE, origin, region, 
        img.w, 0, image_ub, 0, NULL, NULL );
    checkErr(err, __LINE__, "ocl_load_image::clEnqueuWriteImage");
    return img;
}

cl_kernel downfilter_kernel_x;
cl_kernel downfilter_kernel_y;
cl_kernel filter_3x1;
cl_kernel filter_1x3;
cl_kernel filter_G;
cl_kernel lkflow_kernel;
cl_kernel update_motion_kernel;
cl_kernel convert_kernel;

ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_UNSIGNED_INT8> *I;
ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_UNSIGNED_INT8> *J ;
ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_SIGNED_INT16> *Ix ;
ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_SIGNED_INT16> *Iy ;
ocl_pyramid<3, CL_RGBA, CL_SIGNED_INT32> *G; 
ocl_pyramid<3, CL_RGBA, CL_FLOAT> *J_float; 
// load images
ocl_image<SINGLE_CHANNEL_TYPE, CL_UNSIGNED_INT8> images[2];
ocl_buffer flowLvl[3];
cl_mem vbo_cl_mem;

void acquireVBO() {
    cl_int err = clEnqueueAcquireGLObjects( command_queue, 1, &vbo_cl_mem, 0, NULL, NULL );
    checkErr(err, __LINE__, "cl Acquire GL objects");
}
void releaseVBO() {
    cl_int err = clEnqueueReleaseGLObjects( command_queue, 1, &vbo_cl_mem, 0, NULL, NULL );
    checkErr(err, __LINE__, "cl Release GL objects");
}


void updateFlowBuffer(cl_mem pos_mem, cl_mem v_mem, int w, int h) 
{    
    size_t global_work_size[2];
    size_t local_work_size[2];
    cl_int err = CL_SUCCESS;
    static cl_event gpu_timer;
    local_work_size[0] = 32;
    local_work_size[1] = 4;
    global_work_size[0] = local_work_size[0] * DivUp( w, local_work_size[0] ) ;
    global_work_size[1] = local_work_size[1] * DivUp( h, local_work_size[1] ) ;

    int argCnt = 0;
    clSetKernelArg( update_motion_kernel, argCnt++, sizeof(cl_mem), &pos_mem );
    clSetKernelArg( update_motion_kernel, argCnt++, sizeof(cl_mem), &v_mem );
    clSetKernelArg( update_motion_kernel, argCnt++, sizeof(cl_int), &w );
    clSetKernelArg( update_motion_kernel, argCnt++, sizeof(cl_int), &h );
    acquireVBO();
    err = clEnqueueNDRangeKernel( command_queue, update_motion_kernel, 2, 0, 
            global_work_size, local_work_size, 0, NULL, &gpu_timer);
    releaseVBO();
    checkErr(err, __LINE__, "update_motion_kernel");
}


cl_context initOCLFlow(GLuint vbo, int devId)
{
    cl_int err;
    opencl_init(devId);

    // load our filters, downfilter and scharr for building the pyramids
    cl_program lkflow_program = buildProgramFromFile(context,"lkflow.cl");
	cl_program motion_programs = buildProgramFromFile(context,"motion.cl");
    cl_program filter_programs = buildProgramFromFile(context,"filters.cl");
  
    downfilter_kernel_x = clCreateKernel(filter_programs, "downfilter_x_g", &err);
    checkErr(err, __LINE__, "clCreateKernel (downfilter_x)");
    downfilter_kernel_y = clCreateKernel(filter_programs, "downfilter_y_g", &err);
    checkErr(err, __LINE__, "clCreateKernel (downfilter_y)");

    filter_3x1 = clCreateKernel(filter_programs, "filter_3x1_g", &err );
    checkErr(err, __LINE__, "clCreateKrenel (filter3x1)");
    filter_1x3 = clCreateKernel(filter_programs, "filter_1x3_g", &err );
    checkErr(err, __LINE__, "clCreateKrenel (filter1x3)");
    
    filter_G = clCreateKernel(filter_programs, "filter_G", &err );
    checkErr(err, __LINE__, "clCreateKrenel (G_filter)");

    lkflow_kernel = clCreateKernel( lkflow_program, "lkflow", &err );
    checkErr(err, __LINE__, "clCreateKernel (lkflow)");
   
    update_motion_kernel = clCreateKernel( motion_programs, "motion", &err );
    checkErr(err, __LINE__, "clCreateKernel (motion)");

    convert_kernel = clCreateKernel( filter_programs, "convertToRGBAFloat", &err );
    checkErr(err, __LINE__, "clCreateKernel (convert_kernel)");

    images[1] = ocl_load_image( context, "data/minicooper/frame10.pgm", err );
    images[0] = ocl_load_image( context, "data/minicooper/frame11.pgm", err );

    // create pyramids
    I  = new ocl_pyramid<3,SINGLE_CHANNEL_TYPE,CL_UNSIGNED_INT8>(context, command_queue);
    J  = new ocl_pyramid<3,SINGLE_CHANNEL_TYPE,CL_UNSIGNED_INT8>(context, command_queue);
    Ix = new ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_SIGNED_INT16>(context, command_queue);
    Iy = new ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_SIGNED_INT16>(context, command_queue);
    G = new ocl_pyramid<3,CL_RGBA,CL_SIGNED_INT32>(context, command_queue);
    J_float  = new ocl_pyramid<3,CL_RGBA,CL_FLOAT>(context, command_queue);

    // initalize them
    err = I->init( images[0].w, images[0].h, "results/I" );  checkErr(err, __LINE__, "Init I");
    err = J->init( images[1].w, images[1].h, "results/J" );  checkErr(err, __LINE__, "Init J");
    // initialize the Ix,Iy derivatives from the downsampled pyramids
    err = Ix->init( images[0].w, images[0].h, "results/Ix");  checkErr(err, __LINE__, "init Ix");
    err = Iy->init( images[0].w, images[0].h, "results/Iy");  checkErr(err, __LINE__, "init Iy");
    // initialize G 
    err = G->init( images[0].w, images[0].h, "results/G" ) ; checkErr(err, __LINE__, "init G");
    err = J_float->init( images[0].w, images[0].h, "results/J_float" ) ; checkErr(err, __LINE__, "init J_float");

    // simulate a CL_RG buffer in global memory, for lack of support for CL_RG
    for( int i=0 ; i<3; i++ ) {
        flowLvl[i].w = images[0].w>>i;
        flowLvl[i].h = images[0].h>>i;
        flowLvl[i].image_format.image_channel_data_type = CL_FLOAT;
        flowLvl[i].image_format.image_channel_order = CL_RG;
        int size = flowLvl[i].w * flowLvl[i].h* sizeof(cl_float2) ;
        flowLvl[i].mem = clCreateBuffer( context, CL_MEM_READ_WRITE, size, NULL, &err );
        checkErr(err, __LINE__, "creating flow level");
    }

    // get a handle to the VBO that stores point start/end locations 
    vbo_cl_mem = clCreateFromGLBuffer( context, CL_MEM_READ_WRITE, vbo, &err );
    checkErr(err,__LINE__,  "creating vbo handle in CL");

    return context;
}

float computeOCLFlow(int curr, int next)
{
	float t_flow = 0;
    // todo: don't need to refill both images, only the new one. 
    cl_int err;
    I->fill( images[curr], downfilter_kernel_x, downfilter_kernel_y );
    J->fill( images[next], downfilter_kernel_x, downfilter_kernel_y );
    cl_int4 dx_Wx = { -1, 0,  1, 0 };
    cl_int4 dx_Wy = { 3, 10,  3, 0};

    err = Ix->pyrFill( *I, filter_3x1, filter_1x3, dx_Wx, dx_Wy ); checkErr( err, __LINE__,"pyrFill Ix");
    cl_int4 dy_Wx = { 3, 10, 3, 0};
    cl_int4 dy_Wy = { -1, 0, 1, 0}; 
    err = Iy->pyrFill( *I, filter_3x1, filter_1x3, dy_Wx, dy_Wy ); checkErr( err, __LINE__,"pyrFill Iy");

    err = G->G_Fill( *Ix, *Iy, filter_G ); checkErr( err, __LINE__, "G Fill");
    J_float->convFill( *J, convert_kernel );
    t_flow = calc_flow( *I, *J, *Ix, *Iy, *G, *J_float, flowLvl, lkflow_kernel, command_queue );

    updateFlowBuffer(vbo_cl_mem, flowLvl[0].mem, flowLvl[0].w, flowLvl[0].h) ;

    // qeury some data for expected results minicooper data set
    // query_float2_buffer( flowLvl[1], command_queue, 100, 100 );
	//query_float2_buffer( flowLvl[0], command_queue, 200, 200 );
	//query_float_buffer( G->imgLvl[1], command_queue, 100,100 );
	//query_img( Ix->imgLvl[1], command_queue, 100,100 );
	//query_img8( I->imgLvl[1], command_queue, 100,100, "I" );


    return t_flow;
}

