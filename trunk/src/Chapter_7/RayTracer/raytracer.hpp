//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Dan Ginsburg, Timothy Mattson
// ISBN-10:   ??????????
// ISBN-13:   ?????????????
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/??????????
//            http://www.????????.com
//
#ifndef __RAYTRACER_HDR__
#define __RAYTRACKER_HDR__

#if !defined(__OPENCL_VERSION__)
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#else
#define cl_float3 float3
#define cl_float  float
#define cl_int    int
#endif // !defined(__OPENCL_VERSION__)

#define MAX_DEPTH 0

#define WORKGROUP_SIZE_X 8
#define WORKGROUP_SIZE_Y 8

struct __Camera
{
	cl_float3 position_;
	cl_float3 forward_;
	cl_float3 up_;
	cl_float3 right_;
};
typedef struct __Camera Camera;

struct __Light
{
	cl_float3 position_;
	cl_float3 colour_;
};
typedef struct __Light Light;

#define SURFACE_MATT_SHINY   1
#define SURFACE_SHINY        2
#define SURFACE_CHECKERBOARD 3

struct __Surface
{
	cl_int   type_;
	float roughness_;
};
typedef struct __Surface Surface;

#define OBJECT_PLANE  1
#define OBJECT_SPHERE 2

struct __Object 
{
	cl_int type_;
	cl_int p1_;
	cl_int p2_;
	cl_int p3_;
	cl_float3 centerOrNorm_;
	float radiusOrOffset_;
	Surface surface_;
};
typedef struct __Object Object;

struct __Ray
{
	cl_float3 start_;
	cl_float3 direction_;
};
typedef struct __Ray Ray;

struct __InterSection
{
	Ray ray_;
	Object object_;
	float dist_;
};
typedef struct __InterSection InterSection;

#endif // __RAYTRACKER_HDR__
