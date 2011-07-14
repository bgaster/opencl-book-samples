//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Dan Ginsburg, Timothy Mattson
// ISBN-10:   ??????????
// ISBN-13:   ?????????????
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/??????????
//            http://www.????????.com
//

// raytracer.cpp
//
//    This is a (very) simple raytracer that is intended to demonstrate 
//    using OpenCL buffers.

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <GL/glew.h>

#ifdef __APPLE__
#include <GLUT/glut.h>
#include <OpenCL/cl.h>
#else
#include <GL/glut.h>
#include <CL/cl.h>
#endif

#include "info.hpp"

#include "raytracer.hpp"

#define VENDOR 0

#define QUEUE 0

#define MAX_NUM_LIGHTS 20
#define MAX_NUM_OBJECTS 20

// Constants
unsigned int windowWidth  = 512;
unsigned int windowHeight = 512;

// Default Camera configure
cl_float3 cameraPosition = { 6*2.75f, 2.0f, 6.75f, 0.0f };
cl_float3 cameraLookAt   = { -0.6f, -5.0f, 0.0f, 0.0f };

cl_uint numLights;
cl_uint numObjects;

float rotateX;
float rotateY;

float translateZ;
float translateX;
float translateY;

// Function to check and handle OpenCL errors
inline void 
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

struct Vendor
{
    cl_context context_;
    cl_program program_;
    cl_kernel kRender_;
    cl_kernel kInitializeCamera_;
    cl_kernel kInitializeLight_;
    cl_kernel kInitializeObject_;
    cl_mem bImage_;
    cl_mem bCamera_;
    cl_mem bLights_;
    cl_mem bObjects_;
    std::vector<cl_command_queue> queues_;

    Vendor(void) :
        context_(0),
        program_(0),
        kRender_(0),
        kInitializeCamera_(0),
        kInitializeLight_(0),
        kInitializeObject_(0),
        bImage_(0),
        bCamera_(0),
        bLights_(0),
        bObjects_(0)
    {
    }
};

// Global variables...
GLuint glPBO, glTex;
cl_uchar4 * outputImage = 0;
std::vector<Vendor> vendors;

void cleanup(void)
{
  if (outputImage) {
    delete[] outputImage;
  }

  // Add cleanup code for OpenCL
}

void createTexture(void)
{
    if (glTex)
    {
        glDeleteTextures(1, &glTex);
        glTex = 0;
    }

    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &glTex);
    glBindTexture(GL_TEXTURE_2D, glTex);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);

    glTexImage2D(
        GL_TEXTURE_2D, 
        0, 
        GL_RGBA8, 
        windowWidth, 
        windowHeight, 
        0,
        GL_RGBA, 
        GL_UNSIGNED_BYTE, 
        outputImage);
}

void createBuffers(int w, int h)
{
    int errNum;

    Vendor * vendor = &vendors[VENDOR];

    if (outputImage)
    {
        delete[] outputImage;
    }
    outputImage = new cl_uchar4[w*h];

    if (vendor->bImage_)
    {
        clReleaseMemObject(vendor->bImage_);
    }

    vendor->bImage_ = 
        clCreateBuffer(vendor->context_, 
             CL_MEM_READ_WRITE, 
             w*h*sizeof(cl_uchar4), 
             NULL,
             &errNum);
    checkErr(errNum, "clCreateBuffer(bImage_)");

    if (!vendor->bCamera_)
    {
        vendor->bCamera_ = 
        clCreateBuffer(vendor->context_, 
             CL_MEM_READ_WRITE, 
             sizeof(Camera), 
             NULL,
             &errNum);
        checkErr(errNum, "clCreateBuffer(bCamera_)");
    }

    if (!vendor->bLights_)
    {
        vendor->bLights_ = 
        clCreateBuffer(vendor->context_, 
             CL_MEM_READ_WRITE, 
             sizeof(Light) * MAX_NUM_LIGHTS, 
             NULL,
             &errNum);
        checkErr(errNum, "clCreateBuffer(bLights_)");
    }

    if (!vendor->bObjects_)
    {
        vendor->bObjects_ = 
        clCreateBuffer(vendor->context_, 
             CL_MEM_READ_WRITE, 
             sizeof(Object) * MAX_NUM_OBJECTS, 
             NULL,
             &errNum);
        checkErr(errNum, "clCreateBuffer(bObjects_)");
    }
}

void setCamera(cl_float3 position, cl_float3 lookAt)
{
    cl_int errNum;

    Vendor vendor = vendors[VENDOR];

    errNum  = clSetKernelArg(
        vendor.kInitializeCamera_, 
        0, 
        sizeof(cl_float3), 
        (void *)&position);

    errNum  |= clSetKernelArg(
        vendor.kInitializeCamera_, 
        1, 
        sizeof(cl_float3), 
        (void *)&lookAt);

    errNum  |= clSetKernelArg(
        vendor.kInitializeCamera_, 
        2, 
        sizeof(cl_mem), 
        (void *)&vendor.bCamera_);
    checkErr(errNum, "clSetKernelArg(Camera)");

    errNum = clEnqueueTask(
        vendor.queues_[QUEUE], 
        vendor.kInitializeCamera_, 
        0, 
        NULL, 
        NULL);
    checkErr(errNum, "clEnqueueNDRangeKernel");
}

void setLight(cl_float3 position, cl_float3 colour, cl_uint num)
{
    cl_int errNum;

    Vendor vendor = vendors[VENDOR];

    errNum   = clSetKernelArg(
        vendor.kInitializeLight_, 
        0, 
        sizeof(cl_float3), 
        (void *)&position);

    errNum  |= clSetKernelArg(
        vendor.kInitializeLight_, 
        1, 
        sizeof(cl_float3), 
        (void *)&colour);

    errNum  |= clSetKernelArg(
        vendor.kInitializeLight_, 
        2, 
        sizeof(cl_mem), 
        (void *)&vendor.bLights_);

    errNum  |= clSetKernelArg(
        vendor.kInitializeLight_, 
        3, 
        sizeof(cl_uint), 
        (void *)&num);
    checkErr(errNum, "clSetKernelArg(Light)");

    errNum = clEnqueueTask(
        vendor.queues_[QUEUE], 
        vendor.kInitializeLight_, 
        0, 
        NULL, 
        NULL);
    checkErr(errNum, "clEnqueueNDRangeKernel");
}

void setObject(
    cl_int type,
    cl_float3 centerOrNorm,
    float radiusOrOffset,
    cl_int surfaceType,
    cl_float roughness,
    cl_uint num)
{
    cl_int errNum;

    Vendor vendor = vendors[VENDOR];

#if 0
    errNum   = clSetKernelArg(
        vendor.kInitializeObject_, 
        0, 
        sizeof(cl_int), 
        (void *)&type);

    errNum  |= clSetKernelArg(
        vendor.kInitializeObject_, 
        1, 
        sizeof(cl_float3), 
        (void *)&centerOrNorm);

    errNum  |= clSetKernelArg(
        vendor.kInitializeObject_, 
        2, 
        sizeof(float), 
        (void *)&radiusOrOffset);


    Surface surface = { surfaceType, roughness };
    errNum  |= clSetKernelArg(
        vendor.kInitializeObject_, 
        3, 
        sizeof(Surface), 
        (void *)&surface);

    errNum  |= clSetKernelArg(
        vendor.kInitializeObject_, 
        4, 
        sizeof(cl_mem), 
        (void *)&vendor.bObjects_);

    errNum  |= clSetKernelArg(
        vendor.kInitializeObject_, 
        5, 
        sizeof(cl_uint), 
        (void *)&num);
    checkErr(errNum, "clSetKernelArg(Light)");
#else
    errNum  |= clSetKernelArg(
        vendor.kInitializeObject_, 
        0, 
        sizeof(cl_mem), 
        (void *)&vendor.bObjects_);
#endif

    errNum = clEnqueueTask(
        vendor.queues_[QUEUE], 
        vendor.kInitializeObject_, 
        0, 
        NULL, 
        NULL);
    checkErr(errNum, "clEnqueueNDRangeKernel");
}

cl_float3 makeFloat3(float x, float y, float z)
{
    cl_float3 result = { x, y, z};
    return result;
}

void setupScene(void)
{
    setCamera(cameraPosition, cameraLookAt);
    
    numLights = 2;
    setLight(
        makeFloat3(-2.0f, 2.5f, 0.0f),
        makeFloat3(0.5f, 0.45f, 0.41f),
        0);

    setLight(
        makeFloat3(-2.0f, 4.5f, 2.0f),
        makeFloat3(0.99f, 0.95f, 0.8f),
        1);

    numObjects = 3;
    setObject(
        OBJECT_SPHERE,
        makeFloat3(-0.5f, 0.5f, 1.5f),
        1.5f,
        SURFACE_MATT_SHINY,
        250.0f,
        0);

#if 1
    setObject(
        OBJECT_SPHERE,
        makeFloat3(0.0f, 1.0f, -0.25f),
        2.0f,
        SURFACE_SHINY,
        250.0f,
        1);

    setObject(
        OBJECT_PLANE,
        makeFloat3(0.0f, 1.0f, 0.0f),
        0.0f,
        SURFACE_CHECKERBOARD,
        150.0f,
        2);
#endif
}

// OpenGL display function
void displayFunc(void)
{
    cl_int errNum;
    Vendor * vendor = &vendors[VENDOR];

    errNum   = clSetKernelArg(
        vendor->kRender_, 
        0, 
        sizeof(cl_mem), 
        (void *)&vendor->bCamera_);

    errNum  |= clSetKernelArg(
        vendor->kRender_, 
        1, 
        sizeof(cl_mem), 
        (void *)&vendor->bObjects_);

    errNum  |= clSetKernelArg(
        vendor->kRender_, 
        2, 
        sizeof(cl_uint), 
        (void *)&numObjects);

    errNum  |= clSetKernelArg(
        vendor->kRender_, 
        3, 
        sizeof(cl_mem), 
        (void *)&vendor->bImage_);

    errNum  |= clSetKernelArg(
        vendor->kRender_, 
        4, 
        sizeof(cl_uint), 
        (void *)&windowWidth);

    errNum  |= clSetKernelArg(
        vendor->kRender_, 
        5, 
        sizeof(cl_mem), 
        (void *)&vendor->bLights_);

    errNum  |= clSetKernelArg(
        vendor->kRender_, 
        6, 
        sizeof(cl_uint), 
        (void *)&numLights);

    errNum  |= clSetKernelArg(
        vendor->kRender_, 
        7, 
        sizeof(InterSection) * WORKGROUP_SIZE_X * WORKGROUP_SIZE_Y, 
        NULL);

    errNum  |= clSetKernelArg(
        vendor->kRender_, 
        8, 
        sizeof(cl_float3) * WORKGROUP_SIZE_X * WORKGROUP_SIZE_Y, 
        NULL);

    errNum  |= clSetKernelArg(
        vendor->kRender_, 
        9, 
        sizeof(cl_float3) * WORKGROUP_SIZE_X * WORKGROUP_SIZE_Y, 
        NULL);

    errNum  |= clSetKernelArg(
        vendor->kRender_, 
        10, 
        sizeof(cl_float3) * WORKGROUP_SIZE_X * WORKGROUP_SIZE_Y, 
        NULL);

    errNum  |= clSetKernelArg(
        vendor->kRender_, 
        11, 
        sizeof(cl_int) * WORKGROUP_SIZE_X * WORKGROUP_SIZE_Y, 
        NULL);

    checkErr(errNum, "clSetKernelArg(Light)");

    size_t gThreads[2] = { windowWidth, windowHeight };
    // Need to check this is valid!
    size_t lThreads[2] = { WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y };

    errNum = clEnqueueNDRangeKernel(
        vendor->queues_[QUEUE], 
        vendor->kRender_, 
        2, 
        NULL,
        (const size_t*)gThreads, 
        (const size_t*)lThreads, 
                   0, 
                   0, 
                   NULL);
    checkErr(errNum, "clEnqueueNDRangeKernel");

    errNum = clEnqueueReadBuffer(
        vendor->queues_[QUEUE], 
        vendor->bImage_, 
                CL_TRUE, // block until read complete
                0, 
                windowWidth * windowHeight * sizeof(cl_uchar4), 
                (void*)outputImage,
                0, 
                NULL, 
                NULL);
    checkErr(errNum, "clEnqueueReadBuffer");

    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    createTexture();

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();
    glTranslatef( translateX, translateY, translateZ );
 //   glRotatef( rotateX, 0.5f , 0.0f, 0.0f );
 //   glRotatef( rotateY, 0.0f, 0.5f, 0.0f );

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    //glColor3f(0.0, 1.0, 0.0); 

    glBegin(GL_QUADS);
#if 1

#if 0
        glTexCoord2f(1, 0); glVertex2f(-1, -1);
        glTexCoord2f(1, 1); glVertex2f(-1, 1);
        glTexCoord2f(0, 1); glVertex2f(1, 1);
        glTexCoord2f(0, 0); glVertex2f(1, -1);
#endif
        glTexCoord2f(1.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
        glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0f, -1.0f);
        glTexCoord2f(0.0f, 1.0f); glVertex2f( 1.0f,  1.0f);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f,  1.0f);

#else
        glTexCoord2f(0, 0); glVertex2f(-1, -1);
        glTexCoord2f(0, 1); glVertex2f(-1, 1);
        glTexCoord2f(1, 1); glVertex2f(1, 1);
        glTexCoord2f(1, 0); glVertex2f(1, -1);
#endif


    glEnd();

    glutSwapBuffers();
    glutPostRedisplay();

    //printf("Displayed frame\n");
//	std::cout << "Anouther frame" << std::endl;
}

// OpenGL keyboard function
void keyboardFunc(unsigned char k, int, int)
{
    switch (k) {
    case '\033':
    case 'q':
    case 'Q':
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
      std::cout << std::endl << "Executed program succesfully." << std::endl;
      exit(0);
      break;
    default:
      break;
    }
}

void idleFunc(void)
{
  glutPostRedisplay();
}

void reshapeFunc(int w, int h)
{
    glClearColor( 0.0f, 1.0f, 1.f, 0.1f );

    translateX = 0.0f;
    translateY = 0.0f;
    translateZ = -3.0;
    rotateX    = 0;
    rotateY    = 0;

    glViewport(0, 0, w, h);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(
        37.0,
        (GLfloat)w / (GLfloat) h,
        0.1,
        6.0f );

    createBuffers(w, h);

    setupScene();

    windowWidth = w;
    windowHeight = h;
}

///
//	main() for raytracer example
//
int main(int argc, char** argv)
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_uint numDevices;
    cl_platform_id * platformIDs;
    cl_device_id * deviceIDs;

    std::cout << "Simple Ray Tracer Example" << std::endl;

    // First, select an OpenCL platform to run on.  
    errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkErr( 
        (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
        "clGetPlatformIDs"); 
 
    platformIDs = (cl_platform_id *)alloca(
            sizeof(cl_platform_id) * numPlatforms);

    std::cout << "Number of platforms: \t" << numPlatforms << std::endl; 

    errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
    checkErr( 
       (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
       "clGetPlatformIDs");

    std::ifstream srcFile("raytracer.cl");
    checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading raytracer.cl");

    std::string srcProg(
        std::istreambuf_iterator<char>(srcFile),
        (std::istreambuf_iterator<char>()));

    const char * src = srcProg.c_str();
    size_t length = srcProg.length();

    // Iterate through platforms creating a context for each, including all their 
    // devices
    deviceIDs = NULL;
    for (cl_uint i = 0; i < numPlatforms; i++)
    {
        DisplayPlatformInfo(
            platformIDs[i], 
            CL_PLATFORM_VENDOR, 
            "CL_PLATFORM_VENDOR");

        errNum = clGetDeviceIDs(
            platformIDs[i], 
            CL_DEVICE_TYPE_ALL, 
            0,
            NULL,
            &numDevices);
        if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
        {
            checkErr(errNum, "clGetDeviceIDs");
        }
        
        deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
        errNum = clGetDeviceIDs(
            platformIDs[i],
            CL_DEVICE_TYPE_ALL,
            numDevices, 
            &deviceIDs[0], 
            NULL);
        checkErr(errNum, "clGetDeviceIDs");

        cl_context_properties contextProperties[] =
        {
            CL_CONTEXT_PLATFORM,
            (cl_context_properties)platformIDs[i],
            0
        };
        Vendor vendor;

        vendor.context_ = clCreateContext(
            contextProperties, 
            numDevices,
            deviceIDs, 
            NULL,
            NULL, 
            &errNum);
        checkErr(errNum, "clCreateContext");

        // Create command queues
        for (cl_uint j = 0; j < numDevices; j++)
        {
            InfoDevice<cl_device_type>::display(
                deviceIDs[j], 
                CL_DEVICE_TYPE, 
                "CL_DEVICE_TYPE");

            cl_command_queue queue = 
                clCreateCommandQueue(
                    vendor.context_,
                    deviceIDs[j],
                    0,
                    &errNum);
            checkErr(errNum, "clCreateCommandQueue");

            vendor.queues_.push_back(queue);
        }

        // Create program from source
        vendor.program_ = clCreateProgramWithSource(
            vendor.context_, 
            1, 
            &src, 
            &length, 
            &errNum);
        checkErr(errNum, "clCreateProgramWithSource");

        // Build program
        errNum = clBuildProgram(
            vendor.program_,
            numDevices,
            deviceIDs,
            "-I.",
            NULL,
            NULL);
        if (errNum != CL_SUCCESS) {
            // Determine the reason for the error
            char buildLog[16384];
            clGetProgramBuildInfo(
                vendor.program_, 
                deviceIDs[0], 
                CL_PROGRAM_BUILD_LOG,
                sizeof(buildLog), 
                buildLog, 
                NULL);

            std::cerr << "Error in OpenCL C source: " << std::endl;
            std::cerr << buildLog;
            checkErr(errNum, "clBuildProgram");
        }

        vendor.kRender_ = clCreateKernel(
            vendor.program_,
            "render",
            &errNum);
        checkErr(errNum, "clCreateKernel(render)");

        vendor.kInitializeCamera_ = clCreateKernel(
            vendor.program_,
            "initializeCamera",
            &errNum);
        checkErr(errNum, "clCreateKernel(initializeCamera)");

        vendor.kInitializeLight_ = clCreateKernel(
            vendor.program_,
            "initializeLight",
            &errNum);
        checkErr(errNum, "clCreateKernel(initializeLight)");

        vendor.kInitializeObject_ = clCreateKernel(
            vendor.program_,
            "initializeObject",
            &errNum);
        checkErr(errNum, "clCreateKernel(initializeObject)");

        vendors.push_back(vendor);
        //break;
    }

    std::cout << "Devices Initilzied" << std::endl;

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(windowWidth, windowHeight);
    glutInitWindowPosition(0, 0);
    glutCreateWindow(argv[0]);

    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0 "))
    {
        std::cout << "ERROR: Support for necessary OpenGL extensions missing.";
        return false;
    }

    glutDisplayFunc(displayFunc);
    glutIdleFunc(idleFunc);
    glutKeyboardFunc(keyboardFunc);
    glutReshapeFunc(reshapeFunc);
    glewInit();

    atexit(cleanup);
    //amd::fixCloseWindow();
    glutMainLoop();

    return 0;
}