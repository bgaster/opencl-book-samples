//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//

// GLinterop.cpp
//
//    This is a simple example that demonstrates basic OpenCL setup and
//    use.

#include <iostream>
#include <fstream>
#include <sstream>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#include <CL/cl_gl.h>
#endif

#ifdef _WIN32
#include <windows.h>
#endif
#include <GL/glew.h>
#include <GL/freeglut.h>

///
// OpenGL/CL variables, objects
//
GLuint tex = 0;
GLuint vbo = 0;
int vbolen; 
int imWidth = 0;
int imHeight = 0;
cl_kernel kernel = 0;
cl_kernel tex_kernel = 0;
cl_mem cl_vbo_mem, cl_tex_mem;
cl_context context = 0;
cl_command_queue commandQueue = 0;
cl_program program = 0;


///
// Forward declarations
void Cleanup();
cl_int computeVBO();
cl_int computeTexture();

///
// Render the vertex buffer object (VBO) contents
//
void renderVBO( int vbolen ) 
{
	glColor3f(1.0f, 1.0f, 1.0f);
	glLineWidth(2.0f);
    // Draw VBO containing the point list coordinates, to place GL_POINTS at feature locations
    // bind VBOs for vertex array and index array
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo);         // for vertex coordinates
    glEnableClientState(GL_VERTEX_ARRAY);             // activate vertex coords array
    glVertexPointer( 2, GL_FLOAT, 0, 0 );
	 
    // draw lines with endpoints given in the array
    glDrawArrays(GL_LINES, 0, vbolen*2);

    glDisableClientState(GL_VERTEX_ARRAY);            // deactivate vertex array

    // bind with 0, so, switch back to normal pointer operation
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
}

///
// Display the texture in the window
//
void displayTexture(int w, int h)
{
	glEnable(GL_TEXTURE_RECTANGLE_ARB);
	glBindTexture(GL_TEXTURE_RECTANGLE_ARB, tex );
	glBegin(GL_QUADS);
		glTexCoord2f(0, 0);
		glVertex2f(0, 0);
		glTexCoord2f(0, h);
		glVertex2f(0, h);
		glTexCoord2f(w, h);
		glVertex2f(w, h);
		glTexCoord2f(w, 0);
		glVertex2f(w, 0);
	glEnd();
	glDisable(GL_TEXTURE_RECTANGLE_ARB);
}

void reshape(int width, int height) 
{
	glViewport( 0, 0, width, height );
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0,width,height,0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

///
// Main rendering call for the scene
//
void renderScene(void)
{
	glClearColor(0.0f, 0.0f, 1.0f, 1.0f );
	glClear( GL_COLOR_BUFFER_BIT );
	computeTexture();
	computeVBO();

	displayTexture(imWidth,imHeight);
	renderVBO( vbolen );
	glutSwapBuffers();
}

///
// Keyboard events handler
//
void KeyboardGL(unsigned char key, int x, int y)
{
    switch(key) 
    {
        case '\033': // escape quits
        case '\015': // Enter quits    
        case 'Q':    // Q quits
        case 'q':    // q (or escape) quits
            // Cleanup up and quit
	        Cleanup();
            break;
    }
}


void initGlut(int argc, char *argv[], int wWidth, int wHeight)
{
    glutInit(&argc, argv);
    //glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(100,100);
    glutInitWindowSize(wWidth, wHeight);
    glutCreateWindow("GL interop");

    glutDisplayFunc(renderScene);
    glutIdleFunc(renderScene);
    glutReshapeFunc(reshape);
	glutKeyboardFunc(KeyboardGL);
	
    glewInit();
    if (glewIsSupported("GL_VERSION_2_1"))
        printf("Ready for OpenGL 2.1\n");
    else {
		printf("Warning: Detected that OpenGL 2.1 not supported\n");
    }

}

void initTexture( int width, int height )
{
    // make a texture for output
	glGenTextures(1, &tex);              // texture 
    glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE,  GL_REPLACE );
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, tex);
    glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA32F_ARB, width,
            height, 0, GL_LUMINANCE, GL_FLOAT, NULL );
}


GLuint initVBO(int vbolen )
{
    GLint bsize;

    GLuint vbo_buffer; 
    // generate the buffer
    glGenBuffers(1, &vbo_buffer);
    
    // bind the buffer 
    glBindBuffer(GL_ARRAY_BUFFER, vbo_buffer); 
	if( glGetError() != GL_NO_ERROR ) {
		std::cerr<<"Could not bind buffer"<<std::endl;
	}
    
    // create the buffer, this basically sets/allocates the size
	// for our VBO we will hold 2 line endpoints per element
    glBufferData(GL_ARRAY_BUFFER, vbolen*sizeof(float)*4, NULL, GL_STREAM_DRAW);  
	if( glGetError() != GL_NO_ERROR ) {
		std::cerr<<"Could not bind buffer"<<std::endl;
	}
    // recheck the size of the created buffer to make sure its what we requested
    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &bsize); 
    if ((GLuint)bsize != (vbolen*sizeof(float)*4)) {
        printf("Vertex Buffer object (%d) has incorrect size (%d).\n", (unsigned)vbo_buffer, (unsigned)bsize);
    }

    // we're done, so unbind the buffers
    glBindBuffer(GL_ARRAY_BUFFER, 0);                    
	if( glGetError() != GL_NO_ERROR ) {
		std::cerr<<"Could not bind buffer"<<std::endl;
	}
	return vbo_buffer;
}


cl_int computeVBO()
{
	cl_int errNum;

	// a small internal counter for animation
	static cl_int seq = 0;
	seq = (seq+1)%(imWidth);

    // Set the kernel arguments, send the cl_mem object for the VBO
    errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_vbo_mem);
    errNum = clSetKernelArg(kernel, 1, sizeof(cl_int), &imWidth);
    errNum = clSetKernelArg(kernel, 2, sizeof(cl_int), &imHeight);
    errNum = clSetKernelArg(kernel, 3, sizeof(cl_int), &seq);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error setting kernel arguments." << std::endl;
        Cleanup();
        return 1;
    }

    size_t globalWorkSize[1] = { vbolen };
    size_t localWorkSize[1] = { 32 };

	// Acquire the GL Object
	// Note, we should ensure GL is completed with any commands that might affect this VBO
	// before we issue OpenCL commands
	glFinish();
	errNum = clEnqueueAcquireGLObjects(commandQueue, 1, &cl_vbo_mem, 0, NULL, NULL );

    // Queue the kernel up for execution across the array
    errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                    globalWorkSize, localWorkSize,
                                    0, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error queuing kernel for execution." << std::endl;
    }

	// Release the GL Object
	// Note, we should ensure OpenCL is finished with any commands that might affect the VBO
	errNum = clEnqueueReleaseGLObjects(commandQueue, 1, &cl_vbo_mem, 0, NULL, NULL );
	clFinish(commandQueue);
	return 0;
}


///
// Use OpenCL to compute the colors on the texture background
// Bascially the same functionality as the VBO, execpt operating on a texture object
cl_int computeTexture()
{
	cl_int errNum;

	static cl_int seq =0;
	seq = (seq+1)%(imWidth*2);

    errNum = clSetKernelArg(tex_kernel, 0, sizeof(cl_mem), &cl_tex_mem);
    errNum = clSetKernelArg(tex_kernel, 1, sizeof(cl_int), &imWidth);
    errNum = clSetKernelArg(tex_kernel, 2, sizeof(cl_int), &imHeight);
    errNum = clSetKernelArg(tex_kernel, 3, sizeof(cl_int), &seq);
	
	size_t tex_globalWorkSize[2] = { imWidth, imHeight };
	size_t tex_localWorkSize[2] = { 32, 4 } ;

	glFinish();
	errNum = clEnqueueAcquireGLObjects(commandQueue, 1, &cl_tex_mem, 0, NULL, NULL );

    errNum = clEnqueueNDRangeKernel(commandQueue, tex_kernel, 2, NULL,
                                    tex_globalWorkSize, tex_localWorkSize,
                                    0, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error queuing kernel for execution." << std::endl;
    }
	errNum = clEnqueueReleaseGLObjects(commandQueue, 1, &cl_tex_mem, 0, NULL, NULL );
	clFinish(commandQueue);
	return 0;
}

/// 
// Demonstrate usage of the GL Object querying capabilities
//
void performQueries() {
	cl_int errNum;
	std::cout << "Performing queries on OpenGL objects:" << std::endl;
	// example usage of getting information about a GL memory object
	cl_gl_object_type obj_type;
	GLuint objname;
	errNum = clGetGLObjectInfo( cl_tex_mem,  &obj_type, &objname );
	if( errNum != CL_SUCCESS ) {
		std::cerr << "Failed to get object information" << std::endl;
	} else {
		if( obj_type == CL_GL_OBJECT_TEXTURE2D ) {
			std::cout << "Queried a texture object succesfully." << std::endl;
			std::cout << "Object name is: " << objname << std::endl; 
		}
		
	}

	// Example usage of how to get information about the texture object
	GLenum param;
	size_t param_ret_size;
	errNum = clGetGLTextureInfo( cl_tex_mem, CL_GL_TEXTURE_TARGET, sizeof( GLenum ), &param, &param_ret_size );
	if( errNum != CL_SUCCESS ) {
		std::cerr << "Failed to get texture information" << std::endl;
	} else {
		// we have set it to use GL_TEXTURE_RECTANGLE_ARB.  We expect it to be reflectedin the query here
		if( param == GL_TEXTURE_RECTANGLE_ARB ) {
			std::cout << "Texture rectangle ARB is being used." << std::endl;
		}
	}
}


///
//  Create an OpenCL context on the first available platform using
//  either a GPU or CPU depending on what is available.
//
cl_context CreateContext()
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;
    cl_context context = NULL;

    // First, select an OpenCL platform to run on.  For this example, we
    // simply choose the first available platform.  Normally, you would
    // query for all available platforms and select the most appropriate one.
    errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0)
    {
        std::cerr << "Failed to find any OpenCL platforms." << std::endl;
        return NULL;
    }

    // Next, create an OpenCL context on the platform.  Attempt to
    // create a GPU-based context, and if that fails, try to create
    // a CPU-based context.
    cl_context_properties contextProperties[] =
    {
#ifdef _WIN32
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)firstPlatformId,
		CL_GL_CONTEXT_KHR,
		(cl_context_properties)wglGetCurrentContext(),
		CL_WGL_HDC_KHR,
		(cl_context_properties)wglGetCurrentDC(),
#elif defined( __GNUC__)
		CL_CONTEXT_PLATFORM, (cl_context_properties)clSelectedPlatformID, 
		CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(), 
		CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(), 
#elif defined(__APPLE__) 
		//todo
#endif
        0



    };
	cl_uint uiDevCount;
    cl_device_id* cdDevices;
	// Get the number of GPU devices available to the platform
    errNum = clGetDeviceIDs(firstPlatformId, CL_DEVICE_TYPE_GPU, 0, NULL, &uiDevCount);

    // Create the device list
    cdDevices = new cl_device_id [uiDevCount];
    errNum = clGetDeviceIDs(firstPlatformId, CL_DEVICE_TYPE_GPU, uiDevCount, cdDevices, NULL);


    context = clCreateContext(contextProperties, 1, &cdDevices[0], NULL, NULL, &errNum);
	//// alternate:
    //context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
    //                                  NULL, NULL, &errNum);

    if (errNum != CL_SUCCESS)
    {
        std::cout << "Could not create GPU context, trying CPU..." << std::endl;
        context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
                                          NULL, NULL, &errNum);
        if (errNum != CL_SUCCESS)
        {
            std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
            return NULL;
        }
    }

    return context;
}

///
//  Create a command queue on the first device available on the
//  context
//
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device)
{
    cl_int errNum;
    cl_device_id *devices;
    cl_command_queue commandQueue = NULL;
    size_t deviceBufferSize = -1;

    // First get the size of the devices buffer
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
        return NULL;
    }

    if (deviceBufferSize <= 0)
    {
        std::cerr << "No devices available.";
        return NULL;
    }

    // Allocate memory for the devices buffer
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to get device IDs";
        return NULL;
    }

    // In this example, we just choose the first available device.  In a
    // real program, you would likely use all available devices or choose
    // the highest performance device based on OpenCL device queries
    commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);
    if (commandQueue == NULL)
    {
        std::cerr << "Failed to create commandQueue for device 0";
        return NULL;
    }

    *device = devices[0];
    delete [] devices;
    return commandQueue;
}

///
//  Create an OpenCL program from the kernel source file
//
cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName)
{
    cl_int errNum;
    cl_program program;

    std::ifstream kernelFile(fileName, std::ios::in);
    if (!kernelFile.is_open())
    {
        std::cerr << "Failed to open file for reading: " << fileName << std::endl;
        return NULL;
    }

    std::ostringstream oss;
    oss << kernelFile.rdbuf();

    std::string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1,
                                        (const char**)&srcStr,
                                        NULL, NULL);
    if (program == NULL)
    {
        std::cerr << "Failed to create CL program from source." << std::endl;
        return NULL;
    }

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return NULL;
    }

    return program;
}

///
//  Create memory objects used as the arguments to kernels in OpenCL
//  The memory objects are created from existing OpenGL buffers and textures
//
bool CreateMemObjects(cl_context context, GLuint texture, GLuint vbo, cl_mem *p_cl_vbo_mem, cl_mem *p_cl_tex_mem)
{
	cl_int errNum;

	*p_cl_vbo_mem = clCreateFromGLBuffer(context, CL_MEM_READ_WRITE, vbo, &errNum );
	if( errNum != CL_SUCCESS )
	{
		std::cerr<< "Failed creating memory from GL buffer." << std::endl;
		return false;
	}
	
	*p_cl_tex_mem = clCreateFromGLTexture2D(context, CL_MEM_READ_WRITE, GL_TEXTURE_RECTANGLE_ARB, 0, texture, &errNum );
	if( errNum != CL_SUCCESS )
	{
		std::cerr<< "Failed creating memory from GL texture." << std::endl;
		return false;
	}

    return true;
}

///
//  Cleanup any created OpenCL resources
//
void Cleanup()
{
    if (commandQueue != 0)
        clReleaseCommandQueue(commandQueue);

    if (kernel != 0)
        clReleaseKernel(kernel);

    if (program != 0)
        clReleaseProgram(program);

    if (context != 0)
        clReleaseContext(context);

	if( tex_kernel != 0 ) 
		clReleaseKernel(tex_kernel);

	if( cl_vbo_mem != 0 ) 
		clReleaseMemObject(cl_vbo_mem);

	if( cl_tex_mem != 0 )
		clReleaseMemObject(cl_tex_mem);

	// after we have released the OpenCL references, we can delete the underlying OpenGL objects
	if( vbo != 0 )
	{
		glBindBuffer(GL_ARRAY_BUFFER_ARB, vbo);
		glDeleteBuffers(1, &vbo);
	}
	if( tex != 0 ) 
	{
		glBindBuffer(GL_TEXTURE_RECTANGLE_ARB, tex );
		glDeleteBuffers(1, &tex);
	}
	exit(0);
}

///
//	main() for GLinterop example
//
int main(int argc, char** argv)
{
    cl_device_id device = 0;

	imWidth = 256; 
	imHeight = 256;
	vbolen = imHeight;

	initGlut(argc, argv, imWidth, imHeight);
	vbo = initVBO(vbolen);
	initTexture(imWidth,imHeight);

    // Create an OpenCL context on first available platform
    context = CreateContext();
    if (context == NULL)
    {
        std::cerr << "Failed to create OpenCL context." << std::endl;
        return 1;
    }

    // Create a command-queue on the first device available
    // on the created context
    commandQueue = CreateCommandQueue(context, &device);
    if (commandQueue == NULL)
    {
        Cleanup();
        return 1;
    }

    // Create OpenCL program from GLinterop.cl kernel source
    program = CreateProgram(context, device, "GLinterop.cl");
    if (program == NULL)
    {
        Cleanup();
        return 1;
    }

    // Create OpenCL kernel
    kernel = clCreateKernel(program, "init_vbo_kernel", NULL);
    if (kernel == NULL)
    {
        std::cerr << "Failed to create kernel" << std::endl;
        Cleanup();
        return 1;
    }

	tex_kernel = clCreateKernel(program, "init_texture_kernel", NULL);
    if (tex_kernel == NULL)
    {
        std::cerr << "Failed to create kernel" << std::endl;
        Cleanup();
        return 1;
    }

    // Create memory objects that will be used as arguments to
    // kernel
    if (!CreateMemObjects(context, tex, vbo, &cl_vbo_mem, &cl_tex_mem))
    {
        Cleanup();
        return 1;
    }

	// Perform some queries to get information about the OpenGL objects
	performQueries();

	glutMainLoop();

    std::cout << std::endl;
    std::cout << "Executed program succesfully." << std::endl;
    Cleanup();

    return 0;
}
