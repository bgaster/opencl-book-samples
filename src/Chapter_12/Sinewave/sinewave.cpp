//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//

// sinewave.cpp
//
//    Simple OpenCL and OpenGL application to  demonstrate use OpenCL C++ Wrapper API.
#include <GL/glew.h>

#ifndef _WIN32
#include <GL/glx.h>
#endif //!_WIN32

#if defined(__APPLE__) || defined(__MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenGL/opengl.h>
#else
#include <GL/gl.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>

#include "bmpLoader.hpp"  // Header file for Bitmap image

#ifndef _WIN32
#include <GL/glx.h>
#endif //!_WIN32

//------------------------------------------------------------------------

#if 0
#define STRINGIFY(A) #A

const char * vertexShader = STRINGIFY(
void main()
{
    gl_TexCoord[0].st = vec2(gl_Vertex.xy) * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    gl_Position    = ftransform();
}
);

const char * pixelShader = STRINGIFY(
uniform sampler2D tex;
void main()
{
  vec3 color         = vec3(texture2D(tex,gl_TexCoord[0].st));
  gl_FragColor       = vec4(color, 1.0);
}
);

GLuint
compileProgram(const char * vsrc, const char * psrc)
{
    GLint err = 0;

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    GLuint pixelShader  = glCreateShader(GL_FRAGMENT_SHADER);

    glShaderSource(vertexShader, 1, &vsrc, 0);
    glShaderSource(pixelShader, 1, &psrc, 0);

    glCompileShader(vertexShader);

    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &err);

      if (!err) {
          char temp[256];
           glGetShaderInfoLog(vertexShader, 256, 0, temp);
           std::cout << "Failed to compile shader: " << temp << std::endl;
      }

    glCompileShader(pixelShader);

    glGetShaderiv(pixelShader, GL_COMPILE_STATUS, &err);

     if (!err) {
         char temp[256];
          glGetShaderInfoLog(pixelShader, 256, 0, temp);
          std::cout << "Failed to compile shader: " << temp << std::endl;
     }

    GLuint program = glCreateProgram();

    glAttachShader(program, vertexShader);
    glAttachShader(program, pixelShader);

    glLinkProgram(program);

    // check if program linked

    err = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &err);

    if (!err) {
        char temp[256];
         glGetProgramInfoLog(program, 256, 0, temp);
         std::cout << "Failed to link program: " << temp << std::endl;
         glDeleteProgram(program);
         program = 0;
    }

    return program;
}

#endif

//------------------------------------------------------------------------

static bool useGPU = false;
size_t local_work_size[3];

static int numVBOs = 1;
static int currVBO = 0;
static int mapAll = 0;
static std::string platformName = "ATI Stream";

// Global CL values
cl::Context context;
cl::Program program;
cl::Kernel kernel;
cl::CommandQueue queue;
cl::Buffer *pVbo;

const unsigned int windowWidth = 512;
const unsigned int windowHeight = 512;

const unsigned int meshWidth  = 512; //256;
const unsigned int meshHeight = 512; //256;

// vbo variables
GLuint *pvbo;

GLuint texture;

float anim = 0.0;

// mouse controls
int mouseOldX;
int mouseOldY;
int mouseButtons = 0;
float rotateX    = 0.0;
float rotateY    = 0.0;
float translateZ = -3.0;

GLuint glProgram;

#if 0
bool loadTexture(GLuint * texture) {
    char filestr[] ="atiStream.bmp";

    BitMap image(filestr);
    if (!image.isLoaded()) {
        std::cout << "ERROR: could not load bitmap " << "filename" << std::endl;
        return false;
    }

    glGenTextures(1, texture );

    glBindTexture(GL_TEXTURE_2D, *texture);

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
        image.getWidth(),
        image.getHeight(),
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        image.getPixels());

    return true;
}
#endif

void runCL()
{
    std::vector<cl::Memory> v;

    if(mapAll) {
        for(int i = 0; i < numVBOs; i++) {
            v.push_back(pVbo[i]);
        }
    }
    else {
        v.push_back(pVbo[currVBO]);
    }

    queue.enqueueAcquireGLObjects(&v);

    cl::KernelFunctor func = kernel.bind(
        queue,
        cl::NDRange(meshWidth, meshHeight),
        cl::NDRange(local_work_size[0], local_work_size[1]));

    if(mapAll) {
        for(int i = 0; i < numVBOs; i++) {
            cl::Event event = func(pVbo[i], meshWidth, meshHeight, anim );
            event.wait();
        }
    }
    else {
        cl::Event event = func(pVbo[currVBO], meshWidth, meshHeight, anim );
        event.wait();
    }

    queue.enqueueReleaseGLObjects(&v);
}

void display(void)
{
    // run OpenCL kernel to generate vertex positions
    runCL();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translateZ);
    glRotatef(rotateX, 1.0, 0.0, 0.0);
    glRotatef(rotateY, 0.0, 1.0, 0.0);

    // render from the vbo
    glBindBuffer(GL_ARRAY_BUFFER, pvbo[currVBO]);
    glVertexPointer(4, GL_FLOAT, 0, 0);

 //   glActiveTexture(GL_TEXTURE0);
 //   glBindTexture(GL_TEXTURE_2D, texture);

    //glUseProgram(glProgram);
    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(1.0, 0.0, 0.0);
    glDrawArrays(GL_POINTS, 0, meshWidth * meshHeight);
    glDisableClientState(GL_VERTEX_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glutSwapBuffers();
    glutPostRedisplay();

    anim += 0.01;

    if(++currVBO >= numVBOs) currVBO = 0;
}


void keyboard( unsigned char key, int /*x*/, int /*y*/)
{
    switch( key) {
    case('q') :
#ifdef _WIN32
    case VK_ESCAPE:
#endif //_WIN32
        queue.finish();
        exit( 0);
    }
}

void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN) {
        mouseButtons |= 1<<button;
    } else if (state == GLUT_UP) {
        mouseButtons = 0;
    }

    mouseOldX = x;
    mouseOldY = y;
    glutPostRedisplay();
}

void motion(int x, int y)
{
    float dx, dy;
    dx = x - mouseOldX;
    dy = y - mouseOldY;

    if (mouseButtons & 1) {
        rotateX += dy * 0.2;
        rotateY += dx * 0.2;
    } else if (mouseButtons & 4) {
        translateZ += dy * 0.01;
    }

    mouseOldX = x;
    mouseOldY = y;
}

void createVBO(GLuint* vbo, cl::Buffer * buffer)
{
    // create buffer object
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);

    // initialize buffer object
    unsigned int size = meshWidth * meshHeight * 4 * sizeof( float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    *buffer = cl::BufferGL(
        context,
        CL_MEM_READ_WRITE,
        *vbo);
}


int
main(int argc, char ** argv)
{
    cl_int err;

    if(numVBOs < 1) {
        numVBOs = 1;
    }

    // Create GL context
    glutInit( &argc, argv);
    glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize( windowWidth, windowHeight);
    glutCreateWindow( "OpenCL C++ Wrapper Example (with GL interop)");

    // GL init
    glewInit();
    if (! glewIsSupported( "GL_VERSION_2_0 " "GL_ARB_pixel_buffer_object")) {
          std::cerr
              << "Support for necessary OpenGL extensions missing."
              << std::endl;
          return EXIT_FAILURE;
    }

    glEnable(GL_TEXTURE_2D);
    glClearColor( 0.0, 0.0, 0.0, 1.0);
    glDisable( GL_DEPTH_TEST);

    glViewport( 0, 0, windowWidth, windowHeight);

    glMatrixMode( GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(
        60.0,
        (GLfloat)windowWidth / (GLfloat) windowHeight,
        0.1,
        10.0);

    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);

    try {
        cl_uint num_platforms;

        err = clGetPlatformIDs(0, NULL, &num_platforms);
        if(CL_SUCCESS != err) {
            std::cerr
                << "ERROR: "
                << "clGetPlatformIDs() returned code " << err
                << std::endl;
            exit(1);
        }
        if(num_platforms == 0)
        {
            std::cerr
                << "ERROR: "
                << "No CL platform is available."
                << std::endl;
            exit(1);
        }
#ifdef linux
#define _malloca    alloca
#endif //linux
        cl_platform_id *platforms = (cl_platform_id*) _malloca(
            sizeof(cl_platform_id) * num_platforms);
        if(!platforms) {
            std::cerr
                << "ERROR: "
                << "Cannot allocate memory from stack."
                << std::endl;
            exit(1);
        }
        err = clGetPlatformIDs(num_platforms, platforms, &num_platforms);
        if(CL_SUCCESS != err) {
            std::cerr
                << "ERROR: "
                << "clGetPlatformIDs() returned code " << err
                << std::endl;
            exit(1);
        }

        // Find requested platform
        size_t cb;
        char Str[256];
        cl_platform_id *platform = NULL;

        for(cl_uint i = 0; i < num_platforms; i++) {
            err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 256, Str, &cb);
            if(CL_SUCCESS != err) {
                std::cerr
                    << "ERROR: "
                    << "clGetPlatformInfo() returned code " << err
                    << std::endl;
                exit(1);
            }
            if(!strcmp(Str, platformName.data())) {
                platform = &platforms[i];
                break;
            }
        }
        if(!platformName.length()) {
            platform = &platforms[0];
        }
        if(!platform) {
            std::cerr
                << "ERROR: "
                << "No \"" << platformName << "\" CL platform found! Exiting..."
                << std::endl;
            exit(1);
        }

#ifdef _WIN32
        HGLRC glCtx = wglGetCurrentContext();
#else //!_WIN32
        GLXContext glCtx = glXGetCurrentContext();
#endif //!_WIN32

        intptr_t properties[] = {
            CL_CONTEXT_PLATFORM, (intptr_t) *platform,
#ifdef _WIN32
            CL_WGL_HDC_KHR, (intptr_t) wglGetCurrentDC(),
#else //!_WIN32
            CL_GLX_DISPLAY_KHR, (intptr_t) glXGetCurrentDisplay(),
#endif //!_WIN32
            CL_GL_CONTEXT_KHR, (intptr_t) glCtx,
            0, 0
        };

        if (useGPU) {
            context = cl::Context(CL_DEVICE_TYPE_GPU, properties);
         } else {
            context = cl::Context(CL_DEVICE_TYPE_CPU, properties);
         }

        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

        std::ifstream fileSrc("sinewave.cl");
        if (!fileSrc.is_open()) {
            std::cout << "Failed to read sinewave.cl";
            exit(1);
        }

        std::string prog(
            std::istreambuf_iterator<char>(fileSrc),
            (std::istreambuf_iterator<char>()));

        cl::Program::Sources source(
            1,
            std::make_pair(prog.c_str(),prog.length()+1));
        program = cl::Program(context, source);

        program.build(devices);

        kernel = cl::Kernel(program, "simpleGL");

        // create VBOs
        pvbo = new GLuint[numVBOs];
        pVbo = new cl::Buffer[numVBOs];

        for(int i = 0; i < numVBOs; i++) {
            createVBO(&pvbo[i], &pVbo[i]);
        }

//        loadTexture(&texture);

        queue = cl::CommandQueue(context, devices[0], 0, &err);

        size_t maxwgsize;
        cl_int err = devices[0].getInfo<size_t>(
            CL_DEVICE_MAX_WORK_GROUP_SIZE,
            &maxwgsize);

        std::vector<size_t> maxwisizes;
        err = devices[0].getInfo< std::vector<size_t> >(
            CL_DEVICE_MAX_WORK_ITEM_SIZES,
            &maxwisizes);

        size_t i = 1;
        while(i * i <= maxwgsize && i <= maxwisizes.front()) {
            i <<= 1;
        }
        local_work_size[0] = i >> 1;
        local_work_size[1] = local_work_size[0];
        local_work_size[2] = 1;

#if 0
        glProgram = compileProgram(vertexShader, pixelShader);
        if (!glProgram) {
            return EXIT_FAILURE;
        }
#endif

        glutMainLoop();
    }
    catch (cl::Error err) {
         std::cerr
             << "ERROR: "
             << err.what()
             << "("
             << err.err()
             << ")"
             << std::endl;

         return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
