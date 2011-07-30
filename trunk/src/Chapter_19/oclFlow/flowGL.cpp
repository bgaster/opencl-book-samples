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
#include <oclUtils.h>
#include <stdio.h>
#include <shrUtils.h>
#include <iostream>

#ifdef _CH_
#pragma package <opencv>
#endif

#define CV_NO_BACKWARD_COMPATIBILITY

#ifndef _EiC
#include <GL/glew.h>            //GLEW lib
#include <GL/freeglut.h>            //GLUT lib
#include "cv.h"
#include "highgui.h"
#include <stdio.h>
#include <ctype.h>
#endif

// If available/desired, this sample can be compiled with IPP for comparison.
#ifdef USE_IPP
#pragma comment(lib, "ippiemerged.lib");
#pragma comment(lib, "ippsemerged.lib");
#pragma comment(lib, "ippcvemerged.lib");
#pragma comment(lib, "ippimerged.lib");
#pragma comment(lib, "ippcvmerged.lib");
#pragma comment(lib, "ippsmerged.lib");
#pragma comment(lib, "ippcorel.lib");
#endif

IplImage *image = 0;
IplImage *c_image = 0;
CvCapture* capture = 0;

int gw = 640;
int gh = 480;
int displayW = 640;
int displayH = 480;
GLuint tex, vbo;

//
// Header information for communicating the flow object 
//
#ifdef MAC
#define SINGLE_CHANNEL_TYPE CL_R
#else
#define SINGLE_CHANNEL_TYPE CL_INTENSITY
#endif 

template<cl_channel_order co, cl_channel_type dt>
struct ocl_image {
    cl_mem image_mem;
    unsigned int w;
    unsigned int h;
    cl_image_format image_format;
} ;
cl_context clCtx;
void shutdown();
cl_context initOCLFlow(GLuint vbo, int devId);
float computeOCLFlow(int curr, int next);
ocl_image<SINGLE_CHANNEL_TYPE, CL_UNSIGNED_INT8> ocl_set_image( cl_context context, unsigned char *image_grey_ub, int w, int h, cl_int &err );
extern ocl_image<SINGLE_CHANNEL_TYPE, CL_UNSIGNED_INT8> images[2];
void ocl_set_image( ocl_image<SINGLE_CHANNEL_TYPE, CL_UNSIGNED_INT8> img, cl_context context, unsigned char *image_grey_ub, cl_int &err );
ocl_image<SINGLE_CHANNEL_TYPE, CL_UNSIGNED_INT8> ocl_init_image( cl_context context, unsigned char *image_grey_ub, int w, int h, cl_int &err );
extern char device_string[1024];
//
// end flow header info
//
GLuint initVBO( int , int );
int currentFrame = 0;
// IPP version

struct controlState {
    bool use_IPP;
	bool bqatest;
};

controlState state;

// todo: has lots of mallocs, get rid of those. for now show timing for compute only
float calc_flow_IPP( 
    unsigned int width, 
    unsigned int height,
    unsigned char *imgin0,
    unsigned char *imgin1,
    int niter );

// called when window size changes
void changeSize(int w, int h) { 
    //stores the width and height
    displayW = w;
    displayH = h;
    //Set the viewport (fills the entire window)
    glViewport(0,0,displayW,displayH);
    //Go to projection mode and load the identity
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    //Orthographic projection, stretched to fit the screen dimensions 
    gluOrtho2D(0,gw,gh,0);
    //Go back to modelview and load the identity
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void swapIdx( int *currentIdx ) 
{
        if( *currentIdx == 0 ) { 
            *currentIdx = 1;
        } else { 
            *currentIdx = 0;
        }
}

int getNextIdx( int currentIdx )
{
    if( currentIdx == 0 ) return 1;
    if( currentIdx == 1 ) return 0;
    assert (false);
    return -1;
}

void renderBitmapString(
        float x, 
        float y, 
        void *font, 
        char *string) {  
  char *c;
  glRasterPos2f(x, y);
  for (c=string; *c != '\0'; c++) {
    glutBitmapCharacter(font, *c);
  }
}

void renderQuiver(int w, int h) 
{
	// set line width & color for visualization 
    glLineWidth( 1.5f);
    glColor3f(0.0f, 1.0f, 0.0f);

    // Draw VBO containing the point list coordinates, to place GL_POINTS at feature locations
    // bind VBOs for vertex array and index array
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo);         // for vertex coordinates
    glEnableClientState(GL_VERTEX_ARRAY);             // activate vertex coords array
    glVertexPointer( 2, GL_FLOAT, 0, 0 );
	 
    // draw lines with endpoints given in the array
    glDrawArrays(GL_LINES, 0, w*h*2);

    glDisableClientState(GL_VERTEX_ARRAY);            // deactivate vertex array

    // bind with 0, so, switch back to normal pointer operation
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
}

IplImage *opencvImages[2];

float grabFrame()
{
        IplImage* frame = 0;
        /*int i, bin_w, c;*/
        static int currentIdx = 0;
        cl_int err;
        float t_flow = 0.0f;

        if( capture) 
		{
			frame = cvQueryFrame( capture );
			if( !frame ) shutdown();
		}
        if( !image )
        {
				CvSize imSz;
				if( capture ) imSz = cvGetSize(frame);
				else {
					imSz.width = gw;
					imSz.height = gh;
				}
                /* allocate all the buffers */
			    image  = cvCreateImage(imSz, IPL_DEPTH_8U, 1 );
                if( capture ) image->origin = frame->origin;
            
                c_image = cvCreateImage( imSz, IPL_DEPTH_8U, 3);
                if( capture) c_image->origin = frame->origin;
    
                opencvImages[0]  = cvCreateImage( imSz, IPL_DEPTH_8U, 1 );
                if( capture) opencvImages[0]->origin = frame->origin;

                opencvImages[1]  = cvCreateImage( imSz, IPL_DEPTH_8U, 1 );
                if( capture) opencvImages[1]->origin = frame->origin;
				// if capturing, convert the first image
				if( capture ) {
					cvCvtColor( frame, image, CV_RGB2GRAY ); // try hsv later
					cvCvtColor( frame, opencvImages[currentIdx], CV_RGB2GRAY ); // try hsv later
					cvCvtColor( frame, opencvImages[getNextIdx(currentIdx)], CV_RGB2GRAY ); // try hsv later
					ocl_set_image(images[0], clCtx, (unsigned char *)image->imageData, err );
					ocl_set_image(images[1], clCtx, (unsigned char *)image->imageData, err );
				}
        }

        /* CV_RGB2GRAY: convert RGB image to grayscale */
		if( capture ) {
			cvCopy( frame, c_image);
			cvCvtColor( frame, image, CV_RGB2GRAY ); // try hsv later
			ocl_set_image(images[getNextIdx(currentIdx)], clCtx, (unsigned char *)image->imageData, err );
		}

       t_flow = computeOCLFlow( currentIdx, getNextIdx(currentIdx) );

        if( state.use_IPP ) {
                if( capture ) cvCvtColor( frame, opencvImages[getNextIdx(currentIdx)], CV_RGB2GRAY ); // try hsv later
#ifdef USE_IPP 
                t_flow = calc_flow_IPP( image->width, image->height, 
                                (unsigned char *)opencvImages[currentIdx]->imageData,
                                (unsigned char *)opencvImages[getNextIdx(currentIdx)]->imageData,
                                 8 );
#else 
				fprintf(stderr, "IPP not enabled at compile.\n");
#endif
        }
        swapIdx( &currentIdx );
        fprintf(stderr, ".");
        return t_flow;
}

void renderScene(void) 
{

    float t_flow = 0.0f;
    t_flow = grabFrame();

    // display the pixel buffer
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, tex);
    // when the pbo is active, the source for this copy is the pbo
    if( capture) glTexSubImage2D(GL_TEXTURE_RECTANGLE_NV, 0, 0, 0, gw, gh, GL_BGR, GL_UNSIGNED_BYTE, c_image->imageData );
    assert( glGetError() == GL_NO_ERROR );

    //Set the clear color (black)
    glClearColor(0.0,0.0,0.0,1.0);
    //Clear the color buffer
    glClear(GL_COLOR_BUFFER_BIT);

    //stretch to screen
    glViewport(0,0,displayW,displayH);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0,gw,gh,0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

	if( true ) {
		glEnable(GL_TEXTURE_RECTANGLE_NV);
		glBindTexture(GL_TEXTURE_RECTANGLE_NV, tex) ;

		glBegin(GL_QUADS);

		glTexCoord2f(0, 0 );
		glVertex2f(0.0, 0 );

		glTexCoord2f(0, gh);
		glVertex2f(0.0, gh);

		glTexCoord2f(gw, gh);
		glVertex2f(gw,gh);

		glTexCoord2f( gw, 0 );
		glVertex2f(gw, 0);
		glEnd();
		glDisable(GL_TEXTURE_RECTANGLE_NV);

	}

    renderQuiver(gw,gh);
    //swap buffers (double buffering)
    int vertPos = 20;
    char str[256] ;
    glColor3f(.8f, .8f, .2f);
    sprintf(str, "Lucas Kanade Pyramidal Optical Flow,   Dense (%dx%d points)", gw, gh );
    renderBitmapString( 10, vertPos, GLUT_BITMAP_HELVETICA_18,str); vertPos += 20;
    if( state.use_IPP ) {
#ifdef USE_IPP
        sprintf(str, "Hardware: CPU");
#else
		sprintf(str, "IPP Not enabled.");
#endif
	} else {
        sprintf(str, "Hardware: %s", device_string);
    }
    renderBitmapString( 10, vertPos, GLUT_BITMAP_HELVETICA_18,str); vertPos += 20;
    sprintf(str, "Processing Time/frame: %f ms", t_flow );
    renderBitmapString( 10, vertPos, GLUT_BITMAP_HELVETICA_18,str); vertPos +=20;

    glutSwapBuffers();
	if(state.bqatest) exit(0);
}

void keyboard( unsigned char c, int x, int y ) {
       switch( (char) c )
        {
        case 'c':
            state.use_IPP = !state.use_IPP;
            break;
        case 'q':
        case 27:
            shutdown();
#ifdef _MSC_VER
			Sleep(1000);
#else
            sleep(1);
#endif
            exit(0);
            break;
        default:
            ;
        }
}

GLuint initVBO(int imWidth, int imHeight )
{
    int bpp = 4;
    GLint bsize;

    GLuint vbo_buffer; 
    // generate the buffer
    glGenBuffers(1, &vbo_buffer);
    
    // bind the buffer 
    glBindBuffer(GL_ARRAY_BUFFER, vbo_buffer); 
    assert( glGetError() == GL_NO_ERROR );
    
    // create the buffer, this basically sets/allocates the size
    glBufferData(GL_ARRAY_BUFFER, imWidth * imHeight *sizeof(float)*4, NULL, GL_STREAM_DRAW);  
    assert( glGetError() == GL_NO_ERROR );

    // recheck the size of the created buffer to make sure its what we requested
    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &bsize); 
    if ((GLuint)bsize != (imWidth * imHeight*sizeof(float)*4)) {
        printf("Vertex Buffer object (%d) has incorrect size (%d).\n", (unsigned)vbo_buffer, (unsigned)bsize);
    }

    // we're done, so unbind the buffers
    glBindBuffer(GL_ARRAY_BUFFER, 0);                    
    assert( glGetError() == GL_NO_ERROR );
    return vbo_buffer;
}

void initGlut(int argc, char *argv[], int wWidth, int wHeight)
{
    glutInit(&argc, argv);
    //glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(100,100);
    glutInitWindowSize(displayW, displayH);
    glutCreateWindow("GPU LK Optical Flow");

    glutDisplayFunc(renderScene);
    glutIdleFunc(renderScene);
    glutReshapeFunc(changeSize);
    glutKeyboardFunc(keyboard);


    //     We now setup GLEW and see if we hae the necessary OpenGL version support

    glewInit();
    if (glewIsSupported("GL_VERSION_2_1"))
        printf("Ready for OpenGL 2.1\n");
    else {
		printf("Warning: Detected that OpenGL 2.1 not supported\n");
    }
    if (GLEW_ARB_vertex_shader && GLEW_ARB_fragment_shader && GL_EXT_geometry_shader4)
        printf("Ready for GLSL - vertex, fragment, and geometry units\n");
    else {
        printf("Not totally ready :( \n");
        exit(1);
    }
}

void initGLData(int wWidth, int wHeight, void *ptr ) 
{
    // make a texture for the video
    glGenTextures(1, &tex);              // texture 
    glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE,  GL_REPLACE );
    glBindTexture(GL_TEXTURE_RECTANGLE_NV, tex);
    glTexImage2D(GL_TEXTURE_RECTANGLE_NV, 0, GL_RGB, wWidth,
            wHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, ptr );

    vbo =  initVBO( wWidth, wHeight );
}

void shutdown()
{
    cvReleaseCapture( &capture );
}


int main( int argc, char** argv )
{
    bool setQuit = false;
    int capWidth = gw;
    int capHeight = gh;

    state.use_IPP = false;
	state.bqatest = false;
	unsigned int devN = 0;
	if( shrGetCmdLineArgumentu(argc, (const char **)argv, "device", &devN ) ) {
		printf("Using device %d\n", devN);
	}
	if( shrCheckCmdLineFlag( argc,  (const char **)argv, "qatest") ) {
		printf("QA test mode.\n");
		capture = NULL;
		state.bqatest = true;
	} else {

		printf("Attempting to initialize camera\n");
		if( argc == 1 || (argc == 2 && strlen(argv[1]) == 1 && isdigit(argv[1][0])))
			capture = cvCaptureFromCAM( argc == 2 ? argv[1][0] - '0' : 0 );
		else if( argc == 2 )
			capture = cvCaptureFromAVI( argv[1] );
	}

	if( !capture )
	{
		fprintf(stderr,"Could not initialize capturing...\n");
		fprintf(stderr,"Attempting to use PGM files..\n");
	} else { 
		printf("Camera Initialized\n");
		printf("Setting Size\n");
#ifdef _MSC_VER
		Sleep(5000);
#else
		sleep(5); // pause 5 seconds before setting size on Mac , otherwise unstable
#endif
		cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, gw);
		cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, gh);

	}

    initGlut(argc,argv, gw, gh);
	initGLData(gw, gh, NULL);
	// Initialize openCL
	// loads the CL functions from the .cl files
	// also loads reference images../data/minicooper/frame10.pgm
	
    clCtx = initOCLFlow(vbo,devN);
	if( !capture ) {
		if( images[0].w != gw || images[0].h != gh || images[1].w != gw || images[1].h != gh ) {
			fprintf(stderr, "Bad image sizes supplied. Please use %d x %d images\n", gw, gh );
		}

		// load the file into the texture (actually initOCLFLow loaded them into GPU but discarded them
		// so this is justa quick way to load them into the texture as well)
		unsigned int w, h;
		unsigned char *image_ub = NULL;
		shrLoadPGMub( "data/minicooper/frame10.pgm", (unsigned char **)&image_ub, &w, &h );
		glBindTexture(GL_TEXTURE_RECTANGLE_NV, tex);
		glTexSubImage2D(GL_TEXTURE_RECTANGLE_NV, 0, 0, 0, gw, gh, GL_LUMINANCE, GL_UNSIGNED_BYTE, image_ub );
	}
    glutMainLoop();

    return 0;
}

#ifdef _EiC
main(1,"camshiftdemo.c");
#endif
