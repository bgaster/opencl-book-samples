#include <windows.h>
#include <dxgi.h>
#include <d3d10.h>
#include <d3dx10.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <CL/cl.h>
/* OpenCL D3D10 interop functions are available from the header "cl_d3d10.h".  
   Note that the Khronos extensions for D3D10 are available on the Khronos website.  
   On some distributions you may need to download this file.  
   The sample code assumes this is found in the OpenCL include path. */
#include <CL/cl_d3d10.h>
#include <CL/cl_ext.h>
#pragma OPENCL EXTENSION cl_khr_d3d10_sharing  : enable
#ifndef SAFE_RELEASE
#define SAFE_RELEASE(p)      { if (p) { (p)->Release(); (p)=NULL; } }
#endif
clGetDeviceIDsFromD3D10KHR_fn       clGetDeviceIDsFromD3D10KHR      = NULL;
clCreateFromD3D10BufferKHR_fn		clCreateFromD3D10BufferKHR      = NULL;
clCreateFromD3D10Texture2DKHR_fn	clCreateFromD3D10Texture2DKHR   = NULL;
clCreateFromD3D10Texture3DKHR_fn    clCreateFromD3D10Texture3DKHR   = NULL;
clEnqueueAcquireD3D10ObjectsKHR_fn	clEnqueueAcquireD3D10ObjectsKHR = NULL;
clEnqueueReleaseD3D10ObjectsKHR_fn	clEnqueueReleaseD3D10ObjectsKHR = NULL;
#define INITPFN(x) \
    x = (x ## _fn)clGetExtensionFunctionAddress(#x);\
	if(!x) { printf("failed getting %s" #x); }
//--------------------------------------------------------------------------------------
// Global Variables
//--------------------------------------------------------------------------------------
HWND        g_hWnd = NULL;

const unsigned int    g_WindowWidth  = 256;
const unsigned int    g_WindowHeight = 256;

// Global D3D device/context/feature pointers
ID3D10Device *g_pD3DDevice;
IDXGISwapChain *g_pSwapChain;
D3D_FEATURE_LEVEL g_D3DFeatureLevel;
ID3D10RenderTargetView *g_pRenderTargetView;

// stuff for rendering a triangle, the vertex shader, the vertex buffer, and the input layout structure.
ID3D10Buffer*		g_pVertexBuffer = NULL;
ID3D10Buffer*		g_pSineVertexBuffer = NULL;
ID3D10InputLayout*      g_pVertexLayout = NULL;
ID3D10InputLayout*		g_pSineVertexLayout = NULL;
ID3D10Effect*		g_pEffect = NULL;
ID3D10Texture2D*	g_pTexture2D = NULL;
ID3D10EffectTechnique*  g_pTechnique = NULL;
ID3D10EffectShaderResourceVariable* g_pDiffuseVariable = NULL;
ID3D10ShaderResourceView *pSRView = NULL;

struct SimpleVertex
{
    D3DXVECTOR3 Pos;
    D3DXVECTOR2 Tex; // Texture Coordinate
};

struct SimpleSineVertex
{
	D3DXVECTOR4 Pos;
};


bool verbose = true;


// OpenCL global defines
cl_command_queue commandQueue = 0;
cl_program program = 0;
cl_mem g_clTexture2D = 0;
cl_mem g_clBuffer = 0;
cl_kernel tex_kernel = 0;
cl_kernel buffer_kernel = 0;
cl_context context = 0;


//--------------------------------------------------------------------------------------
// Forward declarations
//--------------------------------------------------------------------------------------
HRESULT InitWindow( HINSTANCE hInstance, int nCmdShow );
HRESULT InitTextures(cl_context context);
HRESULT createRenderTargetViewOfSwapChainBackBuffer(int width, int height);
HRESULT InitDeviceAndSwapChain(int width, int height);
void Cleanup();

///
// Desc: Initializes Direct3D Textures (allocation and initialization)
//
HRESULT InitTextures(cl_context context)
{
	cl_int errNum;
	//
	// create the D3D resources we'll be using
	//
	// 2D texture
	D3D10_TEXTURE2D_DESC desc;
	ZeroMemory( &desc, sizeof(D3D10_TEXTURE2D_DESC) );
	desc.Width = g_WindowWidth;
	desc.Height = g_WindowHeight;
	desc.MipLevels = 1;
	desc.ArraySize = 1;
	desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	desc.SampleDesc.Count = 1;
	desc.Usage = D3D10_USAGE_DEFAULT;
	desc.BindFlags = D3D10_BIND_SHADER_RESOURCE;
	if (FAILED(g_pD3DDevice->CreateTexture2D( &desc, NULL, &g_pTexture2D)))
		return E_FAIL;

	if (FAILED(g_pD3DDevice->CreateShaderResourceView(g_pTexture2D, NULL, &pSRView)) )
		return E_FAIL;
	g_pDiffuseVariable->SetResource(pSRView );
	// Create the OpenCL part
	g_clTexture2D = clCreateFromD3D10Texture2DKHR(
		context,
		CL_MEM_READ_WRITE,
		g_pTexture2D,
		0,
		&errNum);
	if (errNum != CL_SUCCESS)
	{
		if( errNum == CL_INVALID_D3D10_RESOURCE_KHR ) {
			std::cerr<<"Invalid d3d10 texture resource"<<std::endl;
		}
		std::cerr << "Error creating 2D CL texture from D3D10" << std::endl;
		return E_FAIL;
	}

	g_clBuffer = clCreateFromD3D10BufferKHR( context, CL_MEM_READ_WRITE, g_pSineVertexBuffer, &errNum );
	if( errNum != CL_SUCCESS)
	{

		std::cerr << "Error creating buffer from D3D10" << std::endl;
		return E_FAIL;
	}

	return S_OK;
}



///
// Use OpenCL to compute the colors on the texture background
cl_int computeTexture()
{
	cl_int errNum;

	static cl_int seq =0;
	seq = (seq+1)%(g_WindowWidth*2);

    errNum = clSetKernelArg(tex_kernel, 0, sizeof(cl_mem), &g_clTexture2D);
    errNum = clSetKernelArg(tex_kernel, 1, sizeof(cl_int), &g_WindowWidth);
    errNum = clSetKernelArg(tex_kernel, 2, sizeof(cl_int), &g_WindowHeight);
    errNum = clSetKernelArg(tex_kernel, 3, sizeof(cl_int), &seq);
	
	size_t tex_globalWorkSize[2] = { g_WindowWidth, g_WindowHeight };
	size_t tex_localWorkSize[2] = { 32, 4 } ;

	errNum = clEnqueueAcquireD3D10ObjectsKHR(commandQueue, 1, &g_clTexture2D, 0, NULL, NULL );

    errNum = clEnqueueNDRangeKernel(commandQueue, tex_kernel, 2, NULL,
                                    tex_globalWorkSize, tex_localWorkSize,
                                    0, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error queuing kernel for execution." << std::endl;
    }
	errNum = clEnqueueReleaseD3D10ObjectsKHR(commandQueue, 1, &g_clTexture2D, 0, NULL, NULL );
	clFinish(commandQueue);
	return 0;
}

///
// Use OpenCL to compute the colors on the texture background
cl_int computeBuffer()
{
	cl_int errNum;

	static cl_int seq =0;
	seq = (seq+1)%(g_WindowWidth*2);

    errNum = clSetKernelArg(buffer_kernel, 0, sizeof(cl_mem), &g_clBuffer);
    errNum = clSetKernelArg(buffer_kernel, 1, sizeof(cl_int), &g_WindowWidth);
    errNum = clSetKernelArg(buffer_kernel, 2, sizeof(cl_int), &g_WindowHeight);
    errNum = clSetKernelArg(buffer_kernel, 3, sizeof(cl_int), &seq);
	
	size_t buffer_globalWorkSize[1] = { g_WindowWidth };
	size_t buffer_localWorkSize[1] = { 32 } ;

	errNum = clEnqueueAcquireD3D10ObjectsKHR(commandQueue, 1, &g_clBuffer, 0, NULL, NULL );

    errNum = clEnqueueNDRangeKernel(commandQueue, buffer_kernel, 1, NULL,
                                    buffer_globalWorkSize, buffer_localWorkSize,
                                    0, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error queuing kernel for execution." << std::endl;
    }
	errNum = clEnqueueReleaseD3D10ObjectsKHR(commandQueue, 1, &g_clBuffer, 0, NULL, NULL );
	clFinish(commandQueue);
	return 0;
}


//--------------------------------------------------------------------------------------
// Render a frame
//--------------------------------------------------------------------------------------
void Render()
{
    // Clear the back buffer 
    float ClearColor[4] = { 0.0f, 0.125f, 0.1f, 1.0f }; // red,green,blue,alpha
	g_pD3DDevice->ClearRenderTargetView( g_pRenderTargetView, ClearColor);
    // Set the input layout
    g_pD3DDevice->IASetInputLayout( g_pVertexLayout );
    // Set vertex buffer
    UINT stride = sizeof( SimpleVertex );
    UINT offset = 0;
    g_pD3DDevice->IASetVertexBuffers( 0, 1, &g_pVertexBuffer, &stride, &offset );

    // Set primitive topology
    //g_pD3DDevice->IASetPrimitiveTopology( D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST );
    g_pD3DDevice->IASetPrimitiveTopology( D3D10_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP );
	//g_pDiffuseVariable = 
	//	g_pEffect->GetVariableByName("txDiffuse")->AsShaderResource();
	computeTexture();


    // Render the quadrilateral
    D3D10_TECHNIQUE_DESC techDesc;
    g_pTechnique->GetDesc( &techDesc );
  //  for( UINT p = 0; p < techDesc.Passes; ++p )
 //   {
        g_pTechnique->GetPassByIndex( 0 )->Apply( 0 );
        g_pD3DDevice->Draw( 4, 0 );
  //  }

    // Set the input layout
    g_pD3DDevice->IASetInputLayout( g_pSineVertexLayout );
    // Set vertex buffer
    stride = sizeof( SimpleSineVertex );
    offset = 0;
    g_pD3DDevice->IASetVertexBuffers( 0, 1, &g_pSineVertexBuffer, &stride, &offset );

    // Set primitive topology
    g_pD3DDevice->IASetPrimitiveTopology( D3D10_PRIMITIVE_TOPOLOGY_LINESTRIP );

	computeBuffer();
        g_pTechnique->GetPassByIndex( 1 )->Apply( 0 );
        g_pD3DDevice->Draw( 256, 0 );


    // Present the information rendered to the back buffer to the front buffer (the screen)
    g_pSwapChain->Present( 0, 0 );
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

int main( int argc, const char** argv[] )
{
    cl_platform_id	cpPlatform;
	cl_int errNum;
	cl_uint num_devices;
	cl_device_id cdDevice;
    cl_uint numPlatforms;

	//
	// Initialization of a D3D program contains the following steps:
	// 1. Initialize a Window, gets you hWnd (handle to the window)
	// 2. Initialize a SwapChain and Device, which can be done in a single call, makes a device points
	// 3. Create a RenderTargetView of the SwapChains BackBuffer.  
	//     a. Setup the viewport & shader effects to pass through
	//
    if( FAILED( InitWindow( GetModuleHandle(NULL), SW_SHOWDEFAULT ) ) )
        return 0;
	if( FAILED( InitDeviceAndSwapChain( 640, 480 ) ) ) 
		return 0;
	if( FAILED( createRenderTargetViewOfSwapChainBackBuffer(640, 480) ) ) 
		return 0;

    // First, select an OpenCL platform to run on.  For this example, we
    // simply choose the first available platform.  Normally, you would
    // query for all available platforms and select the most appropriate one.
    errNum = clGetPlatformIDs(1, &cpPlatform, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0)
    {
        std::cerr << "Failed to find any OpenCL platforms." << std::endl;
        return NULL;
    }
	char extensionString[256];
	size_t extensionSize;
	errNum = clGetPlatformInfo( cpPlatform, CL_PLATFORM_EXTENSIONS, 256, extensionString, &extensionSize );

	// We could parse the returned string to check for the availability of d3d_sharing extensions.
	// Here, we simply print it for code brevity, and assume it is present
	std::cout<<"Extensions:\n\t"<<extensionString<<std::endl;

    //
	// Initialize extension functions for D3D10
	// See the clGetExtensionFunctionAddress() documentation
	// for more details 
	//
	INITPFN(clGetDeviceIDsFromD3D10KHR);
	INITPFN(clCreateFromD3D10BufferKHR);
	INITPFN(clCreateFromD3D10Texture2DKHR);
	INITPFN(clCreateFromD3D10Texture3DKHR);
	INITPFN(clEnqueueAcquireD3D10ObjectsKHR);
	INITPFN(clEnqueueReleaseD3D10ObjectsKHR);

	// 
	// Given a particular OpenCL platform that has D3D sharing capability, this 
	// function will give valid cl_device_ids for D3D sharing.
	// We'll use the returned cl_device_id to use to create a context
	//
    errNum = clGetDeviceIDsFromD3D10KHR(
        cpPlatform,
        CL_D3D10_DEVICE_KHR,
        g_pD3DDevice,
        CL_PREFERRED_DEVICES_FOR_D3D10_KHR,
        1,
        &cdDevice,
        &num_devices);

	if (errNum == CL_INVALID_PLATFORM) {
		printf("Invalid Platform: Specified platform is not valid\n");
	} else if( errNum == CL_INVALID_VALUE) {
		printf("Invalid Value: d3d_device_source, d3d_device_set is not valid or num_entries = 0 and devices != NULL or num_devices == devices == NULL\n");
	} else if( errNum == CL_DEVICE_NOT_FOUND) {
		printf("No OpenCL devices corresponding to the d3d_object were found\n");
	}

	//
    // Next, create an OpenCL context on the OpenCL device ID returned 
	// above (cl_device_id cdDevice)
	//
	// First set the context to include the D3D device being used
    cl_context_properties contextProperties[] =
    {
		CL_CONTEXT_D3D10_DEVICE_KHR, (cl_context_properties)g_pD3DDevice,
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)cpPlatform,
        0
    };


	//
	// Create the context on the appropriate cl_device_id
	// 
	context = clCreateContext( contextProperties, 1, &cdDevice, NULL, NULL, &errNum ) ;
    if (errNum != CL_SUCCESS)
    {
        std::cout << "Could not create GPU context." << std::endl;
    }

	//
	// Create a command queue on the device
	//
    commandQueue = clCreateCommandQueue(context, cdDevice, 0, NULL);
    if (commandQueue == NULL)
    {
        std::cerr << "Failed to create commandQueue for device 0";
        return NULL;
    }
	// 
	// Create the texture.  We can now use this context for D3D sharing.
	//
	if( InitTextures(context) != S_OK ) {
		printf("Failed to initialize the D3D/OCL texture.\n");
	}

	//
	// Create OpenCL program from D3Dinterop.cl kernel source
	//
    program = CreateProgram(context, cdDevice, "D3Dinterop.cl");
    if (program == NULL)
    {
		std::cerr << "Failed to open or compile D3Dinterop.cl" <<std::endl;
        Cleanup();
        return 1;
    }

	//
	// Create the texture processing kernel
	//
	tex_kernel = clCreateKernel(program, "xyz_init_texture_kernel", NULL);
    if (tex_kernel == NULL)
    {
        std::cerr << "Failed to create kernel" << std::endl;
        Cleanup();
        return 1;
    }

	//
	// Create the buffer processing kernel
	//
	buffer_kernel = clCreateKernel(program, "init_vbo_kernel", NULL);
    if (buffer_kernel == NULL)
    {
        std::cerr << "Failed to create kernel" << std::endl;
        Cleanup();
        return 1;
    }
	computeTexture();

	printf("Initialized D3D/OpenCL sharing.\n");

    // Main message loop
    MSG msg = {0};
    while( WM_QUIT != msg.message )
    {
        if( PeekMessage( &msg, NULL, 0, 0, PM_REMOVE ) )
        {
            TranslateMessage( &msg );
            DispatchMessage( &msg );
        }
        else
        {
            Render();
        }
    }
	
    return ( int )msg.wParam;
}

//-----------------------------------------------------------------------------
// Name: MsgProc()
// Desc: The window's message handler
//-----------------------------------------------------------------------------
bool g_bDone = false;
static LRESULT WINAPI MsgProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch(msg)
    {
        case WM_KEYDOWN:
            if(wParam==VK_ESCAPE) 
			{
				g_bDone = true;
                Cleanup();
	            PostQuitMessage(0);
				return 0;
			}
            break;
        case WM_DESTROY:
			g_bDone = true;
            Cleanup();
            PostQuitMessage(0);
            return 0;
        case WM_PAINT:
            ValidateRect(hWnd, NULL);
            return 0;
    }
    return DefWindowProc(hWnd, msg, wParam, lParam);
}
//--------------------------------------------------------------------------------------
// Register class and create window
//--------------------------------------------------------------------------------------
HRESULT InitWindow( HINSTANCE hInstance, int nCmdShow )
{
    // Register the window class
    WNDCLASSEX wc = { sizeof(WNDCLASSEX), CS_CLASSDC, MsgProc, 0L, 0L,
                      GetModuleHandle(NULL), NULL, NULL, NULL, NULL,
                      L"OpenCL/D3D10 Texture InterOP", NULL };
    if( !RegisterClassEx( &wc) )
        return E_FAIL;

	int xBorder = ::GetSystemMetrics(SM_CXSIZEFRAME);
	int yMenu = ::GetSystemMetrics(SM_CYMENU);
	int yBorder = ::GetSystemMetrics(SM_CYSIZEFRAME);

    // Create the application's window (padding by window border for uniform BB sizes across OSs)
    g_hWnd = CreateWindow( wc.lpszClassName, L"OpenCL/D3D10 Texture InterOP",
                              WS_OVERLAPPEDWINDOW, 0, 0, g_WindowWidth + 2*xBorder, g_WindowHeight+ 2*yBorder+yMenu,
                              NULL, NULL, wc.hInstance, NULL );
    if( !g_hWnd )
        return E_FAIL;

    ShowWindow( g_hWnd, nCmdShow );

    return S_OK;
}

HRESULT InitDeviceAndSwapChain(int width, int height)
{
	HRESULT hr;
	DXGI_SWAP_CHAIN_DESC sd;
	ZeroMemory( &sd, sizeof( sd ) );
	sd.BufferCount = 1;
	sd.BufferDesc.Width = width;
	sd.BufferDesc.Height = height;
	sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	sd.BufferDesc.RefreshRate.Numerator = 60;
	sd.BufferDesc.RefreshRate.Denominator = 1;
	sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	sd.OutputWindow = g_hWnd;
	sd.SampleDesc.Count = 1;
	sd.SampleDesc.Quality = 0;
	sd.Windowed = TRUE;
	D3D10_DRIVER_TYPE g_driverType = D3D10_DRIVER_TYPE_HARDWARE;
	UINT createDeviceFlags = NULL;

	hr = D3D10CreateDeviceAndSwapChain( 
		NULL, 
		g_driverType, 
		NULL, 
		createDeviceFlags, 
		D3D10_SDK_VERSION, &sd, &g_pSwapChain, &g_pD3DDevice);

	if( SUCCEEDED( hr ) ) {	
		std::cout<<"Created D3D10 Hardware device."<<std::endl;
	}
	return hr;
}
///
// ..creates a render target view of the swap chain back buffer.
// Also it will setup the vertex shader and triangle strip for
// drawing 2 onscreen triangles that form a quad.
HRESULT createRenderTargetViewOfSwapChainBackBuffer(int width, int height) 
{
	HRESULT hr = S_OK;
	ID3D10Texture2D* pBackBuffer = NULL;
	if(FAILED(hr = g_pSwapChain->GetBuffer( 0, __uuidof( *pBackBuffer ), ( LPVOID* )&pBackBuffer )))
	{
		MessageBox(NULL,L"m_pSwapChain->GetBuffer failed.",L"Swap Chain Error", MB_OK);
		return hr;
	}	

	if(FAILED(hr = g_pD3DDevice->CreateRenderTargetView( pBackBuffer, NULL, &g_pRenderTargetView )))
	{
		MessageBox(NULL,L"m_pDevice->CreateRenderTargetView failed.",L"Create Render Tgt View Error", MB_OK);
		return hr;
	}

	SAFE_RELEASE(pBackBuffer);

	g_pD3DDevice->OMSetRenderTargets( 1, &g_pRenderTargetView, NULL );


	// Setup the viewport
	D3D10_VIEWPORT vp;
	vp.Width		= (UINT)width;
	vp.Height		= (UINT)height;
	vp.MinDepth		= 0.0f;
	vp.MaxDepth		= 1.0f;
	vp.TopLeftX		= 0;
	vp.TopLeftY		= 0;
	g_pD3DDevice->RSSetViewports( 1, &vp );

    // Create the effect
    DWORD dwShaderFlags = D3D10_SHADER_ENABLE_STRICTNESS;
//#if defined( DEBUG ) || defined( _DEBUG )
    // Set the D3D10_SHADER_DEBUG flag to embed debug information in the shaders.
    // Setting this flag improves the shader debugging experience, but still allows 
    // the shaders to be optimized and to run exactly the way they will run in 
    // the release configuration of this program.
    dwShaderFlags |= D3D10_SHADER_DEBUG;
 //   #endif
    hr = D3DX10CreateEffectFromFile( L"D3Dinterop.fx", NULL, NULL, "fx_4_0", dwShaderFlags, 0,
                                         g_pD3DDevice, NULL, NULL, &g_pEffect, NULL, NULL );
    if( FAILED( hr ) )
    {
        MessageBox( NULL,
                    L"The FX file (D3Dinterop.fx) cannot be located.  Please run this executable from the directory that contains the FX file.", L"Error", MB_OK );
        return hr;
    }

    // Obtain the technique
    g_pTechnique = g_pEffect->GetTechniqueByName( "Render" );	

    // Define the input layout
    D3D10_INPUT_ELEMENT_DESC layout[] =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D10_INPUT_PER_VERTEX_DATA, 0 },
	    { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D10_INPUT_PER_VERTEX_DATA, 0 }, 
    };
    UINT numElements = sizeof( layout ) / sizeof( layout[0] );

    // Create the input layout
    D3D10_PASS_DESC PassDesc;
    g_pTechnique->GetPassByIndex( 0 )->GetDesc( &PassDesc );
    hr = g_pD3DDevice->CreateInputLayout( layout, numElements, PassDesc.pIAInputSignature,
                                          PassDesc.IAInputSignatureSize, &g_pVertexLayout );
    if( FAILED( hr ) )
        return hr;

    // Set the input layout
    g_pD3DDevice->IASetInputLayout( g_pVertexLayout );

    // Create vertex buffer
    SimpleVertex vertices[] =
    {
		{ D3DXVECTOR3( -0.5f, -0.5f, 0.5f ), D3DXVECTOR2( 0.0f, 0.0f ) },
		{ D3DXVECTOR3( -0.5f, 0.5f, 0.5f ), D3DXVECTOR2(  0.0f, 1.0f ) },
		{ D3DXVECTOR3( 0.5f, -0.5f, 0.5f ), D3DXVECTOR2( 1.0f, 0.0f ) },
		{ D3DXVECTOR3(  0.5f,  0.5f, 0.5f ), D3DXVECTOR2( 1.0f, 1.0f ) },
	};

    D3D10_BUFFER_DESC bd;
    bd.Usage = D3D10_USAGE_DEFAULT;
    bd.ByteWidth = sizeof( SimpleVertex ) * 4;
    bd.BindFlags = D3D10_BIND_VERTEX_BUFFER;
    bd.CPUAccessFlags = 0;
    bd.MiscFlags = 0;
    D3D10_SUBRESOURCE_DATA InitData;
    InitData.pSysMem = vertices;
    hr = g_pD3DDevice->CreateBuffer( &bd, &InitData, &g_pVertexBuffer );
    if( FAILED( hr ) )
        return hr;

    // Set vertex buffer
    UINT stride = sizeof( SimpleVertex );
    UINT offset = 0;
    g_pD3DDevice->IASetVertexBuffers( 0, 1, &g_pVertexBuffer, &stride, &offset );

    // Set primitive topology
    //g_pD3DDevice->IASetPrimitiveTopology( D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST );
    g_pD3DDevice->IASetPrimitiveTopology( D3D10_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP );
	g_pDiffuseVariable = 
		g_pEffect->GetVariableByName("txDiffuse")->AsShaderResource();

/////////////////////////
    // Define the input layout
    D3D10_INPUT_ELEMENT_DESC sine_layout[] =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D10_INPUT_PER_VERTEX_DATA, 0 },
    };
    numElements = sizeof( sine_layout ) / sizeof( sine_layout[0] );

    // Create the input layout
    g_pTechnique->GetPassByIndex( 1 )->GetDesc( &PassDesc );
    hr = g_pD3DDevice->CreateInputLayout( sine_layout, numElements, PassDesc.pIAInputSignature,
                                          PassDesc.IAInputSignatureSize, &g_pSineVertexLayout );
    if( FAILED( hr ) )
        return hr;

    // Set the input layout
   // g_pD3DDevice->IASetInputLayout( g_pSineVertexLayout );

    // Create vertex buffer
 //   SimpleSineVertex sinevertices[256] =
 //   {
//		{ D3DXVECTOR4( -0.75f, -0.75f, 0.5f, 0.0f ) },
//		{ D3DXVECTOR4(  0.75f,  0.75f, 0.5f, 0.0f ) },
//};

    bd.Usage = D3D10_USAGE_DEFAULT;
    bd.ByteWidth = sizeof( SimpleSineVertex ) * 256;
    bd.BindFlags = D3D10_BIND_VERTEX_BUFFER;
    bd.CPUAccessFlags = 0;
    bd.MiscFlags = 0;
 //   InitData.pSysMem = sinevertices;
    hr = g_pD3DDevice->CreateBuffer( &bd, NULL, &g_pSineVertexBuffer );
    if( FAILED( hr ) )
        return hr;
/*
    // Set vertex buffer
    stride = sizeof( SimpleSineVertex );
    offset = 0;
    g_pD3DDevice->IASetVertexBuffers( 0, 1, &g_pSineVertexBuffer, &stride, &offset );
*/
    return S_OK;	//return hr;
}

///
//  Cleanup any created OpenCL resources
//
void Cleanup()
{
    if (commandQueue != 0) clReleaseCommandQueue(commandQueue);
    if (tex_kernel != 0) clReleaseKernel(tex_kernel);
    if (program != 0) clReleaseProgram(program);
    if (context != 0) clReleaseContext(context);
	if (g_clTexture2D != 0 ) clReleaseMemObject(g_clTexture2D);

	if( pSRView != NULL ) pSRView->Release(); 
	if( g_pTexture2D != NULL ) g_pTexture2D->Release();
	if( g_pVertexLayout != NULL ) g_pVertexLayout->Release();
	if( g_pEffect != NULL ) g_pEffect->Release();
	if( g_pSwapChain != NULL ) g_pSwapChain->Release();
	if( g_pVertexBuffer != NULL ) g_pVertexBuffer->Release();
	if( g_pD3DDevice != NULL ) g_pD3DDevice->Release();


	exit(0);
}