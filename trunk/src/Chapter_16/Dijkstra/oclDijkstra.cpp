//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//


//
//  Description:
//      Implementation of Dijkstra's Single-Source Shortest Path (SSSP) algorithm on the GPU.
//      The basis of this implementation is the paper:
//
//          "Accelerating large graph algorithms on the GPU using CUDA" by
//          Parwan Harish and P.J. Narayanan
//
//      This file is the main driver to test the OpenCL Dijkstra implementation either with
//      randomly generated graph data or pre-canned city data.
//
//  Author:
//      Dan Ginsburg
//      <daniel.ginsburg@childrens.harvard.edu>
//
//  Children's Hospital Boston
//
#include <sstream>
#include <iostream>
#include <boost/program_options.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <stdio.h>
#include "oclDijkstraKernel.h"


///
//  Namespaces
//
namespace po = boost::program_options;
namespace pt = boost::posix_time;


////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////

///
//  Generate a random graph
//
void generateRandomGraph(GraphData *graph, int numVertices, int neighborsPerVertex)
{
    graph->vertexCount = numVertices;
    graph->vertexArray = (int*) malloc(graph->vertexCount * sizeof(int));
    graph->edgeCount = numVertices * neighborsPerVertex;
    graph->edgeArray = (int*)malloc(graph->edgeCount * sizeof(int));
    graph->weightArray = (float*)malloc(graph->edgeCount * sizeof(float));

    for(int i = 0; i < graph->vertexCount; i++)
    {
        graph->vertexArray[i] = i * neighborsPerVertex;
    }

    for(int i = 0; i < graph->edgeCount; i++)
    {
        graph->edgeArray[i] = (rand() % graph->vertexCount);
        graph->weightArray[i] = (float)(rand() % 1000) / 1000.0f;
    }
}

///
//  Parse command line arguments
//
void parseCommandLineArgs(int argc, char **argv, bool &doCPU, bool &doGPU,
                          bool &doMultiGPU, bool &doCPUGPU, bool &doRef,
                          int *sourceVerts,
                          int *generateVerts, int *generateEdgesPerVert)
{
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help",    "Produce help message")
        ("cpu",     "Run CPU version of algorithm")
        ("gpu",     "Run single GPU version of algorithm")
        ("multigpu","Run multi GPU version of algorithm")
        ("cpugpu",  "Run multi GPU+CPU version of algorithm")
        ("ref",     "Run reference version of algorithm")
        ("sources", po::value<int>(), "Number of source vertices to search from (default: 100)")
        ("verts",   po::value<int>(), "Number of vertices in randomly generated graph (default: 100000)")
        ("edges",   po::value<int>(), "Number of edges per vertex in randomly generated graph (default: 10)");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help") || argc == 1)
    {
        std::cout << desc << "\n";
        exit(1);
    }

    // Parse options
    if (vm.count("cpu"))
    {
        doCPU = true;
    }

    if (vm.count("gpu"))
    {
        doGPU = true;
    }

    if (vm.count("multigpu"))
    {
        doMultiGPU = true;
    }

    if (vm.count("cpugpu"))
    {
        doCPUGPU = true;
    }

    if (vm.count("ref"))
    {
        doRef = true;
    }

    if (vm.count("sources"))
    {
        *sourceVerts = vm["sources"].as<int>();
    }

    if (vm.count("verts"))
    {
        *generateVerts = vm["verts"].as<int>();
    }

    if (vm.count("edges"))
    {
        *generateEdgesPerVert = vm["edges"].as<int>();
    }
}

///
/// Gets the id of device with maximal FLOPS from the context (from NVIDIA SDK)
///
static cl_device_id getMaxFlopsDev(cl_context cxGPUContext)
{
    size_t szParmDataBytes;
    cl_device_id* cdDevices;

    // get the list of GPU devices associated with context
    clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, 0, NULL,
            &szParmDataBytes);
    cdDevices = (cl_device_id*) malloc(szParmDataBytes);
    size_t device_count = szParmDataBytes / sizeof(cl_device_id);

    clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, szParmDataBytes,
            cdDevices, NULL);

    cl_device_id max_flops_device = cdDevices[0];
    int max_flops = 0;

    size_t current_device = 0;

    // CL_DEVICE_MAX_COMPUTE_UNITS
    cl_uint compute_units;
    clGetDeviceInfo(cdDevices[current_device], CL_DEVICE_MAX_COMPUTE_UNITS,
            sizeof(compute_units), &compute_units, NULL);

    // CL_DEVICE_MAX_CLOCK_FREQUENCY
    cl_uint clock_frequency;
    clGetDeviceInfo(cdDevices[current_device], CL_DEVICE_MAX_CLOCK_FREQUENCY,
            sizeof(clock_frequency), &clock_frequency, NULL);

    max_flops = compute_units * clock_frequency;
    ++current_device;

    while (current_device < device_count)
    {
        // CL_DEVICE_MAX_COMPUTE_UNITS
        cl_uint compute_units;
        clGetDeviceInfo(cdDevices[current_device], CL_DEVICE_MAX_COMPUTE_UNITS,
                sizeof(compute_units), &compute_units, NULL);

        // CL_DEVICE_MAX_CLOCK_FREQUENCY
        cl_uint clock_frequency;
        clGetDeviceInfo(cdDevices[current_device],
                CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_frequency),
                &clock_frequency, NULL);

        int flops = compute_units * clock_frequency;
        if (flops > max_flops)
        {
            max_flops = flops;
            max_flops_device = cdDevices[current_device];
        }
        ++current_device;
    }

    free(cdDevices);

    return max_flops_device;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    bool doCPU = false;
    bool doGPU = false;
    bool doMultiGPU = false;
    bool doCPUGPU = false;
    bool doRef = false;
    int numSources = 100;
    int generateVerts = 100000;
    int generateEdgesPerVert = 10;

    parseCommandLineArgs(argc, argv, doCPU, doGPU,
                         doMultiGPU, doCPUGPU, doRef,
                         &numSources, &generateVerts, &generateEdgesPerVert);

    cl_platform_id platform;
    cl_context gpuContext;
    cl_context cpuContext;
    cl_int errNum;

    // First, select an OpenCL platform to run on.  For this example, we
    // simply choose the first available platform.  Normally, you would
    // query for all available platforms and select the most appropriate one.
    cl_uint numPlatforms;
    errNum = clGetPlatformIDs(1, &platform, &numPlatforms);
    printf("Number of OpenCL Platforms: %d\n", numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0)
    {
        printf("Failed to find any OpenCL platforms.\n");
        return 1;
    }

    // create the OpenCL context on available GPU devices
    gpuContext = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU, NULL, NULL, &errNum);
    if (errNum != CL_SUCCESS)
    {
        printf("No GPU devices found.\n");
    }

    // Create an OpenCL context on available CPU devices
    cpuContext = clCreateContextFromType(0, CL_DEVICE_TYPE_CPU, NULL, NULL, &errNum);
    if (errNum != CL_SUCCESS)
    {
        printf("No CPU devices found.\n");
    }

    // Allocate memory for arrays
    GraphData graph;
    generateRandomGraph(&graph, generateVerts, generateEdgesPerVert);

    printf("Vertex Count: %d\n", graph.vertexCount);
    printf("Edge Count: %d\n", graph.edgeCount);

    std::vector<int> sourceVertices;


    for(int source = 0; source < numSources; source++)
    {
        sourceVertices.push_back(source % graph.vertexCount);
    }

    int *sourceVertArray = (int*) malloc(sizeof(int) * sourceVertices.size());
    std::copy(sourceVertices.begin(), sourceVertices.end(), sourceVertArray);

    float *results = (float*) malloc(sizeof(float) * sourceVertices.size() * graph.vertexCount);


    // Run Dijkstra's algorithm
    pt::ptime startTimeCPU = pt::microsec_clock::local_time();
    if (doCPU)
    {
        runDijkstra(cpuContext, getMaxFlopsDev(cpuContext), &graph, sourceVertArray,
                    results, sourceVertices.size() );
    }
    pt::time_duration timeCPU = pt::microsec_clock::local_time() - startTimeCPU;

    pt::ptime startTimeGPU = pt::microsec_clock::local_time();
    if (doGPU)
    {
        runDijkstra(gpuContext, getMaxFlopsDev(gpuContext), &graph, sourceVertArray,
                    results, sourceVertices.size() );
    }
    pt::time_duration timeGPU = pt::microsec_clock::local_time() - startTimeGPU;

    pt::ptime startTimeMultiGPU = pt::microsec_clock::local_time();
    if (doMultiGPU)
    {
        runDijkstraMultiGPU(gpuContext, &graph, sourceVertArray,
                            results, sourceVertices.size() );
    }
    pt::time_duration timeMultiGPU = pt::microsec_clock::local_time() - startTimeMultiGPU;


    pt::ptime startTimeGPUCPU = pt::microsec_clock::local_time();
    if (doCPUGPU)
    {
        runDijkstraMultiGPUandCPU(gpuContext, cpuContext, &graph, sourceVertArray,
                                  results, sourceVertices.size() );
    }
    pt::time_duration timeGPUCPU = pt::microsec_clock::local_time() - startTimeGPUCPU;

    pt::ptime startTimeRef = pt::microsec_clock::local_time();
    if (doRef)
    {
        runDijkstraRef( &graph, sourceVertArray,
                        results, sourceVertices.size() );
    }
    pt::time_duration timeRef = pt::microsec_clock::local_time() - startTimeRef;


    if (doCPU)
    {
        printf("\nrunDijkstra - CPU Time:               %f s\n", (float)timeCPU.total_milliseconds() / 1000.0f);
    }

    if (doGPU)
    {
        printf("\nrunDijkstra - Single GPU Time:        %f s\n", (float)timeGPU.total_milliseconds() / 1000.0f);
    }

    if (doMultiGPU)
    {
        printf("\nrunDijkstra - Multi GPU Time:         %f s\n", (float)timeMultiGPU.total_milliseconds() / 1000.0f);
    }

    if (doCPUGPU)
    {
        printf("\nrunDijkstra - Multi GPU and CPU Time: %f s\n", (float)timeGPUCPU.total_milliseconds() / 1000.0f);
    }

    if (doRef)
    {
        printf("\nrunDijkstra - Reference (CPU):        %f s\n", (float)timeRef.total_milliseconds() / 1000.0f);
    }

    free(sourceVertArray);
    free(results);

    clReleaseContext(gpuContext);

    // finish
    //shrEXIT(argc, argv);
 }
