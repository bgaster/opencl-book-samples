# This is an example project to show and test the usage of the FindOpenCL
# script.

cmake_minimum_required( VERSION 2.6 )
project( CL_Book )

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package( OpenCL REQUIRED )

include_directories( ${OPENCL_INCLUDE_DIRS} )
include_directories( "${CMAKE_SOURCE_DIR}/khronos" )

SUBDIRS(src)

