# NV GLUT paths
SET( CMAKE_INCLUDE_PATH $ENV{NVSDKCOMPUTE_ROOT}/shared/inc )

IF("${ISWIN64}" STREQUAL "Win64") 
	SET( CMAKE_LIBRARY_PATH "$ENV{NVSDKCOMPUTE_ROOT}/shared/lib/x64;$ENV{NVSDKCOMPUTE_ROOT}/OpenCL/common/x64" )
ELSE("${ISWIN64}" STREQUAL "Win64") 
	SET( CMAKE_LIBRARY_PATH "$ENV{NVSDKCOMPUTE_ROOT}/shared/lib/Win32;$ENV{NVSDKCOMPUTE_ROOT}/OpenCL/common/Win32" )
ENDIF("${ISWIN64}" STREQUAL "Win64") 

FIND_PACKAGE(GLUT REQUIRED)
FIND_PACKAGE(OpenGL REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)

# Detect 32/64 bit build environment and set glut  and glew libs correctly
IF(WIN32)
	STRING(REGEX MATCH "Win64" ISWIN64 ${CMAKE_GENERATOR})
	IF("${ISWIN64}" STREQUAL "Win64") 
		SET( GLUT_LIBRARIES "freeglut.lib glew64.lib")
		FIND_PATH( GLEW_LIBRARY_PATH glew64.lib PATHS $ENV{NVSDKCOMPUTE_ROOT}/shared/lib/x64)
	ELSE("${ISWIN64}" STREQUAL "Win64") 
		SET( GLUT_LIBRARIES "freeglut.lib glew32.lib")
		FIND_PATH( GLEW_LIBRARY_PATH glew32.lib PATHS $ENV{NVSDKCOMPUTE_ROOT}/shared/lib/Win32 )
	ENDIF("${ISWIN64}" STREQUAL "Win64") 
ENDIF(WIN32)

# find glew include
FIND_PATH( GLEW_INCLUDE_PATH GL/glew.h PATHS $ENV{NVSDKCOMPUTE_ROOT}/shared/inc )
FIND_PATH( OCLUTILS_INCLUDE_PATH oclUtils.h PATHS $ENV{NVSDKCOMPUTE_ROOT}/OpenCL/common/inc )
FIND_PATH( SHRUTILS_INCLUDE_PATH shrUtils.h PATHS $ENV{NVSDKCOMPUTE_ROOT}/shared/inc )
FIND_LIBRARY( OCLUTILS_LIBRARIES oclUtils32.lib PATHS $ENV{NVSDKCOMPUTE_ROOT}/OpenCL/common/lib/Win32 )

MESSAGE("${GLEW_INCLUDE_PATH}")

#We statically link because OpenCV does this by default so to link there we use the same
# comment this out if all other libraries use dynamic linking
if(MSVC)
foreach(flag_var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
    if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
    endif(${flag_var} MATCHES "/MD")
    if(${flag_var} MATCHES "/MDd")
        string(REGEX REPLACE "/MDd" "/MTd" ${flag_var} "${${flag_var}}")
    endif(${flag_var} MATCHES "/MDd")
endforeach(flag_var)
endif(MSVC)

# copy the .cl kernel files and the data files
configure_file(filters.cl ${CMAKE_CURRENT_BINARY_DIR}/filters.cl COPYONLY)
configure_file(motion.cl ${CMAKE_CURRENT_BINARY_DIR}/motion.cl COPYONLY)
configure_file(lkflow.cl ${CMAKE_CURRENT_BINARY_DIR}/lkflow.cl COPYONLY)
configure_file(data/minicooper/frame10.pgm ${CMAKE_CURRENT_BINARY_DIR}/data/minicooper/frame10.pgm COPYONLY)
configure_file(data/minicooper/frame11.pgm ${CMAKE_CURRENT_BINARY_DIR}/data/minicooper/frame11.pgm COPYONLY)
configure_file(readme.txt ${CMAKE_CURRENT_BINARY_DIR}/readme.txt COPYONLY)



link_directories(${CMAKE_LIBRARY_PATH})
add_executable( oclFlow oclFlow.cpp flowGL.cpp)
include_directories( ${GLUT_INCLUDE_DIR} ${OPENGL_INCLUDE_DIR} ${OCLUTILS_INCLUDE_PATH} ${SHRUTILS_INCLUDE_PATH})
target_link_libraries( oclFlow ${OPENCL_LIBRARIES} ${GLUT_LIBRARIES} ${OpenCV_LIBS} ${OCLUTILS_LIBRARIES} )
