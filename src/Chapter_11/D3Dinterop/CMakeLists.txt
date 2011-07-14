# NV GLUT paths
SET( CMAKE_INCLUDE_PATH $ENV{NVSDKCOMPUTE_ROOT}/shared/inc )
SET( CMAKE_LIBRARY_PATH $ENV{NVSDKCOMPUTE_ROOT}/shared/lib )
FIND_PACKAGE(DirectX REQUIRED)
add_definitions(-DUNICODE -D_UNICODE)
link_directories(${CMAKE_LIBRARY_PATH})
add_executable( D3Dinterop D3Dinterop.cpp )
include_directories( ${DX10_INCLUDE_PATH})
target_link_libraries( D3Dinterop ${OPENCL_LIBRARIES} ${D3D10_LIBRARY} ${D3DX10_LIBRARY})
