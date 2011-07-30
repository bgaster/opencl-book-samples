add_executable( HelloBinaryWorld HelloBinaryWorld.cpp )
target_link_libraries( HelloBinaryWorld ${OPENCL_LIBRARIES} )

configure_file(HelloWorld.cl ${CMAKE_CURRENT_BINARY_DIR}/HelloWorld.cl COPYONLY)