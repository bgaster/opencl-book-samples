# Use boost for command-line parsing
find_package( Boost COMPONENTS program_options )

# Also, not on Win32 because this uses pthreads and that is
# Linux/Mac only
IF(NOT WIN32)
       IF (Boost_PROGRAM_OPTIONS_FOUND)
       	  include_directories( ${Boost_INCLUDE_DIRS} ) 
	  add_executable( Dijkstra oclDijkstra.cpp oclDijkstraKernel.cpp )
	  target_link_libraries( Dijkstra ${OPENCL_LIBRARIES} ${Boost_LIBRARIES} )
	  configure_file(dijkstra.cl ${CMAKE_CURRENT_BINARY_DIR}/dijkstra.cl COPYONLY)
	ENDIF()
ENDIF()
