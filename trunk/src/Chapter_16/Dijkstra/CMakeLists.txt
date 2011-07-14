# Use boost for command-line parsing
find_package( Boost 1.40 COMPONENTS program_options REQUIRED )
include_directories( ${Boost_INCLUDE_DIRS} ) 
add_executable( Dijkstra oclDijkstra.cpp oclDijkstraKernel.cpp )
target_link_libraries( Dijkstra ${OPENCL_LIBRARIES} ${Boost_LIBRARIES} )
