IF (NOT WIN32)
	# Does not currently build on Windows because of the use
	# of libgen.h and getopt.h
	add_executable( spmv spmv.c matrix_gen.c )
	target_link_libraries( spmv ${OPENCL_LIBRARIES} )
ENDIF (NOT WIN32)
