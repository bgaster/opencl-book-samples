SUBDIRS(Chapter_2/HelloWorld
	Chapter_3/OpenCLInfo 
	Chapter_3/OpenCLConvolution 
	Chapter_6/HelloBinaryWorld 
	Chapter_7/SimpleBufferSubBuffer
	Chapter_8/ImageFilter2D 
	Chapter_12/VectorAdd
	Chapter_14/histogram 
	Chapter_16/Dijkstra
	Chapter_22/)	

IF(WIN32 OR UNIX AND NOT APPLE)
	# Sinewave uses wgl or glx
	SUBDIRS(Chapter_12/Sinewave)
ENDIF(WIN32 OR UNIX AND NOT APPLE)

IF(WIN32)
	# D3Dinterop is Windows only
	SUBDIRS(Chapter_11/D3Dinterop)
ENDIF(WIN32)
