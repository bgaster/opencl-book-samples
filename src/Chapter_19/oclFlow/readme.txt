This will build the LK optical flow sample
Requires: 

* OpenCV for camera handling 
  * Note: OpenCV paths are stored by CMAKE after you compile OpenCV using CMAKE.
  * so, compile OpenCV then CMAKE should find them. 
  * Alternately set the paths manually in the cmake setup.

* oclUtils/shrUtils from the NVIDIA GPU Computing SDK. 

When running, you will need to set paths to the necessary DLLs, such as: 

PATH=%PATH%;%NVSDKCOMPUTE_ROOT%/OpenCL/bin/win32/Debug;C:\OpenCV2.3\build2\bin\Debug
