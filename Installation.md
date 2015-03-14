# Introduction #

The sample code for the OpenCL Programming Guide was designed to work on a variety of OpenCL 1.1-compatible devices and platforms.  This page details instructions on building the sample code for a select set of platforms.  The code has been tested on Mac OS X 10.7, Microsoft Windows 7, and Ubuntu Linux 11.04.  The sample code has been tested using the AMD, Nvidia, and Apple OpenCL implementations.


# Prerequisites #

The OpenCL sample code depends on the following libraries:

  * An OpenCL v1.1 implementation for your platform/device
  * [FreeImage](http://freeimage.sourceforge.net/) (optional, required for _Chapter\_8/ImageFilter2D_ example only)
  * [Boost](http://www.boost.org/) (optional, required for _Chapter\_16/Dijkstra_ example, Linux/Mac-only)

# Checkout the source code #

Instructions for checking out the source code from Subversion can be found at https://code.google.com/p/opencl-book-samples/source/checkout.  Checkout the code to a local folder using a Subversion client on your platform and return to this page for further instructions.

# Building the source code #

The following sections describe how to build the source code for each of the platforms that the code has been tested on.

## Microsoft Windows 7 w/ Microsoft Visual Studio 2008 ##

Preparation:

  * Download and unzip the Freeimage DLL installation from http://freeimage.sourceforge.net/download.html to a location on your disk such as c:\Freeimage (optional, required for _Chapter\_8/ImageFilter2D_ example only)
  * Download and install cmake from http://www.cmake.org/cmake/resources/software.html using the Win32 Installer.
  * Download and install OpenCL for your platform, for example the NVIDIA GPU Computing SDK or the AMD Stream SDK.

Building:

  * Run cmake-gui and set "Where is the source code" to the location of the root directory where you checked out the tree (e.g., c:\opencl-book-samples)
  * Set "Where to build the binaries" to the folder you want to store the output, such as c:\opencl-book-samples\build
  * Click "Configure"
  * If you wish to build _Chapter\_8/ImageFilter2D_, then switch to "Advanced View" and set FREEIMAGE\_LIBRARY to c:\Freeimage\Dist\Freeimage.lib and set FREEIMAGE\_INCLUDE\_PATH to c:\Freeimage\Dist and FREEIMAGE\_FOUND to True.
  * Click "Configure" and then "Generate"
  * Now open C:\opencl-book-samples\build\ALL\_BUILD.sln in Visual Studio 2008 and click "Build" and it should build the sample code.


## Mac OS X 10.7 ##

The easiest way to install the prerequisites on Mac OS X is to use http://www.macports.org.  In order to use macports, you will need to install Xcode 4.1.  Once macports is installed, the required ports can be installed with the command:

```
  sudo port install cmake freeimage boost
```

You can then build the code with the following commands:

```
 ~/opencl-book-samples$ mkdir build
 ~/opencl-book-samples$ cd build
 ~/opencl-book-samples$ cmake ../
 ~/opencl-book-samples$ make
```

NOTE: The code has been built on Mac OS X 10.7 Lion.  Some of the examples will not build on previous versions of Mac OS X because Lion is the first version to support OpenCL v1.1.

## Ubuntu Linux 11.04 ##

Installation of the prerequisites on Ubuntu can be done using **apt-get**:

```
  sudo apt-get install subversion cmake libboost-all-dev libfreeimage-dev
```

After installation of your OpenCL implementation, you can build simply by doing:

```
 ~/opencl-book-samples$ mkdir build
 ~/opencl-book-samples$ cd build
 ~/opencl-book-samples$ cmake ../
 ~/opencl-book-samples$ make
```

