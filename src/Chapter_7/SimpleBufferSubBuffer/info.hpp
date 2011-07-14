//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Dan Ginsburg, Timothy Mattson
// ISBN-10:   ??????????
// ISBN-13:   ?????????????
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/??????????
//            http://www.????????.com
//
#ifndef __INFO_HDR__
#define __INFO_HDR__

// info.hpp
// Simple C++ code to abstract clGetInfo*, described in chapter 3.

#if defined(linux) || defined(__APPLE__) || defined(__MACOSX)
# include <alloca.h>
#endif // linux

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

///
// Display information for a particular platform.
// Assumes that all calls to clGetPlatformInfo returns
// a value of type char[], which is valid for OpenCL 1.1.
//
static void DisplayPlatformInfo(
	cl_platform_id id, 
	cl_platform_info name,
	std::string str)
{
	cl_int errNum;
	std::size_t paramValueSize;

	errNum = clGetPlatformInfo(
		id,
		name,
		0,
		NULL,
		&paramValueSize);
	if (errNum != CL_SUCCESS)
	{
		std::cerr << "Failed to find OpenCL platform " << str << "." << std::endl;
		return;
	}

	char * info = (char *)alloca(sizeof(char) * paramValueSize);
	errNum = clGetPlatformInfo(
		id,
		name,
		paramValueSize,
		info,
		NULL);
	if (errNum != CL_SUCCESS)
	{
		std::cerr << "Failed to find OpenCL platform " << str << "." << std::endl;
		return;
	}

	std::cout << "\t" << str << ":\t" << info << std::endl; 
}

template<typename T>
void appendBitfield(T info, T value, std::string name, std::string & str)
{
	if (info & value) 
	{
		if (str.length() > 0)
		{
			str.append(" | ");
		}
		str.append(name);
	}
}		

///
// Display information for a particular device.
// As different calls to clGetDeviceInfo may return
// values of different types a template is used. 
// As some values returned are arrays of values, a templated class is
// used so it can be specialized for this case, see below.
//
template <typename T>
class InfoDevice
{
public:
	static void display(
		cl_device_id id, 
		cl_device_info name,
		std::string str)
	{
		cl_int errNum;
		std::size_t paramValueSize;

		errNum = clGetDeviceInfo(
			id,
			name,
			0,
			NULL,
			&paramValueSize);
		if (errNum != CL_SUCCESS)
		{
			std::cerr << "Failed to find OpenCL device info " << str << "." << std::endl;
			return;
		}

		T * info = (T *)alloca(sizeof(T) * paramValueSize);
		errNum = clGetDeviceInfo(
			id,
			name,
			paramValueSize,
			info,
			NULL);
		if (errNum != CL_SUCCESS)
		{
			std::cerr << "Failed to find OpenCL device info " << str << "." << std::endl;
			return;
		}

		// Handle a few special cases
		switch (name)
		{
		case CL_DEVICE_TYPE:
			{
				std::string deviceType;

				appendBitfield<cl_device_type>(
					*(reinterpret_cast<cl_device_type*>(info)),
					CL_DEVICE_TYPE_CPU, 
					"CL_DEVICE_TYPE_CPU", 
					deviceType);

				appendBitfield<cl_device_type>(
					*(reinterpret_cast<cl_device_type*>(info)),
					CL_DEVICE_TYPE_GPU, 
					"CL_DEVICE_TYPE_GPU", 
					deviceType);

				appendBitfield<cl_device_type>(
					*(reinterpret_cast<cl_device_type*>(info)),
					CL_DEVICE_TYPE_ACCELERATOR, 
					"CL_DEVICE_TYPE_ACCELERATOR", 
					deviceType);

				appendBitfield<cl_device_type>(
					*(reinterpret_cast<cl_device_type*>(info)),
					CL_DEVICE_TYPE_DEFAULT, 
					"CL_DEVICE_TYPE_DEFAULT", 
					deviceType);

				std::cout << "\t\t" << str << ":\t" << deviceType << std::endl;
			}
			break;
		case CL_DEVICE_SINGLE_FP_CONFIG:
			{
				std::string fpType;
				
				appendBitfield<cl_device_fp_config>(
					*(reinterpret_cast<cl_device_fp_config*>(info)),
					CL_FP_DENORM, 
					"CL_FP_DENORM", 
					fpType); 

				appendBitfield<cl_device_fp_config>(
					*(reinterpret_cast<cl_device_fp_config*>(info)),
					CL_FP_INF_NAN, 
					"CL_FP_INF_NAN", 
					fpType); 

				appendBitfield<cl_device_fp_config>(
					*(reinterpret_cast<cl_device_fp_config*>(info)),
					CL_FP_ROUND_TO_NEAREST, 
					"CL_FP_ROUND_TO_NEAREST", 
					fpType); 

				appendBitfield<cl_device_fp_config>(
					*(reinterpret_cast<cl_device_fp_config*>(info)),
					CL_FP_ROUND_TO_ZERO, 
					"CL_FP_ROUND_TO_ZERO", 
					fpType); 

				appendBitfield<cl_device_fp_config>(
					*(reinterpret_cast<cl_device_fp_config*>(info)),
					CL_FP_ROUND_TO_INF, 
					"CL_FP_ROUND_TO_INF", 
					fpType); 

				appendBitfield<cl_device_fp_config>(
					*(reinterpret_cast<cl_device_fp_config*>(info)),
					CL_FP_FMA, 
					"CL_FP_FMA", 
					fpType); 

				appendBitfield<cl_device_fp_config>(
					*(reinterpret_cast<cl_device_fp_config*>(info)),
					CL_FP_SOFT_FLOAT, 
					"CL_FP_SOFT_FLOAT", 
					fpType); 

				std::cout << "\t\t" << str << ":\t" << fpType << std::endl;
			}
		case CL_DEVICE_GLOBAL_MEM_CACHE_TYPE:
			{
				std::string memType;
				
				appendBitfield<cl_device_mem_cache_type>(
					*(reinterpret_cast<cl_device_mem_cache_type*>(info)), 
					CL_NONE, 
					"CL_NONE", 
					memType); 
				appendBitfield<cl_device_mem_cache_type>(
					*(reinterpret_cast<cl_device_mem_cache_type*>(info)), 
					CL_READ_ONLY_CACHE, 
					"CL_READ_ONLY_CACHE", 
					memType); 

				appendBitfield<cl_device_mem_cache_type>(
					*(reinterpret_cast<cl_device_mem_cache_type*>(info)), 
					CL_READ_WRITE_CACHE, 
					"CL_READ_WRITE_CACHE", 
					memType); 

				std::cout << "\t\t" << str << ":\t" << memType << std::endl;
			}
			break;
		case CL_DEVICE_LOCAL_MEM_TYPE:
			{
				std::string memType;
				
				appendBitfield<cl_device_local_mem_type>(
					*(reinterpret_cast<cl_device_local_mem_type*>(info)), 
					CL_GLOBAL, 
					"CL_LOCAL", 
					memType);

				appendBitfield<cl_device_local_mem_type>(
					*(reinterpret_cast<cl_device_local_mem_type*>(info)), 
					CL_GLOBAL, 
					"CL_GLOBAL", 
					memType);
				
				std::cout << "\t\t" << str << ":\t" << memType << std::endl;
			}
			break;
		case CL_DEVICE_EXECUTION_CAPABILITIES:
			{
				std::string memType;
				
				appendBitfield<cl_device_exec_capabilities>(
					*(reinterpret_cast<cl_device_exec_capabilities*>(info)), 
					CL_EXEC_KERNEL, 
					"CL_EXEC_KERNEL", 
					memType);

				appendBitfield<cl_device_exec_capabilities>(
					*(reinterpret_cast<cl_device_exec_capabilities*>(info)), 
					CL_EXEC_NATIVE_KERNEL, 
					"CL_EXEC_NATIVE_KERNEL", 
					memType);
				
				std::cout << "\t\t" << str << ":\t" << memType << std::endl;
			}
			break;
		case CL_DEVICE_QUEUE_PROPERTIES:
			{
				std::string memType;
				
				appendBitfield<cl_device_exec_capabilities>(
					*(reinterpret_cast<cl_device_exec_capabilities*>(info)), 
					CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, 
					"CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE", 
					memType);

				appendBitfield<cl_device_exec_capabilities>(
					*(reinterpret_cast<cl_device_exec_capabilities*>(info)), 
					CL_QUEUE_PROFILING_ENABLE, 
					"CL_QUEUE_PROFILING_ENABLE", 
					memType);
				
				std::cout << "\t\t" << str << ":\t" << memType << std::endl;
			}
			break;
		default:
			std::cout << "\t\t" << str << ":\t" << *info << std::endl;
			break;
		}
	}
};

///
// Simple trait class used to wrap base types.
//
template <typename T>
class ArrayType
{
public:
	static bool isChar() { return false; }
};

///
// Specialized for the char (i.e. null terminated string case).
//
template<>
class ArrayType<char>
{
public:
	static bool isChar() { return true; }
};

///
// Specialized instance of class InfoDevice for array types.
//
template <typename T>
class InfoDevice<ArrayType<T> >
{
public:
	static void display(
		cl_device_id id, 
		cl_device_info name,
		std::string str)
	{
		cl_int errNum;
		std::size_t paramValueSize;

		errNum = clGetDeviceInfo(
			id,
			name,
			0,
			NULL,
			&paramValueSize);
		if (errNum != CL_SUCCESS)
		{
			std::cerr 
				<< "Failed to find OpenCL device info " 
				<< str 
				<< "." 
				<< std::endl;
			return;
		}

		T * info = (T *)alloca(sizeof(T) * paramValueSize);
		errNum = clGetDeviceInfo(
			id,
			name,
			paramValueSize,
			info,
			NULL);
		if (errNum != CL_SUCCESS)
		{
			std::cerr 
				<< "Failed to find OpenCL device info " 
				<< str 
				<< "." 
				<< std::endl;
			return;
		}

		if (ArrayType<T>::isChar())
		{
			std::cout << "\t" << str << ":\t" << info << std::endl; 
		}
		else if (name == CL_DEVICE_MAX_WORK_ITEM_SIZES)
		{
			cl_uint maxWorkItemDimensions;

			errNum = clGetDeviceInfo(
				id,
				CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
				sizeof(cl_uint),
				&maxWorkItemDimensions,
				NULL);
			if (errNum != CL_SUCCESS)
			{
				std::cerr 
					<< "Failed to find OpenCL device info " 
					<< "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS." 
					<< std::endl;
				return;
			}
		
			std::cout << "\t" << str << ":\t" ; 
			for (cl_uint i = 0; i < maxWorkItemDimensions; i++)
			{
				std::cout << info[i] << " "; 
			}
			std::cout << std::endl;
		}
	}
};

#endif __INFO_HDR__