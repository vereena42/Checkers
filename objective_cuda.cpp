#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <exception>

#include "cuda.h"

#include "objective_cuda.h"

namespace CuError {
    const char *strerror(CUresult result) { 
        switch(result) { 
            case CUDA_SUCCESS: return "No errors"; 
            case CUDA_ERROR_INVALID_VALUE: return "Invalid value"; 
            case CUDA_ERROR_OUT_OF_MEMORY: return "Out of memory"; 
            case CUDA_ERROR_NOT_INITIALIZED: return "Driver not initialized"; 
            case CUDA_ERROR_DEINITIALIZED: return "Driver deinitialized"; 

            case CUDA_ERROR_NO_DEVICE: return "No CUDA-capable device available"; 
            case CUDA_ERROR_INVALID_DEVICE: return "Invalid device"; 

            case CUDA_ERROR_INVALID_IMAGE: return "Invalid kernel image"; 
            case CUDA_ERROR_INVALID_CONTEXT: return "Invalid context"; 
            case CUDA_ERROR_CONTEXT_ALREADY_CURRENT: return "Context already current"; 
            case CUDA_ERROR_MAP_FAILED: return "Map failed"; 
            case CUDA_ERROR_UNMAP_FAILED: return "Unmap failed"; 
            case CUDA_ERROR_ARRAY_IS_MAPPED: return "Array is mapped"; 
            case CUDA_ERROR_ALREADY_MAPPED: return "Already mapped"; 
            case CUDA_ERROR_NO_BINARY_FOR_GPU: return "No binary for GPU"; 
            case CUDA_ERROR_ALREADY_ACQUIRED: return "Already acquired"; 
            case CUDA_ERROR_NOT_MAPPED: return "Not mapped"; 
            case CUDA_ERROR_NOT_MAPPED_AS_ARRAY: return "Mapped resource not available for access as an array"; 
            case CUDA_ERROR_NOT_MAPPED_AS_POINTER: return "Mapped resource not available for access as a pointer"; 
            case CUDA_ERROR_ECC_UNCORRECTABLE: return "Uncorrectable ECC error detected"; 
            case CUDA_ERROR_UNSUPPORTED_LIMIT: return "CUlimit not supported by device";    

            case CUDA_ERROR_INVALID_SOURCE: return "Invalid source"; 
            case CUDA_ERROR_FILE_NOT_FOUND: return "File not found"; 
            case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND: return "Link to a shared object failed to resolve"; 
            case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED: return "Shared object initialization failed"; 

            case CUDA_ERROR_INVALID_HANDLE: return "Invalid handle"; 

            case CUDA_ERROR_NOT_FOUND: return "Not found"; 

            case CUDA_ERROR_NOT_READY: return "CUDA not ready"; 

            case CUDA_ERROR_LAUNCH_FAILED: return "Launch failed"; 
            case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: return "Launch exceeded resources"; 
            case CUDA_ERROR_LAUNCH_TIMEOUT: return "Launch exceeded timeout"; 
            case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING: return "Launch with incompatible texturing"; 

            case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED: return "Host memory not registered";
            case CUDA_ERROR_NOT_PERMITTED: return "Not permitted";
            case CUDA_ERROR_NOT_SUPPORTED: return "Not supported";

            case CUDA_ERROR_UNKNOWN: return "Unknown error"; 

            default: return "Unknown CUDA error value"; 
        } 
    }

    void cu_assert(CUresult result, const char *message) {
        if (result != CUDA_SUCCESS) {
            throw CuException("%s : %s : %d", message, strerror(result), result);
        }
    }
}

Cuda::Cuda(int device_id, int init_flags) {
    // Init driver
    cuInit(init_flags);

    // Get device handler
    CUresult res = cuDeviceGet(&cuDevice, device_id);
    if (res != CUDA_SUCCESS) {
        throw CuError::DeviceNotExist("Device id: %d\n%s", device_id, CuError::strerror(res));
    }

    // Create context
    res = cuCtxCreate(&cuContext, 0, cuDevice);
    if (res != CUDA_SUCCESS) {
        throw CuError::ContextCreateError("%s", CuError::strerror(res));
    }
}

Cuda::~Cuda() {
    cuCtxDestroy(cuContext);
}

CUmodule Cuda::set_default_module(CUmodule module) {
    // Set default module from existing module
    default_module = module;
    return module;
}

CUmodule Cuda::set_default_module(const char *module_name) {
    // Set default module from path
    return set_default_module(create_module(module_name));
}

CUmodule Cuda::create_module(const char *module_name) {
    // Create module from .ptx
    CUmodule cuModule = (CUmodule)0;
    CUresult res = cuModuleLoad(&cuModule, module_name);
    if (res != CUDA_SUCCESS) {
        throw CuError::ModuleLoadError("Module id: %d\n%s", res, CuError::strerror(res));
    }
    return cuModule;
}

CUfunction Cuda::get_kernel(const char *kernel_name) {
    if (default_module) {
        return get_kernel(kernel_name, default_module);
    } else {
        throw CuError::DefaultModuleNotExist();
    }
}

CUfunction Cuda::get_kernel(const char *kernel_name, CUmodule cuModule) {
    // Get kernel handler from module
    CUfunction kernel;
    CUresult res = cuModuleGetFunction(&kernel, cuModule, kernel_name);
    if (res != CUDA_SUCCESS) {
        throw CuError::KernelGetError("Kernel name: %s\n%s", kernel_name, CuError::strerror(res));
    }
    return kernel;
}

void Cuda::launch_kernel_3d(CUfunction kernel,
        uint grid_dim_x, uint grid_dim_y, uint grid_dim_z,
        uint block_dim_x, uint block_dim_y, uint block_dim_z,
        void ** args,
        uint shared_mem_bytes,
        CUstream h_stream,
        void ** extra) {

    assert(grid_dim_x > 0);
    assert(grid_dim_y > 0);
    assert(grid_dim_z > 0);
    assert(block_dim_x > 0);
    assert(block_dim_y > 0);
    assert(block_dim_z > 0);

    if (args == NULL) {
        args = new void *[0];
    }

    CUresult res = cuLaunchKernel(kernel,
            grid_dim_x, grid_dim_y, grid_dim_z,
            block_dim_x, block_dim_y, block_dim_z,
            shared_mem_bytes,
            h_stream,
            args,
            extra);
    if (res != CUDA_SUCCESS){
        throw CuError::KernelLaunchError("%s", CuError::strerror(res));
    }
}
void Cuda::launch_kernel(CUfunction kernel,
        uint grid_dim_x,
        uint block_dim_x,
        void ** args,
        uint shared_mem_bytes,
        CUstream h_stream,
        void ** extra) {
    launch_kernel_3d(kernel,
            grid_dim_x, 1, 1,
            block_dim_x, 1, 1,
            args,
            shared_mem_bytes,
            h_stream,
            extra);
}
void Cuda::launch_kernel_2d(CUfunction kernel,
        uint grid_dim_x, uint grid_dim_y,
        uint block_dim_x, uint block_dim_y,
        void ** args,
        uint shared_mem_bytes,
        CUstream h_stream,
        void ** extra) {
    launch_kernel_3d(kernel,
            grid_dim_x, grid_dim_y, 1,
            block_dim_x, block_dim_y, 1,
            args,
            shared_mem_bytes,
            h_stream,
            extra);
}

void Cuda::ctx_synchronize() {
    CUresult res = cuCtxSynchronize();
    if (res != CUDA_SUCCESS) {
        throw CuError::ContextSynchronizeError("%s", CuError::strerror(res));
    }
}
