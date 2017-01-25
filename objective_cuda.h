#ifndef OBJECTIVE_CUDA_H
#define OBJECTIVE_CUDA_H

#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <exception>

#include "cuda.h"

#ifndef CUDA_PTX_PREFIX
#define CUDA_PTX_PREFIX
#endif

namespace CuError {
    struct CuException : public std::exception {  // TODO make it nicer
        char *msg;
        mutable char *return_msg;

        CuException() {
            msg = new char[1];
            msg[0] = 0;
            return_msg = NULL;
        }

        CuException(const char* format, ...) {
            char buffer[1024];
            va_list args;
            va_start (args, format);
            vsprintf (buffer, format, args);
            perror (buffer);
            va_end (args);

            msg = new char[strlen(buffer) + 1];
            strcpy(msg, buffer);
            return_msg = NULL;
        }

        ~CuException() throw() {
            if (return_msg) {
                delete [] return_msg;
            }
            if (msg) {
                delete [] msg;
            }
        }

        virtual const char* default_message() const throw() {
            return "";
        }

        virtual const char* what() const throw () {
            if (! return_msg) {
                return_msg = new char[strlen(msg), strlen(default_message()) + 1];
                strcpy(return_msg, default_message());
                strcat(return_msg, msg);
            }
            return return_msg;
        }
    };

    #define CU_EXCEPTION(name,def_msg) \
            struct name : public CuException { \
                name () {} name (const char *format, ...) { \
                    char buffer[1024]; \
                    va_list args; \
                    va_start (args, format); \
                    vsprintf (buffer, format, args); \
                    perror (buffer); \
                    va_end (args); \
                    msg = new char[strlen(buffer) + 1]; \
                    strcpy(msg, buffer); \
                    return_msg = NULL; \
                } \
                virtual const char* default_message() const throw() { return def_msg; } \
            };

        CU_EXCEPTION(DeviceNotExist, "Device does not exist\n");
        CU_EXCEPTION(ContextCreateError, "Context could not be created\n");
        CU_EXCEPTION(ModuleLoadError, "Module could not be loaded\n");
        CU_EXCEPTION(DefaultModuleNotExist, "Default kernel is no set\n");
        CU_EXCEPTION(KernelGetError, "Kernel could not be acquired\n");
        CU_EXCEPTION(KernelLaunchError, "Kernel could not be launched\n");
        CU_EXCEPTION(ContextSynchronizeError, "Context synchronization failed\n");

    #undef CU_EXCEPTION

    const char *strerror(CUresult result);

    void cu_assert(CUresult result, const char *message="");
}

class Cuda{
    private:
        CUdevice cuDevice;
        CUmodule default_module;
        CUcontext cuContext;

    public:
        Cuda(int device_id=0, int init_flags=0);

        ~Cuda();

        CUmodule set_default_module(CUmodule module);

        CUmodule set_default_module(const char *module_name);

        CUmodule create_module(const char *module_name);

        CUfunction get_kernel(const char *kernel_name);

        CUfunction get_kernel(const char *kernel_name, CUmodule cuModule);

        void launch_kernel_3d(CUfunction kernel,
                uint grid_dim_x, uint grid_dim_y, uint grid_dim_z,
                uint block_dim_x, uint block_dim_y, uint block_dim_z,
                void ** args = NULL,
                uint shared_mem_bytes = 0,
                CUstream h_stream = 0,
                void ** extra = NULL);

        void launch_kernel(CUfunction kernel,
                uint grid_dim_x,
                uint block_dim_x,
                void ** args = NULL,
                uint shared_mem_bytes = 0,
                CUstream h_stream = 0,
                void ** extra = NULL);

        void launch_kernel_2d(CUfunction kernel,
                uint grid_dim_x, uint grid_dim_y,
                uint block_dim_x, uint block_dim_y,
                void ** args = NULL,
                uint shared_mem_bytes = 0,
                CUstream h_stream = 0,
                void ** extra = NULL);

        inline void launch_kernel_sync(CUfunction kernel,
                uint grid_dim_x,
                uint block_dim_x,
                void ** args = NULL,
                uint shared_mem_bytes = 0,
                CUstream h_stream = 0,
                void ** extra = NULL) {
            launch_kernel(kernel,
                    grid_dim_x,
                    block_dim_x,
                    args,
                    shared_mem_bytes,
                    h_stream,
                    extra);
            ctx_synchronize();
        }

        inline void launch_kernel_2d_sync(CUfunction kernel,
                uint grid_dim_x, uint grid_dim_y,
                uint block_dim_x, uint block_dim_y,
                void ** args = NULL,
                uint shared_mem_bytes = 0,
                CUstream h_stream = 0,
                void ** extra = NULL) {
            launch_kernel_2d(kernel,
                    grid_dim_x, grid_dim_y,
                    block_dim_x, block_dim_y,
                    args,
                    shared_mem_bytes,
                    h_stream,
                    extra);
            ctx_synchronize();
        }

        inline void launch_kernel_3d_sync(CUfunction kernel,
                uint grid_dim_x, uint grid_dim_y, uint grid_dim_z,
                uint block_dim_x, uint block_dim_y, uint block_dim_z,
                void ** args = NULL,
                uint shared_mem_bytes = 0,
                CUstream h_stream = 0,
                void ** extra = NULL) {
            launch_kernel_3d(kernel,
                    grid_dim_x, grid_dim_y, grid_dim_z,
                    block_dim_x, block_dim_y, block_dim_z,
                    args,
                    shared_mem_bytes,
                    h_stream,
                    extra);
            ctx_synchronize();
        }

        void ctx_synchronize();
};

#endif // OBJECTIVE_CUDA_H
