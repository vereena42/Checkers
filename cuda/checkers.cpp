#include "cuda.h"
#include <cstdio>

struct checkers_point{
    int board[64];
    int how_much_children;
    checkers_point * children = NULL;
    checkers_point * next = NULL;
    checkers_point * parent = NULL;
    bool min_max;
    int value;
    int player;
};


int main(){
    cuInit(0);
    CUdevice cuDevice;
    CUresult res = cuDeviceGet(&cuDevice, 0);
    if (res != CUDA_SUCCESS){
        printf("cannot acquire device 0\n");
        exit(1);
    }

    CUcontext cuContext;
    res = cuCtxCreate(&cuContext, 0, cuDevice);
    if (res != CUDA_SUCCESS){
        printf("cannot create context\n");
        exit(1);
    }

    CUmodule cuModule = (CUmodule)0;
    res = cuModuleLoad(&cuModule, "checkers.ptx");
    if (res != CUDA_SUCCESS) {
        printf("cannot load module: %d\n", res);
        exit(1);
    }

    CUfunction alpha_beta, create_tree, print_tree, nl;
    res = cuModuleGetFunction(&create_tree, cuModule, "create_tree");
    if (res != CUDA_SUCCESS){
        printf("cannot acquire kernel handle\n");
        exit(1);
    }
    res = cuModuleGetFunction(&print_tree, cuModule, "print_tree");
    if (res != CUDA_SUCCESS){
        printf("cannot acquire kernel handle\n");
        exit(1);
    }
    
    res = cuModuleGetFunction(&nl, cuModule, "new_line");
    if (res != CUDA_SUCCESS){
        printf("cannot acquire kernel handle\n");
        exit(1);
    }

    res = cuModuleGetFunction(&alpha_beta, cuModule, "alpha_beta");
    if (res != CUDA_SUCCESS){
        printf("cannot acquire kernel handle\n");
        exit(1);
    }

    
    int how_deep = 4;
    int max_children = 12 * 2;
    int n = max_children;
    for (int i = 0; i < how_deep; i++){
        n *= max_children;
    }
    printf("N: %d\n", n);
    size_t size = sizeof(checkers_point);
    checkers_point * a = (checkers_point*) malloc(size);
    res = cuMemHostRegister(a, size, 0);
    if (res != CUDA_SUCCESS){
        printf("cuMemHostRegister\n");
        exit(1);
    }
    int blocks_per_grid = (n+1023)/1024;
    int threads_per_block = 1024;
    a->children = a->parent = a->next = NULL;
    a->value = 1;
    CUdeviceptr Adev;
    res = cuMemAlloc(&Adev, size);
    if (res != CUDA_SUCCESS){
        printf("cuMemAlloc\n");
        exit(1);
    }
    res = cuMemcpyHtoD(Adev, a, size);
    if (res != CUDA_SUCCESS){
        printf("cuMemcpy\n");
        exit(1);
    }
    int i;
    void* args[] = {&n, &Adev, &i};
    int thread_num = blocks_per_grid * threads_per_block;
	void* args2[] = {&Adev, &thread_num};
    for (i = 1; i < how_deep+1; i++){
        res = cuLaunchKernel(create_tree, blocks_per_grid, 1, 1, threads_per_block, 1, 1, 0, 0, args, 0);
        res = cuLaunchKernel(nl, blocks_per_grid, 1, 1, threads_per_block, 1, 1, 0, 0, args, 0);
    }
    printf("AAA\n");
    if (res != CUDA_SUCCESS){
        printf("cannot run kernel\n");
        exit(1);
    }
    res = cuLaunchKernel(print_tree, blocks_per_grid, 1, 1, threads_per_block, 1, 1, 0, 0, args, 0);
    if (res != CUDA_SUCCESS){
        printf("cannot run kernel\n");
        exit(1);
    }

	res = cuLaunchKernel(alpha_beta, blocks_per_grid, 1, 1, threads_per_block, 1, 1, 0, 0, args2, 0);
    if (res != CUDA_SUCCESS){
        printf("cannot run kernel\n");
        exit(1);
    }

    printf("Alpha-beta runned\n");

	res = cuLaunchKernel(print_tree, blocks_per_grid, 1, 1, threads_per_block, 1, 1, 0, 0, args, 0);
    if (res != CUDA_SUCCESS){
        printf("cannot run kernel\n");
        exit(1);
    }
    cuMemcpyDtoH(a, Adev, size);
    cuMemFree(Adev);
    printf("Result: %d\n", a->value);
    cuCtxDestroy(cuContext);
    return 0;
}
