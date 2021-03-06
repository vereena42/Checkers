#include "cuda.h"
#include <cstdio>

struct checkers_point{
    int board[64];
    int how_much_children;
    checkers_point * children = NULL;
    checkers_point * next = NULL;
    checkers_point * prev = NULL;
    checkers_point * parent = NULL;
    bool min_max;
    int alpha = -1000000000;
    int beta = 1000000000;
    int value;
    int player;
};


#define default_n 8
#define EMPTY 0
#define WHITE 1
#define BLACK 2
#define queenW 11
#define queenB 22

int * computer_turn(int siize, int default_row_with_pawn, int * tab_with_board){
    
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

    CUfunction alpha_beta, create_tree, delete_tree, print_tree, set_root, copy_best_result;
    
    res = cuModuleGetFunction(&alpha_beta, cuModule, "alpha_beta");
    if (res != CUDA_SUCCESS){
        printf("cannot acquire kernel handle\n");
        exit(1);
	}
    res = cuModuleGetFunction(&create_tree, cuModule, "create_tree");
    if (res != CUDA_SUCCESS){
        printf("cannot acquire kernel handle\n");
        exit(1);
    }
	res = cuModuleGetFunction(&delete_tree, cuModule, "delete_tree");
    if (res != CUDA_SUCCESS){
        printf("cannot acquire kernel handle\n");
        exit(1);
	}
    res = cuModuleGetFunction(&print_tree, cuModule, "print_tree"); 
    if (res != CUDA_SUCCESS){
        printf("cannot acquire kernel handle\n");
        exit(1);
    }
    res = cuModuleGetFunction(&set_root, cuModule, "set_root");
    if (res != CUDA_SUCCESS){
        printf("cannot acquire kernel handle\n");
        exit(1);
    }

    res = cuModuleGetFunction(&copy_best_result, cuModule, "copy_best_result");
    if (res != CUDA_SUCCESS){
        printf("cannot acquire kernel handle\n");
        exit(1);
    }

    int how_deep = 6;
    int max_children = 12 * 2;
    int n = max_children;
    for (int i = 0; i < 4; i++){
        n *= max_children;
    }
    //printf("N: %d\n", n);
    size_t size = sizeof(checkers_point)*n;
    size_t size_tab = sizeof(int)*siize*siize;
    checkers_point * a = (checkers_point*) malloc(size);
    res = cuMemHostRegister(a, size, 0);
    if (res != CUDA_SUCCESS){
        printf("cuMemHostRegister\n");
        exit(1);
    }
    res = cuMemHostRegister(tab_with_board, size_tab, 0);
    if (res != CUDA_SUCCESS){
        printf("cuMemHostRegister\n");
        exit(1);
    }

    int blocks_per_grid = (n+1023)/1024;
    int threads_per_block = 1024;
    int blocks_per_grid2 = 100;
    int threads_per_block2 = 100;
	int num_threads = threads_per_block2 * blocks_per_grid2;
    CUdeviceptr Adev, Atab, Vdev;
    res = cuMemAlloc(&Adev, size);
    if (res != CUDA_SUCCESS){
        printf("cuMemAlloc\n");
        exit(1);
    }
    res = cuMemAlloc(&Vdev, num_threads * sizeof(checkers_point*));
    if (res != CUDA_SUCCESS){
        printf("cuMemAlloc\n");
        exit(1);
    }
    res = cuMemAlloc(&Atab, size_tab);
    if (res != CUDA_SUCCESS){
        printf("cuMemAlloc\n");
        exit(1);
    }
    res = cuMemcpyHtoD(Atab, tab_with_board, size_tab);
    if (res != CUDA_SUCCESS){
        printf("cuMemcpy\n");
        exit(1);
    }
    int i = 1;
    void* args[] = {&n, &Adev, &i};
	void* args2[] = {&Adev, &num_threads, &Vdev};
	void* args3[] = {&Adev, &num_threads, &Vdev};
    void* args_root[] = {&Adev, &Atab, &siize};
    res = cuLaunchKernel(set_root, blocks_per_grid, 1, 1, threads_per_block, 1, 1, 0, 0, args_root, 0);
    if (res != CUDA_SUCCESS){
        printf("cannot run kernel\n");
        exit(1);
    }
    for (i = 1; i < how_deep+1; i++){
        res = cuLaunchKernel(create_tree, blocks_per_grid, 1, 1, threads_per_block, 1, 1, 0, 0, args, 0);
	res = cuLaunchKernel(print_tree, blocks_per_grid, 1, 1, threads_per_block, 1, 1, 0, 0, args, 0);
    }
    if (res != CUDA_SUCCESS){
        printf("cannot run kernel\n");
        exit(1);
    }
    res = cuLaunchKernel(alpha_beta, blocks_per_grid2, 1, 1, threads_per_block2, 1, 1, 0, 0, args3, 0);
    if (res != CUDA_SUCCESS){
        printf("cannot run kernel\n");
        exit(1);
    }
	res = cuLaunchKernel(delete_tree, blocks_per_grid2, 1, 1, threads_per_block2, 1, 1, 0, 0, args2, 0);
    if (res != CUDA_SUCCESS){
        printf("cannot run kernel\n");
        exit(1);
	}
    res = cuLaunchKernel(copy_best_result, blocks_per_grid, 1, 1, threads_per_block, 1, 1, 0, 0, args_root, 0);
    if (res != CUDA_SUCCESS){
        printf("cannot run kernel\n");
        exit(1);
    }

	res = cuMemcpyDtoH(tab_with_board, Atab, size_tab);
    if (res != CUDA_SUCCESS){
        printf("cuMemcpy\n");
        exit(1);
	}
    cuMemFree(Adev);
    cuMemFree(Atab);
    cuMemFree(Vdev);
    cuCtxDestroy(cuContext);

    return tab_with_board;
}
