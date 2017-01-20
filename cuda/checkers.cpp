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


#define default_n 8
#define EMPTY 0
#define WHITE 1
#define BLACK 2
#define queenW 11
#define queenB 22

int main(){
    //przekazane powinny zostac: int siize, default_row_with_pawn,     int * tab_with_board;
    int siize = 8;
    int * tab_with_board;
    int default_row_with_pawn = 3;
    tab_with_board = (int*) malloc(sizeof(int)*siize*siize);
    for (int i = 0; i < siize*siize; i++)
        tab_with_board[i] = EMPTY;
    for (int i = 0; i < default_row_with_pawn; ++i){
        for (int j = 0; j < siize/2; ++j){
            tab_with_board[i*siize+2*j+(i%2)] = BLACK;
            tab_with_board[(siize*siize-1)-(i*siize+2*j+(i%2))] = WHITE;
        }
    }
    //^ do usuniecia
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

    CUfunction create_tree, print_tree, set_root, copy_best_result;
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

    int how_deep = 4;
    int max_children = 12 * 2;
    int n = max_children;
    for (int i = 0; i < how_deep; i++){
        n *= max_children;
    }
    printf("N: %d\n", n);
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
    CUdeviceptr Adev, Atab;
    res = cuMemAlloc(&Adev, size);
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
    int i;
    void* args[] = {&n, &Adev, &i};
    void* args_root[] = {&Adev, &Atab, &siize};
    res = cuLaunchKernel(set_root, blocks_per_grid, 1, 1, threads_per_block, 1, 1, 0, 0, args_root, 0);
    if (res != CUDA_SUCCESS){
        printf("cannot run kernel\n");
        exit(1);
    }
    for (i = 1; i < how_deep+1; i++){
        res = cuLaunchKernel(create_tree, blocks_per_grid, 1, 1, threads_per_block, 1, 1, 0, 0, args, 0);
    }
    if (res != CUDA_SUCCESS){
        printf("cannot run kernel\n");
        exit(1);
    }
    res = cuLaunchKernel(print_tree, blocks_per_grid, 1, 1, threads_per_block, 1, 1, 0, 0, args, 0);
    if (res != CUDA_SUCCESS){
        printf("cannot run kernel\n");
        exit(1);
    }
    res = cuLaunchKernel(copy_best_result, blocks_per_grid, 1, 1, threads_per_block, 1, 1, 0, 0, args_root, 0);
    if (res != CUDA_SUCCESS){
        printf("cannot run kernel\n");
        exit(1);
    }
    cuMemFree(Adev);
    cuMemFree(Atab);
    cuCtxDestroy(cuContext);
//    return tab_with_board;
    return 0;
}
