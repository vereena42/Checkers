#include<cstdio>

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


extern "C" {

__device__
    void ramification(checkers_point * ch2, int thid, int how_deep){
        int pseudo_rand = thid % 7 + 2;
	/*
        if (!(thid == 0 && how_deep == 1))
            printf("%d | %d | %d | %d\n", thid, ch2->value, pseudo_rand, ch2->parent->value);
        else {
            printf("%d | %d\n", thid, pseudo_rand);
        }
	*/
        ch2->how_much_children = pseudo_rand;
        ch2->children = new checkers_point;
        ch2->children->value = ch2->value*100+1;
        ch2->children->parent = ch2;
        ch2 = ch2->children;
        for (int j = 1; j < pseudo_rand; j++){
            ch2->next = new checkers_point;
            ch2->next->value = ch2->value+1;
            ch2->next->parent = ch2->parent;
            ch2 = ch2->next;
        }
    }
    
__global__
    void create_tree(int n, checkers_point * ch, int how_deep){
        int thid = (blockIdx.x * blockDim.x) + threadIdx.x;
        int find_me = thid;
        int count_group = n;
            __syncthreads();
        if (thid < n){
            checkers_point * ch2 = ch;
            for (int i = 0; i < how_deep; i++){
                if (find_me == 0 && i + 1 == how_deep){
                    ramification(ch2, thid, how_deep);
                }
                __syncthreads();
                if (i + 1 == how_deep)
                    break;
                count_group = count_group/ch2->how_much_children;
                int group = find_me/count_group;
                if (group >= ch2->how_much_children)
                    break;
                find_me = find_me % count_group;
                ch2 = ch2->children;
                for (int k = 0; k < group; k++)
                    ch2 = ch2->next;
                __syncthreads();
            }
        }
    }
    
    
__device__
    void print_tr(checkers_point * ch){
        if (ch == NULL)
            return;
        print_tr(ch->children);
        print_tr(ch->next);
        printf("%d\n", ch->value);
    }
        
__global__
    void print_tree(int n, checkers_point * ch, int i){
        int thid = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (thid == 0){
            printf("____\n");
            print_tr(ch);
            printf("____\n");
        }
    }

__global__
    void set_root(checkers_point * ch, int * tab){
        int thid = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (thid == 0){
	    ch->value = 1;
	    for (int i = 0; i < 64; ++i)
		ch->board[i] = tab[i]; 
        }
    }

}
