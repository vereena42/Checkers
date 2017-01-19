#include<cstdio>
#include<queue>
#include<vector>

struct checkers_point{
    int board[64];
    int how_much_children;
    checkers_point * children = NULL;
    checkers_point * next = NULL;
    checkers_point * prev = NULL;
    checkers_point * parent = NULL;
    bool min_max;
    int value;
    int player;
};


extern "C" {

__device__
    void ramification(checkers_point * ch2, int thid, int how_deep){
        int pseudo_rand = thid % 7 + 2;
        if (!(thid == 0 && how_deep == 1))
            printf("%d | %d | %d | %d\n", thid, ch2->value, pseudo_rand, ch2->parent->value);
        else {
            printf("%d | %d\n", thid, pseudo_rand);
        }
        ch2->how_much_children = pseudo_rand;
        ch2->children = new checkers_point;
        ch2->children->value = ch2->value*100+1;
        ch2->children->parent = ch2;
        ch2 = ch2->children;
        for (int j = 1; j < pseudo_rand; j++){
            ch2->next = new checkers_point;
            ch2->next->value = ch2->value+1;
            ch2->next->parent = ch2->parent;
            ch2->next->prev = ch2;
            ch2 = ch2->next;
        }
    }

__device__
    void delete_subtree(checkers_point * ch) {
        //detaching subtree from parent
        if(ch->parent != NULL) {
            if(ch->parent->children == ch) {
                ch->parent->children = ch->next;
                if(ch->next != NULL) {
                    ch->next->prev = NULL;
                }
            }   
            else {
                if(ch->prev != NULL) {
                    ch->prev->next = ch->next;
                }
                if(ch->next != NULL) {
                    ch->next->prev = ch->prev;
                }
            }
        } 
        
        //deleting all nodes in BFS order
        std::queue <checkers_point *> Q;
        checkers_point * temp, child;
        Q.push(ch);

        while(!Q.empty()) {
            temp = Q.front();
            Q.pop();

            child = temp->children;
            while(child != NULL) {
                Q.push(child);
                child = child->next;
            }
            delete temp;
         }
    }

__device__
    void change_tree_to_subtree(checkers_point * old_tree, checkers_point * new_tree) {
        int thid = (blockIdx.x * blockDim.x) + threadIdx.x;
        std::vector <checkers_point *> vec;
        if(thid == 0){
            new_tree->parent = NULL;
            checkers_point * child = old_tree->children;
            checkers_point * grandchild;
            checkers_point * ggchild;
            checkers_point * temp;
            while(child != NULL) {
                if(child != new_tree) {
                    grandchild = child->children;
                    while(grandchild!=NULL) {
                        ggchild = grand_child->children;
                        while(ggchild != NULL) {
                            ggchild->parent = NULL;
                            vec.push_back(ggchild);
                            ggchild = ggchild->next;
                        }
                        temp = grandchild;
                        granchild = granchild->next;
                        delete temp;
                    }    
                }
                temp = child;
                child = child->next;
                delete temp;
            }
            delete old_tree;
        }
        __syncthreads();
        if(thid < vec.size()) {
            checkers_point * my_child = vec[thid];
            delete_subtree(my_child);
        }
    }

__global__
	void alpha_beta(checkers_point * ch){
		change_tree_to_subtree(ch, ch->children);
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
    void new_line(int n, checkers_point * ch, int i){
        int thid = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (thid == 0){
            printf("____\n");
        }
    }

}
