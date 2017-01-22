#include<cstdio>

#define EMPTY 0
#define WHITE 1
#define BLACK 2
#define queenW 11
#define queenB 22

struct checkers_point{
    int board[64];
    int how_much_children;
    checkers_point * children = NULL;
    checkers_point * next = NULL;
    checkers_point * prev = NULL;
    checkers_point * parent = NULL;
    bool min_max;
    int value;
    int alpha;
    int beta;
    int player;
};

class Queue{
    private:
        checkers_point * first = NULL;
        checkers_point * last = NULL;
        int size = 0;
    public:
		__device__
			void add_one(checkers_point * point) {
				if(point == NULL)
					return;
				if(first == NULL) {
					this->first = point;
					this->last = point;
				}
				else {
					this->last->next = point;
					this->last = point;
				}
				this->size = this->size + 1;
			}
        __device__
            void add(checkers_point * layer) {
				if(layer == NULL)
					return;
				int counter = 0;
                if(this->first == NULL) {
                    this->first = layer;
                }
                else {
                    this->last->next=layer;
                }
				checkers_point * temp = layer;
				counter+=1;
				while(temp->next != NULL) {
					temp = temp->next;
					counter+=1;
				}
				this->last = temp;
				this->size = this->size + counter;
            }
		
		__device__
			checkers_point * pop() {
				checkers_point * firs,* seco;
				firs = this->first;
				if(firs == NULL)
					return NULL;
				else
					seco = firs->next;
				if(firs->parent != seco->parent) {
					firs->next = NULL;
				}
				this->first = seco;
				this->size = this->size - 1;
				return firs;
			}
		
		__device__
			bool empty() {
				return this->size == 0;
			}

		__device__
			int get_size() {
			return this->size;
		}

		__device__
		    checkers_point * front() {
                return this->first;
            }

		__device__
			void clean() {
				while(this->size > 0)
					this->pop();
			}
};

extern "C" {


__device__
int pawn_owner(int * tab, int x, int y){
    if (tab[x*8+y] == BLACK || tab[x*8+y] == queenB)
        return BLACK;
    if (tab[x*8+y] == WHITE || tab[x*8+y] == queenW)
        return WHITE;
    return EMPTY;
}

__device__
bool is_queen(int * tab, int x, int y){
	int n = 8;
    return (tab[x*n+y] == queenB || tab[x*n+y] == queenW);
}

__device__
bool is_a_pawn(int * tab, int x, int y){
    return !(tab[x*8+y] == EMPTY);
}

__device__
bool correct_kill(int * tab, int x, int y, int x1, int y1){
    if (!is_a_pawn(tab, x1, y1))
	return false;
    if (pawn_owner(tab, x, y) != pawn_owner(tab, x1, y1))
        return true;
    return false;
}


__device__
bool queen_way(int * tab, int x, int y, int x1, int y1){
    int own = pawn_owner(tab, x, y);
    int x_r = x > x1 ? -1 : 1, y_r = y > y1 ? -1 : 1;
    bool next_empty = false;
    x += x_r; y += y_r;
    while (x != x1){
        if (is_a_pawn(tab, x, y)){
            if (next_empty)
                return false;
            next_empty = true;
            if (pawn_owner(tab, x, y) == own)
                return false;
        } else {
            next_empty = false;
        }
        x += x_r; y += y_r;
    }
    return true;
}

__device__
bool is_move_correct(int * tab, int x, int y, int who, int x1, int y1){
    int n = 8;
    if (x < 0 || x >= n || x1 < 0 || x1 >= n || y < 0 || y >= n || y1 < 0 || y1 >= n ){
//	printf("WRONG VALUE");
        return false;
    }
    if (std::abs(x-x1) != std::abs(y-y1)){
//	printf("ABS PROBLEM");
        return false;
    }
    int pwn_wnr = pawn_owner(tab, x, y);
    if (pwn_wnr == EMPTY){
//	printf("PAWN OWNER EMPTY");
        return false;
    }
    if (pwn_wnr != who){
//	printf("pwn_wnr != who");
        return false;
    }
    if (is_a_pawn(tab, x1, y1)){
//	printf("pawn in _");
        return false;
    }
    if (x < x1 && who == WHITE && tab[x*n+y] != queenW){
//	printf("WHITE WRONG WAY");
        return false;
    }
    if (x > x1 && who == BLACK && tab[x*n+y] != queenB){
//	printf("BLACK WRONG WAY");
        return false;
    }
    if ((tab[x*n+y] == queenW || tab[x*n+y] == queenB) && (!queen_way(tab, x, y, x1, y1))){
//      printf("queen problem");
	return false;
    }
    if (!is_queen(tab, x, y) && std::abs((x-x1)) > 1 && !correct_kill(tab, x, y, (x1+x)/2, (y1+y)/2)){
//      printf("Correct kill problem");
	return false;
    }
    return true;
}

__device__
	void copy_board(checkers_point * ch, checkers_point * ch2){
		for (int i = 0; i < 64; i++){
			ch2->board[i] = ch->board[i];
		}
	}

__device__
	checkers_point * pawn(checkers_point * ch, int x, int y, int x1, int y1, bool &nxt, checkers_point * chprev, int & rand, bool iskillsomethingnow){
		if (chprev != NULL)
		return ch;
		int * tab = ch->board;
		if (ch->parent != NULL)
			tab = ch->parent->board;
		if (is_move_correct(tab, x, y, pawn_owner(tab, x, y), x1, y1) == true){
//			printf("correct ");
			checkers_point * chld, * chld_now;
                        if (!nxt){
//				printf("chld ");
                                ch->children = new checkers_point;
				ch->children->parent = ch;
				ch->children->prev = NULL;
				chld = ch->children;
                        } else {
//				printf("next ");
				ch->next = new checkers_point;
				ch->next->parent = ch->parent;
				ch->next->prev = ch;
                        	chld = ch->next;
			}
			chld->how_much_children = 0;
			chld->next = chld->children = NULL;
			if (chprev != NULL)
				copy_board(chprev, chld);
			else
				copy_board(chld->parent, chld);
			chld->parent->how_much_children++;
			chld->value = rand++;
                        chld->board[x1*8+y1] = chld->board[x*8+y];
                        chld->board[x*8+y] = EMPTY;
			if (iskillsomethingnow)
                        chld->board[(x+x1)/2*8+(y+y1)/2] = EMPTY;
			ch = chld;
			chld_now = ch;
			nxt = true;
//			printf("%d, %d -> %d, %d\n", x, y, x1, y1);
			/*
			if (iskillsomethingnow){
				if (ch->board[x1*8+y1] == WHITE){
		                	ch = pawn(ch, x1, y1, x1-2, y1-2, nxt, chld_now, rand, true);
        		        	ch = pawn(ch, x1, y1, x1-2, y1+2, nxt, chld_now, rand, true);
				} else {
		        	        ch = pawn(ch, x1, y1, x1+2, y1-2, nxt, chld_now, rand, true);
  		                	ch = pawn(ch, x1, y1, x1+2, y1+2, nxt, chld_now, rand, true);
				}
			}
			*/
		}
		return ch;
	}

__device__
    checkers_point * dismember_child(checkers_point * ch, int x, int y, int turn_no, bool &nxt, int &rand){
	checkers_point * chb = ch->parent;
	if (!nxt){
//		printf(" NO PARENT ");
		chb = ch;
	}
	/*
	printf("NR %d\n", chb->value);
	for (int i = 0; i < 64; i++)
		printf("%d ", chb->board[i]);
	printf("\n");
	*/
	switch(chb->board[x*8+y]){
	    case WHITE:
		if (turn_no % 2 == 0){
//		printf("WHITE ");
		ch = pawn(ch, x, y, x-1, y-1, nxt, NULL, rand, false);
                ch = pawn(ch, x, y, x-1, y+1, nxt, NULL, rand, false);
		ch = pawn(ch, x, y, x-2, y-2, nxt, NULL, rand, true);
		ch = pawn(ch, x, y, x-2, y+2, nxt, NULL, rand, true);
		}
		break;
	    case BLACK:
		if (turn_no % 2 == 1){
//		printf("BLACK %d %d", x, y);
		ch = pawn(ch, x, y, x+1, y-1, nxt, NULL, rand, false);
                ch = pawn(ch, x, y, x+1, y+1, nxt, NULL, rand, false);
		ch = pawn(ch, x, y, x+2, y-2, nxt, NULL, rand, true);
                ch = pawn(ch, x, y, x+2, y+2, nxt, NULL, rand, true);
		}
		break;
	    default:
		break;
	}
	return ch;
    }

__device__
//add global size
    void ramification(checkers_point * ch2, int thid, int how_deep){
	bool nxt = false;
	int rand = ch2->value;
	printf("!%d!\n", how_deep);
	for (int i = 0; i < 8*8; i++){
	    if (ch2->board[i] != EMPTY){
		ch2 = dismember_child(ch2, i/8, i % 8, how_deep, nxt, rand);
	    }
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

//<<<<<<< HEAD
//=======
__global__
    void delete_tree(checkers_point * ch, int thread_num, checkers_point ** V) {
        int thid = (blockIdx.x * blockDim.x) + threadIdx.x;
		int count;
		if(thid == 0){
            checkers_point * child = ch->children;
            checkers_point * temp;
            Queue Q;
            int count = 0;

            Q.add(child);
            while(!Q.empty() && Q.get_size()+Q.front()->how_much_children < thread_num) {
                temp = Q.pop();
                if(temp->children !=NULL)
                    Q.add(temp->children);
				delete temp;
            }

			count = 0;
            while(!Q.empty()) {
				temp = Q.pop();
				V[count]=temp;
				count++;
			}
        }
        __syncthreads();
        if(thid < count) {
            checkers_point * my_child = V[thid];
            Queue Q;
        	checkers_point * temp, * child;
        	Q.add_one(my_child);

        	while(!Q.empty()) {
            	temp = Q.pop();

            	child = temp->children;
				if(child != NULL)
            		Q.add(child);
            	delete temp;
			}
		}
    }
__device__
	void minmax(checkers_point * ch) {
		//zjedz do lisci i wrzuc je na kolejke
		Queue tempQueue;
		Queue Q;
		checkers_point * temp;
		tempQueue.add_one(ch);
		while(!tempQueue.empty()) {
			temp = tempQueue.pop();
			if(temp->alpha!=-1000000000 || temp->beta!=1000000000)
				Q.add_one(temp);
			if(temp->children==NULL) {
                int wynik = 0;//policz stan planszy
                temp->alpha = wynik;
                temp->beta = wynik;
                Q.add_one(temp);
            }
			if(temp->children!=NULL)
				tempQueue.add(temp->children);
		}
		//pamietaj parenta pierwszego z kolejki
		checkers_point * parent = Q.front()->parent;		
		//lecac po kolejce modyfikuj parenta danego liscia
			//jak parent nowego goscia jest inny niz poprzedni dorzuc poprzedni na kolejke i zastap go w zmiennej nowym
		while(!Q.empty()) {
			temp = Q.pop();
			if(temp->parent!=NULL) {
				if(temp->min_max)
					temp->parent->beta = min(temp->alpha,temp->parent->beta);
				else
					temp->parent->alpha = max(temp->beta,temp->parent->alpha);
			}	
			if(parent!=temp->parent) {
				Q.add_one(parent);
				parent = temp->parent;
			}
		}
		//tadam!

	}

__global__
    void alpha_beta(checkers_point * ch, int thread_num, checkers_point ** V) {
        //rozdziel i wrzuc do V
		int thid = (blockIdx.x * blockDim.x) + threadIdx.x;
		int count;
		if(thid == 0){
            checkers_point * temp;
            Queue Q;
            int count = 0;

            Q.add(ch);
            while(!Q.empty() && Q.get_size()+Q.front()->how_much_children < thread_num) {
                temp = Q.pop();
                if(temp->children !=NULL)
                    Q.add(temp->children);
				temp->alpha=-1000000000;
				temp->beta=1000000000;
            }

			count = 0;
            while(!Q.empty()) {
				temp = Q.pop();
				temp->alpha=-1000000000;
				temp->beta=1000000000;
				V[count]=temp;
				count++;
			}
        }
		__syncthreads();
        //policz dla tych w V
		if(thid<count)
			minmax(V[thid]);
		__syncthreads();
        //policz w gore
		if(thid == 0) {
			minmax(ch);
		}
		//zwroc wynik (?)
	}
    
//>>>>>>> d7adecfe7e0407bbe669ef75a06f1ba72ca587af
__device__
    void print_tr(checkers_point * ch){
        if (ch == NULL)
            return;
	if (ch->children != NULL){
	printf("(c) %d ", ch->value);
        print_tr(ch->children);
	}
	if (ch->next != NULL){
	printf("(n) %d ", ch->value);
        print_tr(ch->next);
	}
	if (ch->next == NULL && ch->children == NULL){
        printf("%d\n", ch->value);
	for (int i = 0; i < 64; i++){
		printf("%d", ch->board[i]);
		if (i % 8 == 7)
			printf("\n");
	}
	printf("\n");
	}
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
    void set_root(checkers_point * ch, int * tab, int size){
        int thid = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (thid == 0){
	    ch->value = 1;
	    ch->children = NULL;
	    ch->next = NULL;
	    ch->prev = NULL;
	    ch->how_much_children = 0;
	    for (int i = 0; i < size*size; ++i)
		ch->board[i] = tab[i]; 
        }
    }

__global__
    void copy_best_result(checkers_point * ch, int * tab, int size){
        int thid = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (thid == 0){
	//find the best board!
            for (int i = 0; i < 64; ++i)
                tab[i] = ch->board[i];
        }
    }

}
