#include <cstdio>

#define EMPTY 0
#define WHITE 1
#define BLACK 2
#define QUEENW 11
#define QUEENB 22

struct checkers_point{
    int board[64];
    int how_much_children;
    checkers_point * children = NULL;
    checkers_point * next = NULL;
    checkers_point * prev = NULL;
    checkers_point * parent = NULL;
    bool min_max;
    int value;
    int alpha = -1000000000;
    int beta = 1000000000;
    int player;
};

struct next_kill{
    int t[4];
    next_kill * next = NULL;
    int * parent_tab;
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
				if(seco==NULL || firs->parent != seco->parent) {
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
    if (tab[x*8+y] == BLACK || tab[x*8+y] == QUEENB)
        return BLACK;
    if (tab[x*8+y] == WHITE || tab[x*8+y] == QUEENW)
        return WHITE;
    return EMPTY;
}

__device__
bool create_queen(int * tab, int x, int y){
    int n = 8;
    if ((x != 0 && x != n-1) ||
        (tab[x*n+y] != WHITE && tab[x*n+y] != BLACK) ||
        (x == 0 && tab[x*n+y] == BLACK) ||
        (x == n-1 && tab[x*n+y] == WHITE))
        return false;
    if (tab[x*n+y] == WHITE)
        tab[x*n+y] = QUEENW;
    else
        tab[x*n+y] = QUEENB;
    return true;
}

__device__
bool is_queen(int * tab, int x, int y){
	int n = 8;
    return (tab[x*n+y] == QUEENB || tab[x*n+y] == QUEENW);
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
        if (!(tab[x*8+y] == EMPTY)){
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
    if (x < x1 && who == WHITE && tab[x*n+y] != QUEENW){
//	printf("WHITE WRONG WAY");
        return false;
    }
    if (x > x1 && who == BLACK && tab[x*n+y] != QUEENB){
//	printf("BLACK WRONG WAY");
        return false;
    }
    if ((tab[x*n+y] == QUEENW || tab[x*n+y] == QUEENB) && (!queen_way(tab, x, y, x1, y1))){
        printf("queen problem");
	return false;
    }
    if (!is_queen(tab, x, y) && std::abs((x-x1)) > 1 && !correct_kill(tab, x, y, (x1+x)/2, (y1+y)/2)){
//      printf("Correct kill problem");
	return false;
    }
    return true;
}

__device__
	next_kill * create_next_move(int x, int y, int * par_tb, int x1, int y1){
	next_kill * res = new next_kill;
        res->t[0] = x; res->t[1] = y;
        res->t[2] = x1; res->t[3] = y1;
	res->parent_tab = par_tb;
	return res;
}

__device__
	void copy_board(int * ch, checkers_point * ch2){
		for (int i = 0; i < 64; i++){
			ch2->board[i] = ch[i];
		}
	}


__device__
	void kill(int x, int y, int * tab){
    	    tab[x*8+y] = EMPTY;
	}

__device__
	bool has_next_move(int x, int y, int x1, int y1, int * tab, bool kiiil){
        int kll = 0;
        int x_r = x > x1 ? -1 : 1, y_r = y > y1 ? -1 : 1;
            while (x != x1){
                if (is_a_pawn(tab, x, y)){
		    if (kiiil)
                        kill(x, y, tab);
                    kll++;
                }
            x += x_r; y += y_r;
            }
        return (kll > 0);
	}

__device__
	checkers_point * again_queen(checkers_point * ch, next_kill * first, next_kill * last){
	return ch;
	}

__device__
	next_kill * queen_move_again_again(int * tab, int x1, int y1, int x, int y, next_kill * last){
		if (is_move_correct(tab, x1, y1, pawn_owner(tab, x1, y1), x, y)
                         && has_next_move(x1, y1, x, y, tab, false)) {
                     last->next = create_next_move(x1, y1, tab, x, y);
                     last = last->next;
                }
		return last;
	}

__device__
	checkers_point * again(checkers_point * ch, next_kill * first, next_kill * last, int pm){
	int x = first->t[0], y = first->t[1], x1 = first->t[2], y1 = first->t[3], * tab = first->parent_tab;
	if(is_move_correct(tab, x, y, pawn_owner(tab, x, y), x1, y1)){
		checkers_point * chld;
		ch->next = new checkers_point;
                ch->next->parent = ch->parent;
                ch->next->prev = ch;
                chld = ch->next;
		chld->min_max = !chld->parent->min_max;
		chld->alpha = -1000000000;
		chld->beta = 1000000000;
		copy_board(tab, chld);
		chld->parent->how_much_children++;
		chld->board[x1*8+y1] = chld->board[x*8+y];
                chld->board[x*8+y] = EMPTY;
                chld->board[(x+x1)/2*8+(y+y1)/2] = EMPTY;
                ch = chld;
		create_queen(ch->board, x1, y1);
		last->next = create_next_move(x1, y1, ch->board, x1+pm, y1+2);
		last = last->next;
		last->next = create_next_move(x1, y1, ch->board, x1+pm, y1-2);
		last = last->next;
	}
	return ch;
}

__device__
	checkers_point * pawn(checkers_point * ch, int x, int y, int x1, int y1, bool &nxt, bool iskillsomethingnow, bool queen){
		int * tab = ch->board;
		if (ch->parent != NULL)
			tab = ch->parent->board;
		if (is_move_correct(tab, x, y, pawn_owner(tab, x, y), x1, y1) == true){
//			printf("correct ");
			checkers_point * chld;
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
			chld->min_max = !chld->parent->min_max;
			chld->alpha = -1000000000;
			chld->beta = 1000000000;
			chld->how_much_children = 0;
			chld->next = chld->children = NULL;
			copy_board(chld->parent->board, chld);
			chld->parent->how_much_children++;
                        chld->board[x1*8+y1] = chld->board[x*8+y];
                        chld->board[x*8+y] = EMPTY;
			if (iskillsomethingnow && queen == false)
                            chld->board[(x+x1)/2*8+(y+y1)/2] = EMPTY;
			ch = chld;
			nxt = true;
			if (!iskillsomethingnow)
			    create_queen(ch->board, x1, y1);
//			printf("%d, %d -> %d, %d\n", x, y, x1, y1);
			if (iskillsomethingnow && queen == false && (create_queen(ch->board, x1, y1) == false)){
				int pm;
				if (ch->board[x1*8+y1] == WHITE){
					pm = -2;
				} else {
					pm = 2;
				}
				next_kill * first, * last, * temp;
				first = create_next_move(x1, y1, ch->board, x1+pm, y1+2);
				first->next = last = create_next_move(x1, y1, ch->board, x1+pm, y1-2);
				while (first != NULL){
					ch = again(ch, first, last, pm);
					while (last->next != NULL)
						last = last->next;
					temp = first;
					first = first->next;
					delete temp;
				}
			}
		
			if (queen && has_next_move(x1, y1, x, y, tab, true)){
				next_kill * first, * last, * temp;
                                first = create_next_move(x1, y1, ch->board, x1+1, y1+1);
				first->next = last = create_next_move(x1, y1, ch->board, x1-1, y1+1);
                                last->next = create_next_move(x1, y1, ch->board, x1+1, y1-1);
				last = last->next;
				last->next = create_next_move(x1, y1, ch->board, x1-1, y1-1);
				last = last->next;
				for (int i = 2; i < 8; i++){
					last = queen_move_again_again(tab, x1, y1, x1+i, y1+i, last);
					last = queen_move_again_again(tab, x1, y1, x1+i, y1-i, last);
					last = queen_move_again_again(tab, x1, y1, x1-i, y1+i, last);
					last = queen_move_again_again(tab, x1, y1, x1-i, y1-i, last);
				}
                                while (first != NULL){
                                        ch = again_queen(ch, first, last);
                                        while (last->next != NULL)
                                                last = last->next;
                                        temp = first;
                                        first = first->next;
                                        delete temp;
                                }
			}

		}
		return ch;
	}

__device__
    checkers_point * dismember_child(checkers_point * ch, int x, int y, int turn_no, bool &nxt, int player){
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
        int ww = 1, bb = 0;
        if (player == BLACK){
	    ww = 0;
	    bb = 1;
	}
	switch(chb->board[x*8+y]){
	    case WHITE:
		if (turn_no % 2 == ww){
//		printf("WHITE ");
		ch = pawn(ch, x, y, x-1, y-1, nxt, false, false);
                ch = pawn(ch, x, y, x-1, y+1, nxt, false, false);
		ch = pawn(ch, x, y, x-2, y-2, nxt, true, false);
		ch = pawn(ch, x, y, x-2, y+2, nxt, true, false);
		}
		break;
	    case BLACK:
		if (turn_no % 2 == bb){
//		printf("BLACK %d %d", x, y);
		ch = pawn(ch, x, y, x+1, y-1, nxt, false, false);
                ch = pawn(ch, x, y, x+1, y+1, nxt, false, false);
		ch = pawn(ch, x, y, x+2, y-2, nxt, true, false);
                ch = pawn(ch, x, y, x+2, y+2, nxt, true, false);
		}
		break;
	    case QUEENB:
		if (turn_no % 2 == bb){
		    for (int i = 0; i < 8; i++){
			ch = pawn(ch, x, y, x+i, y-i, nxt, false, true);
			ch = pawn(ch, x, y, x+i, y+i, nxt, false, true);
			ch = pawn(ch, x, y, x-i, y-i, nxt, false, true);
			ch = pawn(ch, x, y, x-i, y+i, nxt, false, true);
		    }
		}
		break;
	    case QUEENW:
                if (turn_no % 2 == ww){
                    for (int i = 0; i < 8; i++){
			ch = pawn(ch, x, y, x+i, y-i, nxt, false, true);
                        ch = pawn(ch, x, y, x+i, y+i, nxt, false, true);
                        ch = pawn(ch, x, y, x-i, y-i, nxt, false, true);
                        ch = pawn(ch, x, y, x-i, y+i, nxt, false, true);
                    }
                }
		break;
	    default:
		break;
	}
	return ch;
    }

__device__
//add global size
    void ramification(checkers_point * ch2, int thid, int how_deep, int player){
	bool nxt = false;
	//printf("!%d!\n", how_deep);
	for (int i = 0; i < 8*8; i++){
	    if (ch2->board[i] != EMPTY){
		ch2 = dismember_child(ch2, i/8, i % 8, how_deep, nxt, player);
	    }
	}
	if (nxt == false)
		printf("????");
    }

__global__
    void create_tree(int n, checkers_point * ch, int how_deep, int player){
        int thid = (blockIdx.x * blockDim.x) + threadIdx.x;
        int find_me = thid;
        int count_group = n;
            __syncthreads();
        if (thid < n){
            checkers_point * ch2 = ch;
            for (int i = 0; i < how_deep; i++){
                if (find_me == 0 && i + 1 == how_deep){
                    ramification(ch2, thid, how_deep, player);
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
		if (ch2 == NULL)
			break;
                for (int k = 0; k < group; k++){
                    ch2 = ch2->next;
		    if (ch2 == NULL)
			break;
		}
		if (ch2 == NULL)
			break;
                __syncthreads();
            }
        }
    }

__global__
    void delete_tree(checkers_point * ch, int thread_num, checkers_point ** V) {
        int thid = (blockIdx.x * blockDim.x) + threadIdx.x;
		__shared__ int count;
		count = 0;
		if(thid == 0){
		printf("delete_tree");
            checkers_point * child = ch->children;
            checkers_point * temp;
            Queue Q;

            Q.add(child);
            while(!Q.empty() && Q.get_size()+Q.front()->how_much_children < thread_num) {
                temp = Q.pop();
                if(temp->children !=NULL)
                    Q.add(temp->children);
				delete temp;
            }

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
	int calculate_pawns_value(int * tab){
    	int count = 0;
		int n=8;
    	for(int i=0;i<n*n;i++){
        	if(tab[i] == WHITE)
            	count+=3;
        	else if(tab[i] == QUEENW)
            	count+=5;
        	else if(tab[i] == BLACK)
            	count-=3;
        	else if(tab[i] == QUEENB)
        	    count-=5;
    	}
    	return (int)(((49.5*count)/57.0) + 49.5);
	}

__device__
	int calculate_dist_to_be_queen(int * tab){
		int n=8;
    	int black_count=0,white_count=0,black_dist=0,white_dist=0;
    	for(int row=0;row<n;row++){
        	for(int col=0;col<n;col++){
            	if(tab[row*n+col] == WHITE){
                	white_count++;
                	white_dist+=(n-1-row);
            	}
            	else if(tab[row*n+col] == BLACK){
                	black_count++;
                	black_dist+=row;
            	}
        	}
    	}
    	double black_val, white_val;
    	if(white_count == 0)
        	white_val = 0;
    	else
        	white_val = (double)white_dist/(double)white_count;
    	if(black_count == 0)
        	black_val = 0;
    	else
        	black_val = (double)black_dist/(double)black_count;
    	double value = black_val - white_val;
    	return (int)(0.495*value + 49.5);
	}

__device__
	int calculate_future_queen_kills(int * tab){
	int n=8;
    int * kill_tab = new int [n*n];
    int temp_row,temp_col,white_count=0,black_count=0,white_dead=0,black_dead=0;
    for(int row=0;row<n;row++){
        for(int col=0;col<n;col++){
            kill_tab[row*n+col] = EMPTY;
            if(tab[row*n+col] == QUEENB || tab[row*n+col] == QUEENW){
                temp_row = row-1;
                temp_col = col-1;
                while(temp_row >= 1 && temp_col >= 1){
                    if(is_a_pawn(tab,temp_row,temp_col)){
                        if(pawn_owner(tab,row,col) == pawn_owner(tab,temp_row,temp_col))
                            break;
                        else if(is_a_pawn(tab,temp_row-1,temp_col-1))
                            break;
                        else{
                            kill_tab[temp_row*n+temp_col] = pawn_owner(tab,temp_row,temp_col);
                            temp_row--;
                            temp_col--;
                        }
                    }
                    temp_row--;
                    temp_col--;
                }
                temp_row = row-1;
                temp_col = col+1;
                while(temp_row >= 1 && temp_col <= n-2){
                    if(is_a_pawn(tab,temp_row,temp_col)){
                        if(pawn_owner(tab,row,col) == pawn_owner(tab,temp_row,temp_col))
                            break;
                        else if(is_a_pawn(tab,temp_row-1,temp_col+1))
                            break;
                        else{
                            kill_tab[temp_row*n+temp_col] = pawn_owner(tab,temp_row,temp_col);
                            temp_row--;
                            temp_col++;
                        }
                    }
                    temp_row--;
                    temp_col++;
                }
                temp_row = row+1;
                temp_col = col-1;
                while(temp_row <= n-2 && temp_col >= 1){
                    if(is_a_pawn(tab,temp_row,temp_col)){
                        if(pawn_owner(tab,row,col) == pawn_owner(tab,temp_row,temp_col))
                            break;
                        else if(is_a_pawn(tab,temp_row+1,temp_col-1))
                            break;
                        else{
                            kill_tab[temp_row*n+temp_col] = pawn_owner(tab,temp_row,temp_col);
                            temp_row++;
                            temp_col--;
                        }
                    }
                    temp_row++;
                    temp_col--;
                }
                temp_row = row+1;
                temp_col = col+1;
                while(temp_row <= n-2 && temp_col <= n-2){
                    if(is_a_pawn(tab,temp_row,temp_col)){
                        if(pawn_owner(tab,row,col) == pawn_owner(tab,temp_row,temp_col))
                            break;
                        else if(is_a_pawn(tab,temp_row+1,temp_col+1))
                            break;
                        else{
                            kill_tab[temp_row*n+temp_col] = pawn_owner(tab,temp_row,temp_col);
                            temp_row++;
                            temp_col++;
                        }
                    }
                    temp_row++;
                    temp_col++;
                }
            }
        }
    }
    for(int row=0;row<n;row++){
        for(int col=0;col<n;col++){
            if(kill_tab[row*n+col] == WHITE)
                white_dead++;
            else if(kill_tab[row*n+col] == BLACK)
                black_dead++;
            if(is_a_pawn(tab,row,col)){
                if(pawn_owner(tab,row,col) == WHITE)
                    white_count++;
                else
                    black_count++;
            }
        }
    }
    double black_percent,white_percent;
    if(black_count == 0)
        black_percent = 1.0;
    else
        black_percent = (double)black_dead/(double)black_count;
    if(white_count == 0)
        white_percent = 1.0;
    else
        white_percent = (double)white_dead/(double)white_count;
    int value = white_percent - black_percent;
	delete kill_tab;
    return 49.5*value+49.5;
}

__device__
    int calculate_board_value(int * Board){
	    int value;
	    value=calculate_pawns_value(Board);
    	value=100*value+calculate_dist_to_be_queen(Board);
    	value=100*value+calculate_future_queen_kills(Board);
    	//value=10*value+(rand()%10);
    	return value;
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
            else if(temp->children==NULL) {
                int wynik = calculate_board_value(temp->board);//policz stan planszy
                temp->alpha = wynik;
                temp->beta = wynik;
                Q.add_one(temp);
            }
            else if(temp->children!=NULL)
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
		__shared__ int count;
		count = 0;
		if(thid == 0){
		    checkers_point * temp;
            Queue Q;
            Q.add(ch);
            while(!Q.empty() && Q.get_size()+Q.front()->how_much_children < thread_num) {
                temp = Q.pop();
                if(temp->children !=NULL)
                    Q.add(temp->children);
				temp->alpha=-1000000000;
				temp->beta=1000000000;
            }
            
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
    void print_tree(int n, checkers_point * ch, int i, int player){
        int thid = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (thid == 0){
            printf("____\n");
//            print_tr(ch);
//            printf("____\n");
        }
    }

__global__
    void set_root(checkers_point * ch, int * tab, int size, int player){
        int thid = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (thid == 0){
	    ch->value = 1;
	    ch->children = NULL;
	    ch->next = NULL;
	    ch->prev = NULL;
	    ch->parent = NULL;
		ch->alpha = -1000000000;
		ch->beta = 1000000000;
	    if(player == WHITE)
	        ch->min_max = true;
        else
            ch->min_max = false;
	    ch->how_much_children = 0;
	    for (int i = 0; i < size*size; ++i)
		ch->board[i] = tab[i]; 
        }
    }

__global__
    void copy_best_result(checkers_point * ch, int * tab, int size, int player){
        int thid = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (thid == 0){
	    checkers_point * ch2 = ch->children;
	    if (ch2 == NULL){
		printf("No result -> root->chidlren = NULL\n");
		return;
            }
	    if (player == WHITE){
	    while (ch->alpha != ch2->beta){
		if (ch2->next == NULL)
			break;
		ch2 = ch2->next;
	    }
	    } else {
	    while (ch->beta != ch2->alpha){
		if (ch2->next == NULL)
			break;
                ch2 = ch2->next;
	    }
	    }
            for (int i = 0; i < 64; ++i)
                tab[i] = ch2->board[i];
        }
    }

}
