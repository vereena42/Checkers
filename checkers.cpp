//
//  checkers.cpp
//  Warcaby
//
//


#include "checkers.hpp"
#include "objective_cuda.h"
#include <set>
#define STR(k) XSTR(k)
#define XSTR(k) #k

#define cu_auto_assert(k) CuError::cu_assert(k, __FILE__ ":" STR(__LINE__))

std::set<int> cycle;
int zobrist_table[64][4];

int zobrist_int(int * tab){
    int res = 0;
    for (int i = 0; i < 64; i++){
	switch(tab[i]){
	case BLACK:
		res = res ^ zobrist_table[i][0];
		break;
	case WHITE:
		res = res ^ zobrist_table[i][1];
		break;
	case QUEENB:
		res = res ^ zobrist_table[i][2];
		break;
	case QUEENW:
		res = res ^ zobrist_table[i][3];
		break;
	default:
		break;
	}
    }
    //std::cout << res << "\n";
    return res;
}

void init_zobrist(){
	for (int i = 0 ; i < 64; i++){
		for (int j = 0; j < 4; j++)
			zobrist_table[i][j] = rand();
	}
}


checkers::checkers(){
    this->n = default_n;
    this->row_with_pawn = default_row_with_pawn;
    this->tab = new int [this->n*this->n];
    this->new_game();
}

checkers::checkers(int n, int rwp){
    this->n = n;
    this->row_with_pawn = rwp;
    this->tab = new int [this->n*this->n];
    this->new_game();
}

checkers::~checkers(){
    delete [] this->tab;
}

void checkers::new_game(){
    for (int i = 0; i < n*n; i++)
        this->tab[i] = EMPTY;
    for (int i = 0; i < row_with_pawn; ++i){
        for (int j = 0; j < n/2; ++j){
            tab[i*n+2*j+(i%2)] = BLACK;
            tab[(n*n-1)-(i*n+2*j+(i%2))] = WHITE;
        }
    }/*
    tab[9] = tab[11] = tab[13] = tab[16] = tab[20] = EMPTY;
    tab[9] = tab[32] = tab[36] = WHITE;
    tab[43] = tab[47] = tab[48] = tab[50] = tab[54] = EMPTY;
    tab[38] = tab[47] = BLACK;*/
   
   // tab[0] = tab[9] = tab[11] = tab[13] = tab[16] = tab[18] = tab[20] = tab[22] = EMPTY;
    //tab[25] = BLACK;
    //tab[18] = WHITE;
   
  //  for (int i = 0; i < 64; i++)
//	tab[i] = EMPTY;
    //tab[0] = tab[2] = tab[4] = tab[6] = QUEENW;
    //tab[9] = tab[11] = tab[13] = WHITE;
    //tab[59] = QUEENB;
}

int checkers::pawn_owner(int x, int y, int * t){
    int n = 8;
    if (t[x*n+y] == BLACK || t[x*n+y] == QUEENB)
        return BLACK;
    if (t[x*n+y] == WHITE || t[x*n+y] == QUEENW)
        return WHITE;
    return EMPTY;
}

int checkers::move(int x, int y, int who, int x1, int y1, int kll){
    if (!is_move_correct(x, y, who, x1, y1, kll, tab, true))
        return 0;
    if (has_next_move(x, y, x1, y1, tab, true)){
        create_queen(x1, y1, tab);
        return 2;
    }
    create_queen(x1, y1, tab);
    return 1;
}

bool checkers::is_move_correct(int x, int y, int who, int x1, int y1, int kll, int * t, bool kill_in_f){
    int n = 8;
    if (x < 0 || x >= n || x1 < 0 || x1 >= n || y < 0 || y >= n || y1 < 0 || y1 >= n ){
        std::cout << "WRONG!\n";
        return false;
    }
    if (std::abs(x-x1) != std::abs(y-y1)){
        std::cout << "WRONG!\n";
        return false;
    }
    int pwn_wnr = pawn_owner(x, y, t);
    if (pwn_wnr == EMPTY){
        std::cout << "There's no pawn!\n";
        return false;
    }
    if (pwn_wnr != who){
        std::cout << "This is not your pawn!\n";
        return false;
    }
    if (is_a_pawn(x1, y1, t)){
        std::cout << "Target field is not empty\n";
        return false;
    }
    if (x < x1 && who == WHITE && t[x*n+y] != QUEENW){
        std::cout << "This is not a Queen!\n";
        return false;
    }
    if (x > x1 && who == BLACK && t[x*n+y] != QUEENB){
        std::cout << "This is not a Queen!\n";
        return false;
    }
    if ((t[x*n+y] == QUEENW || t[x*n+y] == QUEENB) && (!queen_way(x, y, x1, y1, t))){
        std::cout << "Something is wrong :C\n";
        return false;
    }
    if (!is_queen(x, y, t) && std::abs((x-x1)) > 1 && !correct_kill(x, y, (x1+x)/2, (y1+y)/2, t)){
        std::cout << "NOPE!\n";
        return false;
    }
    if (kll){
        kll = 0;
        int x_r = x > x1 ? -1 : 1, y_r = y > y1 ? -1 : 1;
        int own = pawn_owner(x, y, t);
        x += x_r; y += y_r;
        while (x != x1){
            std::cout << pawn_owner(x, y, t) << " " << x << " " << y << "\n";
            if (is_a_pawn(x, y, t) && pawn_owner(x, y, t) != own){
                kll++;
            }
            x += x_r; y += y_r;
        }
        return (kll > 0);
    }
    return true;
}

bool checkers::queen_way(int x, int y, int x1, int y1, int * t){
    int own = pawn_owner(x, y, t);
    int x_r = x > x1 ? -1 : 1, y_r = y > y1 ? -1 : 1;
    bool next_empty = false;
    x += x_r; y += y_r;
    while (x != x1){
        if (is_a_pawn(x, y, t)){
            if (next_empty)
                return false;
            next_empty = true;
            if (pawn_owner(x, y, t) == own)
                return false;
        } else {
            next_empty = false;
        }
        x += x_r; y += y_r;
    }
    return true;
}

bool checkers::is_a_pawn(int x, int y, int * t){
    int n = 8;
    return !(t[x*n+y] == EMPTY);
}

void checkers::kill(int x, int y, int * t){
    int n = 8;
    t[x*n+y] = EMPTY;
}

bool checkers::has_next_move(int x, int y, int x1, int y1, int * t, bool kill_in_f){
    int n = 8;
    t[x1*n+y1] = t[x*n+y];
    kill(x, y, t);
    if (is_queen(x1, y1, t)){
        int kll = 0;
        int x_r = x > x1 ? -1 : 1, y_r = y > y1 ? -1 : 1;
        while (x != x1){
            if (is_a_pawn(x, y, t)){
                if (kill_in_f)
                    kill(x, y, t);
                kll++;
            }
            x += x_r; y += y_r;
        }
        return (kll > 0);
    }
    if (std::abs((x-x1)) == 1)
        return false;
    kill((x+x1)/2, (y+y1)/2, t);
    return true;
}

bool checkers::is_queen(int x, int y, int * t){
    int n = 8;
    return (t[x*n+y] == QUEENB || t[x*n+y] == QUEENW);
}

bool checkers::correct_kill(int x, int y, int x1, int y1, int * t){
    if (pawn_owner(x1, y1, t) == EMPTY)
        return false;
    if (pawn_owner(x, y, t) != pawn_owner(x1, y1, t))
        return true;
    return false;
}

bool checkers::create_queen(int x, int y, int * t){
    int n = 8;
    if ((x != 0 && x != n-1) ||
        (t[x*n+y] != WHITE && t[x*n+y] != BLACK) ||
        (x == 0 && t[x*n+y] == BLACK) ||
        (x == n-1 && t[x*n+y] == WHITE))
        return false;
    if (t[x*n+y] == WHITE)
        t[x*n+y] = QUEENW;
    else
        t[x*n+y] = QUEENB;
    return true;
}

bool checkers::is_end_of_game(){
    if (is_there_winner() ||
            is_no_pawns() ||
            is_game_blocked())
	return true;
    int res = zobrist_int(tab);
    if (cycle.find(res) != cycle.end())
        return true;
    cycle.insert(res);
    return false;
}

bool checkers::is_no_pawns(){
    for(int i=0;i<n*n;i++){
        if (tab[i] == WHITE || tab[i] == BLACK)
            return false;
    }
    return true;
}

bool checkers::is_game_blocked(){
    int temp_col, temp_row;
    for(int row=0;row<n;row++){
        for(int col=0;col<n;col++){
            if(tab[row*n+col] == WHITE){
                if(col > 0 && row > 0 && !is_a_pawn(row-1,col-1, tab))
                    return false;
                if(col < n-1 && row > 0 && !is_a_pawn(row-1,col+1, tab))
                    return false;
                if(col > 1 && row > 1 && pawn_owner(row-1,col-1, tab) == BLACK && !is_a_pawn(row-2,col-2, tab))
                    return false;
                if(col < n-2 && row > 1 && pawn_owner(row-1,col+1, tab) == BLACK && !is_a_pawn(row-2,col+2, tab))
                    return false;
            }
            else if(tab[row*n+col] == BLACK){
                if(col > 0 && row < n-1  && !is_a_pawn(row+1,col-1, tab))
                    return false;
                if(col < n-1 && row < n-1 && !is_a_pawn(row+1,col+1, tab))
                    return false;
                if(col > 1 && row < n-2 && pawn_owner(row+1,col-1, tab) == WHITE && !is_a_pawn(row+2,col-2, tab))
                    return false;
                if(col < n-2 && row < n-2 && pawn_owner(row+1,col+1, tab) == WHITE && !is_a_pawn(row+2,col+2, tab))
                    return false;
            }
            else if(tab[row*n+col] == QUEENW || tab[row*n+col] == QUEENB){
                temp_col = col-1;
                temp_row = row-1;
                while(temp_col >= 0 && temp_row >= 0){
                    if(!is_a_pawn(temp_row,temp_col, tab) && queen_way(row,col,temp_row,temp_col, tab))
                        return false;
                    temp_col--;
                    temp_row--;
                }
                temp_col = col+1;
                temp_row = row-1;
                while(temp_col <= n-1 && temp_row >= 0){
                    if(!is_a_pawn(temp_row,temp_col, tab) && queen_way(row,col,temp_row,temp_col,tab))
                        return false;
                    temp_col++;
                    temp_row--;
                }
                temp_col = col-1;
                temp_row = row+1;
                while(temp_col >= 0 && temp_row <= n-1){
                    if(!is_a_pawn(temp_row,temp_col, tab) && queen_way(row,col,temp_row,temp_col,tab))
                        return false;
                    temp_col--;
                    temp_row++;
                }
                temp_col = col+1;
                temp_row = row+1;
                while(temp_col <= n-1 && temp_row <= n-1){
                    if(!is_a_pawn(temp_row,temp_col,tab) && queen_way(row,col,temp_row,temp_col,tab))
                        return false;
                    temp_col++;
                    temp_row++;
                }
            }
        }
    }
    return true;
}

bool checkers::is_there_winner(){
    int black_count = 0,white_count = 0;
    for(int i=0;i<n*n;i++){
        if(tab[i] == WHITE || tab[i] == QUEENW)
            white_count++;
        else if(tab[i] == BLACK || tab[i] == QUEENB)
            black_count++;
    }
    return black_count == 0 || white_count == 0;
}

std::string win_symbol(int win){
	if (win == BLACK)
		return outB;
	else
		return outW;
}

std::string checkers::check_who_won(){
    int res = zobrist_int(tab);
    if (cycle.find(res) != cycle.end())
	return "REMIS";
    if (is_no_pawns())
        return win_symbol(who_got_more_queens());
    if (is_game_blocked())
        return win_symbol(who_got_more_points());
    return win_symbol(who_got_pawns());
}

int checkers::who_got_more_queens(){
    int black_count = 0,white_count = 0;
    for(int i=0;i<n*n;i++){
        if(tab[i] == QUEENW)
            white_count++;
        else if(tab[i] == QUEENB)
            black_count++;
    }
    if(black_count > white_count)
        return BLACK;
    if(white_count > black_count)
        return WHITE;
    return -1;
}

int checkers::who_got_more_points(){
    int black_count = 0,white_count = 0;
    for(int i=0;i<n*n;i++){
        if(tab[i] == WHITE)
            white_count+=3;
        else if(tab[i] == QUEENW)
            white_count+=5;
        else if(tab[i] == BLACK)
            black_count+=3;
        else if(tab[i] == QUEENB)
            black_count+=5;
    }
    if(black_count > white_count)
        return BLACK;
    if(white_count > black_count)
        return WHITE;
    return -1;
}

int checkers::who_got_pawns(){
    for(int i=0;i<n*n;i++){
        if(tab[i] == QUEENW || tab[i] == WHITE)
            return WHITE;
        else if(tab[i] == QUEENB || tab[i] == BLACK)
            return BLACK;
    }
    return 0;
}

int checkers::calculate_board_value(){
    int value;
    value=calculate_pawns_value();
    value=100*value+calculate_dist_to_be_queen();
    value=100*value+calculate_future_queen_kills();
    value=10*value+(std::rand()%10);
    return value;
}

int checkers::calculate_pawns_value(){
    int count = 0;
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

int checkers::calculate_dist_to_be_queen(){
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

int checkers::calculate_future_queen_kills(){
    int * kill_tab = new int [n*n];
    int temp_row,temp_col,white_count=0,black_count=0,white_dead=0,black_dead=0;
    for(int row=0;row<n;row++){
        for(int col=0;col<n;col++){
            kill_tab[row*n+col] = EMPTY;
            if(tab[row*n+col] == QUEENB || tab[row*n+col] == QUEENW){
                temp_row = row-1;
                temp_col = col-1;
                while(temp_row >= 1 && temp_col >= 1){
                    if(is_a_pawn(temp_row,temp_col,tab)){
                        if(pawn_owner(row,col,tab) == pawn_owner(temp_row,temp_col,tab))
                            break;
                        else if(is_a_pawn(temp_row-1,temp_col-1,tab))
                            break;
                        else{
                            kill_tab[temp_row*n+temp_col] = pawn_owner(temp_row,temp_col,tab);
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
                    if(is_a_pawn(temp_row,temp_col,tab)){
                        if(pawn_owner(row,col,tab) == pawn_owner(temp_row,temp_col,tab))
                            break;
                        else if(is_a_pawn(temp_row-1,temp_col+1,tab))
                            break;
                        else{
                            kill_tab[temp_row*n+temp_col] = pawn_owner(temp_row,temp_col,tab);
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
                    if(is_a_pawn(temp_row,temp_col,tab)){
                        if(pawn_owner(row,col,tab) == pawn_owner(temp_row,temp_col,tab))
                            break;
                        else if(is_a_pawn(temp_row+1,temp_col-1,tab))
                            break;
                        else{
                            kill_tab[temp_row*n+temp_col] = pawn_owner(temp_row,temp_col,tab);
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
                    if(is_a_pawn(temp_row,temp_col,tab)){
                        if(pawn_owner(row,col,tab) == pawn_owner(temp_row,temp_col,tab))
                            break;
                        else if(is_a_pawn(temp_row+1,temp_col+1,tab))
                            break;
                        else{
                            kill_tab[temp_row*n+temp_col] = pawn_owner(temp_row,temp_col,tab);
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
            if(is_a_pawn(row,col,tab)){
                if(pawn_owner(row,col,tab) == WHITE)
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
    return 49.5*value+49.5;
}

std::string checkers::player_symbol(int i){
    if (i == WHITE)
        return outW;
    if (i == BLACK)
        return outB;
    return outE;
}

void checkers::move_switch(checkers &ch, int player){
    bool brk;
    while (true){
        std::cout << "\n" << ch << "\n";
        char input = getchar();
        if (input == 'q')
            break;
        switch (input) {
            case 'a':
                for (int i = ch.a_y-1; i >= 0; i--){
                    if (ch.pawn_owner(ch.a_x, i, ch.tab) == player){
                        ch.a_y = i;
                        break;
                    }
                }
                break;
            case 's':
                brk = false;
                for (int i = ch.a_x+1; i < ch.n; i++){
                    for (int j = 0; j < ch.n; j++){
                        if (ch.pawn_owner(i, j, ch.tab) == player){
                            ch.a_x = i;
                            ch.a_y = j;
                            brk = true;
                            break;
                        }
                    }
                    if (brk)
                        break;
                }
                break;
            case 'd':
                for (int i = ch.a_y+1; i < ch.n; i++){
                    if (ch.pawn_owner(ch.a_x, i, ch.tab) == player){
                        ch.a_y = i;
                        break;
                    }
                }
                break;
            case 'w':
                brk = false;
                for (int i = ch.a_x-1; i >= 0; i--){
                    for (int j = 0; j < ch.n; j++){
                        if (ch.pawn_owner(i, j, ch.tab) == player){
                            ch.a_x = i;
                            ch.a_y = j;
                            brk = true;
                            break;
                        }
                    }
                    if (brk)
                        break;
                }
                break;
            case 'z':
                ch.a_x = ch.a_y = -1;
                return;
            default:
                break;
        }
    }
}

void checkers::move(checkers &ch, int * xy, int player, bool next_move){
    int i, j = 0;
    do {
        if (next_move == false){
            std::cout << "Tura gracza: " << checkers::player_symbol(player) << "\n";
            for (int i = 0; i < ch.n*ch.n; ++i){
                if (ch.pawn_owner(i/ch.n, i%ch.n, ch.tab) == player){
                    ch.a_x = i/ch.n;
                    ch.a_y = i%ch.n;
                    break;
                }
            }
            move_switch(ch, player);
            xy[0] = ch.a_x;
            xy[1] = ch.a_y;
        }
        for (i = ch.a_x-1; i < ch.n; ++i){
            for (j = ch.a_y+1; j < ch.n; ++j){
                if (ch.pawn_owner(i, j, ch.tab) == EMPTY){
                    ch.a_x = i;
                    ch.a_y = j;
                }
            }
        }
        if (i == ch.n && j == ch.n){
            for (i = 0; i < ch.n; ++i){
                for (j = 0; j < ch.n; ++j){
                    if (ch.pawn_owner(i, j, ch.tab) == EMPTY){
                        ch.a_x = i;
                        ch.a_y = j;
                    }
                }
            }
        }
        move_switch(ch, EMPTY);
        xy[2] = ch.a_x;
        xy[3] = ch.a_y;
    } while (next_move == false && xy[2] == -1);
}

void checkers::play_computer_vs_computer(checkers &ch){
    int hd1, hd2;
    std::cout << "Wprowadź dwie liczby oznaczające zagłębienie jednego i drugiego gracza:\n";
    std::cin >> hd1 >> hd2;
//    while (hd1 < 1 || hd2 < 1 || hd1 > 2 || hd2 > 2){
//	std::cout << "Muszą byc z zakresu 1-2\n";
//	std::cin >> hd1 >> hd2;
 //   }
 //   bool wait = false;
    int iwhite = hd1, iblack = hd2;
    int i = WHITE, i2 = BLACK;
    while (true){
	   // system("sleep 1");
            int * new_board = computer_turn(ch.n, ch.row_with_pawn, ch.tab, i, hd1);
            for (int k = 0; k < ch.n*ch.n; k++)
                ch.tab[k] = new_board[k];
            std::swap(i, i2);
            std::swap(hd1, hd2);
		std::cout << "\n" << ch << "\n";
        if (ch.is_end_of_game())
            break;
    }
    std::cout << "Zagłębienie dla gracza " << outW << " " << iwhite << "\n";
    std::cout << "Zagłębienie dla gracza " << outB << " " << iblack << "\n";
    std::string end = ch.check_who_won();
    if (end == "REMIS")
       std::cout << "No i mamy remis, każdy gracz gra optymalnie i nie pozwala sie zbić\n";
    else std::cout << "\n" << ch << "\nPlayer " << ch.check_who_won() << "  won!\n";
}

void checkers::play(checkers &ch){
    std::cout << ch << "\n";
    init_zobrist();
    int comp = WHITE, player = BLACK;
    bool plvspl = false;
    bool cudda = false;
    int * new_board = new int [64];
    std::string input;
    std::cout << "Wybierz tryb gry: \n1 -> Player vs Player\n2 -> Player vs Computer\n3 -> Computer vs Computer\n";
    std::cin >> input;
    if (input == "3"){
        play_computer_vs_computer(ch);
        return;
    } else if (input == "1"){
        plvspl = true;
    } else {
        std::cout << "Wybierz 1 jeśli ruch gracza komputerowego ma być liczony na cudzie, 2 jeśli nie.\n";
        std::cin >> input;
        if (true || input == "1"){
            cudda = true;
            std::cout << "Wybierz pionki, b jeśli biały, c jeśli czarny\n";
            std::cin >> input;
            if (input == "b")
                std::swap(comp, player);
        }
    }
    
    int i = WHITE, i2 = BLACK, x, y, x1, y1;
    int xy[4];
    bool check_winer = true;
    while (true){
	if (i == player || plvspl){
	        move(ch, xy, i, false);
        	x = xy[0]; y = xy[1]; x1 = xy[2]; y1 = xy[3];
        	int mv = ch.move(x, y, i, x1, y1, 0);
                if (mv == 0)
                   check_winer = false;
        	if (mv == 2){
           	    while (mv == 2 || mv == 0){
                        std::cout << ch << "\n";
               		if (mv == 2){
              	            x = x1; y = y1;
                	}
                	std::cout << "Your next move " << x << " " << y << " ->\n";
                	move(ch, xy, i, true);
                	x1 = xy[2]; y1 = xy[3];
                	if (x1 == -1){
                    	    mv = 1;
                    	    break;
                	}
                    mv = ch.move(x, y, i, x1, y1, 1);
            	}
            }
            if (mv == 1){
                std::swap(i, i2);
	    }
	} else {
        for (int i = 0; i < 64; i++)
            new_board[i] = ch.tab[i];
        if (cudda)
            new_board = computer_turn(ch.n, ch.row_with_pawn, new_board, comp, 3);
        else
            new_board = computer_turn2(ch.n, ch.row_with_pawn, new_board, comp, 3);
		for (int k = 0; k < ch.n*ch.n; k++)
		    ch.tab[k] = new_board[k];
		std::swap(i, i2);
	}
        if (check_winer)
	        if (ch.is_end_of_game())
        	    break;
    	check_winer = true;
    }
    std::cout << "\n" << ch << "\nPlayer " << ch.check_who_won() << "  won!\n";
    delete [] new_board;
}

std::string symobol(int i){
    switch(i){
        case 0:
            return z0;
        case 1:
            return z1;
        case 2:
            return z2;
        case 3:
            return z3;
        case 4:
            return z4;
        case 5:
            return z5;
        case 6:
            return z6;
        case 7:
            return z7;
        default:
            return "";
    }
    return "";
}

std::ostream& operator<<(std::ostream& os, const checkers& ch){
    os << zs << SSPACE;
    for (int i = 0; i < ch.n; i++)
        os << symobol(i) << SSPACE;
    os << "\n";
    for (int i = 0; i < ch.n; ++i){
        os << symobol(i) << SSPACE;
        for (int j = 0; j < ch.n; ++j){
            if (i == ch.a_x && j == ch.a_y){
                switch (ch.tab[i*ch.n + j]){
                    case WHITE:
                        os << outWW << SSPACE;
                        break;
                    case BLACK:
                        os << outBB << SSPACE;
                        break;
                    case QUEENB:
                        os << outQbb << SSPACE;
                        break;
                    case QUEENW:
                        os << outQww << SSPACE;
                        break;
                    default:
                        os << outEe << SSPACE;
                }
            } else {
                switch (ch.tab[i*ch.n + j]){
                    case WHITE:
                        os << outW << SSPACE;
                        break;
                    case BLACK:
                        os << outB << SSPACE;
                        break;
                    case QUEENB:
                        os << outQb << SSPACE;
                        break;
                    case QUEENW:
                        os << outQw << SSPACE;
                        break;
                    default:
                        os << outE << SSPACE;
                }
            }
        }
        os << "\n";
    }
    return os;
}

CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction alpha_beta, create_tree, delete_tree, print_tree, set_root, copy_best_result;
CUdeviceptr Adev, Atab, Vdev;
int how_deep = 3;
int max_children = 12 * 2;
int cuda_n = max_children;
int blocks_per_grid, threads_per_block, blocks_per_grid2, threads_per_block2, num_threads;
size_t size, size_tab;
checkers_point * a;

void cuda_start(){
//	std::cout << "?";
    cu_auto_assert(cuInit(0));
    cu_auto_assert(cuDeviceGet(&cuDevice, 0));
    cu_auto_assert(cuCtxCreate(&cuContext, 0, cuDevice));
    cuModule = (CUmodule)0;
    cu_auto_assert(cuModuleLoad(&cuModule, "checkers.ptx"));
    cu_auto_assert(cuModuleGetFunction(&alpha_beta, cuModule, "alpha_beta"));
    cu_auto_assert(cuModuleGetFunction(&create_tree, cuModule, "create_tree"));
    cu_auto_assert(cuModuleGetFunction(&delete_tree, cuModule, "delete_tree"));
    cu_auto_assert(cuModuleGetFunction(&print_tree, cuModule, "print_tree"));
    cu_auto_assert(cuModuleGetFunction(&set_root, cuModule, "set_root"));
    cu_auto_assert(cuModuleGetFunction(&copy_best_result, cuModule, "copy_best_result"));
    for (int i = 0; i < how_deep+1; i++){
        cuda_n *= max_children;
    }
    blocks_per_grid = (cuda_n+1023)/1024;
    threads_per_block = 1024;
    blocks_per_grid2 = 1;
    threads_per_block2 = 1000;
    num_threads = threads_per_block2 * blocks_per_grid2;
    size = sizeof(checkers_point)*cuda_n;
    size_tab = sizeof(int)*64;
    a = (checkers_point*) malloc(size);
    cu_auto_assert(cuMemHostRegister(a, size, 0));
    cu_auto_assert(cuMemAlloc(&Adev, size));
    cu_auto_assert(cuMemAlloc(&Vdev, num_threads * sizeof(checkers_point*)));
    cu_auto_assert(cuMemAlloc(&Atab, size_tab));
}

void cuda_stop(){
    cu_auto_assert(cuMemHostUnregister(a));
    cuMemFree(Adev);
    cuMemFree(Atab);
    cuMemFree(Vdev);
    cuCtxDestroy(cuContext);
}


struct next_kill{
    int t[4];
    next_kill * next = NULL;
    int * parent_tab;
};

next_kill * create_next_move(int x, int y, int * par_tb, int x1, int y1){
    next_kill * res = new next_kill;
    res->t[0] = x; res->t[1] = y;
    res->t[2] = x1; res->t[3] = y1;
    res->parent_tab = par_tb;
    return res;
}

void copy_board(int * ch, checkers_point * ch2){
    for (int i = 0; i < 64; i++){
        ch2->board[i] = ch[i];
    }
}

checkers_point * again(checkers_point * ch, next_kill * first, next_kill * last, int pm){
    int x = first->t[0], y = first->t[1], x1 = first->t[2], y1 = first->t[3], * tab = first->parent_tab;
    if(checkers::is_move_correct(x, y, checkers::pawn_owner(x, y, tab), x1, y1, false, tab, false)){
        checkers_point * chld;
        ch->next = new checkers_point;
        ch->next->parent = ch->parent;
        ch->next->prev = ch;
        chld = ch->next;
        chld->min_max = !chld->parent->min_max;
        copy_board(tab, chld);
        chld->parent->how_much_children++;
        chld->board[x1*8+y1] = chld->board[x*8+y];
        chld->board[x*8+y] = EMPTY;
        chld->board[(x+x1)/2*8+(y+y1)/2] = EMPTY;
        ch = chld;
        checkers::create_queen(x1, y1, ch->board);
        last->next = create_next_move(x1, y1, ch->board, x1+pm, y1+2);
        last = last->next;
        last->next = create_next_move(x1, y1, ch->board, x1+pm, y1-2);
        last = last->next;
        std::cout << "AGAIN";
    }
    return ch;
}

checkers_point * create_node(checkers_point * ch, int x, int y, int x1, int y1, bool &nxt, bool queen, bool kill){
    int * tab = ch->board;
    if (ch->parent != NULL)
        tab = ch->parent->board;
    if (checkers::is_move_correct(x, y, checkers::pawn_owner(x, y, tab), x1, y1, false, tab, false)){
        checkers_point * chld;
        if (!nxt){
            std::cout << "child ";
            ch->children = new checkers_point;
            ch->children->parent = ch;
            ch->children->prev = NULL;
            chld = ch->children;
        } else {
            std::cout << "next ";
            ch->next = new checkers_point;
            ch->next->parent = ch->parent;
            ch->next->prev = ch;
            chld = ch->next;
        }
        std::cout << "correct  ";
        chld->min_max = !chld->parent->min_max;
        chld->how_much_children = 0;
        chld->next = chld->children = NULL;
        copy_board(chld->parent->board, chld);
        chld->parent->how_much_children++;
        chld->board[x1*8+y1] = chld->board[x*8+y];
        chld->board[x*8+y] = EMPTY;
        if (kill && queen == false){
            std::cout << "HAHAKIL";
            chld->board[(x+x1)/2*8+(y+y1)/2] = EMPTY;
        }
        ch = chld;
        nxt = true;
        std::cout << x << " " << y << " - > " << x1 << " " << y1 << "\n";
        if (kill && !queen && !(checkers::create_queen(x1, y1, (ch->board)))){
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
    }
    return ch;
}

void create_tree_linear(checkers_point * x, int how_deep, int now) {
    int * tab = x->board;
    for (int i = 0 ; i < 64; i++)
        std::cout << tab[i];
    std::cout << "\n";
    if (how_deep == now)
        return;
    bool nxt = false;
    for (int i = 0; i < 64; ++i){
        switch (tab[i]) {
            case WHITE:
                if (now % 2 == 0){
                    x = create_node(x, i/8, i%8, ((i/8)-1), ((i%8)-1), nxt, false, false);
                    x = create_node(x, i/8, i%8, (i/8)-1, (i%8)+1, nxt, false, false);
                    x = create_node(x, i/8, i%8, (i/8)-2, (i%8)-2, nxt, false, true);
                    x = create_node(x, i/8, i%8, (i/8)-2, (i%8)+2, nxt, false, true);
                }
                break;
            case BLACK:
                if (now % 2 == 1){
                    x = create_node(x, i/8, i%8, i/8+1, (i%8)-1, nxt, false, false);
                    x = create_node(x, i/8, i%8, i/8+1, (i%8)+1, nxt, false, false);
                    x = create_node(x, i/8, i%8, i/8+2, (i%8)-2, nxt, false, true);
                    x = create_node(x, i/8, i%8, i/8+2, (i%8)+2, nxt, false, true);
                }
                break;
            case QUEENW:
                if (now % 2 == 0){
                    
                }
                break;
            case QUEENB:
                if (now % 2 == 1){
                    
                }
                break;
            default:
                break;
        }
    }
    std::cout << "how much children " << x->parent->how_much_children << "\n";
    x = x->parent;
    x = x->children;
    while (x != NULL){
        create_tree_linear(x, how_deep, now+1);
        x = x->next;
    }
}

void set_root_linear(checkers_point *x,int * tab){
	for(int i = 0; i < 64; i++) {
		x->board[i]=tab[i];
	}
    x->how_much_children = 0;
	x->min_max=true;
	x->player=WHITE;
}

void delete_tree_linear(checkers_point * x){
    if (x == NULL)
        return;
    std::cout << x->how_much_children << " ";
    checkers_point * child = x->children;
//	while (child != NULL) {
//		delete_tree_linear(child);
//		child = child->next;
//	}
//	delete x;
}

int alpha_beta_linear(checkers_point * x){
	if(x->children==NULL) {
		checkers check;
		check.tab=x->board;
		x->value=check.calculate_board_value();
		return x->value;
	}
	else if(x->min_max) {
		checkers_point * child = x->children;
		while(child!=NULL) {
			x->value=std::min(x->value,alpha_beta_linear(child));
			child=child->next;
		}
		return x->value;
	}
	else {
		checkers_point * child = x->children;
        while(child!=NULL) {
            	x->value=std::max(x->value,alpha_beta_linear(child));
            	child=child->next;
        	}
        	return x->value;
		}
}

void get_best(checkers_point *x, int *tab) {
	int best_val=x->value;
	checkers_point * child = x->children;
    std::cout << x->value << " ";
	while(child!=NULL) {
        std::cout << child->value << " ";
		if(child->value==x->value)
			break;
		child=child->next;
	}
    if (child == NULL){
        std::cout << "No result???";
        return;
    }
	for(int i=0;i<64;i++) {
		tab[i]=child->board[i];
	}
}

int * computer_turn2(int siize, int row_with_pawn, int * tab_with_board, int player, int hd){
	checkers_point * x = new checkers_point;
	set_root_linear(x,tab_with_board);
	create_tree_linear(x, hd+1, 0);
	alpha_beta_linear(x);
    get_best(x,tab_with_board);
    delete_tree_linear(x);
	return tab_with_board;
}

int * computer_turn(int siize, int row_with_pawn, int * tab_with_board, int player, int hd){
    how_deep = hd;
    cu_auto_assert(cuMemcpyHtoD(Atab, tab_with_board, size_tab));
    int i = 1;
    void* args[] = {&cuda_n, &Adev, &i, &player};
    void* args2[] = {&Adev, &num_threads, &Vdev};
    void* args_root[] = {&Adev, &Atab, &siize, &player};
    cu_auto_assert(cuLaunchKernel(set_root, blocks_per_grid, 1, 1, threads_per_block, 1, 1, 0, 0, args_root, 0));
    for (i = 1; i < how_deep+1; i++){
        cu_auto_assert(cuLaunchKernel(create_tree, blocks_per_grid, 1, 1, threads_per_block, 1, 1, 0, 0, args, 0));
				cu_auto_assert(cuCtxSynchronize());
				cu_auto_assert(cuLaunchKernel(print_tree, blocks_per_grid, 1, 1, threads_per_block, 1, 1, 0, 0, args, 0));
				cu_auto_assert(cuCtxSynchronize());
    }
    cu_auto_assert(cuLaunchKernel(print_tree, blocks_per_grid, 1, 1, threads_per_block, 1, 1, 0, 0, args, 0));
		cu_auto_assert(cuCtxSynchronize());
    cu_auto_assert(cuLaunchKernel(alpha_beta, blocks_per_grid2, 1, 1, threads_per_block2, 1, 1, 0, 0, args2, 0));
		cu_auto_assert(cuCtxSynchronize());
    // cu_auto_assert(cuLaunchKernel(delete_tree, blocks_per_grid2, 1, 1, threads_per_block2, 1, 1, 0, 0, args2, 0));
		cu_auto_assert(cuCtxSynchronize());
    cu_auto_assert(cuLaunchKernel(copy_best_result, blocks_per_grid, 1, 1, threads_per_block, 1, 1, 0, 0, args_root, 0));
		cu_auto_assert(cuCtxSynchronize());
    cu_auto_assert(cuLaunchKernel(delete_tree, blocks_per_grid2, 1, 1, threads_per_block2, 1, 1, 0, 0, args2, 0));
    cu_auto_assert(cuMemcpyDtoH(tab_with_board, Atab, size_tab));
    return tab_with_board;
}
