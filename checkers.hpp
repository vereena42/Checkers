//
//  checkers.hpp
//  Warcaby
//
//

#ifndef checkers_hpp
#define checkers_hpp

#include <iostream>
#include <stdio.h>
#include "cuda.h"
#define default_n 8
#define default_row_with_pawn 3
#define EMPTY 0
#define WHITE 1
#define BLACK 2
#define queenW 11
#define queenB 22

//X OS
/*
#define outE "⬜️"
#define outEe "⬛️"
#define outW "♥️"
#define outWW "💙"
#define outB "♠️"
#define outBB "♣️"
#define outQw "👸🏻"
#define outQww "👩🏼"
#define outQb "👸🏿"
#define outQbb "👩🏿"
#define zs "⚜️"
#define z0 "0️⃣"
#define z1 "1️⃣"
#define z2 "2️⃣"
#define z3 "3️⃣"
#define z4 "4️⃣"
#define z5 "5️⃣"
#define z6 "6️⃣"
#define z7 "7️⃣"
 */
//LINUX

#define outE "_"
#define outW "O"
#define outB "X"
#define outQw "U"
#define outQb "Y"
#define zs "~"
#define z0 "0"
#define z1 "1"
#define z2 "2"
#define z3 "3"
#define z4 "4"
#define z5 "5"
#define z6 "6"
#define z7 "7"

class checkers{
public:
    int n;
    int row_with_pawn;
    int * tab;
    int a_x, a_y;
    checkers();
    checkers(int n, int rwp);
    ~checkers();
    void new_game();
    int move(int x, int y, int who, int x1, int y1, int kll);
    int pawn_owner(int x, int y);
    bool is_move_correct(int x, int y, int who, int x1, int y1, int kll);
    bool has_next_move(int x, int y, int x1, int y1);
    bool is_a_pawn(int x, int y);
    bool correct_kill(int x, int y, int x1, int y1);
    void kill(int x, int y);
    bool queen_way(int x, int y, int x1, int y1);
    bool create_queen(int x, int y);
    bool is_queen(int x, int y);
    bool is_end_of_game();
    bool is_no_pawns();
    bool is_game_blocked();
    bool is_there_winner();
    int check_who_won();
    int who_got_more_queens();
    int who_got_more_points();
    int who_got_pawns();
    int calculate_board_value();
    int calculate_pawns_value();
    int calculate_dist_to_be_queen();
    int calculate_future_queen_kills();
    static std::string player_symbol(int k);
    static void play(checkers &ch);
    static void move(checkers &ch, int * tab, int player, bool next_move);
    static void move_switch(checkers &ch, int player);
};
std::ostream& operator<<(std::ostream& os, const checkers& ch);

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

int * computer_turn(int siize, int default_row_with_pawn, int * tab_with_board);

#endif /* checkers_hpp */
