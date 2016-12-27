//
//  checkers.cpp
//  Warcaby
//
//


#include "checkers.hpp"

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
    }
    
}

int checkers::pawn_owner(int x, int y){
    if (tab[x*n+y] == BLACK || tab[x*n+y] == queenB)
        return BLACK;
    if (tab[x*n+y] == WHITE || tab[x*n+y] == queenW)
        return WHITE;
    return EMPTY;
}

int checkers::move(int x, int y, int who, int x1, int y1, int kll){
    if (!is_move_correct(x, y, who, x1, y1, kll))
        return 0;
    if (has_next_move(x, y, x1, y1)){
        create_queen(x1, y1);
        return 2;
    }
    create_queen(x1, y1);
    return 1;
}

bool checkers::is_move_correct(int x, int y, int who, int x1, int y1, int kll){
    if (x < 0 || x >= n || x1 < 0 || x1 >= n || y < 0 || y >= n || y1 < 0 || y1 >= n ){
        std::cout << "WRONG!\n";
        return false;
    }
    if (std::abs(x-x1) != std::abs(y-y1)){
        std::cout << "WRONG!\n";
        return false;
    }
    int pwn_wnr = pawn_owner(x, y);
    if (pwn_wnr == EMPTY){
        std::cout << "There's no pawn!\n";
        return false;
    }
    if (pwn_wnr != who){
        std::cout << "This is not your pawn!\n";
        return false;
    }
    if (is_a_pawn(x1, y1)){
        std::cout << "Target field is not empty\n";
        return false;
    }
    if (x < x1 && who == WHITE && tab[x*n+y] != queenW){
        std::cout << "This is not a Queen!\n";
        return false;
    }
    if (x > x1 && who == BLACK && tab[x*n+y] != queenB){
        std::cout << "This is not a Queen!\n";
        return false;
    }
    if ((tab[x*n+y] == queenW || tab[x*n+y] == queenB) && (!queen_way(x, y, x1, y1))){
        std::cout << "Something is wrong :C\n";
        return false;
    }
    if (!is_queen(x, y) && std::abs((x-x1)) > 1 && !correct_kill(x, y, (x1+x)/2, (y1+y)/2)){
        std::cout << "NOPE!\n";
        return false;
    }
    if (kll){
        kll = 0;
        int x_r = x > x1 ? -1 : 1, y_r = y > y1 ? -1 : 1;
        int own = pawn_owner(x, y);
        x += x_r; y += y_r;
        while (x != x1){
            std::cout << pawn_owner(x, y) << " " << x << " " << y << "\n";
            if (is_a_pawn(x, y) && pawn_owner(x, y) != own){
                kll++;
            }
            x += x_r; y += y_r;
        }
        return (kll > 0);
    }
    return true;
}

bool checkers::queen_way(int x, int y, int x1, int y1){
    int own = pawn_owner(x, y);
    int x_r = x > x1 ? -1 : 1, y_r = y > y1 ? -1 : 1;
    bool next_empty = false;
    x += x_r; y += y_r;
    while (x != x1){
        if (is_a_pawn(x, y)){
            if (next_empty)
                return false;
            next_empty = true;
            if (pawn_owner(x, y) == own)
                return false;
        } else {
            next_empty = false;
        }
        x += x_r; y += y_r;
    }
    return true;
}

bool checkers::is_a_pawn(int x, int y){
    return !(tab[x*n+y] == EMPTY);
}

void checkers::kill(int x, int y){
    tab[x*n+y] = EMPTY;
}

bool checkers::has_next_move(int x, int y, int x1, int y1){
    tab[x1*n+y1] = tab[x*n+y];
    kill(x, y);
    if (is_queen(x1, y1)){
        int kll = 0;
        int x_r = x > x1 ? -1 : 1, y_r = y > y1 ? -1 : 1;
        while (x != x1){
            if (is_a_pawn(x, y)){
                kill(x, y);
                kll++;
            }
            x += x_r; y += y_r;
        }
        return (kll > 0);
    }
    if (std::abs((x-x1)) == 1)
        return false;
    kill((x+x1)/2, (y+y1)/2);
    return true;
}

bool checkers::is_queen(int x, int y){
    return (tab[x*n+y] == queenB || tab[x*n+y] == queenW);
}

bool checkers::correct_kill(int x, int y, int x1, int y1){
    if (pawn_owner(x, y) != pawn_owner(x1, y1))
        return true;
    return false;
}

bool checkers::create_queen(int x, int y){
    if ((x != 0 && x != n-1) ||
        (tab[x*n+y] != WHITE && tab[x*n+y] != BLACK) ||
        (x == 0 && tab[x*n+y] == BLACK) ||
        (x == n-1 && tab[x*n+y] == WHITE))
        return false;
    if (tab[x*n+y] == WHITE)
        tab[x*n+y] = queenW;
    else
        tab[x*n+y] = queenB;
    return true;
}

std::string checkers::player_symbol(int i){
    if (i == WHITE)
        return outW;
    if (i == BLACK)
        return outB;
    return outE;
}

void checkers::play(checkers &ch){
    int i = WHITE, i2 = BLACK, x, y, x1, y1;
    while (true){
        std::cout << ch << "\n";
        std::cout << "Tura gracza: " << checkers::player_symbol(i) << "\n";
        std::cin >> x >> y >> x1 >> y1;
        int move = ch.move(x, y, i, x1, y1, 0);
        if (move == 2){
            while (move == 2 || move == 0){
                std::cout << ch << "\n";
                if (move == 2){
                    x = x1; y = y1;
                }
                std::cout << "Your next move " << x << " " << y << " ->\n";
                std::cin >> x1 >> y1;
                if (x1 == -1){
                    move = 1;
                    break;
                }
                move = ch.move(x, y, i, x1, y1, 1);
            }
        }
        if (move == 1)
            std::swap(i, i2);
    }
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
    os << zs;
    for (int i = 0; i < ch.n; i++)
        os << symobol(i);
    os << "\n";
    for (int i = 0; i < ch.n; ++i){
        os << symobol(i);
        for (int j = 0; j < ch.n; ++j){
            switch (ch.tab[i*ch.n + j]){
                case WHITE:
                    os << outW;
                    break;
                case BLACK:
                    os << outB;
                    break;
                case queenB:
                    os << outQb;
                    break;
                case queenW:
                    os << outQw;
                    break;
                default:
                    os << outE;
            }
        }
        os << "\n";
    }
    return os;
}
