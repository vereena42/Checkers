//
//  main.cpp
//  Warcaby
//
//

#include <iostream>
#include <cstdlib>
#include <ctime>
#include "checkers.hpp"

int main(int argc, const char * argv[]) {
    std::cout << "Warcaby\nSterowanie:\na + enter -> wyranie następnego pionka/pola po lewej\nd + enter -> wybranie następnego pionka/pola po prawej\nw + enter -> wyrabie następnego pionka/pola powyżej\ns + enter -> wybranie nastęnego pionka/pola poniżej\n";
    std::cout << "q + enter -> zatwierdzenie wyboru pionka\n";
    std::cout << "z + enter -> rezygnacja z następnego ruchu/zmiana pionka\n";

    //system("stty raw");
    cuda_start();
    srand( time( NULL ) );
    checkers warcaby;
    //std::cout << "?";
    checkers::play(warcaby);
    cuda_stop();
    return 0;
}
