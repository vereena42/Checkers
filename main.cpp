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
    //system("stty raw");
    cuda_start();
    srand( time( NULL ) );
    checkers warcaby;
    std::cout << "?";
    checkers::play(warcaby);
    cuda_stop();
    return 0;
}
