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
    system("stty raw");
	srand( time( NULL ) );
    checkers warcaby;
    checkers::play(warcaby);
    return 0;
}
