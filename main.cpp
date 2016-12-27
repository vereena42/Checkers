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
	srand( time( NULL ) );
    checkers warcaby;
    checkers::play(warcaby);
    return 0;
}
