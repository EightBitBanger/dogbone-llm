#include "Print.h"
#include <iostream>
#include <sstream>

unsigned int mConsolPosition = 0;
unsigned int mConsoleWidth = 120;

void print(std::string str) {
    mConsolPosition += str.length();
    if (mConsolPosition > mConsoleWidth) 
        printLn();
    std::cout << str;
}

void printInt(int value) {
    std::stringstream stream;
    stream << value;
    mConsolPosition += stream.str().length();
    if (mConsolPosition > mConsoleWidth) 
        printLn();
    std::cout << stream.str();
}

void printFloat(float value) {
    std::stringstream stream;
    stream << value;
    mConsolPosition += stream.str().length();
    if (mConsolPosition > mConsoleWidth) 
        printLn();
    std::cout << stream.str();
}

void printLn() {
    mConsolPosition = 0;
    std::cout << "\n";
}
