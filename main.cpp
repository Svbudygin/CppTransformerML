#include <iostream>
#include <fstream>
#include <istream>
#include <string>
#include "modules/nnlayers.h"



class Bolvanka {
public:
    LinearLayer ll;

    void forward() 
    {
        //
    }

    void backward() 
    {
        //
    }
};

int main()
{
    // read lines from file
    std::ifstream file("input.txt");
    std::string line;
    while (std::getline(file, line))
    {
        // one processing
    }

    //
    Bolvanka b;
    b.forward();
    b.backward();
    // some kind of b.printloss on test data

    return 0;
}