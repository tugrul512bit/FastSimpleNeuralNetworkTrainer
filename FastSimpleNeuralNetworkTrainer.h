#pragma once

#include<vector>
#include<iostream>
template<int ... ARGS>
class FastSimpleNeuralNetworkTrainer
{
public:
    FastSimpleNeuralNetworkTrainer()
    {
        architecture = { ARGS... };
        //std::cout << sizeof...(ARGS) << std::endl;
        
    }
private:
    std::vector<int> architecture;
};
