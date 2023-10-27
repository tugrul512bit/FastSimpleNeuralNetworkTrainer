#include <iostream>
#include"FastSimpleNeuralNetworkTrainer.h"
int main()
{
    constexpr int numParallelSimulations = 5000;
    constexpr int numInputs = 2;
    constexpr int numOutputs = 1;
    GPGPU::ActivationFunction act("x*exp(x)", [](float x) {return x * exp(x); });
    GPGPU::FastSimpleNeuralNetworkTrainer<numParallelSimulations, numInputs, 10, 20, 10, numOutputs> nn(256,1.1f,act);
    GPGPU::TrainingData<numInputs, numOutputs> td;



    // training data for: y = x1 or x2 where 1 means x>=0.5 and 0 means x<0.5
    constexpr int numDataX = 32;
    constexpr int numDataY = 32;
    for (int i = 0; i < numDataX * numDataY; i++)
    {
        std::vector<float> x(numInputs);
        std::vector<float> y(numOutputs);

        x[0] = (i % numDataX) / (float)numDataX;
        x[1] = (i / numDataX) / (float)numDataY;

        y[0] = x[0] >= 0.5f or x[1] >= 0.5f;

        td.AddInputOutputPair(x, y);
    }

    // neural network learns how to compute y = x1 or x2
    // more data = better fit, too much data = overfit, less data = generalization, too few data = not learning good
    std::vector<float> testInput = { 0.75f, 0.75f };
    auto model = nn.Train(td, testInput, [testInput](std::vector<float> testOutput)
        {
            std::cout << "training: now " << (testInput[0] >= 0.5f) << " or " << (testInput[1] >= 0.5f) << " is " << (testOutput[0] >= 0.5f) << std::endl;
        });

    auto result = model.Run({ 0.3f, 0.3f });
    std::cout << " 0 or 0  = " << (result[0] >= 0.5f) << std::endl;
    result = model.Run({ 0.3f, 0.6f });
    std::cout << " 0 or 1  = " << (result[0] >= 0.5f) << std::endl;
    result = model.Run({ 0.6f, 0.3f });
    std::cout << " 1 or 0  = " << (result[0] >= 0.5f) << std::endl;
    result = model.Run({ 0.6f, 0.6f });
    std::cout << " 1 or 1  = " << (result[0] >= 0.5f) << std::endl;
    {
        constexpr int numDataX = 500;
        constexpr int numDataY = 500;
        int err = 0;
        for (int i = 0; i < numDataX * numDataY; i++)
        {
            std::vector<float> x(numInputs);
            std::vector<float> y(numOutputs);

            x[0] = (i % numDataX) / (float)numDataX;
            x[1] = (i / numDataX) / (float)numDataY;

            y[0] = x[0] >= 0.5f or x[1] >= 0.5f;

            auto z = model.Run(x);
            if (std::abs(y[0] - z[0]) > 0.01)
            {
                err++;
                
            }
        }
        if (err > 0)
        {
            std::cout << "error: found training error. "<< numDataX * numDataX<<" samples have "<<err<<" errors" << std::endl;
            
        }
    }
    return 0;
}
