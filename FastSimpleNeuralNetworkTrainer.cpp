#include <iostream>
#include"FastSimpleNeuralNetworkTrainer.h"
int main()
{
    constexpr int numInputs = 2;
    constexpr int numOutputs = 1;
    GPGPU::FastSimpleNeuralNetworkTrainer<numInputs, 10, 20, 10, numOutputs> nn;
    GPGPU::TrainingData<numInputs, numOutputs> td;



    // training data for: y = x1 or x2 where 1 means x>=0.5 and 0 means x<0.5
    constexpr int numDataX = 50;
    constexpr int numDataY = 50;
    for (int i = 0; i < numDataX*numDataY; i++)
    {
        std::vector<float> x(numInputs);
        std::vector<float> y(numOutputs);

        x[0] = (i % numDataX)/(float)numDataX;
        x[1] = (i / numDataX)/(float)numDataY;

        y[0] = x[0]>=0.5f or x[1]>=0.5f;

        td.AddInputOutputPair(x, y);
    }

    // neural network learns how to compute y = x1 or x2
    // more data = better fit, too much data = overfit, less data = generalization, too few data = not learning good
    std::vector<float> testInput = { 0.75f, 0.75f };
    auto model = nn.Train(td, testInput, [testInput](std::vector<float> testOutput) 
        {
            std::cout<<"training: now "<<(testInput[0]>=0.5f)<<" or "<<(testInput[1]>=0.5f) << " is " << (testOutput[0]>=0.5f) << std::endl;
        });

    auto result = model.Run({ 0.3f, 0.3f });
    std::cout << " 0 or 0  = " << (result[0]>=0.5f) << std::endl;
    result = model.Run({ 0.3f, 0.6f });
    std::cout << " 0 or 1  = " << (result[0] >= 0.5f) << std::endl;
    result = model.Run({ 0.6f, 0.3f });
    std::cout << " 1 or 0  = " << (result[0] >= 0.5f) << std::endl;
    result = model.Run({ 0.6f, 0.6f });
    std::cout << " 1 or 1  = " << (result[0] >= 0.5f) << std::endl;
    return 0;
}
