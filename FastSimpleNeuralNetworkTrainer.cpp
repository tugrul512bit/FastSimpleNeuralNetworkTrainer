#include <iostream>
#include"FastSimpleNeuralNetworkTrainer.h"
int main()
{
    constexpr int numParallelSimulations = 1000;
    constexpr int numInputs = 1;
    constexpr int numOutputs = 1;
    GPGPU::FastSimpleNeuralNetworkTrainer<numParallelSimulations, numInputs, 10, 20, 10, numOutputs> nn;
    GPGPU::TrainingData<numInputs, numOutputs> td;



    // training data for: y = sqrt(x)
    constexpr int numData = 2500;
    for (int i = 0; i < numData; i++)
    {
        std::vector<float> x(numInputs);
        std::vector<float> y(numOutputs);

        x[0] = i / (float)numData;
        y[0] = std::sqrt(x[0]);

        td.AddInputOutputPair(x, y);
    }

    // neural network learns how to compute y = sqrt(x)
    // more data = better fit, too much data = overfit, less data = generalization, too few data = not learning good

    std::vector<float> testInput = { 0.5f };
    auto model = nn.Train(td, testInput, [testInput](std::vector<float> testOutput)
        {
            std::cout << "training: now square root of " << testInput[0] << " is " << testOutput[0] << std::endl;
        });



    double err = 0.0;
    int ctr = 0;
    for (float i = 0.0001f; i < 1.0f; i += 0.0001f)
    {

        auto result = model.Run({ i });
        err += std::abs(result[0] - sqrt(i)) / std::abs(sqrt(i));
        ctr++;
    }

    std::cout<<ctr<<" samples have " << err / ctr << "% average error. "<< std::endl;
    return 0;
}

