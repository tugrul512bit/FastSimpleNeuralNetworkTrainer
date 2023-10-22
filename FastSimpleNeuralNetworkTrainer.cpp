#include <iostream>
#include"FastSimpleNeuralNetworkTrainer.h"
int main()
{
    constexpr int numInputs = 1;
    constexpr int numOutputs = 1;
    GPGPU::FastSimpleNeuralNetworkTrainer<numInputs, 10, 20, 10, numOutputs> nn;
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
    float startTemperature = 1.0f;
    float stopTemperature = 0.0001f;
    float coolingRate = 1.1f;
    int numReheating = 5;
    bool debugPerformance = true;
    bool debugDevice = true;
    bool debugEnergy = true;
    auto model = nn.Train(
        td, 
        testInput, 
        [testInput](std::vector<float> testOutput)
        {
            std::cout << "training: now square root of " << testInput[0] << " is " << testOutput[0] << std::endl;
        },
        startTemperature,
        stopTemperature,
        coolingRate,
        numReheating,
        debugPerformance,
        debugDevice,
        debugEnergy
    );

    auto result = model.Run({ 0.49 });
    std::cout << "Square root of 0.49 = " << result[0] << std::endl;
    return 0;
}
