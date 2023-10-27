#include <iostream>
#include"FastSimpleNeuralNetworkTrainer.h"
int main()
{
    constexpr int numParallelSimulations = 500;
    constexpr int numInputs = 1;
    constexpr int numOutputs = 1;
    const int numThreadsPerSimulation = 256;
    const float parameterScaling = 1.1f;
    //const GPGPU::ActivationFunction activation("tanh(x)", [](float x) { return tanh(x); });
    const GPGPU::ActivationFunction activation("tanh(x)", [](float x) { return tanh(x); });
    
    GPGPU::FastSimpleNeuralNetworkTrainer<numParallelSimulations, numInputs,10, 20, 10,  numOutputs> nn(
        numThreadsPerSimulation, parameterScaling, activation
    );
    GPGPU::TrainingData<numInputs, numOutputs> td;
    


    // training data for: y = sqrt(x)
    constexpr int numData =2252;
    for (int i = 0; i < numData; i++)
    {
        std::vector<float> x(numInputs);
        std::vector<float> y(numOutputs);

        // x
        x[0] = i / (float)numData;
        // weighting close-to-zero x values more because of floating-point accuracy
        // y
        y[0] = std::sqrt(x[0]);

        td.AddInputOutputPair(x, y);
    }

    // neural network learns how to compute y = sqrt(x)
    // more data = better fit, too much data = overfit, less data = generalization, too few data = not learning good

    std::vector<float> testInput = { 0.5f };
    float startTemperature = 1.0f;
    float stopTemperature = 0.00001f;
    float coolingRate = 1.5f;
    int numRepeats = 5;
    auto model = nn.Train(td, testInput, [testInput](std::vector<float> testOutput)
        {
            std::cout << "training: now square root of " << testInput[0] << " is " << testOutput[0] << std::endl;
        }, startTemperature, stopTemperature, coolingRate, numRepeats);



    double errPercent = 0.0;
    double errTotal = 0.0;
    double errMax = 0.0;
    int ctr = 0;
    for (double i = 0.00001; i < 1.0; i += 0.00001)
    {

        auto result = model.Run({ (float)i });
        double err = std::abs(result[0] - sqrt(i));
        errPercent += 100.0 * err / std::abs(sqrt(i));
        errTotal += err;
        if (errMax < err)
            errMax = err;
        ctr++;
    }

    std::cout << ctr << " samples between [0,1] have " << errPercent / ctr << "% average error, " << errTotal << " total error, "<<errMax<<" maximum error." << std::endl;
    return 0;
}