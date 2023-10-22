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
    nn.Train(td);

    return 0;
}
