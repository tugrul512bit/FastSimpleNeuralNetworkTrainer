#include <iostream>
#include<algorithm>
#include <random>
#include"FastSimpleNeuralNetworkTrainer.h"
int main()
{
    constexpr int numParallelSimulations =500;
    constexpr int numInputs = 3;
    constexpr int numOutputs = 3;
    const GPGPU::ActivationFunction act("tanh(x)", [](float x) { return tanh(x); });
    GPGPU::FastSimpleNeuralNetworkTrainer<numParallelSimulations, numInputs, 10, 20, 10, numOutputs> nn(256,1.1f,act);
    GPGPU::TrainingData<numInputs, numOutputs> td;


    std::random_device rd;
    std::mt19937 rng{ rd() };
    std::uniform_real_distribution<float> val(0, 1);


    // training data for: unsorted array ==> sorted array
    constexpr int numData = 4000;

    for (int i = 0; i < numData; i++)
    {
        std::vector<float> x(numInputs);

        x[0] = val(rng);
        x[1] = val(rng);
        x[2] = val(rng);

        std::vector<float> y = x;
        std::sort(y.begin(), y.end());

        td.AddInputOutputPair(x, y);
    }

    // neural network learns how to sort an array    
    std::vector<float> testInput = { 0.75f, 0.45f, 0.9f };
    auto model = nn.Train(td, testInput, [testInput](std::vector<float> testOutput)
        {
            std::cout << "training: { 0.75f, 0.45f, 0.9f } ==> {" << testOutput[0] << "," << testOutput[1] << "," << testOutput[2] << "}" << std::endl;
        },1,0.00001f,1.05f,500);

    for (int i = 0; i < 20; i++)
    {
        float v1, v2, v3;
        auto result = model.Run({ v1 = val(rng), v2 = val(rng), v3 = val(rng) });
        std::cout << v1 << " " << v2 << " " << v3 << " = > " << result[0] << " " << result[1] << " " << result[2] << std::endl;
    }
    return 0;
}

