# FastSimpleNeuralNetworkTrainer

This GPGPU library uses multiple GPUs to train small-scale neural-networks, fast and in a simple way.

## Dependencies

- Vcpkg (for OpenCL)
- OpenCL (for GPGPU in CPUs, GPUs and FPGAs)
- C++17

  ## Algorithm

  It works on a simple parallelized simulated annealing algorithm which supports load-balancing between multiple GPUs and CPUs. Every pipeline of GPU runs same neural network but with different input-output data pair. This makes single-pass training for whole data set. But parameters are stored inside in-chip fast memory so the number of parameters can not exceed this small amount (possible as low as 64kB).

## Hello World

Following square-root approximation test completes in 8 seconds using two GPUs (~10000 CUDA cores)

```C++
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
    auto model = nn.Train(td, testInput, [testInput](std::vector<float> testOutput) 
        {
            std::cout<<"training: now square root of "<<testInput[0]<<" is " << testOutput[0] << std::endl;
        });

    auto result = model.Run({ 0.49 });
    std::cout << "Square root of 0.49 = " << result[0] << std::endl;
    return 0;
}

```

output:

```
lower energy found: 104.923
training: now square root of 0.5 is 0.707106
reheating. num reheats left=4
reheating. num reheats left=3
lower energy found: 104.902
training: now square root of 0.5 is 0.707107
reheating. num reheats left=2
lower energy found: 104.891
training: now square root of 0.5 is 0.707107
lower energy found: 104.855
training: now square root of 0.5 is 0.707107
reheating. num reheats left=1
total computation-time=8.48946 seconds (this includes debugging console-output that is slow)
---------------
OpenCL device info:
NVIDIA GeForce RTX 4070 computed 27.84% of total work
NVIDIA GeForce RTX 4060 Ti computed 21.78% of total work
---------------
Square root of 0.49 = 0.7
```
