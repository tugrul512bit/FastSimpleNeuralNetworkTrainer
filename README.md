# FastSimpleNeuralNetworkTrainer

This GPGPU library uses multiple GPUs to train small-scale neural-networks, fast and in a simple way.

## Dependencies

- Vcpkg (for OpenCL)
- OpenCL (for GPGPU in CPUs, GPUs and FPGAs)
- C++17

## Algorithm

It works on parallelized simulated annealing algorithm which supports load-balancing between multiple GPUs and CPUs. Every pipeline of GPU runs same neural network but with different input-output data pair. This makes single-pass training for whole data set with thousands of simulations in parallel. But parameters are stored on in-chip fast memory so the number of parameters can not exceed this small amount (sometimes as low as 64kB).

![training](https://github.com/tugrul512bit/FastSimpleNeuralNetworkTrainer/blob/master/neural-network-training.png)

### Hello World

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

---

### Example: Y = X1 or X2

```C++
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

```

output:

```
lower energy found: 277.04
training: now 1 or 1 is 1
lower energy found: 277.035
training: now 1 or 1 is 1
lower energy found: 277.035
training: now 1 or 1 is 1
lower energy found: 277.028
training: now 1 or 1 is 1
total computation-time=7.64115 seconds (this includes debugging console-output that is slow)
---------------
OpenCL device info:
NVIDIA GeForce RTX 4070 computed 26.96% of total work
NVIDIA GeForce RTX 4060 Ti computed 23.94% of total work
---------------
 0 or 0  = 0
 0 or 1  = 1
 1 or 0  = 1
 1 or 1  = 1
```
---

### Example: Array Sorting

```C++
#include <iostream>
#include<algorithm>
#include <random>
#include"FastSimpleNeuralNetworkTrainer.h"
int main()
{
    constexpr int numInputs = 3;
    constexpr int numOutputs = 3;
    GPGPU::FastSimpleNeuralNetworkTrainer<numInputs, 10, 20, 10, numOutputs> nn;
    GPGPU::TrainingData<numInputs, numOutputs> td;


    std::random_device rd;
    std::mt19937 rng{ rd() };
    std::uniform_real_distribution<float> val(0, 1);


    // training data for: unsorted array ==> sorted array
    constexpr int numData = 10000;

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
            std::cout<<"training: { 0.75f, 0.45f, 0.9f } ==> {"<<testOutput[0]<<","<<testOutput[1]<<","<<testOutput[2]<<"}" << std::endl;
        });

    auto result = model.Run({ 0.3f, 0.3f, 0.5f });
    std::cout << " 0.3 0.3 0.5  => " << result[0]<<" "<<result[1]<<" "<<result[2] << std::endl;
    
    return 0;
}

```
output:
```
lower energy found: 4520.95
training: { 0.75f, 0.45f, 0.9f } ==> {0.456367,0.749462,0.916047}
lower energy found: 4520.94
training: { 0.75f, 0.45f, 0.9f } ==> {0.456301,0.749611,0.916149}
lower energy found: 4520.94
training: { 0.75f, 0.45f, 0.9f } ==> {0.456391,0.749787,0.916117}
lower energy found: 4520.86
training: { 0.75f, 0.45f, 0.9f } ==> {0.456467,0.749952,0.916225}
lower energy found: 4520.77
training: { 0.75f, 0.45f, 0.9f } ==> {0.456201,0.749887,0.916184}
reheating. num reheats left=1
total computation-time=209.195 seconds (this includes debugging console-output that is slow)
---------------
OpenCL device info:
NVIDIA GeForce RTX 4070 computed 25.04% of total work
NVIDIA GeForce RTX 4060 Ti computed 23.74% of total work
---------------
 0.3 0.3 0.5  => 0.238977 0.351452 0.445364

```
