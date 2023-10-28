# FastSimpleNeuralNetworkTrainer

This GPGPU library uses multiple GPUs to train small-scale neural-networks with variable number of fully connected layers, fast and in a simple way.

## Dependencies

- Vcpkg (for OpenCL)
- OpenCL (for GPGPU in CPUs, GPUs and FPGAs)
- C++17

## Algorithm

It works on parallelized simulated annealing algorithm which supports load-balancing between multiple GPUs and CPUs. Every pipeline of GPU runs same neural network but with different input-output data pair. This makes single-pass training for whole data set with thousands of simulations in parallel. But parameters are stored on in-chip fast memory so the number of parameters can not exceed this small amount (sometimes as low as 64kB).

![training](https://github.com/tugrul512bit/FastSimpleNeuralNetworkTrainer/blob/master/neural-network-training.png)

## How To Use?

Class ```GPGPU::FastSimpleNeuralNetworkTrainer``` takes multiple template parameters starting with number of parallel simulations, ending with topology of the neural network to train & infer. To create a neural network with two hidden layers(of 10 neurons each), 1 input, 2 outputs following code is valid: 

```C++
    // 500 simulations in parallel, each with 1:10:10:2 topology & different parameters
    GPGPU::FastSimpleNeuralNetworkTrainer<500, 1, 10, 10, 2> nn;
```

then training data needs to be prepared:

```C++
    // 1 input element, 2 output elements
    GPGPU::TrainingData<1, 2> td;
    // x and y are vectors for 1 input element, 2 output elements for network
    td.AddInputOutputPair(x, y); 
    td.AddInputOutputPair(x2, y2);
```

once data is prepared, training can start (it takes a callback function that is called whenever a better solution is found, until simulation ends):

```C++
    std::vector<float> testInput = { 0.5f };
    auto model = nn.Train(td, testInput, [testInput](std::vector<float> testOutput)
        {
            std::cout << "training: now square root of " << testInput[0] << " is " << testOutput[0] << std::endl;
        });
```

finally, the trained network can start computing test data:

```C++
    auto result = model.Run({ 0.49 });
    std::cout << "Square root of 0.49 = " << result[0] << std::endl;
```

### Hello World

Following square-root approximation training completes in ~1.5 seconds using two GPUs (~10000 CUDA cores) with 1 : 10 : 20 : 10 : 1 neural network topology

```C++
#include <iostream>
#include"FastSimpleNeuralNetworkTrainer.h"
int main()
{
    constexpr int numParallelSimulations = 1500;
    constexpr int numInputs = 1;
    constexpr int numOutputs = 1;
    const int numThreadsPerSimulation = 256;
    const float parameterScaling =1.1f;
    //const GPGPU::ActivationFunction activation("sin(x)", [](float x) { return sin(x); });
    //const GPGPU::ActivationFunction activation("tanh(x)", [](float x) { return tanh(x); });
    //const GPGPU::ActivationFunction activation("x/(1.0f + exp(-x))", [](float x) { return x / (1.0f + exp(-x)); });
    //const GPGPU::ActivationFunction activation("exp(-x*x)", [](float x) { return exp(-x*x); });
    //const GPGPU::ActivationFunction activation("4.0f*tanh(x)", [](float x) { return 4.0f*tanh(x); });
    //const GPGPU::ActivationFunction activation("exp(x)", [](float x) { return exp(x); });
    const GPGPU::ActivationFunction activation("x*exp(x)", [](float x) { return x*exp(x); });

    GPGPU::FastSimpleNeuralNetworkTrainer<numParallelSimulations, numInputs,10, 20, 10,  numOutputs> nn(
        numThreadsPerSimulation, parameterScaling, activation
    );
    GPGPU::TrainingData<numInputs, numOutputs> td;
    


    // training data for: y = sqrt(x)
    constexpr int numData =256;
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
    float stopTemperature = 0.0001f;
    float coolingRate = 1.1f;
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
```

output:

```
lower energy found: 0.00489567
training: now square root of 0.5 is 0.704874
lower energy found: 0.00489242
training: now square root of 0.5 is 0.705003
lower energy found: 0.0048903
training: now square root of 0.5 is 0.705251
lower energy found: 0.00488862
training: now square root of 0.5 is 0.705173
total computation-time=1.48203 seconds (this includes debugging console-output that is slow)
---------------
OpenCL device info:
NVIDIA GeForce RTX 4070 computed 40.8667% of total work
NVIDIA GeForce RTX 4060 Ti computed 25.7333% of total work
NVIDIA GeForce RTX 4070 computed 21.5333% of total work
NVIDIA GeForce RTX 4060 Ti computed 11.8667% of total work
---------------
100000 samples between [0,1] have 0.708837% average error, 295.043 total error, 0.0253749 maximum error.
```

---

### Example: Y = X1 or X2 learned in 7.5 seconds (Topology = 2 : 10 : 20 : 10 : 1)

```C++
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

```

output:

```
lower energy found: 6.75636e-06
training: now 1 or 1 is 1
lower energy found: 6.72371e-06
training: now 1 or 1 is 1
lower energy found: 6.57961e-06
training: now 1 or 1 is 1
total computation-time=7.58356 seconds (this includes debugging console-output that is slow)
---------------
OpenCL device info:
NVIDIA GeForce RTX 4070 computed 25.9% of total work
NVIDIA GeForce RTX 4060 Ti computed 20.96% of total work
---------------
 0 or 0  = 0
 0 or 1  = 1
 1 or 0  = 1
 1 or 1  = 1
error: found training error. 250000 samples have 3907 errors
```
---

### Example: 3-Element Array Sorting (topology = 3 : 10 : 20 : 10 : 3)

```C++
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



```
output:
```
training: { 0.75f, 0.45f, 0.9f } ==> {0.44334,0.764172,0.896679}
lower energy found: 2.38113
training: { 0.75f, 0.45f, 0.9f } ==> {0.443298,0.76415,0.896682}
reheating. num reheats left=1
lower energy found: 2.38113
training: { 0.75f, 0.45f, 0.9f } ==> {0.443315,0.764151,0.896683}
lower energy found: 2.38112
training: { 0.75f, 0.45f, 0.9f } ==> {0.443317,0.764172,0.896672}
total computation-time=1123.08 seconds (this includes debugging console-output that is slow)
---------------
OpenCL device info:
NVIDIA GeForce RTX 4070 computed 26.2% of total work
NVIDIA GeForce RTX 4060 Ti computed 22.8% of total work
NVIDIA GeForce RTX 4070 computed 29.4% of total work
NVIDIA GeForce RTX 4060 Ti computed 21.6% of total work
---------------
0.0581711 0.0220006 0.347504 = > 0.0141105 0.0670803 0.346016
0.0576147 0.0856164 0.0753108 = > 0.0212964 0.0836086 0.120738
0.699539 0.761743 0.32585 = > 0.329718 0.701058 0.75616
0.644279 0.929641 0.146303 = > 0.13175 0.661939 0.917223
0.43991 0.503481 0.939782 = > 0.436406 0.490325 0.935676
0.855676 0.365408 0.30679 = > 0.305092 0.371161 0.869841
0.928948 0.198322 0.144337 = > 0.135829 0.208741 0.928802
0.295339 0.467584 0.216512 = > 0.216227 0.298812 0.45997
0.988677 0.958554 0.206755 = > 0.184551 0.907675 0.973348
0.956393 0.0890611 0.482762 = > 0.0811025 0.469526 0.953046
0.766326 0.209337 0.368451 = > 0.220405 0.360087 0.774145
0.741995 0.111297 0.623226 = > 0.113182 0.629861 0.727476
0.461 0.722884 0.0677963 = > 0.0647887 0.47715 0.713786
0.773039 0.464691 0.795905 = > 0.453112 0.751761 0.81975
0.441087 0.440932 0.0803012 = > 0.0680988 0.417058 0.475103
0.39298 0.905184 0.489139 = > 0.402733 0.482062 0.916494
0.221201 0.808543 0.310247 = > 0.223572 0.300826 0.815217
0.278789 0.311042 0.451181 = > 0.267732 0.338308 0.42525
0.305794 0.937993 0.220696 = > 0.221226 0.312084 0.94049
0.43699 0.432962 0.0989809 = > 0.0895862 0.408906 0.467968
```
