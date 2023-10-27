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

Following square-root approximation test completes in ~2.5 seconds using two GPUs (~10000 CUDA cores) with 1 : 10 : 20 : 10 : 1 neural network topology

```C++
#include <iostream>
#include"FastSimpleNeuralNetworkTrainer.h"
int main()
{
    constexpr int numParallelSimulations = 1000;
    constexpr int numInputs = 1;
    constexpr int numOutputs = 1;
    const int numThreadsPerSimulation = 256;
    const float parameterScaling = 1.1f;
    GPGPU::FastSimpleNeuralNetworkTrainer<numParallelSimulations, numInputs, 10,20,10, numOutputs> nn(numThreadsPerSimulation,parameterScaling);
    GPGPU::TrainingData<numInputs, numOutputs> td;



    // training data for: y = sqrt(x)
    constexpr int numData = 4000;
    for (int i = 0; i < numData; i++)
    {
        std::vector<float> x(numInputs);
        std::vector<float> y(numOutputs);

        // x
        x[0] = i / (float)numData; 
        // weighting close-to-zero x values more because of floating-point accuracy
        x[0] *= x[0] * x[0] * x[0];
        // y
        y[0] = std::sqrt(x[0]);

        td.AddInputOutputPair(x, y);
    }

    // neural network learns how to compute y = sqrt(x)
    // more data = better fit, too much data = overfit, less data = generalization, too few data = not learning good

    std::vector<float> testInput = { 0.5f };
    float startTemperature = 1.0f;
    float stopTemperature = 0.000001f;
    float coolingRate = 2.0f;
    int numRepeats = 5;
    auto model = nn.Train(td, testInput, [testInput](std::vector<float> testOutput)
        {
            std::cout << "training: now square root of " << testInput[0] << " is " << testOutput[0] << std::endl;
        }, startTemperature, stopTemperature, coolingRate, numRepeats);



    double errPercent = 0.0;
    double errTotal = 0.0;
    int ctr = 0;
    for (double i = 0.00001; i < 1.0; i += 0.00001)
    {

        auto result = model.Run({ (float)i });
        errPercent += 100.0 * std::abs(result[0] - sqrt(i)) / std::abs(sqrt(i));
        errTotal += std::abs(result[0] - sqrt(i));
        ctr++;
    }

    std::cout << ctr << " samples between [0,1] have " << errPercent/ctr << "% average error, "<<errTotal<<" total error." << std::endl;
    return 0;
}
```

output:

```
lower energy found: 404.237
training: now square root of 0.5 is 0.707107
lower energy found: 404.231
training: now square root of 0.5 is 0.707107
lower energy found: 404.229
training: now square root of 0.5 is 0.707107
lower energy found: 404.226
training: now square root of 0.5 is 0.707107
lower energy found: 404.224
training: now square root of 0.5 is 0.707106
lower energy found: 404.223
training: now square root of 0.5 is 0.707106
total computation-time=2.27395 seconds (this includes debugging console-output that is slow)
---------------
OpenCL device info:
NVIDIA GeForce RTX 4070 computed 32.7% of total work
NVIDIA GeForce RTX 4060 Ti computed 25% of total work
---------------
100000 samples between [0,1] have 1.57069% average error, 577.963 total error.
```

---

### Example: Y = X1 or X2 (Topology = 2 : 10 : 20 : 10 : 1)

```C++
#include <iostream>
#include"FastSimpleNeuralNetworkTrainer.h"
int main()
{
    constexpr int numParallelSimulations = 5000;
    constexpr int numInputs = 2;
    constexpr int numOutputs = 1;
    GPGPU::FastSimpleNeuralNetworkTrainer<numParallelSimulations, numInputs, 10, 20, 10, numOutputs> nn;
    GPGPU::TrainingData<numInputs, numOutputs> td;



    // training data for: y = x1 or x2 where 1 means x>=0.5 and 0 means x<0.5
    constexpr int numDataX = 50;
    constexpr int numDataY = 50;
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
    return 0;
}

```

output:

```
lower energy found: 191.707
training: now 1 or 1 is 1
lower energy found: 191.705
training: now 1 or 1 is 1
lower energy found: 191.643
training: now 1 or 1 is 1
reheating. num reheats left=4
reheating. num reheats left=3
reheating. num reheats left=2
reheating. num reheats left=1
total computation-time=7.19534 seconds (this includes debugging console-output that is slow)
---------------
OpenCL device info:
NVIDIA GeForce RTX 4070 computed 27% of total work
NVIDIA GeForce RTX 4060 Ti computed 21.84% of total work
---------------
 0 or 0  = 0
 0 or 1  = 1
 1 or 0  = 1
 1 or 1  = 1
```
---

### Example: 3-Element Array Sorting (topology = 3 : 10 : 10 : 10 : 3)

```C++
#include <iostream>
#include<algorithm>
#include <random>
#include"FastSimpleNeuralNetworkTrainer.h"
int main()
{
    constexpr int numParallelSimulations = 1000;
    constexpr int numInputs = 3;
    constexpr int numOutputs = 3;
    GPGPU::FastSimpleNeuralNetworkTrainer<numParallelSimulations, numInputs, 10, 10, 10, numOutputs> nn;
    GPGPU::TrainingData<numInputs, numOutputs> td;


    std::random_device rd;
    std::mt19937 rng{ rd() };
    std::uniform_real_distribution<float> val(0, 1);


    // training data for: unsorted array ==> sorted array
    constexpr int numData = 50000;

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
        });

    for (int i = 0; i < 20; i++)
    {
        float v1, v2, v3;
        auto result = model.Run({ v1=val(rng), v2=val(rng), v3=val(rng) });
        std::cout<<v1<<" "<<v2<<" "<<v3 << " = > " << result[0] << " " << result[1] << " " << result[2] << std::endl;
    }
    return 0;
}


```
output:
```
lower energy found: 21787.1
training: { 0.75f, 0.45f, 0.9f } ==> {0.468531,0.742753,0.896765}
lower energy found: 21787
training: { 0.75f, 0.45f, 0.9f } ==> {0.468401,0.742808,0.896771}
lower energy found: 21786.8
training: { 0.75f, 0.45f, 0.9f } ==> {0.468678,0.742671,0.89682}
lower energy found: 21786.3
training: { 0.75f, 0.45f, 0.9f } ==> {0.468513,0.742799,0.896646}
lower energy found: 21786.1
training: { 0.75f, 0.45f, 0.9f } ==> {0.468555,0.742762,0.896568}
lower energy found: 21785.9
training: { 0.75f, 0.45f, 0.9f } ==> {0.468648,0.742738,0.896601}
total computation-time=156.615 seconds (this includes debugging console-output that is slow)
---------------
OpenCL device info:
NVIDIA GeForce RTX 4070 computed 25.7% of total work
NVIDIA GeForce RTX 4060 Ti computed 24.8% of total work
---------------
0.199261 0.076507 0.15168 = > 0.0600224 0.138361 0.204892
0.45549 0.188664 0.496166 = > 0.199949 0.416709 0.500581
0.967422 0.000415819 0.49834 = > 0.07027 0.523895 0.857152
0.926756 0.329457 0.2655 = > 0.231931 0.330157 0.876284
0.919132 0.15296 0.89446 = > 0.145453 0.822045 0.909742
0.873497 0.433849 0.196184 = > 0.195906 0.448797 0.842662
0.0115518 0.44483 0.81149 = > 0.0284795 0.500942 0.799638
0.105706 0.918242 0.140533 = > 0.109146 0.120049 0.866248
0.358427 0.555395 0.954306 = > 0.373149 0.560527 0.869426
0.655331 0.401202 0.613121 = > 0.420308 0.585037 0.698692
0.195866 0.00679714 0.737391 = > 0.0139784 0.175823 0.781814
0.880138 0.205239 0.595874 = > 0.195103 0.624835 0.844253
0.919363 0.949601 0.782634 = > 0.734794 0.852594 0.970217
0.0437144 0.399191 0.896938 = > 0.0698933 0.439391 0.830835
0.315253 0.608907 0.340029 = > 0.273511 0.394141 0.598521
0.68282 0.466698 0.875696 = > 0.485868 0.69535 0.86767
0.624292 0.391674 0.0525485 = > 0.0538038 0.410066 0.631914
0.447762 0.373671 0.270433 = > 0.261963 0.350505 0.429356
0.509768 0.480715 0.682073 = > 0.458925 0.553615 0.665736
0.328889 0.115437 0.16312 = > 0.097182 0.175621 0.279622
```
