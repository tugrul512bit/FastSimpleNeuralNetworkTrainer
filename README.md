# FastSimpleNeuralNetworkTrainer

This GPGPU library uses multiple GPUs to train small-scale neural-networks, fast and in a simple way.

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
    nn.Train(td);

    return 0;
}
```

output:

```
reheating. num reheats left=2
lower energy found: 104.891
sqrt (0.5) =0.707107
------
lower energy found: 104.855
sqrt (0.5) =0.707107
------
reheating. num reheats left=1
total computation-time=8.28626 seconds (this includes debugging console-output that is slow)
---------------
OpenCL device info:
NVIDIA GeForce RTX 4070 computed 27.34% of total work
NVIDIA GeForce RTX 4060 Ti computed 20.56% of total work
---------------
------
sqrt(0.15)=0.387022  error = -0.0714089%
------
sqrt(0.25)=0.499811  error = -0.0377417%
------
sqrt(0.35)=0.591629  error = 0.00362701%
------
sqrt(0.45)=0.67082  error = -1.77707e-05%
------
sqrt(0.55)=0.74162  error = 1.60742e-05%
------
sqrt(0.65)=0.806252  error = 0.00320858%
------
sqrt(0.75)=0.865667  error = -0.0413366%
------
sqrt(0.85)=0.917586  error = -0.473796%

```
