#pragma once

#include<vector>
#include<iostream>
#include<memory>
#include<functional>
#include"UfSaCL.h"
namespace GPGPU
{
    namespace Util
    {
        template<int ... ARGS>
        constexpr int ComputeNumberOfNeuralNetworkParameters()
        {
            int result = 0;
            constexpr int vec[sizeof...(ARGS)] = { ARGS... };
            int lastWidth = 1;
            for (int i = 0; i < sizeof...(ARGS); i++)
            {
                result += lastWidth * vec[i] + vec[i];
                lastWidth = vec[i];
            }
            return result;
        }

        template<int ... ARGS>
        constexpr int ComputeSizeOfFirstLayer()
        {
            constexpr int vec[sizeof...(ARGS)] = { ARGS... };
            return vec[0];
        }

        template<int ... ARGS>
        constexpr int ComputeSizeOfLastLayer()
        {
            constexpr int vec[sizeof...(ARGS)] = { ARGS... };
            return vec[(sizeof...(ARGS)) - 1];
        }

        template<int ... ARGS>
        constexpr int ComputeLargestLayerSize()
        {
            int result = 0;
            constexpr int vec[sizeof...(ARGS)] = { ARGS... };
            
            for (int i = 0; i < sizeof...(ARGS); i++)
            {
                if (result < vec[i])
                    result = vec[i];
            }
            return result;
        }
    }

    template<int INPUT_SIZE, int OUTPUT_SIZE>
    class TrainingData
    {
    public:
        TrainingData():_sz(0)
        {
            
        }

        void AddInputOutputPair(std::vector<float> inputElements, std::vector<float> outputElements)
        {
            _sz++;
            for (int i = 0; i < INPUT_SIZE; i++)
            {
                _inputs.push_back(inputElements[i]);                
            }

            for (int i = 0; i < OUTPUT_SIZE; i++)
            {                
                _outputs.push_back(outputElements[i]);
            }
        }

        int Size()
        {
            return _sz;
        }

        std::vector<float> GetInputs()
        {
            return _inputs;
        }

        std::vector<float> GetOutputs()
        {
            return _outputs;
        }
    private:
        int _sz;
        std::vector<float> _inputs;
        std::vector<float> _outputs;
    };


    class TrainedModel
    {
    private:
        int _numInputs;
        int _numOutputs;
        std::vector<float> _parameters;
        std::vector<int> _architecture;
        std::function<std::vector<float>(std::vector<float> inputs)> _run;

    public:
        TrainedModel() { _numInputs = 0; _numOutputs = 0; }
        TrainedModel(std::vector<float> parameters, std::vector<int> architecture,
            std::function<std::vector<float>(std::vector<float> inputs)> run)
        {
            _numInputs = architecture[0];
            _numOutputs = architecture[architecture.size() - 1];
            _parameters = parameters;
            _architecture = architecture;
            _run = run;
        }

        // inference
        std::vector<float> Run(std::vector<float> inputs)
        {
            if (inputs.size() != _numInputs)
            {
                std::cout << "Error: input size is not same as neural architecture" << std::endl;
                exit(1);
            }
            return _run(inputs);
        }

        // save parameters to use elsewhere
        std::vector<float> GetParameters()
        {
            return _parameters;
        }

        // to use code directly in GPU kernels with C-style function
        std::string GetFunctionCodeString()
        {
            return "";
        }
    };

    /*  GPU - based Trainer Tool For Simple Neural Networks
        Every GPU has thousands of pipelines
        Every pipeline runs the same neural architecture but with different data
        Each inpt-output training data pair flows through a different GPU pipeline        
        Possible to take millions of training data pairs
        Parameters are stored in in-chip fast shared-memory(local memory)
    */ 
    template<int ... ARGS>
    class FastSimpleNeuralNetworkTrainer
    {
    private:
        std::vector<int> _architecture;
        std::shared_ptr<UFSACL::UltraFastSimulatedAnnealing<Util::ComputeNumberOfNeuralNetworkParameters<ARGS...>(), 5000>> _sim;
        bool _built;
    public:
        FastSimpleNeuralNetworkTrainer()
        {
            _architecture = { ARGS... };
            _built = false;
            try
            {
                std::string constantsDefines = std::string("#define NUM_NETWORK_INPUTS ")+std::to_string(Util::ComputeSizeOfFirstLayer<ARGS...>())+std::string(R"(
                )");
                constantsDefines += std::string("#define NUM_NETWORK_OUTPUTS ") + std::to_string(Util::ComputeSizeOfLastLayer<ARGS...>()) + std::string(R"(
                )");
                constantsDefines += std::string("#define NUM_NETWORK_LARGEST_LAYER ") + std::to_string(Util::ComputeLargestLayerSize<ARGS...>()) + std::string(R"(
                )");


                // gpu-accelerated simulated-annealing that launches 1 block per simulation
                _sim = std::make_shared<UFSACL::UltraFastSimulatedAnnealing<Util::ComputeNumberOfNeuralNetworkParameters<ARGS...>(), 5000>>(

                    R"(    
                        const int nData = settings[0];
                        const int nLayers = settings[1];
                        float energyLocal = 0.0f;

                        // do same work for each pair of input-output & compute error (as energy for simulated annealing)
                        // this parallel for loop is not per workitem but works on all workitems at once, so any local variables (energyLocal) only visible by themselves
                        parallelFor(nData,
                        {
                                int i=loopId;
                                float trainingDataInputTmp[NUM_NETWORK_INPUTS];
                                float trainingDataOutputTmp[NUM_NETWORK_OUTPUTS];

                                for(int itInp = 0; itInp<NUM_NETWORK_INPUTS; itInp++)
                                    trainingDataInputTmp[itInp] = trainingDataInput[i*NUM_NETWORK_INPUTS + itInp];
                                trainingDataOutputTmp[0] = 0.0f;
                        
                                Compute(architecture, trainingDataInputTmp, trainingDataOutputTmp, nLayers, parameters);

                                for(int itOutp = 0; itOutp<NUM_NETWORK_OUTPUTS; itOutp++)
                                {
                                    float diff = (trainingDataOutput[i*NUM_NETWORK_OUTPUTS + itOutp] - trainingDataOutputTmp[itOutp]);
                                    energy += pow(fabs(diff),0.5f);
                                }
                        
                        });
                        energy += energyLocal;                
                )");

                _sim->addFunctionDefinition(constantsDefines+R"(  
                        void Compute(global int * architecture, float * input, float * output, int numLayers, local float * parameters)
                        {
                            int parameterCtr = 0;
                            float layerVal[NUM_NETWORK_LARGEST_LAYER];
                            float layerValTmp[NUM_NETWORK_LARGEST_LAYER];
                            for(int i=0;i<numLayers;i++)
                            {
                                if(i==0)
                                {
                                    // input layer
                                    int n = architecture[i];
                                    for(int j=0;j<n;j++)
                                    {
                                        const float bias = parameters[parameterCtr++]*2.0f - 1.0f;

                                        // neuron input multiplier
                                        const float mult = parameters[parameterCtr++]*2.0f - 1.0f;

                                        // neuron output
                                        layerVal[j] = tanh(mult * input[j] + bias);  
                                    }                       

                                }
                                else if(i==numLayers-1)
                                {
                                    // output layer
                                    int n = architecture[i];
                                    int n0 = architecture[i-1];
                                    for(int j=0;j<n;j++)
                                    {
                                        const float bias = parameters[parameterCtr++]*2.0f - 1.0f;
                                        float acc = 0.0f;
                                        for(int k=0;k<n0;k++)
                                        {
                                            // neuron input multiplier
                                            const float mult = parameters[parameterCtr++]*2.0f - 1.0f;

                                            // neuron output
                                            acc += mult * layerVal[k];  
                                        }

                                        output[j] = tanh(acc + bias);
                                    }  
                                }
                                else
                                {
                                    // hidden layer
                                    int n = architecture[i];
                                    int n0 = architecture[i-1];
                                    for(int j=0;j<n;j++)
                                    {
                                        const float bias = parameters[parameterCtr++]*2.0f - 1.0f;
                                        float acc = 0.0f;
                                        for(int k=0;k<n0;k++)
                                        {
                                            // neuron input multiplier
                                            const float mult = parameters[parameterCtr++]*2.0f - 1.0f;

                                            // neuron output
                                            acc += mult * layerVal[k];  
                                        }

                                        layerValTmp[j] = tanh(acc + bias);
                                    }        

                                    for(int j=0;j<n;j++)               
                                        layerVal[j]=layerValTmp[j];

                                }
                            }
    
                        }
                )");


            }
            catch (std::exception& ex)
            {
                std::cout << ex.what() << std::endl;
            }
        }

        /*
            trainingData: input-output data pairs for training
            testInput: sample input from user to be used in callback that is called whenever a better energy is found by simulated annealing
            callbackBetterEnergyFound:  this is called whenever solver(simulated annealing) finds a better set of parameters
                                        returns outputs of neural network with input given by testInput
        */
        TrainedModel Train
        (
            TrainingData<Util::ComputeSizeOfFirstLayer<ARGS...>(), Util::ComputeSizeOfLastLayer<ARGS...>()> trainingData,
            std::vector<float> testInput,
            std::function<void(std::vector<float>)> callbackBetterEnergyFound,
            float startTemperature = 1.0f,
            float stopTemperature = 0.0001f,
            float coolingRate = 1.1f,
            int numReHeating = 5,
            bool debugPerformance = false,
            bool debugDevice = false,
            bool debugEnergy = true
        )
        {
            const int nTrainingData = trainingData.Size();
            std::vector<float> trainingDataInput = trainingData.GetInputs();
            std::vector<float> trainingDataOutput = trainingData.GetOutputs();
            std::vector<int> settings = { nTrainingData,(int)_architecture.size() };
           
            _sim->addUserInput("architecture", _architecture);
            _sim->addUserInput("trainingDataInput", trainingDataInput);
            _sim->addUserInput("trainingDataOutput", trainingDataOutput);
            _sim->addUserInput("settings", settings);
            if (!_built)
            {
                _built = true;
                _sim->build();
            }
           
            int numInputs = Util::ComputeSizeOfFirstLayer<ARGS...>();
            int numOutputs = Util::ComputeSizeOfLastLayer<ARGS...>();
            int numLargestLayerSize = Util::ComputeLargestLayerSize<ARGS...>();
            std::vector<int> architecture = _architecture;
            std::vector<float> prm = _sim->run(
                startTemperature, stopTemperature, coolingRate, numReHeating,
                debugPerformance, debugDevice, debugEnergy,
                [numInputs,numOutputs,numLargestLayerSize, 
                callbackBetterEnergyFound, testInput,architecture]
                (float* optimizedParameters) 
                {
                    std::vector<float> output(numOutputs,0.0f);
                    
                    float* parameters = optimizedParameters;
                    {
                        int parameterCtr = 0;
                        std::vector<float> layerVal(numLargestLayerSize);
                        std::vector<float> layerValTmp(numLargestLayerSize);
                        for (int i = 0; i < architecture.size(); i++)
                        {
                            if (i == 0)
                            {
                                // input layer
                                int n = architecture[i];
                                for (int j = 0; j < n; j++)
                                {
                                    const float bias = parameters[parameterCtr++] * 2.0f - 1.0f;

                                    // neuron input multiplier
                                    const float mult = parameters[parameterCtr++] * 2.0f - 1.0f;

                                    // neuron output
                                    layerVal[j] = tanh(mult * testInput[j] + bias);
                                }

                            }
                            else if (i == architecture.size() - 1)
                            {
                                // output layer
                                int n = architecture[i];
                                int n0 = architecture[i - 1];
                                for (int j = 0; j < n; j++)
                                {
                                    const float bias = parameters[parameterCtr++] * 2.0f - 1.0f;
                                    float acc = 0.0f;
                                    for (int k = 0; k < n0; k++)
                                    {
                                        // neuron input multiplier
                                        const float mult = parameters[parameterCtr++] * 2.0f - 1.0f;

                                        // neuron output
                                        acc += mult * layerVal[k];
                                    }

                                    output[j] = tanh(acc + bias);
                                }
                            }
                            else
                            {
                                // hidden layer
                                int n = architecture[i];
                                int n0 = architecture[i - 1];
                                for (int j = 0; j < n; j++)
                                {
                                    const float bias = parameters[parameterCtr++] * 2.0f - 1.0f;
                                    float acc = 0.0f;
                                    for (int k = 0; k < n0; k++)
                                    {
                                        // neuron input multiplier
                                        const float mult = parameters[parameterCtr++] * 2.0f - 1.0f;

                                        // neuron output
                                        acc += mult * layerVal[k];
                                    }

                                    layerValTmp[j] = tanh(acc + bias);
                                }

                                for (int j = 0; j < n; j++)
                                    layerVal[j] = layerValTmp[j];
                            }
                        }
                        callbackBetterEnergyFound(output);
                    }                    
                }
            );


            TrainedModel model(
                prm, 
                architecture, 
                [numInputs, numOutputs, numLargestLayerSize,
                callbackBetterEnergyFound, architecture, prm]
                (std::vector<float> input)
                {
                    std::vector<float> output(numOutputs, 0.0f);

                    const float* parameters = prm.data();
                    {
                        int parameterCtr = 0;
                        std::vector<float> layerVal(numLargestLayerSize);
                        std::vector<float> layerValTmp(numLargestLayerSize);
                        for (int i = 0; i < architecture.size(); i++)
                        {
                            if (i == 0)
                            {
                                // input layer
                                int n = architecture[i];
                                for (int j = 0; j < n; j++)
                                {
                                    const float bias = parameters[parameterCtr++] * 2.0f - 1.0f;

                                    // neuron input multiplier
                                    const float mult = parameters[parameterCtr++] * 2.0f - 1.0f;

                                    // neuron output
                                    layerVal[j] = tanh(mult * input[j] + bias);
                                }

                            }
                            else if (i == architecture.size() - 1)
                            {
                                // output layer
                                int n = architecture[i];
                                int n0 = architecture[i - 1];
                                for (int j = 0; j < n; j++)
                                {
                                    const float bias = parameters[parameterCtr++] * 2.0f - 1.0f;
                                    float acc = 0.0f;
                                    for (int k = 0; k < n0; k++)
                                    {
                                        // neuron input multiplier
                                        const float mult = parameters[parameterCtr++] * 2.0f - 1.0f;

                                        // neuron output
                                        acc += mult * layerVal[k];
                                    }

                                    output[j] = tanh(acc + bias);
                                }
                            }
                            else
                            {
                                // hidden layer
                                int n = architecture[i];
                                int n0 = architecture[i - 1];
                                for (int j = 0; j < n; j++)
                                {
                                    const float bias = parameters[parameterCtr++] * 2.0f - 1.0f;
                                    float acc = 0.0f;
                                    for (int k = 0; k < n0; k++)
                                    {
                                        // neuron input multiplier
                                        const float mult = parameters[parameterCtr++] * 2.0f - 1.0f;

                                        // neuron output
                                        acc += mult * layerVal[k];
                                    }

                                    layerValTmp[j] = tanh(acc + bias);
                                }

                                for (int j = 0; j < n; j++)
                                    layerVal[j] = layerValTmp[j];
                            }
                        }
                        return output;
                    }
                }
            );
         
            return model;
           
        }
    };
}