#pragma once

#include<vector>
#include<iostream>
#include<memory>
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


                // gpu-accelerated simulated-annealing that launches 1 block per simulation
                _sim = std::make_shared<UFSACL::UltraFastSimulatedAnnealing<Util::ComputeNumberOfNeuralNetworkParameters<ARGS...>(), 5000>>(
                    constantsDefines + 
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

                _sim->addFunctionDefinition(R"(  
                        void Compute(global int * architecture, float * input, float * output, int numLayers, local float * parameters)
                        {
                            int parameterCtr = 0;
                            float layerVal[256];
                            float layerValTmp[256];
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

        void Train(TrainingData<Util::ComputeSizeOfFirstLayer<ARGS...>(), Util::ComputeSizeOfLastLayer<ARGS...>()> trainingData)
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
            float startTemperature = 1.0f;
            float stopTemperature = 0.0001f;
            float coolingRate = 1.1f;
            bool debugPerformance = false;
            bool debugDevice = false;
            bool debugEnergy = true;
            int numReHeating = 5;
            std::vector<float> prm = _sim->run(
                startTemperature, stopTemperature, coolingRate, numReHeating,
                debugPerformance, debugDevice, debugEnergy,
                [&](float* optimizedParameters) {

                    float input[1] = { 0.5f };
                    float output[1] = { 0.0f };
                    float* parameters = optimizedParameters;
                    {
                        int parameterCtr = 0;
                        float layerVal[256];
                        float layerValTmp[256];
                        for (int i = 0; i < _architecture.size(); i++)
                        {
                            if (i == 0)
                            {
                                // input layer
                                int n = _architecture[i];
                                for (int j = 0; j < n; j++)
                                {
                                    const float bias = parameters[parameterCtr++] * 2.0f - 1.0f;

                                    // neuron input multiplier
                                    const float mult = parameters[parameterCtr++] * 2.0f - 1.0f;

                                    // neuron output
                                    layerVal[j] = tanh(mult * input[j] + bias);
                                }

                            }
                            else if (i == _architecture.size() - 1)
                            {
                                // output layer
                                int n = _architecture[i];
                                int n0 = _architecture[i - 1];
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
                                int n = _architecture[i];
                                int n0 = _architecture[i - 1];
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


                        std::cout << "sqrt (0.5) =" << output[0] << std::endl;

                    }
                    std::cout << "------" << std::endl;
                }
            );

            for (float inp = 0.15; inp < 0.95; inp += 0.1)
            {


                float input[1] = { inp };
                float output[1] = { 0.0f };
                float* parameters = prm.data();
                {
                    int parameterCtr = 0;
                    float layerVal[256];
                    float layerValTmp[256];
                    for (int i = 0; i < _architecture.size(); i++)
                    {
                        if (i == 0)
                        {
                            // input layer
                            int n = _architecture[i];
                            for (int j = 0; j < n; j++)
                            {
                                const float bias = parameters[parameterCtr++] * 2.0f - 1.0f;

                                // neuron input multiplier
                                const float mult = parameters[parameterCtr++] * 2.0f - 1.0f;

                                // neuron output
                                layerVal[j] = tanh(mult * input[j] + bias);
                            }

                        }
                        else if (i == _architecture.size() - 1)
                        {
                            // output layer
                            int n = _architecture[i];
                            int n0 = _architecture[i - 1];
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
                            int n = _architecture[i];
                            int n0 = _architecture[i - 1];
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
                }
                std::cout << "------" << std::endl;
                std::cout << "sqrt(" << input[0] << ")=" << output[0] << "  error = " << (output[0] - std::sqrt(input[0])) / std::sqrt(input[0]) * 100 << "%" << std::endl;
            }
        }
    };
}