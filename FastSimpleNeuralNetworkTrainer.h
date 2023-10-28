#pragma once
#include<cmath>
#include<vector>
#include<iostream>
#include<memory>
#include<functional>
#include<exception>
#include"UfSaCL.h"
// todo: simulated annealing <---- momentum search 
// todo2: optional activation functions
// todo3: unroll kernel loop programmatically
namespace GPGPU
{
    namespace Util
    {
        template<int ... NEURAL_NETWORK_ARCHITECTURE>
        constexpr int ComputeNumberOfNeuralNetworkParameters()
        {
            int result = 0;
            constexpr int vec[sizeof...(NEURAL_NETWORK_ARCHITECTURE)] = { NEURAL_NETWORK_ARCHITECTURE... };
            int lastWidth = 1;
            for (int i = 0; i < sizeof...(NEURAL_NETWORK_ARCHITECTURE); i++)
            {
                result += lastWidth * vec[i] + vec[i];
                lastWidth = vec[i];
            }
            return result;
        }

        template<int ... NEURAL_NETWORK_ARCHITECTURE>
        constexpr int ComputeSizeOfFirstLayer()
        {
            constexpr int vec[sizeof...(NEURAL_NETWORK_ARCHITECTURE)] = { NEURAL_NETWORK_ARCHITECTURE... };
            return vec[0];
        }

        template<int ... NEURAL_NETWORK_ARCHITECTURE>
        constexpr int ComputeSizeOfLastLayer()
        {
            constexpr int vec[sizeof...(NEURAL_NETWORK_ARCHITECTURE)] = { NEURAL_NETWORK_ARCHITECTURE... };
            return vec[(sizeof...(NEURAL_NETWORK_ARCHITECTURE)) - 1];
        }

        template<int ... NEURAL_NETWORK_ARCHITECTURE>
        constexpr int ComputeLargestLayerSize()
        {
            int result = 0;
            constexpr int vec[sizeof...(NEURAL_NETWORK_ARCHITECTURE)] = { NEURAL_NETWORK_ARCHITECTURE... };
            
            for (int i = 0; i < sizeof...(NEURAL_NETWORK_ARCHITECTURE); i++)
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

    // custom activation function (hidden layers, input layer) to be given to constructor of FastSimpleNeuralNetworkTrainer
    // by default, it has swish activation function
    // inKernelCode: the code that will be run inside GPU kernel
    // inHostCode: the function that will be called in inference & callback inside simulated annealing
    class ActivationFunction
    {
    private:
        std::string _kernelStringForDevice;
        std::function<float(float)> _hostFunction;
    public:
        ActivationFunction(std::string inKernelCode = "x*exp(x)", std::function<float(float)> inHostCode = [](float x) { return x * std::exp(x); })
        {
            _kernelStringForDevice = inKernelCode;
            _hostFunction = inHostCode;
        }

        float operator ()(float input)
        {
            return _hostFunction(input);
        }

        std::string GetKernelString()
        {
            return _kernelStringForDevice;
        }
    };

    class TrainedModel
    {
    private:
        int _numInputs;
        int _numOutputs;
        std::vector<float> _parameters;
        std::vector<int> _architecture;
        std::function<std::vector<float>(std::vector<float> inputs)> _run;
        std::string _functionCode;
        ActivationFunction _activation;

    public:
        TrainedModel() { _numInputs = 0; _numOutputs = 0; }
        TrainedModel(std::vector<float> parameters, std::vector<int> architecture,
            std::function<std::vector<float>(std::vector<float> inputs)> run,
            std::string functionCode,
            ActivationFunction activationFunction)
        {
            _numInputs = architecture[0];
            _numOutputs = architecture[architecture.size() - 1];
            _parameters = parameters;
            _architecture = architecture;
            _run = run;
            _activation = activationFunction;
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

        // infer on GPU for many different inputs (higher throughput, higher latency)
        // designed for millions of NPCs running their own brains
        std::vector<std::vector<float>> RunMultiple(std::vector<std::vector<float>> inputSets)
        {
            throw std::logic_error("not implemented");
            return std::vector<std::vector<float>>();
        }

        // save parameters to use elsewhere
        std::vector<float> GetParameters()
        {
            return _parameters;
        }

        // to use code directly in GPU kernels with C-style function
        std::string GetFunctionCodeString()
        {
            return _functionCode;
        }
    };

    /*  GPU - based Trainer Tool For Simple Neural Networks
        Every GPU has thousands of pipelines
        Every pipeline runs the same neural architecture but with different data
        Each inpt-output training data pair flows through a different GPU pipeline        
        Possible to take millions of training data pairs
        Parameters are stored in in-chip fast shared-memory(local memory)
    */ 
    /*
        template parameters
        NUM_PARALLEL_SIMULATIONS: number of simulated annealing simulations running in parallel on GPUs
        NEURAL_NETWORK_ARCHITECTURE: number of neurons in layers. nInput,nHidden1,nHidden2,...,nHiddenK,nOutput 

        constructor parameters
        numThreadsPerBlock: number of gpu threads per simulated annealing simulation (256 default)
        parameterScaling: range of parameters of neural network (2.0f default ==> [-2, +2] range for all parameters)
        activation: custom activation function for hidden & input layers. default = swish function
    */
    template<int NUM_PARALLEL_SIMULATIONS,int ... NEURAL_NETWORK_ARCHITECTURE>
    class FastSimpleNeuralNetworkTrainer
    {
    private:
        std::vector<int> _architecture;
        std::string _constantsDefines;
        int _numThreadsPerBlock;
        std::shared_ptr<UFSACL::UltraFastSimulatedAnnealing<Util::ComputeNumberOfNeuralNetworkParameters<NEURAL_NETWORK_ARCHITECTURE...>(), NUM_PARALLEL_SIMULATIONS>> _sim;
        bool _built;
        float _prmScalingMult;
        float _prmScalingAdd;
        ActivationFunction _activation;
    public:
        FastSimpleNeuralNetworkTrainer(
            const int numThreadsPerBlock = 256, const float parameterScaling = 1.1f, ActivationFunction activation=ActivationFunction())
        {
            _numThreadsPerBlock = numThreadsPerBlock;
            _architecture = { NEURAL_NETWORK_ARCHITECTURE... };
            _built = false;
            _prmScalingMult = parameterScaling*2;
            _prmScalingAdd = parameterScaling;
            _activation = activation;
            try
            {
                _constantsDefines = std::string("#define NUM_NETWORK_INPUTS ")+std::to_string(Util::ComputeSizeOfFirstLayer<NEURAL_NETWORK_ARCHITECTURE...>())+std::string(R"(
                )");
                _constantsDefines += std::string("#define NUM_NETWORK_OUTPUTS ") + std::to_string(Util::ComputeSizeOfLastLayer<NEURAL_NETWORK_ARCHITECTURE...>()) + std::string(R"(
                )");
                _constantsDefines += std::string("#define NUM_NETWORK_LARGEST_LAYER ") + std::to_string(Util::ComputeLargestLayerSize<NEURAL_NETWORK_ARCHITECTURE...>()) + std::string(R"(
                )");
                _constantsDefines += std::string("#define NUM_NETWORK_PARAMETERS ") + std::to_string(Util::ComputeNumberOfNeuralNetworkParameters<NEURAL_NETWORK_ARCHITECTURE...>()) + std::string(R"(
                )");
                _constantsDefines += std::string("#define NUM_NETWORK_LAYERS ") + std::to_string(_architecture.size()) + std::string(R"(
                )");
                _constantsDefines += std::string("#define NUM_THREADS_PER_BLOCK ") + std::to_string(numThreadsPerBlock) + std::string(R"(
                )");
                _constantsDefines += std::string("#define NUM_PARALLEL_SIMULATIONS ") + std::to_string(NUM_PARALLEL_SIMULATIONS) + std::string(R"(
                )");

                _constantsDefines += std::string("#define PARAMETER_SCALING_MULT ") + std::to_string(2.0f*parameterScaling) + std::string(R"(f
                )");

                _constantsDefines += std::string("#define PARAMETER_SCALING_ADD ") + std::to_string(parameterScaling) + std::string(R"(f
                )");

                for (int i = 0; i < _architecture.size(); i++)
                {
                    _constantsDefines += std::string("#define NUM_LAYER_")+std::to_string(i) + std::string("_NEURON_NUM ") + std::to_string(_architecture[i]) + std::string(R"(
                    )");
                }

                // generating compile-time loops for full loop unrolling
                std::string loop = "";
                int parameterCtr = 0;
                for (int i = 0; i < _architecture.size(); i++)
                {
                    if (i == 0)
                    {
                        
                        // input layer
                        loop += R"( { 
                        )";
                            loop += std::string("#define LOCAL_LOOP_N ")+std::to_string(_architecture[i])+std::string(R"( 
                            )");
                            for (int j = 0; j < _architecture[i]; j++)
                            {
                                loop += std::string(R"(
                                    // #pragma unroll
                                    // for (int j = 0; j < LOCAL_LOOP_N; j++)
                                    {
                                        const float bias = parameters[)") + std::to_string(parameterCtr + j*2) + std::string(R"(];

                                        // neuron input multiplier
                                        const float mult = parameters[)") + std::to_string(parameterCtr + j * 2 + 1) + std::string(R"(];

                                        // neuron output
                                        const float x = (mult * input[)") + std::to_string(j) + std::string(R"(] + bias);
                                        layerVal[)") + std::to_string(j) + std::string(R"(] = )") + _activation.GetKernelString() + std::string(R"(;
                                    }
                      
                                )");
                            }
                            loop += R"( 
                        #undef LOCAL_LOOP_N
                        #undef LOCAL_LOOP2_N
                        } 
                        )";
                            parameterCtr += _architecture[i] * 2;

                    }
                    else if (i == _architecture.size() - 1)
                    {
                        // output layer
                        loop += R"( { 
                        )";


                        loop += std::string("#define LOCAL_LOOP_N ") + std::to_string(_architecture[i]) + std::string(R"( 
                            )");
                            loop += std::string("#define LOCAL_LOOP2_N ") + std::to_string(_architecture[i-1]) + std::string(R"( 
                                )");
                            for (int j = 0; j < _architecture[i]; j++)
                            {

                                loop += std::string(R"(
                            //for (int j = 0; j < LOCAL_LOOP_N; j++)
                            {
                                const float bias = parameters[)") + std::to_string(parameterCtr + j * (_architecture[i - 1] + 1)) + std::string(R"(];
                                float acc = 0.0f;
                                //#pragma unroll
                                // for (int k = 0; k < LOCAL_LOOP2_N; k++)

                                )");

                                for(int k=0;k< _architecture[i - 1];k++)
                                {

                                    loop += R"( { 
                                    )";

                                    // neuron input multiplier
                                    loop += std::string(R"(const float mult = parameters[)") + std::to_string(parameterCtr + j * (_architecture[i - 1] + 1) + k + 1) + std::string(R"(];
                                    )");

                                    // neuron output
                                    loop += std::string(R"(acc += mult * layerVal[)") + std::to_string(k)+std::string(R"(];
                                    )");


                                    loop += R"( }
                                    )";
 
                                }
  
                                loop += std::string(R"(output[)")+std::to_string(j)+std::string(R"(] = tanh(acc + bias);
                                    }
                                )");
                                
                            

                            }


                        loop += R"( 
                        #undef LOCAL_LOOP_N
                        #undef LOCAL_LOOP2_N
                        } 
                        )";
                        parameterCtr += _architecture[i] * _architecture[i - 1] + _architecture[i];


                    }
                    else
                    {
                        // hidden layer
                        loop += R"( { 
                        )";
                        loop += std::string("#define LOCAL_LOOP_N ") + std::to_string(_architecture[i]) + std::string(R"( 
                            )");
                        loop += std::string("#define LOCAL_LOOP2_N ") + std::to_string(_architecture[i - 1]) + std::string(R"( 
                                )");
                        for (int j = 0; j < _architecture[i]; j++)
                        {
                            loop += std::string(R"(
                            //for (int j = 0; j < LOCAL_LOOP_N; j++)
                            {
                                const float bias = parameters[)") + std::to_string(parameterCtr + j * (_architecture[i - 1] + 1)) + std::string(R"(];
                                
                                float acc = 0.0f;
                                )");

                            //#pragma unroll
                            for (int k = 0; k < _architecture[i - 1]; k++)
                            {
                                loop += std::string(R"({
                                )");
                                // neuron input multiplier
                                loop += std::string(R"(const float mult = parameters[)") + std::to_string(parameterCtr + j * (_architecture[i - 1] + 1) + 1 + k) + std::string(R"(];

                                    // neuron output
                                    acc += mult * layerVal[)") + std::to_string(k) + std::string(R"(];
                                    )");
                                loop += std::string(R"(}
                                )");
                            }

                            loop += std::string(R"(
                                const float x = (acc + bias);
                                layerValTmp[)")+std::to_string(j)+std::string(R"(] = )") + _activation.GetKernelString() + std::string(R"(;
                            })");
                        }
                            loop+=std::string(R"(
                            for (int j = 0; j < LOCAL_LOOP_N; j++)
                                layerVal[j] = layerValTmp[j];
                                )");
                            loop += R"( 
                        #undef LOCAL_LOOP_N
                        #undef LOCAL_LOOP2_N
                        } 
                        )";
                            parameterCtr += _architecture[i] * _architecture[i-1] + _architecture[i];
                    }
                }
                
                // gpu-accelerated simulated-annealing that launches 1 block per simulation
                _sim = std::make_shared<UFSACL::UltraFastSimulatedAnnealing<Util::ComputeNumberOfNeuralNetworkParameters<NEURAL_NETWORK_ARCHITECTURE...>(), NUM_PARALLEL_SIMULATIONS>>(

                    R"(    
                        const int nData = settings[0];
                        const int nLayers = settings[1];
                        float energyLocal = 0.0f;

                        // from SA space to problem space
                        parallelFor(    NUM_NETWORK_PARAMETERS,
                        {
                            parameters[loopId] = parameters[loopId]*PARAMETER_SCALING_MULT - PARAMETER_SCALING_ADD;                          
                        });
                        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);


                        // do same work for each pair of input-output & compute error (as energy for simulated annealing)
                        // this parallel for loop is not per workitem but works on all workitems at once, so any local variables (energyLocal) only visible by themselves
                        parallelFor(nData,
                        {
                                int i=loopId;
                                float trainingDataInputTmp[NUM_NETWORK_INPUTS];
                                float trainingDataOutputTmp[NUM_NETWORK_OUTPUTS];
                                
                                for(int itInp = 0; itInp<NUM_NETWORK_INPUTS; itInp++)
                                    trainingDataInputTmp[itInp] = trainingDataInput[i*NUM_NETWORK_INPUTS + itInp];
                              
                        
                                Compute(
                                    architecture,  
                                    trainingDataInputTmp, 
                                    trainingDataOutputTmp, 
                                    nLayers, 
                                    parameters,
                                    trainingDataOutput+(i*NUM_NETWORK_OUTPUTS)
                                );

                                for(int itOutp = 0; itOutp<NUM_NETWORK_OUTPUTS; itOutp++)
                                {
                                    float diff = (trainingDataOutput[i*NUM_NETWORK_OUTPUTS + itOutp] - trainingDataOutputTmp[itOutp]);
                                    energy += diff * diff; /* pow(fabs(diff),0.5f);*/
                                }
                        
                        });
                            
                        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

                        // from problem space to SA space
                        parallelFor(NUM_NETWORK_PARAMETERS,
                        {
                            parameters[loopId] = (parameters[loopId]+ PARAMETER_SCALING_ADD)/PARAMETER_SCALING_MULT;
                        });

                        energy += energyLocal;                
                )", numThreadsPerBlock);

                _sim->addFunctionDefinition(_constantsDefines+R"(  
                        
                        void Compute(
                            global int * architecture, 
                            float * input, 
                            float * output, 
                            int numLayers, 
                            local float * parameters,
                            global float * target)
                        {
                            int parameterCtr = 0;
                            float layerVal[NUM_NETWORK_LARGEST_LAYER];
                            float layerValTmp[NUM_NETWORK_LARGEST_LAYER];

                        
                        
                )" + loop + std::string(R"(
                        }
                )"));


            }
            catch (std::exception& ex)
            {
                std::cout << ex.what() << std::endl;
            }
        }


        /*
            ---- neural network parameters ----
            trainingData: input-output data pairs for training
            testInput: sample input from user to be used in callback that is called whenever a better energy is found by simulated annealing
            callbackBetterEnergyFound:  this is called whenever solver(simulated annealing) finds a better set of parameters
                                        returns outputs of neural network with input given by testInput
            ---- simulated annealing parameters ----
            startTemperature: usually between 1 and 0.1, decides maximum randomness added to parameters
            stopTemperature: usually close to 0, decides minimum randomness added to parameters
            coolingRate: how fast temperature goes down. generally between 1.00001f and 2.0f.
            numReheating: once cooling is completed, result may not be the global one, so it retries while keeping the best parameter set, for this amount of times
        */
        TrainedModel Train
        (
            TrainingData<Util::ComputeSizeOfFirstLayer<NEURAL_NETWORK_ARCHITECTURE...>(), Util::ComputeSizeOfLastLayer<NEURAL_NETWORK_ARCHITECTURE...>()> trainingData,
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
            // num parallelism (blocks) x num threads per block x network bias+multiplier size
            /*
            std::vector<float> trainingBackpropagationMemory(
                NUM_PARALLEL_SIMULATIONS * 
                _numThreadsPerBlock * 
                Util::ComputeNumberOfNeuralNetworkParameters<NEURAL_NETWORK_ARCHITECTURE...>()
            );
            */

            std::vector<float> trainingDataInput = trainingData.GetInputs();
            std::vector<float> trainingDataOutput = trainingData.GetOutputs();
            std::vector<int> settings = { nTrainingData,(int)_architecture.size() };
           
            _sim->addUserInput("architecture", _architecture);
            _sim->addUserInput("trainingDataInput", trainingDataInput);
            _sim->addUserInput("trainingDataOutput", trainingDataOutput);
            _sim->addUserInput("settings", settings);
            /*
            _sim->addUserInput(
                "gradients", 
                trainingBackpropagationMemory,
                NUM_PARALLEL_SIMULATIONS * 
                Util::ComputeNumberOfNeuralNetworkParameters<NEURAL_NETWORK_ARCHITECTURE...>(),
                true
            );
            */

            if (!_built)
            {
                _built = true;
                _sim->build();
            }
           
            int numInputs = Util::ComputeSizeOfFirstLayer<NEURAL_NETWORK_ARCHITECTURE...>();
            int numOutputs = Util::ComputeSizeOfLastLayer<NEURAL_NETWORK_ARCHITECTURE...>();
            int numLargestLayerSize = Util::ComputeLargestLayerSize<NEURAL_NETWORK_ARCHITECTURE...>();
            std::vector<int> architecture = _architecture;
            float prmScalingAdd = _prmScalingAdd;
            float prmScalingMult = _prmScalingMult;
            ActivationFunction activation = _activation;
            std::vector<float> prm = _sim->run(
                startTemperature, stopTemperature, coolingRate, numReHeating,
                debugPerformance, debugDevice, debugEnergy,
                [numInputs,numOutputs,numLargestLayerSize, 
                callbackBetterEnergyFound, testInput,architecture, prmScalingMult, prmScalingAdd,activation]
                (float* optimizedParameters) mutable
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
                                    const float bias = parameters[parameterCtr++] * prmScalingMult - prmScalingAdd;

                                    // neuron input multiplier
                                    const float mult = parameters[parameterCtr++] * prmScalingMult - prmScalingAdd;

                                    // neuron output

                                    const float x = (mult * testInput[j] + bias);
                                    layerVal[j] = activation(x);
                                }

                            }
                            else if (i == architecture.size() - 1)
                            {
                                // output layer
                                int n = architecture[i];
                                int n0 = architecture[i - 1];
                                for (int j = 0; j < n; j++)
                                {
                                    const float bias = parameters[parameterCtr++] * prmScalingMult - prmScalingAdd;
                                    float acc = 0.0f;
                                    for (int k = 0; k < n0; k++)
                                    {
                                        // neuron input multiplier
                                        const float mult = parameters[parameterCtr++] * prmScalingMult - prmScalingAdd;

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
                                    const float bias = parameters[parameterCtr++] * prmScalingMult - prmScalingAdd;
                                    float acc = 0.0f;
                                    for (int k = 0; k < n0; k++)
                                    {
                                        // neuron input multiplier
                                        const float mult = parameters[parameterCtr++] * prmScalingMult - prmScalingAdd;

                                        // neuron output
                                        acc += mult * layerVal[k];
                                    }


                                    const float x = (acc + bias);
                                    layerValTmp[j] = activation(x);
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
                callbackBetterEnergyFound, architecture, prm,prmScalingMult, prmScalingAdd, activation]
            (std::vector<float> input) mutable
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
                                    const float bias = parameters[parameterCtr++] * prmScalingMult - prmScalingAdd;

                                    // neuron input multiplier
                                    const float mult = parameters[parameterCtr++] * prmScalingMult - prmScalingAdd;

                                    // neuron output

                                    const float x = (mult * input[j] + bias);
                                    layerVal[j] = activation(x);
                                }

                            }
                            else if (i == architecture.size() - 1)
                            {
                                // output layer
                                int n = architecture[i];
                                int n0 = architecture[i - 1];
                                for (int j = 0; j < n; j++)
                                {
                                    const float bias = parameters[parameterCtr++] * prmScalingMult - prmScalingAdd;
                                    float acc = 0.0f;
                                    for (int k = 0; k < n0; k++)
                                    {
                                        // neuron input multiplier
                                        const float mult = parameters[parameterCtr++] * prmScalingMult - prmScalingAdd;

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
                                    const float bias = parameters[parameterCtr++] * prmScalingMult - prmScalingAdd;
                                    float acc = 0.0f;
                                    for (int k = 0; k < n0; k++)
                                    {
                                        // neuron input multiplier
                                        const float mult = parameters[parameterCtr++] * prmScalingMult - prmScalingAdd;

                                        // neuron output
                                        acc += mult * layerVal[k];
                                    }


                                    const float x = (acc + bias);
                                    layerValTmp[j] = activation(x);
                                }

                                for (int j = 0; j < n; j++)
                                    layerVal[j] = layerValTmp[j];
                            }
                        }
                        return output;
                    }
                },
                "",_activation);
         
            

            return model;
           
        }
    };
}