#pragma once
#include"libGPGPU/gpgpu.hpp"
#include<memory>
#include<random>


namespace GPGPU
{
	// brute-force gpu-accelerated neural network with simple backpropagation
	template<int NUM_PARALLELISM>
	class NeuralNetwork
	{
	public:
		NeuralNetwork(
			// {2,1,1,1,1,3} means 2 inputs, 4 hidden layers with 1 neuron each, 3 outputs
			std::vector<int> numNeuronsInLayers,

			// {{0,0},{0,1},{1,0},{1,1}} means 4 training inputs with two values for each sample
			std::vector<std::vector<float>> trainingInputs,

			// {{0,0,0},{0,0,0},{0,0,0},{1,1,1}} means 4 training outputs with 3 values for each sample
			std::vector<std::vector<float>> trainingOutputs
		)
		{

			_nLayers = numNeuronsInLayers.size();
			int nInp = 0;
			for (auto& e : trainingInputs)
			{
				nInp += e.size();
			}
			int nOutp = 0;
			for (auto& e : trainingOutputs)
			{
				nOutp += e.size();
			}
			_nOutputs = numNeuronsInLayers[numNeuronsInLayers.size() - 1];
			int nTmpW = numNeuronsInLayers[0];
			int nTmpB = numNeuronsInLayers[0];
			std::vector<int> layerStartWeight;
			std::vector<int> layerStartBias;
			layerStartWeight.push_back(0);
			layerStartBias.push_back(0);
			for (int i = 1; i < numNeuronsInLayers.size(); i++)
			{
				layerStartWeight.push_back(nTmpW);
				layerStartBias.push_back(nTmpB);
				nTmpW += numNeuronsInLayers[i] * numNeuronsInLayers[i - 1];
				nTmpB += numNeuronsInLayers[i];
			}
			_nWeight = nTmpW;
			_nBias = nTmpB;
			// unified compute device (all gpus,cpus combined)
			_computer = std::make_shared<GPGPU::Computer>(GPGPU::Computer::DEVICE_ALL);


			_settingsFloat = std::make_shared<GPGPU::HostParameter>();
			*_settingsFloat = _computer->createArrayInput<float>("settingsFloat", 1000);

			// weights
			_weight = std::make_shared<GPGPU::HostParameter>();
			*_weight = _computer->createArrayInput<float>("weight", _nWeight);
			_weightDelta = std::make_shared<GPGPU::HostParameter>();
			*_weightDelta = _computer->createArrayInput<float>("weightDelta", _nWeight);

			// weight index checkpoints per layer
			_weightLayerStart = std::make_shared<GPGPU::HostParameter>();
			*_weightLayerStart = _computer->createArrayInput<int>("weightLayerStart", layerStartWeight.size());

			// biases
			_bias = std::make_shared<GPGPU::HostParameter>();
			*_bias = _computer->createArrayInput<float>("bias", _nBias);
			_biasDelta = std::make_shared<GPGPU::HostParameter>();
			*_biasDelta = _computer->createArrayInput<float>("biasDelta", _nBias);

			// bias index checkpoints per layer
			_biasLayerStart = std::make_shared<GPGPU::HostParameter>();
			*_biasLayerStart = _computer->createArrayInput<int>("biasLayerStart", layerStartBias.size());

			// training input data
			_trainingInput = std::make_shared<GPGPU::HostParameter>();
			*_trainingInput = _computer->createArrayInput<float>("trainingInput", nInp);

			// training output data
			_trainingOutput = std::make_shared<GPGPU::HostParameter>();
			*_trainingOutput = _computer->createArrayInput<float>("trainingOutput", nOutp);


			std::random_device rd; // random device engine, usually based on /dev/random on UNIX-like systems
			// initialize Mersennes' twister using rd to generate the seed
			std::mt19937 rng{ rd() };

			std::uniform_real_distribution<float> gen(-1.0f, 1.0f);
			for (int i = 0; i < _nWeight; i++)
			{
				_weight->access<float>(i) = gen(rng);
				_weightDelta->access<float>(i) = 0.0f;
			}
			for (int i = 0; i < _nBias; i++)
			{
				_bias->access<float>(i) = gen(rng);
				_biasDelta->access<float>(i) = 0.0f;
			}

			for (int i = 0; i < layerStartWeight.size(); i++)
			{
				_weightLayerStart->access<int>(i) = layerStartWeight[i];
			}

			for (int i = 0; i < layerStartBias.size(); i++)
			{
				_biasLayerStart->access<int>(i) = layerStartBias[i];
			}

			for (int i = 0; i < trainingInputs.size(); i++)
			{
				for (int j = 0; j < trainingInputs[i].size(); j++)
					_trainingInput->access<float>(i * trainingInputs[i].size() + j) = trainingInputs[i][j];
			}

			for (int i = 0; i < trainingOutputs.size(); i++)
			{
				for (int j = 0; j < trainingOutputs[i].size(); j++)
				{
					_trainingOutput->access<float>(i * trainingOutputs[i].size() + j) = trainingOutputs[i][j];

				}
			}

			_topology = std::make_shared<GPGPU::HostParameter>();
			*_topology = _computer->createArrayInput<int>("topology", numNeuronsInLayers.size());
			for (int i = 0; i < numNeuronsInLayers.size(); i++)
				_topology->access<int>(i) = numNeuronsInLayers[i];

			_settings = std::make_shared<GPGPU::HostParameter>();
			*_settings = _computer->createArrayInput<int>("settings", 1000);
			_settings->access<int>(0) = trainingInputs.size();
			_settings->access<int>(1) = numNeuronsInLayers.size();

			_neuronInputValues = std::make_shared<GPGPU::HostParameter>();
			*_neuronInputValues = _computer->createArrayInput<float>("neuronInputValue", _nBias);
			_neuronOutputValues = std::make_shared<GPGPU::HostParameter>();
			*_neuronOutputValues = _computer->createArrayInput<float>("neuronOutputValue", _nBias);
			_neuronErrorValues = std::make_shared<GPGPU::HostParameter>();
			*_neuronErrorValues = _computer->createArrayOutput<char>("neuronErrorValue", 256 * NUM_PARALLELISM);
			_networkErrorValues = std::make_shared<GPGPU::HostParameter>();
			*_networkErrorValues = _computer->createArrayOutput<float>("networkErrorValue", 256 * NUM_PARALLELISM);



			_kernelParameters = std::make_shared<GPGPU::HostParameter>();


			for (int i = 0; i < _nBias; i++)
			{
				_neuronInputValues->access<float>(i) = 0.0f;
				_neuronOutputValues->access<float>(i) = 0.0f;
				_neuronErrorValues->access<char>(i) = 0;
			}


			*_kernelParameters = _trainingInput->next(*_trainingOutput).next(*_weight).
				next(*_bias).next(*_weightLayerStart).next(*_biasLayerStart).
				next(*_weightDelta).next(*_biasDelta).next(*_topology).
				next(*_settings).next(*_neuronInputValues).next(*_neuronOutputValues).
				next(*_neuronErrorValues).next(*_networkErrorValues).next(*_settingsFloat);

			_computer->compile(R"(

				void kernel training(
					global float * trainingInput,
					global float * trainingOutput,
					global float * weight,
					global float * bias,
					global int * weightLayerStart,
					global int * biasLayerStart,
					global float * weightDelta,
					global float * biasDelta,
					global int * topology,
					global int * settings,
					global float * neuronInputValue,
					global float * neuronOutputValue,
					global char * neuronErrorValue,
					global float * networkErrorValue,
					global float * settingsFloat
				)
				{
					const int id = get_global_id(0);
	
					const float trainingSpeed = settingsFloat[0];
					if(id>0) 
						return;
					const int numInputs = topology[0];
					const int numLayers = settings[1];
					const int numOutputs= topology[numLayers-1];
					const int numTrainingData = settings[0];
					const bool weightToUpdate = (settings[2]==0?true:false);
					const int weightIndexToUpdate = settings[3];
					const int biasIndexToUpdate = settings[4];
		
					float errorLeft = 0.0f;
					float errorRight = 0.0f;
					
					float originalParameter = 0.0f;
					if(weightToUpdate)
					{
						originalParameter = weight[weightIndexToUpdate];
					}
					else
					{
						originalParameter = bias[biasIndexToUpdate];
					}

						
					// scan all inputs/outputs
					for(int i=0;i<numTrainingData;i++)
					{
						for(int dir = -1; dir<=1;dir+=2)
						{
							float dirVal = dir * trainingSpeed;

					
							if(weightToUpdate)
							{
								weight[weightIndexToUpdate] = originalParameter+dirVal;
							}
							else
							{
								bias[biasIndexToUpdate] = originalParameter+dirVal;
							}

							// FEED FORWARD
							// put inputs into first layer input & calculate output
							for(int neu=0;neu<numInputs;neu++)
							{
								int biasIndex = neu;
								int weightIndex = neu;
								neuronInputValue[biasIndex]=trainingInput[i*numInputs + neu];
								neuronOutputValue[biasIndex]=tanh(
									weight[weightIndex]*neuronInputValue[biasIndex]+bias[biasIndex]
								);
							}

			

							// input --> weighted sum --> activation --> output
							for(int layer = 1;layer<numLayers;layer++)
							{
								int numPreviousNeurons = topology[layer-1];
								int numCurrentNeurons = topology[layer];
								for(int neu=0;neu<numCurrentNeurons;neu++)
								{
									int biasIndex = biasLayerStart[layer] + neu;
									
									
									float accumulator = 0.0f;
									for(int pNeu=0;pNeu<numPreviousNeurons;pNeu++)
									{
										int weightIndex = weightLayerStart[layer]+neu*numPreviousNeurons+pNeu;
										int biasIndexPrevious = biasLayerStart[layer-1] + pNeu;
										accumulator += neuronOutputValue[biasIndexPrevious]*weight[weightIndex];
									}
									neuronInputValue[biasIndex]=accumulator;
									neuronOutputValue[biasIndex]=tanh(
										neuronInputValue[biasIndex]+bias[biasIndex]
									);
								}
							}
	
							float error = 0.0f;
							for(int out=0;out<numOutputs;out++)
							{
								int biasIndex = biasLayerStart[numLayers-1] + out;
								float val = neuronOutputValue[biasIndex]; 
								float output = trainingOutput[i*numOutputs + out];
								float err = (val-output);
								err *= err;
								error += err;
							}

							if(dir == -1)
								errorLeft += error;
							else
								errorRight += error;
						}

					}
									
					const int groupId = id/256;				
					const int localId = id % 256;
					if(localId == 0)
					{
						neuronErrorValue[groupId*256]=(errorLeft<errorRight?-1:1);
						networkErrorValue[groupId*256]=(errorLeft<errorRight?errorLeft:errorRight);
					}
					
				}
			)", "training");
		}

		// dampening: less than 1 greater than 0
		void Train(float trainingSpeed, int epochs, bool dampenTrainingSpeed, float dampening = 0.001f)
		{
			int weightCtr = 0;
			int biasCtr = 0;
			for (int i = 0; i < epochs; i++)
			{
				if (dampenTrainingSpeed)
					trainingSpeed *= (1.0f - dampening);
				_settingsFloat->access<float>(0) = trainingSpeed;
				const bool weightToUpdate = (i % 2 == 0);
				if (weightToUpdate)
				{
					_settings->access<int>(2) = 0;
					_settings->access<int>(3) = weightCtr++ % _nWeight;
				}
				else
				{
					_settings->access<int>(2) = 1;
					_settings->access<int>(4) = biasCtr++ % _nBias;
				}


				_computer->compute(*_kernelParameters, "training", 0, 256 * NUM_PARALLELISM, 256);

				if (i % 1000 == 0)
				{
					std::cout <<
						(int)_neuronErrorValues->access<char>(0) << ": " <<
						_networkErrorValues->access<float>(0) << "  " <<
						trainingSpeed << std::endl;

					/*
					for (int i = 0; i < _nWeight; i++)
						if (_weight->access<float>(i) > 1.0 || _weight->access<float>(i) < -1.0)
							std::cout << _weight->access<float>(i) << "!!!" << std::endl;

					for (int i = 0; i < _nBias; i++)
						if (_bias->access<float>(i) > 1.0 || _bias->access<float>(i) < -1.0)
							std::cout << _bias->access<float>(i) << "!!!" << std::endl;
							*/
				}
				if (_neuronErrorValues->access<char>(0) == -1)
				{
					if (weightToUpdate)
					{
						_weight->access<float>((weightCtr - 1) % _nWeight) -= trainingSpeed;
					}
					else
					{
						_bias->access<float>((biasCtr - 1) % _nBias) -= trainingSpeed;
					}
				}
				else
				{
					if (weightToUpdate)
					{
						_weight->access<float>((weightCtr - 1) % _nWeight) += trainingSpeed;
					}
					else
					{
						_bias->access<float>((biasCtr - 1) % _nBias) += trainingSpeed;
					}
				}

			}
		}

		std::vector<float> Infer(std::vector<float> inputs)
		{
			std::vector<float> outputs(_nOutputs);
			int numInputs = inputs.size();
			if (numInputs != _topology->access<int>(0))
			{
				std::cout << "Error: wrong input size" << std::endl;
				return std::vector<float>();
			}
			// FEED FORWARD
			// put inputs into first layer input & calculate output
			for (int neu = 0; neu < numInputs; neu++)
			{
				int biasIndex = neu;
				int weightIndex = neu;
				_neuronInputValues->access<float>(biasIndex) = inputs[neu];
				_neuronOutputValues->access<float>(biasIndex) = std::tanh(
					_weight->access<float>(weightIndex) * _neuronInputValues->access<float>(biasIndex) + _bias->access<float>(biasIndex)
				);
			}



			// input --> weighted sum --> activation --> output
			for (int layer = 1; layer < _nLayers; layer++)
			{
				int numPreviousNeurons = _topology->access<int>(layer - 1);
				int numCurrentNeurons = _topology->access<int>(layer);
				for (int neu = 0; neu < numCurrentNeurons; neu++)
				{
					int biasIndex = _biasLayerStart->access<int>(layer) + neu;


					float accumulator = 0.0f;
					for (int pNeu = 0; pNeu < numPreviousNeurons; pNeu++)
					{
						int weightIndex = _weightLayerStart->access<int>(layer) + neu * numPreviousNeurons + pNeu;
						int biasIndexPrevious = _biasLayerStart->access<int>(layer - 1) + pNeu;
						accumulator += _neuronOutputValues->access<float>(biasIndexPrevious) * _weight->access<float>(weightIndex);
					}
					_neuronInputValues->access<float>(biasIndex) = accumulator;
					_neuronOutputValues->access<float>(biasIndex) = std::tanh(
						_neuronInputValues->access<float>(biasIndex) + _bias->access<float>(biasIndex)
					);
					if (layer == _nLayers - 1)
					{
						outputs[neu] = _neuronOutputValues->access<float>(biasIndex);
					}
				}
			}

			return outputs;
		}
	private:
		int _nWeight;
		int _nBias;
		int _nOutputs;
		int _nLayers;
		std::shared_ptr<GPGPU::Computer> _computer;
		std::shared_ptr<GPGPU::HostParameter> _kernelParameters;

		std::shared_ptr<GPGPU::HostParameter> _settingsFloat;
		std::shared_ptr<GPGPU::HostParameter> _settings;
		std::shared_ptr<GPGPU::HostParameter> _topology;
		std::shared_ptr<GPGPU::HostParameter> _trainingInput;
		std::shared_ptr<GPGPU::HostParameter> _trainingOutput;
		// memory layout for weights is packed without any padding
		std::shared_ptr<GPGPU::HostParameter> _weight;
		std::shared_ptr<GPGPU::HostParameter> _weightDelta;
		std::shared_ptr<GPGPU::HostParameter> _weightLayerStart;
		// similar to weight except only 1 bias per neuron		
		std::shared_ptr<GPGPU::HostParameter> _bias;
		std::shared_ptr<GPGPU::HostParameter> _biasDelta;
		std::shared_ptr<GPGPU::HostParameter> _biasLayerStart;

		// input/outputs of each neuron
		std::shared_ptr<GPGPU::HostParameter> _neuronInputValues;
		std::shared_ptr<GPGPU::HostParameter> _neuronOutputValues;

		// error of neuron
		std::shared_ptr<GPGPU::HostParameter> _neuronErrorValues;
		std::shared_ptr<GPGPU::HostParameter> _networkErrorValues;
	};

}