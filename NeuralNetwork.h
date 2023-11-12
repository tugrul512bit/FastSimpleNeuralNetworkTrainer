#pragma once

#include"libGPGPU/gpgpu.hpp"
#include<memory>
#include<random>


namespace GPGPU
{
	/*
		brute-force gpu-accelerated neural network
		NUM_PARALLELISM: number of parameters tuned at once as a global step
	*/
	class NeuralNetwork
	{
	private:
		int _paddedNumElementsPerGroupForWeight;
		int _paddedNumElementsPerGroupForBias;
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
			_numParallelism = _nWeight + _nBias;
			int multipleW = 1 + _nWeight / 256;
			int multipleB = 1 + _nBias / 256;
			_paddedNumElementsPerGroupForWeight = multipleW * 256;
			_paddedNumElementsPerGroupForBias = multipleB * 256;
			// unified compute device (all gpus,cpus combined)
			_computer = std::make_shared<GPGPU::Computer>(GPGPU::Computer::DEVICE_ALL);


			_settingsFloat = std::make_shared<GPGPU::HostParameter>();
			*_settingsFloat = _computer->createArrayInput<float>("settingsFloat", 1000);

			// weights
			_weightParallel = std::make_shared<GPGPU::HostParameter>();
			*_weightParallel = _computer->createArrayInput<float>("weight", _paddedNumElementsPerGroupForWeight * _numParallelism);
			_weightDeltaParallel = std::make_shared<GPGPU::HostParameter>();
			*_weightDeltaParallel = _computer->createArrayInput<float>("weightDelta", _paddedNumElementsPerGroupForWeight * _numParallelism);

			// weight index checkpoints per layer
			_weightLayerStart = std::make_shared<GPGPU::HostParameter>();
			*_weightLayerStart = _computer->createArrayInput<int>("weightLayerStart", layerStartWeight.size());

			// biases
			_biasParallel = std::make_shared<GPGPU::HostParameter>();
			*_biasParallel = _computer->createArrayInput<float>("bias", _paddedNumElementsPerGroupForBias * _numParallelism);
			_biasDeltaParallel = std::make_shared<GPGPU::HostParameter>();
			*_biasDeltaParallel = _computer->createArrayInput<float>("biasDelta", _paddedNumElementsPerGroupForBias * _numParallelism);

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
			for (int i = 0; i < _paddedNumElementsPerGroupForWeight * _numParallelism; i++)
			{
				if (i < _paddedNumElementsPerGroupForWeight)
				{
					_weightParallel->access<float>(i) = gen(rng);
				}
				else
				{
					_weightParallel->access<float>(i) = _weightParallel->access<float>(i % _paddedNumElementsPerGroupForWeight);
				}
				_weightDeltaParallel->access<float>(i) = 0.0f;
			}


			for (int i = 0; i < _paddedNumElementsPerGroupForBias * _numParallelism; i++)
			{
				if (i < _paddedNumElementsPerGroupForBias)
				{
					_biasParallel->access<float>(i) = gen(rng);
				}
				else
				{
					_biasParallel->access<float>(i) = _biasParallel->access<float>(i % _paddedNumElementsPerGroupForBias);
				}
				_biasDeltaParallel->access<float>(i) = 0.0f;
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

			_neuronInputValuesParallel = std::make_shared<GPGPU::HostParameter>();
			*_neuronInputValuesParallel = _computer->createArrayInput<float>("neuronInputValue", _paddedNumElementsPerGroupForBias * _numParallelism);
			_neuronOutputValuesParallel = std::make_shared<GPGPU::HostParameter>();
			*_neuronOutputValuesParallel = _computer->createArrayInput<float>("neuronOutputValue", _paddedNumElementsPerGroupForBias * _numParallelism);
			_neuronErrorValues = std::make_shared<GPGPU::HostParameter>();
			*_neuronErrorValues = _computer->createArrayOutput<char>("neuronErrorValue", 256 * _numParallelism);
			_networkErrorValues = std::make_shared<GPGPU::HostParameter>();
			*_networkErrorValues = _computer->createArrayOutput<float>("networkErrorValue", 256 * _numParallelism);



			_kernelParameters = std::make_shared<GPGPU::HostParameter>();


			for (int i = 0; i < _paddedNumElementsPerGroupForBias * _numParallelism; i++)
			{
				_neuronInputValuesParallel->access<float>(i) = 0.0f;
				_neuronOutputValuesParallel->access<float>(i) = 0.0f;
			}
			for (int i = 0; i < 256 * _numParallelism; i++)
				_neuronErrorValues->access<char>(i) = 0;

			_settings->access<int>(5) = _nWeight + _nBias;
			_settings->access<int>(6) = _paddedNumElementsPerGroupForWeight;
			_settings->access<int>(7) = _paddedNumElementsPerGroupForBias;
			*_kernelParameters = _trainingInput->next(*_trainingOutput).next(*_weightParallel).
				next(*_biasParallel).next(*_weightLayerStart).next(*_biasLayerStart).
				next(*_weightDeltaParallel).next(*_biasDeltaParallel).next(*_topology).
				next(*_settings).next(*_neuronInputValuesParallel).next(*_neuronOutputValuesParallel).
				next(*_neuronErrorValues).next(*_networkErrorValues).next(*_settingsFloat);

			std::string wgt = std::string("#define workGroupThreads ") + std::to_string(256);
			wgt = std::string(R"(
			)") + wgt + std::string(R"(
			)");
			std::string wgt2 = std::string("#define numWeight ") + std::to_string(_nWeight);
			wgt2 = std::string(R"(
			)") + wgt2 + std::string(R"(
			)");
			std::string wgt3 = std::string("#define numBias ") + std::to_string(_nBias);
			wgt3 = std::string(R"(
			)") + wgt3 + std::string(R"(
			)");
			_computer->compile(wgt + wgt2 + wgt3 + std::string(R"(

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
					local float errors[workGroupThreads];
					const int id = get_global_id(0);
					const int groupId = id/workGroupThreads;				
					const int localId = id % workGroupThreads;
			

					const float trainingSpeed = settingsFloat[0];
					const int numInputs = topology[0];
					const int numLayers = settings[1];
					const int numOutputs= topology[numLayers-1];
					const int numTrainingData = settings[0];

					/*
					const bool weightToUpdate = (settings[2]==0?true:false);
					const int weightIndexToUpdate = settings[3];
					const int biasIndexToUpdate = settings[4];
					*/
					const bool weightToUpdate = (groupId<numWeight?true:false);
					const int weightIndexToUpdate = groupId;
					const int biasIndexToUpdate = groupId - numWeight;


					const int numParameters = settings[5];
					const int numParametersPaddedWeight = settings[6];
					const int numParametersPaddedBias = settings[7];
					const int offsetGroupWeight = numParametersPaddedWeight * groupId;
					const int offsetGroupBias = numParametersPaddedBias * groupId;
					float errorLeft = 0.0f;
					float errorRight = 0.0f;
					
					float originalParameter = 0.0f;
					if(weightToUpdate)
					{
						originalParameter = weight[offsetGroupWeight+weightIndexToUpdate];
					}
					else
					{
						originalParameter = bias[offsetGroupBias+biasIndexToUpdate];
					}

						
					// scan all inputs/outputs
					for(int i=0;i<numTrainingData;i++)
					{
						for(int dir = -1; dir<=1;dir+=2)
						{
							float dirVal = dir * trainingSpeed;

							if(localId == 0)
							{
								if(weightToUpdate)
								{								
									weight[offsetGroupWeight+weightIndexToUpdate] = originalParameter+dirVal;
								}
								else
								{
									bias[offsetGroupBias+biasIndexToUpdate] = originalParameter+dirVal;
								}
							}
							barrier(CLK_GLOBAL_MEM_FENCE);


							// FEED FORWARD
							// put inputs into first layer input & calculate output

							{
								const int numLoopIter = (numInputs / workGroupThreads) + 1;     
								for(int iGPGPU=0;iGPGPU<numLoopIter;iGPGPU++)                          
								{                                                       
									const int loopId = localId + workGroupThreads * iGPGPU; 
									if(loopId < numInputs)                                 
									{                   
									                               
										int biasIndex = loopId;
										int weightIndex = loopId;
										neuronInputValue[offsetGroupBias+biasIndex]=trainingInput[i*numInputs + loopId];
										neuronOutputValue[offsetGroupBias+biasIndex]=tanh(
											weight[offsetGroupWeight+weightIndex]*neuronInputValue[offsetGroupBias+biasIndex]+bias[offsetGroupBias+biasIndex]
										);                                        
									}                                                   
								}                                                       
							}
							barrier(CLK_GLOBAL_MEM_FENCE);
			

							// input --> weighted sum --> activation --> output
							for(int layer = 1;layer<numLayers;layer++)
							{
								int numPreviousNeurons = topology[layer-1];
								int numCurrentNeurons = topology[layer];

								const int numLoopIter = (numCurrentNeurons / workGroupThreads) + 1;     
								for(int iGPGPU=0;iGPGPU<numLoopIter;iGPGPU++)                          
								{                                                       
									const int loopId = localId + workGroupThreads * iGPGPU; 
									if(loopId < numCurrentNeurons)                                 
									{                   
										int biasIndex = biasLayerStart[layer] + loopId;
									
									
										float accumulator = 0.0f;
										for(int pNeu=0;pNeu<numPreviousNeurons;pNeu++)
										{
											int weightIndex = weightLayerStart[layer]+loopId*numPreviousNeurons+pNeu;
											int biasIndexPrevious = biasLayerStart[layer-1] + pNeu;
											accumulator += neuronOutputValue[offsetGroupBias+biasIndexPrevious]*weight[offsetGroupWeight+weightIndex];
										}
										neuronInputValue[offsetGroupBias+biasIndex]=accumulator;
										neuronOutputValue[offsetGroupBias+biasIndex]=tanh(
											neuronInputValue[offsetGroupBias+biasIndex]+bias[offsetGroupBias+biasIndex]
										);																	                               
										                                        
									}                                                   
								}                                                       
								barrier(CLK_GLOBAL_MEM_FENCE);
							}


							float error = 0.0f;

							{
								const int numLoopIter = (numOutputs / workGroupThreads) + 1;     
								for(int iGPGPU=0;iGPGPU<numLoopIter;iGPGPU++)                          
								{                                                       
									const int loopId = localId + workGroupThreads * iGPGPU; 
									if(loopId < numOutputs)                                 
									{                   
										int biasIndex = biasLayerStart[numLayers-1] + loopId;
										float val = neuronOutputValue[offsetGroupBias+biasIndex]; 
										float output = trainingOutput[i*numOutputs + loopId];
										float err = (val-output);
										err *= err;
										error += err;																				                                        
									}                                                   
								}                                                       
								barrier(CLK_GLOBAL_MEM_FENCE);
							}
							errors[localId] = error;
							barrier(CLK_LOCAL_MEM_FENCE);
			                for(unsigned int i=workGroupThreads/2;i>=1;i>>=1)
							{
								unsigned int reduceId = i + localId;
								if(localId<i)
									errors[localId] += errors[reduceId]; 
								barrier(CLK_LOCAL_MEM_FENCE);
							}

							if(localId == 0)
							{
								if(dir == -1)
									errorLeft += errors[0];
								else
									errorRight += errors[0];
							}
						}

					}
									

					if(localId == 0)
					{
						neuronErrorValue[groupId*workGroupThreads]=(errorLeft<errorRight?-1:1);
						networkErrorValue[groupId*workGroupThreads]=(errorLeft<errorRight?errorLeft:errorRight);
					}
					
				}
			)"), "training");
		}

		// dampening: less than 1 greater than 0
		void Train(float trainingSpeed, int epochs, bool dampenTrainingSpeed, float dampening = 0.001f, bool profileEpoch = false)
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


				size_t t;
				{
					GPGPU::Bench benchmark(&t);
					_computer->compute(*_kernelParameters, "training", 0, 256 * _numParallelism, 256);
				}
				if (profileEpoch)
					std::cout << "epoch: " << t / 1000000000.0f << "s" << std::endl;
				if (i % 10 == 0)
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

				for (int k = 0; k < _numParallelism; k++)
				{
					if (k < _nWeight)
					{
						if (_neuronErrorValues->access<char>(k * 256) == -1)
						{
							_weightParallel->access<float>(k) -= trainingSpeed;
						}
						else // 1
						{
							_weightParallel->access<float>(k) += trainingSpeed;
						}
					}
					else // bias
					{
						if (_neuronErrorValues->access<char>(k * 256) == -1)
						{
							_biasParallel->access<float>(k - _nWeight) -= trainingSpeed;
						}
						else // 1
						{
							_biasParallel->access<float>(k - _nWeight) += trainingSpeed;
						}
					}

					/*
					if (_neuronErrorValues->access<char>(0) == -1)
					{
						if (weightToUpdate)
						{
							_weightParallel->access<float>((weightCtr - 1) % _nWeight) -= trainingSpeed;
						}
						else
						{
							_biasParallel->access<float>((biasCtr - 1) % _nBias) -= trainingSpeed;
						}
					}
					else
					{
						if (weightToUpdate)
						{
							_weightParallel->access<float>((weightCtr - 1) % _nWeight) += trainingSpeed;
						}
						else
						{
							_biasParallel->access<float>((biasCtr - 1) % _nBias) += trainingSpeed;
						}
					}
					*/
				}

				for (int k = 1; k < _numParallelism; k++)
				{
					for (int m = 0; m < _paddedNumElementsPerGroupForBias; m++)
					{
						_biasParallel->access<float>(m + k * _paddedNumElementsPerGroupForBias)
							=
							_biasParallel->access<float>(m);
					}

					for (int m = 0; m < _paddedNumElementsPerGroupForWeight; m++)
					{
						_weightParallel->access<float>(m + k * _paddedNumElementsPerGroupForWeight)
							=
							_weightParallel->access<float>(m);
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
				_neuronInputValuesParallel->access<float>(biasIndex) = inputs[neu];
				_neuronOutputValuesParallel->access<float>(biasIndex) = std::tanh(
					_weightParallel->access<float>(weightIndex) * _neuronInputValuesParallel->access<float>(biasIndex) + _biasParallel->access<float>(biasIndex)
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
						accumulator += _neuronOutputValuesParallel->access<float>(biasIndexPrevious) * _weightParallel->access<float>(weightIndex);
					}
					_neuronInputValuesParallel->access<float>(biasIndex) = accumulator;
					_neuronOutputValuesParallel->access<float>(biasIndex) = std::tanh(
						_neuronInputValuesParallel->access<float>(biasIndex) + _biasParallel->access<float>(biasIndex)
					);
					if (layer == _nLayers - 1)
					{
						outputs[neu] = _neuronOutputValuesParallel->access<float>(biasIndex);
					}
				}
			}

			return outputs;
		}
	private:
		int _numParallelism;
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

		// parallel-postfix means N copies will be made for computing all parameters at once
		// memory layout for weights is packed without any padding
		std::shared_ptr<GPGPU::HostParameter> _weightParallel;
		std::shared_ptr<GPGPU::HostParameter> _weightDeltaParallel;
		std::shared_ptr<GPGPU::HostParameter> _weightLayerStart;
		// similar to weight except only 1 bias per neuron		
		std::shared_ptr<GPGPU::HostParameter> _biasParallel;
		std::shared_ptr<GPGPU::HostParameter> _biasDeltaParallel;
		std::shared_ptr<GPGPU::HostParameter> _biasLayerStart;

		// input/outputs of each neuron
		std::shared_ptr<GPGPU::HostParameter> _neuronInputValuesParallel;
		std::shared_ptr<GPGPU::HostParameter> _neuronOutputValuesParallel;

		// error of neuron
		std::shared_ptr<GPGPU::HostParameter> _neuronErrorValues;
		std::shared_ptr<GPGPU::HostParameter> _networkErrorValues;
	};

}