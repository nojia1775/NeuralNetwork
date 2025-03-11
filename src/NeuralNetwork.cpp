#include "../include/NeuralNetwork.hpp"

NN::NN(const size_t& nbrInputs, const size_t& nbrHiddenLayers, const size_t& nbrHiddenCells, const size_t& nbrOutputs)
{
	_id = std::rand();
	_learningRate = 0.1;
	for (size_t i = 0 ; i < nbrInputs ; i++)
		_inputs.push_back(Input(nbrHiddenCells));
	std::vector<HiddenCell> cells;
	if (nbrHiddenLayers == 1)
	{
		for (size_t i = 0 ; i < nbrHiddenCells ; i++)
			cells.push_back(HiddenCell(nbrOutputs, i));
		_hiddenCells.push_back(cells);
	}
	else
	{
		for (size_t i = 0 ; i < nbrHiddenLayers ; i++)
		{
			for (size_t j = 0 ; j < nbrHiddenCells ; j++)
			{
				if (i < nbrHiddenLayers - 1)
					cells.push_back(HiddenCell(nbrHiddenCells, j));
				else
					cells.push_back(HiddenCell(nbrOutputs, j));
			}
			_hiddenCells.push_back(cells);
		}
	}
	for (size_t i = 0 ; i < nbrOutputs ; i++)
		_outputs.push_back(Output(i));
}

NN::NN(const NN& other)
{
	_id = std::rand();
	_learningRate = other.getLearningRate();
	_inputs = other._inputs;
	_hiddenCells = other._hiddenCells;
	_outputs = other._outputs;
}

NN&	NN::operator=(const NN& other)
{
	if (this != &other)
	{
		_id = std::rand();
		_learningRate = other.getLearningRate();
		_inputs = other._inputs;
		_hiddenCells = other._hiddenCells;
		_outputs = other._outputs;
	}
	return *this;
}

void	NN::displayValues(void) const
{
	std::cout << "Inputs : \n";
	for (size_t i = 0 ; i < getNbrInputs() ; i++)
		std::cout << "[" << i << "] : " << _inputs[i].getValue() << "\n";
	std::cout << "\n";
	for (size_t i = 0 ; i < getNbrHiddenLayers() ; i++)
	{
		std::cout << "Layer " << i + 1 << " : \n";
		for (size_t j = 0 ; j < getNbrHiddenCells() ; j++)
			std::cout << "[" << j << "] : " << _hiddenCells[i][j].getValue() << "\n";
		std::cout << "\n";
	}
	std::cout << "Outputs : \n";
	for (size_t i = 0 ; i < getNbrOutputs() ; i++)
		std::cout << "[" << i << "] : " << _outputs[i].getValue() << "\n";
}

void	NN::initNN(void)
{
	for (size_t i = 0 ; i < getNbrInputs() ; i++)
		_inputs[i].randomWeights();
	for (size_t i = 0 ; i < getNbrHiddenLayers() ; i++)
		for (size_t j = 0 ; j < getNbrHiddenCells() ; j++)
			_hiddenCells[i][j].randomWeights();
	for (size_t i = 0 ; i < getNbrOutputs() ; i++)
		_outputs[i].randomBias();
}

void	NN::feedForward(float (*activHL)(float), float (*activO)(float))
{
	if (DEBUG)
	{
		std::cout << "------------- FEED FORWARD -------------\n\n";
		std::cout << "Compute hidden cells values : \n";
	}
	for (size_t i = 0 ; i < getNbrHiddenLayers() ; i++)
	{
		for (size_t j = 0 ; j < getNbrHiddenCells() ; j++)
		{
			if (i == 0)
				_hiddenCells[i][j].computeValue(_inputs, activHL);
			else
				_hiddenCells[i][j].computeValue(_hiddenCells[i - 1], activHL);
		}
	}
	if (DEBUG)
		std::cout << "Compute outputs values : \n";
	for (size_t i = 0 ; i < getNbrOutputs() ; i++)
	{
		if (getNbrHiddenLayers() == 0)
			_outputs[i].computeValue(_inputs, activO);
		else
			_outputs[i].computeValue(_hiddenCells[getNbrHiddenLayers() - 1], activO);
	}
}

void	NN::initInputs(const std::vector<float>& inputs)
{
	if (inputs.size() == getNbrInputs())
	{
		for (size_t i = 0 ; i < getNbrInputs() ; i++)
			_inputs[i].initValue(inputs[i]);
	}
	else
		throw Input::DifferentNumberOfWeights();
}

std::vector<float>	NN::getOutputs(void) const
{
	std::vector<float> outputs;
	for (auto value : _outputs)
		outputs.push_back(value.getValue());
	return outputs;
}

std::vector<float>	NN::updateLastLayerWeights(float (*derivatedLoss)(float, float), float (*derivatedActivO)(float), const std::vector<float>& targets)
{
	if (DEBUG)
		std::cout << "--------------------- UPDATE LAST LAYER ---------------------\n\n";
	std::vector<float> dErrorsOutputs(getNbrOutputs());
	for (size_t i = 0 ; i < getNbrOutputs() ; i++)
	{
		dErrorsOutputs[i] = derivatedLoss(_outputs[i].getValue(), targets[i]);
		if (DEBUG)
		{
			std::cout << "dErrorOutput[" << i << "] = derivLoss(output[" << i << "]) -> \n" << dErrorsOutputs[i] << " = " << "derivLoss(" << _outputs[i].getValue() << ")\n\n";
		}
	}

	std::vector<float> dErrorsZOutputs(getNbrOutputs());
	for (size_t i = 0 ; i < getNbrOutputs() ; i++)
	{
		dErrorsZOutputs[i] = dErrorsOutputs[i] * derivatedActivO(_outputs[i].getZ());
		if (DEBUG)
		{
			std::cout << "dErrorZOutput[" << i << "] = dErrorOutput[" << i << "] * derivO(output[" << i << "].Z) -> \n" << dErrorsZOutputs[i] << " = " << dErrorsOutputs[i] << " * " << "derivO(" << _outputs[i].getZ() << ")\n\n";
		}
	}

	std::vector<std::vector<float>> dErrorsWeightsLastLayer(getNbrOutputs());
	for (size_t i = 0 ; i < getNbrOutputs() ; i++)
	{
		std::vector<float> dErrorsWeights(getNbrHiddenCells(), 0);
		for (size_t j = 0 ; j < getNbrHiddenCells() ; j++)
		{
			dErrorsWeights[j] = dErrorsZOutputs[i] * _hiddenCells[getNbrHiddenLayers() - 1][j].getValue();
			if (DEBUG)
			{
				std::cout << "dErrorWeight[" << i << "][" << j << "] = dErrorZOutput[" << i << "] * hiddenCells[" << getNbrHiddenLayers() - 1 << "][" << j << "] ->\n" << dErrorsWeights[j] << " = " << dErrorsZOutputs[i] << " * " << _hiddenCells[getNbrHiddenLayers() - 1][j].getValue() << "\n\n";
			}
		}
		dErrorsWeightsLastLayer[i] = dErrorsWeights;
	}

	std::vector<std::vector<float>> newWeights(getNbrOutputs());
	for (size_t i = 0 ; i < getNbrOutputs() ; i++)
	{
		std::vector<float> weights(getNbrHiddenCells());
		for (size_t j = 0 ; j < getNbrHiddenCells() ; j++)
		{
			float currentWeight = _hiddenCells[getNbrHiddenLayers() - 1][j].getWeight(_outputs[i].getIndex());
			weights[j] = currentWeight - _learningRate * dErrorsWeightsLastLayer[i][j];
			if (DEBUG)
			{
				std::cout << "newWeight[" << i << "][" << j << "] = currentWeight - learningRate * dErrorWeightLastLayer[" << i << "][" << j << "] -> \n" << weights[j] << " = " << currentWeight << " - " << _learningRate << " * " << dErrorsWeightsLastLayer[i][j] << "\n\n";
			}
		}
		newWeights[i] = weights;
		float currentBias = _outputs[i].getBias();
		_outputs[i].setBias(currentBias - _learningRate * dErrorsZOutputs[i]);
		if (DEBUG)
		{
			std::cout << "newBias[" << i << "] = currentBias[" << i << "] - learningRate * dErrorZOutput[" << i << "] -> \n" << currentBias - _learningRate * dErrorsZOutputs[i] << " = " << currentBias << " - " << _learningRate << " * " << dErrorsZOutputs[i] << "\n\n";
		}
	}

	std::vector<float> dErrorsALastLayer(getNbrHiddenCells());
	for (size_t i = 0 ; i < getNbrHiddenCells() ; i++)
	{
		float sum = 0;
		for (size_t j = 0 ; j < getNbrOutputs() ; j++)
		{
			sum += dErrorsZOutputs[j] * _hiddenCells[getNbrHiddenLayers() - 1][i].getWeight(_outputs[j].getIndex());
		}
		dErrorsALastLayer[i] = sum;
		if (DEBUG)
		{
			std::cout << "dErrorsALastLayer[" << i << "] = somme(dErrorZOutput[] * hiddenCells[][].weight[]) ->\n";
			std::cout << "dErrorALastLayer[" << i << "] = " << sum << "\n\n";
		}
	}
	for (size_t i = 0 ; i < getNbrOutputs() ; i++)
	{
		for (size_t j = 0 ; j < getNbrHiddenCells() ; j++)
			_hiddenCells[getNbrHiddenLayers() - 1][j].setWeight(_outputs[i].getIndex(), newWeights[i][j]);
	}
	return dErrorsALastLayer;
}

std::vector<float>	NN::updateHiddenLayersWeights(float (*derivatedHL)(float), const std::vector<float>& dErrorsALastLayer)
{
	if (DEBUG)
		std::cout << "--------------------- UPDATE HIDDEN LAYERS ---------------------\n\n";
	std::vector<float> dErrors = dErrorsALastLayer;
	for (int layer = getNbrHiddenLayers() - 1 ; layer > 0 ; layer--)
	{
		std::vector<float> dErrorsZLayer(getNbrHiddenCells());
		for (size_t i = 0 ; i < getNbrHiddenCells() ; i++)
		{
			dErrorsZLayer[i] = dErrors[i] * derivatedHL(_hiddenCells[layer][i].getZ());
			if (DEBUG)
			{
				std::cout << "dErrorZLayer[" << i << "] = dErrorALastLayer[" << i << "] * derivHL(hiddenCells[" << layer << "][" << i << "].Z) ->\n" << dErrorsZLayer[i] << " = " << dErrors[i] << " * " << derivatedHL(_hiddenCells[layer][i].getZ()) << "\n\n";
			}
		}

		std::vector<std::vector<float>> dErrorsWeights(getNbrHiddenCells());
		for (size_t i = 0 ; i < getNbrHiddenCells() ; i++)
		{
			std::vector<float> dWeights(getNbrHiddenCells());
			for (size_t j = 0 ; j < getNbrHiddenCells() ; j++)
			{
				dWeights[j] = dErrorsZLayer[i] * _hiddenCells[layer - 1][j].getValue();
				if (DEBUG)
				{
					std::cout << "dWeight[" << i << "][" << j << "] = dErrorZLayer[" << i << "] * hiddenCells[" << layer - 1 << "][" << j << "] -> \n" << dWeights[j] << " = " << dErrorsZLayer[i] << " * " << _hiddenCells[layer - 1][j].getValue() << "\n\n";
				}
			}
			dErrorsWeights[i] = dWeights;
		}

		std::vector<std::vector<float>> newWeights(getNbrHiddenCells());
		for (size_t i = 0 ; i < getNbrHiddenCells() ; i++)
		{
			std::vector<float> weights(getNbrHiddenCells());
			for (size_t j = 0 ; j < getNbrHiddenCells() ; j++)
			{
				float currentWeight = _hiddenCells[layer - 1][j].getWeight(_hiddenCells[layer][i].getIndex());
				weights[j] = currentWeight - _learningRate * dErrorsWeights[i][j];
				if (DEBUG)
				{
					std::cout << "newWeight[" << i << "][" << j << "] = currentWeight - learningRate * dErrorweight[" << i << "][" << j << "] ->\n" << weights[j] << " = " << currentWeight << " - " << _learningRate << " * " << dErrorsWeights[i][j] << "\n\n";
				}
			}
			newWeights[i] = weights;
			float currentBias = _hiddenCells[layer][i].getBias();
			_hiddenCells[layer][i].setBias(currentBias - _learningRate * dErrorsZLayer[i]);
			if (DEBUG)
				std::cout << "newBias[" << layer << "][" << i << "] = currentBias - learningRate * dErrorZLayer[" << i << "] -> \n" << currentBias - _learningRate * dErrorsZLayer[i] << " = " << currentBias << " - " << _learningRate << " * " << dErrorsZLayer[i] << "\n\n";
		}

		for (size_t i = 0 ; i < getNbrHiddenCells() ; i++)
		{
			float sum = 0;
			for (size_t j = 0 ; j < getNbrHiddenCells() ; j++)
				sum += dErrorsZLayer[i] * _hiddenCells[layer - 1][i].getWeight(_hiddenCells[layer][j].getIndex());
			dErrors[i] = sum;
			if (DEBUG)
			{
				std::cout << "dErrorALayer[" << i << "] = somme(dErrorZLayer[" << i << "] * hiddenCells[][].weight[]) ->\ndErrorZLayer[" << i << "] = " << sum << "\n\n";
			}
		}
		for (size_t i = 0 ; i < getNbrHiddenCells() ; i++)
		{
			for (size_t j = 0 ; j < getNbrHiddenCells() ; j++)
				_hiddenCells[layer - 1][j].setWeight(_hiddenCells[layer][i].getIndex(), newWeights[i][j]);
		}
	}
	return dErrors;
}

void	NN::updateInputsWeights(float (*derivatedHL)(float), const std::vector<float>& dErrorsAFirstLayer)
{
	if (DEBUG)
		std::cout << "--------------------- UPDATE FIRST LAYER ---------------------\n\n";
	std::vector<float> dErrorsZ(getNbrHiddenCells());
	for (size_t i = 0 ; i < getNbrHiddenCells() ; i++)
	{
		dErrorsZ[i] = dErrorsAFirstLayer[i] * derivatedHL(_hiddenCells[0][i].getZ());
		if (DEBUG)
		{
			std::cout << "dErrorZ[" << i << "] = dErrorAFirstLayer[" << i << "] * derivHL(hiddenCells[0][" << i << "].Z) ->\n" << dErrorsZ[i] << " = " << dErrorsAFirstLayer[i] << " * " << derivatedHL(_hiddenCells[0][i].getZ()) << "\n\n";
		}
	}
	
	std::vector<std::vector<float>> dErrorsWeightsInputs(getNbrHiddenCells());
	for (size_t i = 0 ; i < getNbrHiddenCells() ; i++)
	{
		std::vector<float> dErrorsWeights(getNbrInputs());
		for (size_t j = 0 ; j < getNbrInputs() ; j++)
		{
			dErrorsWeights[j] = dErrorsZ[i] * _inputs[j].getValue();
			if (DEBUG)
			{
				std::cout << "dErrorsWeights[" << i << "][" << j << "] = dErrorZ[" << i << "] * inputs[" << j << "] ->\n" << dErrorsWeights[j] << " = " << dErrorsZ[i] << " * " << _inputs[j].getValue() << "\n\n";
			}	
		}
		dErrorsWeightsInputs[i] = dErrorsWeights;
	}

	for (size_t i = 0 ; i < getNbrHiddenCells() ; i++)
	{
		for (size_t j = 0 ; j < getNbrInputs() ; j++)
		{
			float currentWeight = _inputs[j].getWeight(_hiddenCells[0][i].getIndex());
			_inputs[j].setWeight(_hiddenCells[0][i].getIndex(), currentWeight - _learningRate * dErrorsWeightsInputs[i][j]);
			if (DEBUG)
			{
				std::cout << "newWeight[" << i << "] = currentWeight - learningRate * dErrorWeightInput[" << i << "][" << j << "] ->\n" << currentWeight - _learningRate * dErrorsWeightsInputs[i][j] << " = " << currentWeight << " - " << _learningRate << " * " << dErrorsWeightsInputs[i][j] << "\n\n";
			}
		}
		float currentBias = _hiddenCells[0][i].getBias();
		_hiddenCells[0][i].setBias(currentBias - _learningRate * dErrorsZ[i]);
		if (DEBUG)
		{
			std::cout << "newBias[" << i << "] = currentBias - learningRate * dErrorZ[" << i << "] ->\n" << currentBias - _learningRate * dErrorsZ[i] << " = " << currentBias << " - " << _learningRate << " * " << dErrorsZ[i] << "\n\n";
		}
	}
}

float	NN::backPropagation(float (*loss)(float, float), float (*derivatedLoss)(float, float), float (*derivatedActivHL)(float), float (*derivatedActivO)(float), const std::vector<float>& targets)
{
	if (DEBUG)
		std::cout << "------------- BACK PROPAGATION -------------\n\n";
	float accuracy = 0;
	for (size_t i = 0 ; i < getNbrOutputs() ; i++)
		accuracy += loss(_outputs[i].getValue(), targets[i]);
	accuracy /= getNbrOutputs();

	std::vector<float> dErrorsALastLayer = updateLastLayerWeights(derivatedLoss, derivatedActivO, targets);
	std::vector<float> dErrorsAFirstLayer = updateHiddenLayersWeights(derivatedActivHL, dErrorsALastLayer);
	updateInputsWeights(derivatedActivHL, dErrorsAFirstLayer);
	_loss.push_back(accuracy);
	return accuracy;
}

void	NN::train(const size_t& epochs, const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& expectedOutputs, float (*loss)(float, float), float (*derivatedLoss)(float, float), float (*func1)(float), float (*func2)(float), float (*derivFunc1)(float), float (*derivFunc2)(float))
{
	float accuracy = 0;
	for (size_t i = 0 ; i < epochs ; i++)
	{
		for (size_t j = 0 ; j < inputs.size() ; j++)
		{
			initInputs(inputs[j]);
			feedForward(func1, func2);
			accuracy = backPropagation(loss, derivatedLoss, derivFunc1, derivFunc2, expectedOutputs[j]);
		}
	}
	std::cout << "Accuracy = " << 1 - accuracy << "\n";
}

std::vector<float>	NN::use(const std::vector<float>& inputs, float (*activHL)(float), float (*activO)(float))
{
	initInputs(inputs);
	feedForward(activHL, activO);
	return getOutputs();
}

void	NN::getJSON(void) const
{
	std::string fileName("nn" + std::to_string(_id) + ".json");
	nlohmann::json jsonData;
	jsonData["inputs"] = getNbrInputs();
	jsonData["hidden_layers"] = getNbrHiddenLayers();
	jsonData["hidden_neurals"] = getNbrHiddenCells();
	jsonData["outputs"] = getNbrOutputs();
	jsonData["layers"] = nlohmann::json::array();

	nlohmann::json inputsData;
	inputsData["type"] = "inputs";
	inputsData["weights"] = nlohmann::json::array();
	for (size_t i = 0 ; i < getNbrInputs() ; i++)
		inputsData["weights"].push_back(_inputs[i].getWeights());
	jsonData["layers"].push_back(inputsData);

	nlohmann::json hiddenData;
	for (size_t i = 0 ; i < getNbrHiddenLayers() ; i++)
	{
		hiddenData["bias"] = getHiddenLayersBias(i);
		hiddenData["type"] = "hidden_layer_" + std::to_string(i);
		hiddenData["weights"] = nlohmann::json::array();
		for (size_t j = 0 ; j < getNbrHiddenCells() ; j++)
			hiddenData["weights"].push_back(_hiddenCells[i][j].getWeights());
		jsonData["layers"].push_back(hiddenData);
	}

	nlohmann::json outputsData;
	outputsData["type"] = "outputs";
	outputsData["bias"] = getOutputsBias();
	jsonData["layers"].push_back(outputsData);
	jsonData["loss"] = {getLoss()};
	std::ofstream file("nn" + std::to_string(_id) + ".json");
	if (file.is_open())
	{
		file << jsonData.dump(4);
		std::cout << "Log saved in " << fileName << "\n";
		file.close();
	}
	else
		std::cout << "Error\n";
}

void	NN::getJSON(const std::string& fileName) const
{
	nlohmann::json jsonData;
	jsonData["inputs"] = getNbrInputs();
	jsonData["hidden_layers"] = getNbrHiddenLayers();
	jsonData["hidden_neurals"] = getNbrHiddenCells();
	jsonData["outputs"] = getNbrOutputs();
	jsonData["layers"] = nlohmann::json::array();

	nlohmann::json inputsData;
	inputsData["type"] = "inputs";
	inputsData["weights"] = nlohmann::json::array();
	for (size_t i = 0 ; i < getNbrInputs() ; i++)
		inputsData["weights"].push_back(_inputs[i].getWeights());
	jsonData["layers"].push_back(inputsData);

	nlohmann::json hiddenData;
	for (size_t i = 0 ; i < getNbrHiddenLayers() ; i++)
	{
		hiddenData["bias"] = getHiddenLayersBias(i);
		hiddenData["type"] = "hidden_layer_" + std::to_string(i);
		hiddenData["weights"] = nlohmann::json::array();
		for (size_t j = 0 ; j < getNbrHiddenCells() ; j++)
			hiddenData["weights"].push_back(_hiddenCells[i][j].getWeights());
		jsonData["layers"].push_back(hiddenData);
	}

	nlohmann::json outputsData;
	outputsData["type"] = "outputs";
	outputsData["bias"] = getOutputsBias();
	jsonData["layers"].push_back(outputsData);
	jsonData["loss"] = {getLoss()};
	std::ofstream file(fileName);
	if (file.is_open())
	{
		file << jsonData.dump(4);
		std::cout << "Log saved in " << fileName << "\n";
		file.close();
	}
	else
		std::cout << "Error\n";
}

std::vector<float>	NN::getHiddenLayersBias(const size_t& layer) const
{
	if (layer > getNbrHiddenLayers() - 1)
	{
		throw Input::OutOfRange();
		return {0};
	}
	std::vector<float> bias(getNbrHiddenCells());
	for (size_t i = 0 ; i < getNbrHiddenCells() ; i++)
		bias[i] = _hiddenCells[layer][i].getBias();
	return bias;
}

std::vector<float>	NN::getOutputsBias(void) const
{
	std::vector<float> bias(getNbrOutputs());
	for (size_t i = 0 ; i < getNbrOutputs() ; i++)
		bias[i] = _outputs[i].getBias();
	return bias;
}

NN::NN(const std::string& fileName)
{
	std::ifstream file(fileName);
	if (!file.is_open())
	{
		std::cout << "Impossible to open " << fileName << "\n";
		return;
	}

	nlohmann::json jsonData;
	try
	{
		file >> jsonData;
	}
	catch (const nlohmann::json::parse_error& e)
	{
		std::cout << e.what() << "\n";
	}
	size_t nbrInputs = jsonData["inputs"];
	size_t nbrHiddenLayers = jsonData["hidden_layers"];
	size_t nbrHiddenNeurals = jsonData["hidden_neurals"];
	size_t nbrOutputs = jsonData["outputs"];

	for (size_t i = 0 ; i < nbrInputs ; i++)
		_inputs.push_back(Input(nbrHiddenNeurals));
	
	for (size_t i = 0 ; i < nbrHiddenLayers ; i++)
	{
		std::vector<HiddenCell> layer;
		for (size_t j = 0 ; j < nbrHiddenNeurals ; j++)
		{
			if (i == nbrHiddenLayers - 1)
				layer.push_back(HiddenCell(nbrOutputs, j));
			else
				layer.push_back(HiddenCell(nbrHiddenNeurals, j));
		}
		_hiddenCells.push_back(layer);
		layer.clear();
	}

	for (size_t i = 0 ; i < nbrOutputs ; i++)
		_outputs.push_back(Output(i));

	auto layers = jsonData["layers"];
	size_t index = 0;
	for (auto& layer : layers)
	{
		if (layer["type"] == "inputs")
		{
			for (size_t i = 0 ; i < nbrInputs ; i++)
			{
				for (size_t j = 0 ; j < nbrHiddenNeurals ; j++)
					_inputs[i].setWeight(j, layer["weights"][i][j]);
			}
		}
		else if (layer["type"] == "hidden_layer_" + std::to_string(index - 1))
		{
			for (size_t i = 0 ; i < nbrHiddenNeurals ; i++)
			{
				if (index == nbrHiddenLayers)
				{
					for (size_t j = 0 ; j < nbrOutputs ; j++)
						_hiddenCells[index - 1][i].setWeight(j, layer["weights"][i][j]);
				}
				else
				{
					for (size_t j = 0 ; j < nbrHiddenNeurals ; j++)
						_hiddenCells[index - 1][i].setWeight(j, layer["weights"][i][j]);
				}
				_hiddenCells[index - 1][i].setBias(layer["bias"][i]);
			}
		}
		else if (layer["type"] == "outputs")
		{
			for (size_t i = 0 ; i < nbrOutputs ; i++)
				_outputs[i].setBias(layer["bias"][i]);
		}
		index++;
	}
}