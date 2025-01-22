#include "../include/NeuralNetwork.hpp"

NN::NN(const size_t& nbrInputs, const size_t& nbrHiddenLayers, const size_t& nbrHiddenCells, const size_t& nbrOutputs)
{
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
	_learningRate = other.getLearningRate();
	_inputs = other._inputs;
	_hiddenCells = other._hiddenCells;
	_outputs = other._outputs;
}

NN&	NN::operator=(const NN& other)
{
	if (this != &other)
	{
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

std::vector<float>	NN::updateLastLayerWeights(float (*derivatedActivO)(float), const std::vector<float>& targets)
{
	std::vector<float> dErrorsOutputs(getNbrOutputs());
	for (size_t i = 0 ; i < getNbrOutputs() ; i++)
		dErrorsOutputs[i] = _outputs[i].getValue() - targets[i];

	std::vector<float> dErrorsZOutputs(getNbrOutputs());
	for (size_t i = 0 ; i < getNbrOutputs() ; i++)
		dErrorsZOutputs[i] = dErrorsOutputs[i] * derivatedActivO(_outputs[i].getZ());

	std::vector<std::vector<float>> dErrorsWeightsLastLayer(getNbrOutputs());
	for (size_t i = 0 ; i < getNbrOutputs() ; i++)
	{
		std::vector<float> dErrorsWeights(getNbrHiddenCells(), 0);
		for (size_t j = 0 ; j < getNbrHiddenCells() ; j++)
			dErrorsWeights[j] = dErrorsZOutputs[i] * _hiddenCells[getNbrHiddenLayers() - 1][j].getValue();
		dErrorsWeightsLastLayer[i] = dErrorsWeights;
	}
	for (size_t i = 0 ; i < getNbrOutputs() ; i++)
	{
		for (size_t j = 0 ; j < getNbrHiddenCells() ; j++)
		{
			float currentWeight = _hiddenCells[getNbrHiddenLayers() - 1][j].getWeight(_outputs[i].getIndex());
			_hiddenCells[getNbrHiddenLayers() - 1][j].setWeight(_outputs[i].getIndex(), currentWeight - _learningRate * dErrorsWeightsLastLayer[i][j]);
		}
		float currentBias = _outputs[i].getBias();
		_outputs[i].setBias(currentBias - _learningRate * dErrorsZOutputs[i]);
	}
	std::vector<float> dErrorsALastLayer(getNbrHiddenCells());
	for (size_t i = 0 ; i < getNbrHiddenCells() ; i++)
	{
		float sum = 0;
		for (size_t j = 0 ; j < getNbrOutputs() ; j++)
			sum += dErrorsZOutputs[j] * _hiddenCells[getNbrHiddenLayers() - 1][i].getWeight(_outputs[j].getIndex());
		dErrorsALastLayer[i] = sum;
	}
	return dErrorsALastLayer;
}

std::vector<float>	NN::updateHiddenLayersWeights(float (*derivatedHL)(float), const std::vector<float>& dErrorsALastLayer)
{
	std::vector<float> dErrors = dErrorsALastLayer;
	for (int layer = getNbrHiddenLayers() - 1 ; layer > 0 ; layer--)
	{
		std::vector<float> dErrorsZLayer(getNbrHiddenCells());
		for (size_t i = 0 ; i < getNbrHiddenCells() ; i++)
			dErrorsZLayer[i] = dErrors[i] * derivatedHL(_hiddenCells[layer][i].getZ());

		std::vector<std::vector<float>> dErrorsWeights(getNbrHiddenCells());
		for (size_t i = 0 ; i < getNbrHiddenCells() ; i++)
		{
			std::vector<float> dWeights(getNbrHiddenCells());
			for (size_t j = 0 ; j < getNbrHiddenCells() ; j++)
				dWeights[j] = dErrorsZLayer[i] * _hiddenCells[layer - 1][j].getValue();
			dErrorsWeights[i] = dWeights;
		}

		for (size_t i = 0 ; i < getNbrHiddenCells() ; i++)
		{
			for (size_t j = 0 ; j < getNbrHiddenCells() ; j++)
			{
				float currentWeight = _hiddenCells[layer - 1][j].getWeight(_hiddenCells[layer][i].getIndex());
				_hiddenCells[layer - 1][j].setWeight(_hiddenCells[layer][i].getIndex(), currentWeight - _learningRate * dErrorsWeights[i][j]);
			}
			float currentBias = _hiddenCells[layer][i].getBias();
			_hiddenCells[layer][i].setBias(currentBias - _learningRate * dErrorsZLayer[i]);
		}

		for (size_t i = 0 ; i < getNbrHiddenCells() ; i++)
		{
			float sum = 0;
			for (size_t j = 0 ; j < getNbrHiddenCells() ; j++)
				sum += dErrorsZLayer[i] * _hiddenCells[layer - 1][i].getWeight(_hiddenCells[layer][j].getIndex());
			dErrors[i] = sum;
		}
	}
	return dErrors;
}

void	NN::updateInputsWeights(float (*derivatedHL)(float), const std::vector<float>& dErrorsAFirstLayer)
{
	std::vector<float> dErrorsZ(getNbrHiddenCells());
	for (size_t i = 0 ; i < getNbrHiddenCells() ; i++)
		dErrorsZ[i] = dErrorsAFirstLayer[i] * derivatedHL(_hiddenCells[0][i].getZ());
	
	std::vector<std::vector<float>> dErrorsWeightsInputs(getNbrHiddenCells());
	for (size_t i = 0 ; i < getNbrHiddenCells() ; i++)
	{
		std::vector<float> dErrorsWeights(getNbrInputs());
		for (size_t j = 0 ; j < getNbrInputs() ; j++)
			dErrorsWeights[i] = dErrorsZ[i] * _inputs[j].getValue();
		dErrorsWeightsInputs[i] = dErrorsWeights;
	}

	for (size_t i = 0 ; i < getNbrHiddenCells() ; i++)
	{
		for (size_t j = 0 ; j < getNbrInputs() ; j++)
		{
			float currentWeight = _inputs[j].getWeight(_hiddenCells[0][i].getIndex());
			_inputs[j].setWeight(_hiddenCells[0][i].getIndex(), currentWeight - _learningRate * dErrorsWeightsInputs[i][j]);
		}
		float currentBias = _hiddenCells[0][i].getBias();
		_hiddenCells[0][i].setBias(currentBias - _learningRate * dErrorsZ[i]);
	}
}

void	NN::backPropagation(float (*derivatedActivHL)(float), float (*derivatedActivO)(float), const std::vector<float>& targets)
{
	std::vector<float> losses;
	for (size_t i = 0 ; i < getNbrOutputs() ; i++)
		losses.push_back(pow(_outputs[i].getValue() - targets[i], 2) / 2);

	std::vector<float> dErrorsALastLayer = updateLastLayerWeights(derivatedActivO, targets);
	std::vector<float> dErrorsAFirstLayer = updateHiddenLayersWeights(derivatedActivHL, dErrorsALastLayer);
}