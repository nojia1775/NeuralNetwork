#include "NeuralNetwork.hpp"

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

void	NN::train(const size_t& epochs, const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& expectedOutputs, float (*loss)(const float&, const float&), float (*derivatedLoss)(const float&, const float&), float (*func1)(const float&), float (*func2)(const float&), float (*derivFunc1)(const float&), float (*derivFunc2)(const float&))
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

void	NN::trainMultiClass(const size_t& epochs, const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& expectedOutputs, float (*f)(const float&), float (*derivatedF)(const float&))
{
	float accuracy = 0;
	for (size_t i = 0 ; i < epochs ; i++)
	{
		for (size_t j = 0 ; j < inputs.size() ; j++)
		{
			initInputs(inputs[j]);
			feedForwardMultiClass(f);
			accuracy = backPropagationMultiClass(derivatedF, expectedOutputs[j]);
		}
	}
	std::cout << "Accuracy = " << (1 - accuracy) * 100 << "%\n";
}

std::vector<float>	NN::use(const std::vector<float>& inputs, float (*activHL)(const float&), float (*activO)(const float&))
{
	initInputs(inputs);
	feedForward(activHL, activO);
	return getOutputs();
}

std::vector<float>	NN::useMultiClass(const std::vector<float>& inputs, float (*f)(const float&))
{
	initInputs(inputs);
	feedForwardMultiClass(f);
	return getOutputs();
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