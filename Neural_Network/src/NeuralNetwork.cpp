#include "../include/NeuralNetwork.hpp"

NN::NN(const size_t& nbrInputs, const size_t& nbrHiddenLayers, const size_t& nbrHiddenCells, const size_t& nbrOutputs)
{
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
	_inputs = other._inputs;
	_hiddenCells = other._hiddenCells;
	_outputs = other._outputs;
}

NN&	NN::operator=(const NN& other)
{
	if (this != &other)
	{
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
	std::cout << "\n";
	std::cout << "Outputs : \n";
	for (size_t i = 0 ; i < getNbrOutputs() ; i++)
		std::cout << "[" << i << "] : " << _outputs[i].getValue() << "\n";
}

void	NN::initRandomValues(void)
{
	for (size_t i = 0 ; i < getNbrInputs() ; i++)
		_inputs[i].randomWeights();
	for (size_t i = 0 ; i < getNbrHiddenLayers() ; i++)
	{
		for (size_t j = 0 ; j < getNbrHiddenCells() ; j++)
			_hiddenCells[i][j].randomWeights();
	}
	for (size_t i = 0 ; i < getNbrOutputs() ; i++)
		_outputs[i].randomBias();
}

void	NN::frontPropagation(float (*activHL)(float), float (*activO)(float))
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
	throw Input::DifferentNumberOfWeights();
	return;
}