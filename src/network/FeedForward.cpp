#include "NeuralNetwork.hpp"

void	NN::feedForward(float (*activHL)(const float&), float (*activO)(const float&))
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

void	NN::feedForwardMultiClass(float (*f)(const float&))
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
				_hiddenCells[i][j].computeValue(_inputs, f);
			else
				_hiddenCells[i][j].computeValue(_hiddenCells[i - 1], f);
		}
	}
	if (DEBUG)
		std::cout << "Compute outputs values : \n";
	for (size_t i = 0 ; i < getNbrOutputs() ; i++)
	{
		if (getNbrHiddenLayers() == 0)
			_outputs[i].computeValue(_inputs);
		else
			_outputs[i].computeValue(_hiddenCells[getNbrHiddenLayers() - 1]);
	}
	std::vector<float> Zs(getNbrOutputs());
	for (size_t i = 0 ; i < getNbrOutputs() ; i++)
		Zs[i] = _outputs[i].getZ();
	std::vector<float> values = softMax(Zs);
	for (size_t i = 0 ; i < getNbrOutputs() ; i++)
		_outputs[i].setValue(values[i]);
}