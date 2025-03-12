#include "../../include/NeuralNetwork.hpp"

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