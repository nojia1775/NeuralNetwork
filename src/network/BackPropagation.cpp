#include "../../include/NeuralNetwork.hpp"

std::vector<float>	NN::updateLastLayerWeights(float (*derivatedLoss)(const float&, const float&), float (*derivatedActivO)(const float&), const std::vector<float>& targets)
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

std::vector<float>	NN::updateLastLayerWeightsMultiClass(const std::vector<float>& targets)
{
	if (DEBUG)
		std::cout << "--------------------- UPDATE LAST LAYER ---------------------\n\n";
	std::vector<float> dErrorsZOutputs(getNbrOutputs());
	for (size_t i = 0 ; i < getNbrOutputs() ; i++)
	{
		dErrorsZOutputs[i] = _outputs[i].getValue() - targets[i];
		if (DEBUG)
			std::cout << "dErrorZOutput[" << i << "] = output[" << i << "] - targets[" << i << "] ->\n" << dErrorsZOutputs[i] << " = " << _outputs[i].getValue() << " - " << targets[i] << "\n\n";
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

std::vector<float>	NN::updateHiddenLayersWeights(float (*derivatedHL)(const float&), const std::vector<float>& dErrorsALastLayer)
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

void	NN::updateInputsWeights(float (*derivatedHL)(const float&), const std::vector<float>& dErrorsAFirstLayer)
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

float	NN::backPropagation(float (*loss)(const float&, const float&), float (*derivatedLoss)(const float&, const float&), float (*derivatedActivHL)(const float&), float (*derivatedActivO)(const float&), const std::vector<float>& targets)
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

float	NN::backPropagationMultiClass(float (*f)(const float&), float (*derivatedF)(const float&), const std::vector<float>& targets)
{
	if (DEBUG)
		std::cout << "------------- BACK PROPAGATION -------------\n\n";
	std::vector<float> predicted(getNbrOutputs());
	for (size_t i = 0 ; i < getNbrOutputs() ; i++)
	predicted[i] = _outputs[i].getValue();
	float accuracy = crossEntropy(predicted, targets);
	std::vector<float> dErrorsALastLayer = updateLastLayerWeightsMultiClass(targets);
	std::vector<float> dErrorsAFirstLayer = updateHiddenLayersWeights(derivatedF, dErrorsALastLayer);
	updateInputsWeights(derivatedF, dErrorsAFirstLayer);
	_loss.push_back(accuracy);
	return accuracy;
}