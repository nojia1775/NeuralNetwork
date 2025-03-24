#include "functions.hpp"

std::vector<float>	softMax(const std::vector<float>& input)
{
	std::vector<float> output(input.size());
	float maxVal = *std::max_element(input.begin(), input.end());
	float sumExp = 0.0;
	for (size_t i = 0; i < input.size(); ++i)
	{
		output[i] = std::exp(input[i] - maxVal);
		sumExp += output[i];
	}
	for (size_t i = 0; i < output.size(); ++i)
		output[i] /= sumExp;
	return output;
}

std::vector<float>	derivatedSoftMax(const std::vector<float>& inputs)
{
	std::vector<float> result(inputs.size());

	for (size_t i = 0 ; i < inputs.size() ; i++)
	{
		float sum = 0;
		for (size_t j = 0 ; j < inputs.size() ; j++)
			sum += exp(inputs[j]);
		result[i] = exp(inputs[i]) / sum;
	}
	return result;
}

float	crossEntropy(const std::vector<float>& yPred, const std::vector<float>& yTrue)
{
	float loss = 0.0f;
	for (size_t i = 0; i < yTrue.size(); ++i)
		if (yTrue[i] > 0)
			loss -= std::log(yPred[i] + EPSILON);
	return loss;
}

float	derivatedCrossEntropy(const std::vector<float>& yPred, const std::vector<float>& yTrue)
{
	float loss = 0;
	for (size_t i = 0 ; i < yPred.size() ; i++)
		loss += yTrue[i] * std::log(yPred[i] + EPSILON);
	return -loss;
}