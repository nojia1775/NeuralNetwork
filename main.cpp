#include "Neural_Network/include/NeuralNetwork.hpp"

float	ReLU(float x) { return x < 0 ? 0 : x; }

float	derivatedReLU(float x) { return x < 0 ? 0 : 1; }

float	sigmoid(float x) { return 1 / (1 + exp(-x)); }

float	derivatedSigmoid(float x) { return sigmoid(x) * (1 - sigmoid(x)); }

float	BCE(float yPred, float yTrue) { return - yTrue * log(yPred) - (1 - yTrue) * log(1 - yPred); }

float	derivatedBCE(float yPred, float yTrue) { return - yTrue / (yPred + 1e-7) + (1 - yTrue) / (1 - yPred + 1e-7); }

float	activation(float x) { return x; }

float	derivatedAction(float x) { return x / abs(x); }

float	MSE(float yPred, float yTrue) { return pow(yPred - yTrue, 2) / 2; }

float	derivatedMSE(float yPred, float yTrue) { return yPred - yTrue; }

int	main(void)
{
	std::srand(std::time(NULL));
	std::vector<std::vector<float>> datas = {
		{1.0f, 1.0f},
		{0.0f, 1.0f},
		{1.0f, 0.0f},
		{0.0f, 0.0f}
	};
	std::vector<std::vector<float>> expected = {
		{0.0f},
		{1.0f},
		{1.0f},
		{0.0f}
	};
	try
	{
		NN nn(2, 1, 2, 1);
		nn.initNN();
		nn.setLearningRate(0.1f);
		std::vector<float> outputs = nn.use({0.0f, 1.0f}, ReLU, sigmoid);
		std::cout << "avant entrainement : " << outputs[0] << "\n";
		nn.train(10000, datas, expected, BCE, derivatedBCE, ReLU, sigmoid, derivatedReLU, derivatedSigmoid);
		outputs = nn.use({0.0f, 1.0f}, ReLU, sigmoid);
		std::cout << "Output = " << outputs[0] << "\n";
		outputs = nn.use({1.0f, 0.0f}, ReLU, sigmoid);
		std::cout << "Output = " << outputs[0] << "\n";
		outputs = nn.use({1.0f, 1.0f}, ReLU, sigmoid);
		std::cout << "Output = " << outputs[0] << "\n";
		outputs = nn.use({0.0f, 0.0f}, ReLU, sigmoid);
		std::cout << "Output = " << outputs[0] << "\n";
		outputs = nn.use({0.0f, 1.0f}, ReLU, sigmoid);
		std::cout << "Output = " << outputs[0] << "\n";
		nn.getJSON();
	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << "\n";
	}
	return 0;
}