#include "Neural_Network/include/NeuralNetwork.hpp"

float	ReLU(float x) { return x < 0 ? 0 : x; }

float	derivatedReLU(float x) { return x < 0 ? 0 : 1; }

float	sigmoid(float x) { return 1 / (1 + exp(-x)); }

float	derivatedSigmoid(float x) { return sigmoid(x) * (1 - sigmoid(x)); }

int	main(void)
{
	NN nn(3, 1, 2, 1);
	try
	{
		nn.initNN();
	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << "\n";
	}
	return 0;
}