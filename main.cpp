#include "Neural_Network/include/NeuralNetwork.hpp"

float	ReLU(float x)
{
	return x < 0 ? 0 : x;
}

float	sigmoid(float x)
{
	return 1 / (1 + exp(-x));
}

int	main(void)
{
	NN nn(3, 1, 2, 1);
	std::cout << nn.getNbrInputs() << "\n";
	try
	{
		nn.initNN();
		nn.initInputs({1, 2, 3});
		nn.feedForward(ReLU, sigmoid);
		nn.displayValues();
	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << "\n";
	}
	return 0;
}