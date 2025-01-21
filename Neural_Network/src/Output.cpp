#include "../include/Output.hpp"

Output::Output(const size_t& index) : _value(0), _bias(0), _index(index) {}

Output::Output(const Output& other) : _value(other.getValue()), _bias(other.getBias()), _index(other.getIndex()) {}

Output&	Output::operator=(const Output& other)
{
	if (this != &other)
	{
		_value = other.getValue();
		_bias = other.getBias();
		_index = other.getIndex();
	}
	return *this;
}

void	Output::computeValue(const std::vector<HiddenCell>& hiddenCells, float (*activation)(float))
{
	float activate = 0;
	try
	{
		for (auto cell : hiddenCells)
			activate += cell.getValue() * cell.getWeight(_index);
		_value = activation(activate + _bias);
	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << "\n";
	}
}

void	Output::computeValue(const std::vector<Input>& inputs, float (*activation)(float))
{
	float activate = 0;
	try
	{
		for (auto input : inputs)
			activate += input.getValue() * input.getWeight(_index);
		_value = activation(activate + _bias);
	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << "\n";
	}
}

void	Output::randomBias(void)
{
	std::srand(std::time(NULL));
	_bias = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) * 2 - 1;
}