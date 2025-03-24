#include "../include/Output.hpp"
#include "../include/NeuralNetwork.hpp"

Output::Output(const size_t& index) : _value(0), _bias(0), _index(index), _z(0) {}

Output::Output(const Output& other) : _value(other.getValue()), _bias(other.getBias()), _index(other.getIndex()), _z(other.getZ()) {}

Output&	Output::operator=(const Output& other)
{
	if (this != &other)
	{
		_value = other.getValue();
		_bias = other.getBias();
		_index = other.getIndex();
		_z = other.getZ();
	}
	return *this;
}

void	Output::computeValue(const std::vector<HiddenCell>& hiddenCells, float (*activation)(const float&))
{
	float activate = 0;
	try
	{
		for (auto cell : hiddenCells)
			activate += cell.getValue() * cell.getWeight(_index);
		_z = activate + _bias;
		_value = activation(_z);
		if (DEBUG)
		{
			std::cout << "activate = somme(cell * cell.weight) ->\n";
			std::cout << "activate" << " = " << activate << "\n";
			std::cout << "Z = activate + bias ->\n" << _z << " = " << activate << " + " << _bias << "\n";
			std::cout << "value = function(activate) ->\nvalue = " << _value << "\n\n";
		}
	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << "\n";
	}
}

void	Output::computeValue(const std::vector<HiddenCell>& hiddenCells)
{
	float activate = 0;
	try
	{
		for (auto cell : hiddenCells)
			activate += cell.getValue() * cell.getWeight(_index);
		_z = activate + _bias;
		if (DEBUG)
		{
			std::cout << "activate = somme(cell * cell.weight) ->\n";
			std::cout << "activate" << " = " << activate << "\n";
			std::cout << "Z = activate + bias ->\n" << _z << " = " << activate << " + " << _bias << "\n";
		}
	}
	catch (const std::exception& e) { std::cout << e.what() << "\n"; }
}

void	Output::computeValue(const std::vector<Input>& inputs, float (*activation)(const float&))
{
	float activate = 0;
	try
	{
		for (auto input : inputs)
			activate += input.getValue() * input.getWeight(_index);
		_z = activate + _bias;
		_value = activation(_z);
	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << "\n";
	}
}

void	Output::computeValue(const std::vector<Input>& inputs)
{
	float activate = 0;
	try
	{
		for (auto input : inputs)
			activate += input.getValue() * input.getWeight(_index);
		_z = activate + _bias;
	}
	catch (const std::exception& e) { std::cout << e.what() << "\n"; }
}

void	Output::randomBias(void)
{
	_bias = 0;
	while (_bias == 0)
		_bias = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) * 2 - 1;
}