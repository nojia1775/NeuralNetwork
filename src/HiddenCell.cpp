#include "HiddenCell.hpp"
#include "NeuralNetwork.hpp"

HiddenCell::HiddenCell(const size_t& nbrWeights, const size_t& index)
{
	_bias = 0;
	_value = 0;
	_index = index;
	_z = 0;
	for (size_t i = 0 ; i < nbrWeights ; i++)
		_weights.push_back(0);
}

HiddenCell::HiddenCell(const HiddenCell& other)
{
	_value = other.getValue();
	_bias = other.getBias();
	_index = other.getIndex();
	_z = other.getZ();
	for (size_t i = 0 ; i < other.getNbrWeights() ; i++)
		_weights.push_back(other.getWeight(i));
}

HiddenCell&	HiddenCell::operator=(const HiddenCell& other)
{
	_weights.clear();
	if (this != &other)
	{
		_value = other.getValue();
		_bias = other.getBias();
		_index = other.getIndex();
		_z = other.getZ();
		for (size_t i = 0 ; i < other.getNbrWeights() ; i++)
			_weights[i] = other._weights[i];
	}
	return *this;
}

void	HiddenCell::randomWeights(void)
{
	for (size_t i = 0 ; i < getNbrWeights() ; i++)
	{
		_weights[i] = 0;
		while (_weights[i] == 0)
			_weights[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) * 2 - 1;
	}
	_bias = 0;
	while (_bias == 0)
		_bias = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) * 2 - 1;
}
float	HiddenCell::getWeight(const size_t& index) const
{
	if (index < getNbrWeights())
		return _weights[index];
	throw OutOfRange();
	return -1.0;
}

void	HiddenCell::computeValue(const std::vector<Input>& inputs, float (*activation)(const float&))
{
	float activate = 0;
	try
	{
		for (auto input : inputs)
			activate += input.getValue() * input.getWeight(_index);
		_z = activate + _bias;
		_value = activation(_z);
		if (DEBUG)
		{
			std::cout << "activate = somme(input * input.weight) ->\n";
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

void	HiddenCell::computeValue(const std::vector<HiddenCell>& cells, float (*activation)(const float&))
{
	float activate = 0;
	try
	{
		for (auto cell : cells)
			activate += cell.getValue() * cell.getWeight(_index);
		_z = activate + _bias;
		_value = activation(_z);
	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << "\n";
	}
}

const char	*HiddenCell::OutOfRange::what(void) const throw() { return "Index out of range\n"; }

void	HiddenCell::setWeight(const size_t& index, const float& weight)
{
	if (index < getNbrWeights())
		_weights[index] = weight;
	else
		throw OutOfRange();
}