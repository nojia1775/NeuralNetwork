#include "../include/HiddenCell.hpp"

HiddenCell::HiddenCell(const size_t& nbrWeights, const size_t& index)
{
	_bias = 0;
	_value = 0;
	_index = index;
	for (size_t i = 0 ; i < nbrWeights ; i++)
		_weights.push_back(0);
}

HiddenCell::HiddenCell(const HiddenCell& other)
{
	_value = other.getValue();
	_bias = other.getBias();
	_index = other.getIndex();
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
		for (size_t i = 0 ; i < other.getNbrWeights() ; i++)
			_weights[i] = other._weights[i];
	}
	return *this;
}

void	HiddenCell::randomWeights(void)
{
	std::srand(std::time(NULL));
	for (size_t i = 0 ; i < getNbrWeights() ; i++)
		_weights[i] = std::rand() / RAND_MAX * 2 - 1;
	_bias = std::rand() / RAND_MAX * 2 - 1;
}
float	HiddenCell::getWeight(const size_t& index) const
{
	if (index < getNbrWeights())
		return _weights[index];
	throw OutOfRange();
	return -1.0;
}

void	HiddenCell::computeValue(const std::vector<Input>& inputs, float (*activation)(float))
{
	float activate = 0;
	try
	{
		for (auto input : inputs)
			activate += input.getValue() * input.getWeight(_index);
		std::cout << "activate et bias hidden layer -> " << activate << " " << _bias << "\n";
		_value = activation(activate + _bias);
	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << "\n";
	}
}

void	HiddenCell::computeValue(const std::vector<HiddenCell>& cells, float (*activation)(float))
{
	float activate = 0;
	try
	{
		for (auto cell : cells)
			activate += cell.getValue() * cell.getWeight(_index);
		_value = activation(activate + _bias);
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