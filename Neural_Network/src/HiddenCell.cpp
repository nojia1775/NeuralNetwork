#include "../include/HiddenCell.hpp"

HiddenCell::HiddenCell(const size_t& nbrWeights)
{
	_bias = 0;
	_value = 0;
	for (size_t i = 0 ; i < nbrWeights ; i++)
		_weights.push_back(0);
}

HiddenCell::HiddenCell(const HiddenCell& other)
{
	_value = other.getValue();
	_bias = other.getBias();
	for (size_t i = 0 ; i < other.getNbrWeights() ; i++)
		_weights.push_back(other[i]);
}

HiddenCell&	HiddenCell::operator=(const HiddenCell& other)
{
	_weights.clear();
	if (this != &other)
	{
		_value = other.getValue();
		_bias = other.getBias();
		for ()
	}
}