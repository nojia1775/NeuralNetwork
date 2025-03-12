#include "../include/Input.hpp"

Input::Input(const size_t& nbr_weights)
{
	_value = 0;
	for (size_t i = 0 ; i < nbr_weights ; i++)
		_weights.push_back(0);
}

Input::Input(const Input& other)
{
	_value = other.getValue();
	try
	{
		for (size_t i = 0 ; i < other.getNbrWeights() ; i++)
			_weights.push_back(other.getWeight(i));
	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << "\n";
	}
}

Input&	Input::operator=(const Input& other)
{
	if (this != &other)
	{
		_value = other.getValue();
		_weights.clear();
		for (size_t i = 0 ; i < other.getNbrWeights() ; i++)
			_weights.push_back(other.getWeight(i));
	}
	return *this;
}

float	Input::getWeight(const size_t& index) const
{
	if (index < getNbrWeights())
		return _weights[index];
	throw OutOfRange();
	return -1.0;
}

void	Input::randomWeights(void)
{
	for (size_t i = 0 ; i < getNbrWeights() ; i++)
	{
		_weights[i] = 0;
		while (_weights[i] == 0)
			_weights[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) * 2 - 1;
	}
}

void	Input::initWeights(const std::vector<float>& weights)
{
	if (weights.size() != _weights.size())
	{
		throw DifferentNumberOfWeights();
		return;
	}
	for (size_t i = 0 ; i < weights.size() ; i++)
		_weights[i] = weights[i];
}

const char	*Input::OutOfRange::what(void) const throw() { return "Index out of range\n"; }

const char	*Input::DifferentNumberOfWeights::what(void) const throw() { return "The array has a different size\n"; }

void	Input::setWeight(const size_t& index, const float& weight)
{
	if (index < getNbrWeights())
		_weights[index] = weight;
	else
		throw OutOfRange();
}