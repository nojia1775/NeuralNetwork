#ifndef OUTPUT_HPP

# define OUTPUT_HPP

# include <iostream>
# include <cstdlib>
# include <ctime>
# include <cmath>
# include <vector>
# include "Input.hpp"
# include "HiddenCell.hpp"

# pragma once

class	Output
{
	private:
		float		_value;
		float		_bias;
		size_t		_index;

	public:
				Output(const size_t& index);
				Output(const Output& other);
		Output&		operator=(const Output& other);

		void		computeValue(const std::vector<HiddenCell>& hiddenCells, float (*activation)(float));
		void		computeValue(const std::vector<Input>& inputs, float (*activation)(float));
		void		randomBias(void);

		float		getValue(void) const { return _value; }
		float		getBias(void) const { return _bias; }
		float		getIndex(void) const { return _index; }
		void		setIndex(const size_t& index) { _index = index; }
};

#endif