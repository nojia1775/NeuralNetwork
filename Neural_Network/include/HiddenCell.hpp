#ifndef HIDDENCELL_HPP

# define HIDDDENCELL_HPP

# include <iostream>
# include <cstdlib>
# include <ctime>
# include <cmath>
# include <vector>
# include "Input.hpp"

# pragma once

class	HiddenCell
{
	private:
		float			_value;
		std::vector<float>	_weights;
		float			_bias;
		size_t			_index;

	public:
					HiddenCell(const size_t& nbrWeights, const size_t& index);
					HiddenCell(const HiddenCell& other);
		HiddenCell&		operator=(const HiddenCell& other);

		void			randomWeights(void);
		void			computeValue(const std::vector<Input>& inputs, float (*activation)(float));
		void			computeValue(const std::vector<HiddenCell>& cells, float (*activation)(float));

		float			getValue(void) const { return _value; }
		float			getBias(void) const { return _bias; }
		size_t			getNbrWeights(void) const { return _weights.size(); }
		float			getWeight(const size_t& index) const;
		size_t			getIndex(void) const { return _index; }
		void			setIndex(const size_t& index) { _index = index; }

		class			OutOfRange : public std::exception 	
		{
			const char	*what(void) const throw();
		};
};

#endif