#ifndef HIDDENCELL_HPP

# define HIDDENCELL_HPP

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
		float			_z;

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
		std::vector<float>	getWeights(void) const { return _weights; }
		size_t			getIndex(void) const { return _index; }
		float			getZ(void) const { return _z; }
		void			setIndex(const size_t& index) { _index = index; }
		void			setWeight(const size_t& index, const float& weight);
		void			setBias(const float& bias) { _bias = bias; }


		class			OutOfRange : public std::exception 	
		{
			const char	*what(void) const throw();
		};
};

#endif