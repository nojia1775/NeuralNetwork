#ifndef HIDDENCELL_HPP

# define HIDDDENCELL_HPP

# include <iostream>
# include <cstdlib>
# include <ctime>
# include <cmath>
# include <vector>

#pragma once

class	HiddenCell
{
	private:
		float			_value;
		std::vector<float>	_weights;
		float			_bias;

	public:
					HiddenCell(const size_t& nbrWeights);
					HiddenCell(const HiddenCell& other);
		HiddenCell&		operator=(const HiddenCell& other);

		void			randomWeights(void);

		float			getValue(void) const { return _value; }
		float			getBias(void) const { return _bias; }
		size_t			getNbrWeights(void) const { return _weights.size(); }

		const float&		operator[](const size_t& index) const;

		class			OutOfRange : public std::exception 	
		{
			const char	*what(void) const throw();
		};
};

#endif