#ifndef INPUT_HPP

# define INPUT_HPP

# include <iostream>
# include <cstdlib>
# include <cmath>
# include <ctime>
# include <vector>

#pragma once


class	Input
{
	private:
		float			_value;
		std::vector<float>	_weights;

	public:
					Input(const size_t& nbr_weight);
					Input(const Input& other);
		Input&			operator=(const Input& other);

		void			randomWeights(void);
		void			initWeights(const std::vector<float>& weights);

		size_t			getNbrWeights(void) const { return _weights.size(); };
		float			getValue(void) const { return _value; };

		const float&		operator[](const size_t& index) const;

		class			OutOfRange : public std::exception 	
		{
			const char	*what(void) const throw();
		};

		class			DifferentNumberOfWeights : public std::exception
		{
			const char	*what(void) const throw();
		};
};

#endif