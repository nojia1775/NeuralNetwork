#ifndef INPUT_HPP

# define INPUT_HPP

# include <iostream>
# include <cstdlib>
# include <ctime>
# include <cmath>
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
		void			initValue(const float& value) { _value = value; }

		size_t			getNbrWeights(void) const { return _weights.size(); };
		float			getValue(void) const { return _value; };
		float			getWeight(const size_t& index) const;
		void			setWeight(const size_t& index, const float& weight);
		
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