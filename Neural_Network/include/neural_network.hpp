#ifndef NEURAL_NETWORK_HPP

# define  NEURAL_NETWORK_HPP

# include <iostream>
# include <cstdlib>
# include <ctime>
# include <cmath>
# include <vector>

class	NN
{
	private:
		size_t			_inputs;
		size_t			_outputs;
		size_t			_hidden_layers;
		size_t			_cells_per_layers;
		std::vector<float>	_inputs_values;
		std::vector<float>	_w;
};

#endif