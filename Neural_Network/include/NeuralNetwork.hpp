#ifndef NEURALNETWORK_HPP

# define  NEURALNETWORK_HPP

# include <iostream>
# include <cstdlib>
# include <ctime>
# include <cmath>
# include <vector>
# include "Input.hpp"
# include "HiddenCell.hpp"
# include "Output.hpp"

# pragma once

class	NN
{
	private:
		std::vector<Input>			_inputs;
		std::vector<std::vector<HiddenCell>>	_hiddenCells;
		std::vector<Output>			_outputs;
		float					_learningRate;

	public:
							NN(const size_t& nbrInputs, const size_t& nbrHiddenLayers, const size_t& nbrHiddenCells, const size_t& nbrOutputs);
							NN(const NN& other);
		NN&					operator=(const NN& other);

		void					displayValues(void) const;
		void					initNN(void);
		void					feedForward(float (*activHL)(float), float (*activO)(float));
		void					initInputs(const std::vector<float>& inputs);

		size_t					getNbrInputs(void) const { return _inputs.size(); }
		size_t					getNbrHiddenLayers(void) const { return _hiddenCells.size(); }
		size_t					getNbrHiddenCells(void) const { return _hiddenCells[0].size(); }
		size_t					getNbrOutputs(void) const { return _outputs.size(); }
		std::vector<float>			getOutputs(void) const;
		float					getLearningRate(void) const { return _learningRate; }
		void					setLearningRate(float learningRate) { _learningRate = learningRate; }
};

#endif