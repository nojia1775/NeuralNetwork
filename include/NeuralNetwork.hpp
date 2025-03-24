#ifndef NEURALNETWORK_HPP

# define  NEURALNETWORK_HPP

# define DEBUG 0

# include <iostream>
# include <cstdlib>
# include <ctime>
# include <cmath>
# include <vector>
# include <fstream>
# include "Json.hpp"
# include "Input.hpp"
# include "HiddenCell.hpp"
# include "Output.hpp"
# include "functions.hpp"

class	NN
{
	private:
		std::vector<Input>			_inputs;
		std::vector<std::vector<HiddenCell>>	_hiddenCells;
		std::vector<Output>			_outputs;
		float					_learningRate;
		size_t					_id;
		std::vector<float>			_loss;

		std::vector<float>			updateLastLayerWeights(float (*derivatedLoss)(const float&, const float&), float (*derivatedAtivO)(const float&), const std::vector<float>& targets);
		std::vector<float>			updateHiddenLayersWeights(float (*derivatedHL)(const float&), const std::vector<float>& dErrorsALastLayer);
		void					updateInputsWeights(float (*derivatedHL)(const float&), const std::vector<float>& dErrorsAFirstLayer);
		std::vector<float>			updateLastLayerWeightsMultiClass(const std::vector<float>& targets);

	public:
							NN(const size_t& nbrInputs, const size_t& nbrHiddenLayers, const size_t& nbrHiddenCells, const size_t& nbrOutputs);
							NN(const std::string& fileName);
							NN(const NN& other);
		NN&					operator=(const NN& other);

		void					displayValues(void) const;
		void					initNN(void);
		void					feedForward(float (*activHL)(const float&), float (*activO)(const float&));
		void					feedForwardMultiClass(float (*f)(const float&));
		void					initInputs(const std::vector<float>& inputs);
		float					backPropagation(float (*loss)(const float&, const float&), float (*derivatedLoss)(const float&, const float&), float (*derivatedActivHL)(const float&), float (*derivatedActivO)(const float&), const std::vector<float>& targets);
		float					backPropagationMultiClass(float (*f)(const float&), float (*derivatedF)(const float&), const std::vector<float>& targets);
		void					train(const size_t& epochs, const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& expectedOutputs, float (*loss)(const float&, const float&), float (*derivatedLoss)(const float&, const float&), float (*func1)(const float&), float (*func2)(const float&), float (*derivFunc1)(const float&), float (*derivFunc2)(const float&));
		void					trainMultiClass(const size_t& epochs, const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& expectedOutputs, float (*f)(const float&), float (*derivatedF)(const float&));
		std::vector<float>			use(const std::vector<float>& inputs, float (*activHL)(const float&), float (*activO)(const float&));
		std::vector<float>			useMultiClass(const std::vector<float>& inputs, float (*f)(const float&));

		size_t					getNbrInputs(void) const { return _inputs.size(); }
		size_t					getNbrHiddenLayers(void) const { return _hiddenCells.size(); }
		size_t					getNbrHiddenCells(void) const { return _hiddenCells[0].size(); }
		size_t					getNbrOutputs(void) const { return _outputs.size(); }
		std::vector<float>			getOutputs(void) const;
		float					getLearningRate(void) const { return _learningRate; }
		void					getJSON(void) const;
		void					getJSON(const std::string& name) const;
		size_t					getId(void) const { return _id; }
		std::vector<float>			getHiddenLayersBias(const size_t& layer) const;
		std::vector<float>			getOutputsBias(void) const;
		std::vector<float>			getLoss(void) const { return _loss; }
		void					setLearningRate(float learningRate) { _learningRate = learningRate; }
};

#endif