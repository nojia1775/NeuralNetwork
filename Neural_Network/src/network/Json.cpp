#include "../../include/NeuralNetwork.hpp"

void	NN::getJSON(void) const
{
	std::string fileName("nn" + std::to_string(_id) + ".json");
	nlohmann::json jsonData;
	jsonData["inputs"] = getNbrInputs();
	jsonData["hidden_layers"] = getNbrHiddenLayers();
	jsonData["hidden_neurals"] = getNbrHiddenCells();
	jsonData["outputs"] = getNbrOutputs();
	jsonData["layers"] = nlohmann::json::array();

	nlohmann::json inputsData;
	inputsData["type"] = "inputs";
	inputsData["weights"] = nlohmann::json::array();
	for (size_t i = 0 ; i < getNbrInputs() ; i++)
		inputsData["weights"].push_back(_inputs[i].getWeights());
	jsonData["layers"].push_back(inputsData);

	nlohmann::json hiddenData;
	for (size_t i = 0 ; i < getNbrHiddenLayers() ; i++)
	{
		hiddenData["bias"] = getHiddenLayersBias(i);
		hiddenData["type"] = "hidden_layer_" + std::to_string(i);
		hiddenData["weights"] = nlohmann::json::array();
		for (size_t j = 0 ; j < getNbrHiddenCells() ; j++)
			hiddenData["weights"].push_back(_hiddenCells[i][j].getWeights());
		jsonData["layers"].push_back(hiddenData);
	}

	nlohmann::json outputsData;
	outputsData["type"] = "outputs";
	outputsData["bias"] = getOutputsBias();
	jsonData["layers"].push_back(outputsData);
	jsonData["loss"] = {getLoss()};
	std::ofstream file("nn" + std::to_string(_id) + ".json");
	if (file.is_open())
	{
		file << jsonData.dump(4);
		std::cout << "Log saved in " << fileName << "\n";
		file.close();
	}
	else
		std::cout << "Error\n";
}

void	NN::getJSON(const std::string& fileName) const
{
	nlohmann::json jsonData;
	jsonData["inputs"] = getNbrInputs();
	jsonData["hidden_layers"] = getNbrHiddenLayers();
	jsonData["hidden_neurals"] = getNbrHiddenCells();
	jsonData["outputs"] = getNbrOutputs();
	jsonData["layers"] = nlohmann::json::array();

	nlohmann::json inputsData;
	inputsData["type"] = "inputs";
	inputsData["weights"] = nlohmann::json::array();
	for (size_t i = 0 ; i < getNbrInputs() ; i++)
		inputsData["weights"].push_back(_inputs[i].getWeights());
	jsonData["layers"].push_back(inputsData);

	nlohmann::json hiddenData;
	for (size_t i = 0 ; i < getNbrHiddenLayers() ; i++)
	{
		hiddenData["bias"] = getHiddenLayersBias(i);
		hiddenData["type"] = "hidden_layer_" + std::to_string(i);
		hiddenData["weights"] = nlohmann::json::array();
		for (size_t j = 0 ; j < getNbrHiddenCells() ; j++)
			hiddenData["weights"].push_back(_hiddenCells[i][j].getWeights());
		jsonData["layers"].push_back(hiddenData);
	}

	nlohmann::json outputsData;
	outputsData["type"] = "outputs";
	outputsData["bias"] = getOutputsBias();
	jsonData["layers"].push_back(outputsData);
	jsonData["loss"] = {getLoss()};
	std::ofstream file(fileName);
	if (file.is_open())
	{
		file << jsonData.dump(4);
		std::cout << "Log saved in " << fileName << "\n";
		file.close();
	}
	else
		std::cout << "Error\n";
}

NN::NN(const std::string& fileName)
{
	std::ifstream file(fileName);
	if (!file.is_open())
	{
		std::cout << "Impossible to open " << fileName << "\n";
		return;
	}

	nlohmann::json jsonData;
	try
	{
		file >> jsonData;
	}
	catch (const nlohmann::json::parse_error& e)
	{
		std::cout << e.what() << "\n";
	}
	size_t nbrInputs = jsonData["inputs"];
	size_t nbrHiddenLayers = jsonData["hidden_layers"];
	size_t nbrHiddenNeurals = jsonData["hidden_neurals"];
	size_t nbrOutputs = jsonData["outputs"];

	for (size_t i = 0 ; i < nbrInputs ; i++)
		_inputs.push_back(Input(nbrHiddenNeurals));
	
	for (size_t i = 0 ; i < nbrHiddenLayers ; i++)
	{
		std::vector<HiddenCell> layer;
		for (size_t j = 0 ; j < nbrHiddenNeurals ; j++)
		{
			if (i == nbrHiddenLayers - 1)
				layer.push_back(HiddenCell(nbrOutputs, j));
			else
				layer.push_back(HiddenCell(nbrHiddenNeurals, j));
		}
		_hiddenCells.push_back(layer);
		layer.clear();
	}

	for (size_t i = 0 ; i < nbrOutputs ; i++)
		_outputs.push_back(Output(i));

	auto layers = jsonData["layers"];
	size_t index = 0;
	for (auto& layer : layers)
	{
		if (layer["type"] == "inputs")
		{
			for (size_t i = 0 ; i < nbrInputs ; i++)
			{
				for (size_t j = 0 ; j < nbrHiddenNeurals ; j++)
					_inputs[i].setWeight(j, layer["weights"][i][j]);
			}
		}
		else if (layer["type"] == "hidden_layer_" + std::to_string(index - 1))
		{
			for (size_t i = 0 ; i < nbrHiddenNeurals ; i++)
			{
				if (index == nbrHiddenLayers)
				{
					for (size_t j = 0 ; j < nbrOutputs ; j++)
						_hiddenCells[index - 1][i].setWeight(j, layer["weights"][i][j]);
				}
				else
				{
					for (size_t j = 0 ; j < nbrHiddenNeurals ; j++)
						_hiddenCells[index - 1][i].setWeight(j, layer["weights"][i][j]);
				}
				_hiddenCells[index - 1][i].setBias(layer["bias"][i]);
			}
		}
		else if (layer["type"] == "outputs")
		{
			for (size_t i = 0 ; i < nbrOutputs ; i++)
				_outputs[i].setBias(layer["bias"][i]);
		}
		index++;
	}
}