#ifndef FUNCTIONS_HPP

# define FUNCTIONS_HPP

# include <iostream>
# include <cmath>
# include <vector>
# include <algorithm>

# define EPSILON 1e-9

// fonctions d'activaiton

inline float	ReLU(const float& x) { return x <= 0 ? 0 : x; }
inline float	derivatedReLU(const float& x) { return x <= 0 ? 0 : 1; }

inline float	leakyReLU(const float& x) { return x <= 0 ? x * 0.01 : x; }
inline float	derivatedLeakyReLU(const float& x) { return x <= 0 ? 0.01 : 1; }

inline float	sigmoid(const float& x) { return 1 / (1 + exp(-x)); }
inline float	derivatedSigmoid(const float& x) { return sigmoid(x) * (1 - sigmoid(x)); }

inline float	identity(const float& x) { return x; }
inline float	derivatedIdentity(const float& x) { return x / std::abs(x); }

inline float	tanH(const float& x) { return (exp(x) - exp(-x)) / (exp(x) + exp(-x)); }
inline float	derivatedTanH(const float& x) { return 1 - pow(tanH(x), 2); }

std::vector<float>	softMax(const std::vector<float>& input);
std::vector<float>	derivatedSoftMax(const std::vector<float>& inputs);

// fonctions de perte

inline float	BCE(const float& yPred, const float& yTrue) { return - yTrue * log(yPred) - (1 - yTrue) * log(1 - yPred); }
inline float	derivatedBCE(const float& yPred, const float& yTrue) { return - yTrue / (yPred + 1e-7) + (1 - yTrue) / (1 - yPred + 1e-7); }

inline float	MSE(const float& yPred, const float& yTrue) { return pow(yPred - yTrue, 2) / 2; }
inline float	derivatedMSE(const float& yPred, const float& yTrue) { return yPred - yTrue; }

float	crossEntropy(const std::vector<float>& yPred, const std::vector<float>& yTrue);
float	derivatedCrossEntropy(const std::vector<float>& yPred, const std::vector<float>& yTrue);

#endif