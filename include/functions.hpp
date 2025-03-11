#ifndef FUNCTIONS_HPP

# define FUNCTIONS_HPP

# include <iostream>
# include <cmath>

// fonctions d'activaiton

float	ReLU(float x) { return x <= 0 ? 0 : x; }
float	derivatedReLU(float x) { return x <= 0 ? 0 : 1; }

float	leakyReLU(float x) { return x <= 0 ? x * 0.01 : x; }
float	derivatedLeakyReLU(float x) { return x <= 0 ? 0.01 : 1; }

float	sigmoid(float x) { return 1 / (1 + exp(-x)); }
float	derivatedSigmoid(float x) { return sigmoid(x) * (1 - sigmoid(x)); }

float	identity(float x) { return x; }
float	derivatedIdentity(float x) { return x / abs(x); }

float	tanH(float x) { return (exp(x) - exp(-x)) / (exp(x) + exp(-x)); }

float	derivatedTanH(float x) { return 1 - pow(tanH(x), 2); }

// fonctions de perte

float	BCE(float yPred, float yTrue) { return - yTrue * log(yPred) - (1 - yTrue) * log(1 - yPred); }
float	derivatedBCE(float yPred, float yTrue) { return - yTrue / (yPred + 1e-7) + (1 - yTrue) / (1 - yPred + 1e-7); }

float	MSE(float yPred, float yTrue) { return pow(yPred - yTrue, 2) / 2; }
float	derivatedMSE(float yPred, float yTrue) { return yPred - yTrue; }

#endif