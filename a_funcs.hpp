#pragma once
#include <cmath>

namespace rum
{    
    //relu
    constexpr float relu(float x) { return (x > 0) * x; }
    constexpr float relu_prime(float x) { return x > 0; }

    //leaky relu
    constexpr float relu_leaky(float x) { if (x < 0) { return x * 0.0001; } else { return x; } }
    constexpr float relu_leaky_prime(float x) { if (x < 0) { return 0.0001; } else { return 1; } }

    //tanh
    constexpr float tanh_prime(float x) { return 1 - std::pow(tanh(x), 2); }

    //sigmoid
    constexpr float sigmoid(float x) { return 1 / (1 + exp(-x)); }
    constexpr float sigmoid_prime(float x) { return exp(-x) / std::pow(1 + exp(-x), 2); }

    //swish
    constexpr float swish(float x) { return x * sigmoid(x); }
    constexpr float swish_prime(float x) { return (exp(x) * (exp(x) + x + 1)) / std::pow(exp(x) + 1, 2);}
}
