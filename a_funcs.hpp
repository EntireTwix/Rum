#pragma once
#include <cmath>
#include "third_party/Matrix/std/dependencies/pow.hpp"

namespace rum
{
    //relu
    constexpr float relu(float x) { return (x > 0) * x; }
    constexpr float relu_prime(float x) { return x > 0; }

    //leaky relu
    constexpr float relu_leaky(float x) { if (x < 0) { return x * 0.01f; } else { return x; } }
    constexpr float relu_leaky_prime(float x) { if (x < 0) { return 0.01f; } else { return 1.0f; } }

    //tanh
    constexpr float tanh_prime(float x) { return 1 - pow2<float>(tanh(x)); }

    //sigmoid
    constexpr float sigmoid(float x) { return 1 / (1 + exp(-x)); }
    constexpr float sigmoid_prime(float x) { float temp = exp(-x); return temp / pow2<float>(1 + temp); }

    //swish
    constexpr float swish(float x) { return x * sigmoid(x); }
    constexpr float swish_prime(float x) { return (exp(x) * (exp(x) + x + 1)) / pow2<float>(exp(x) + 1);}
}
