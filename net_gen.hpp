#pragma once
#include "third_party/Matrix/std/ml_ops.hpp"

#define LAYER_N(name, number) name##_layer_##number

#define RUM_FF(name, input_size, hidden_size, output_size)\
IMat<input_size> LAYER_N(name, 0);                        \
WMat<input_size, hidden_size> LAYER_N(name, 1);           \
BMat<hidden_size> LAYER_N(name, 2);                       \
WMat<hidden_size, output_size> LAYER_N(name, 3);          \
BMat<output_size> LAYER_N(name, 4);                       \
OMat<output_size> LAYER_N(name, 5);                       \
OMat<hidden_size> LAYER_N(name, 6);                       \
OMat<hidden_size> LAYER_N(name, 7);                       \
OMat<output_size> LAYER_N(name, 8);                       \
OMat<output_size> LAYER_N(name, 9);                       