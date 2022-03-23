#pragma once
#include "third_party/Matrix/std/ml_ops.hpp"

#define LAYER_N(net_name, name) net_name##_layer_##name

#define RUM_FF(name, input_size, hidden_size, output_size)\
IMat<input_size> LAYER_N(name, inp);                      \
WMat<input_size, hidden_size> LAYER_N(name, w1);          \
BMat<hidden_size> LAYER_N(name, b1);                      \
WMat<hidden_size, output_size> LAYER_N(name, w2);         \
BMat<output_size> LAYER_N(name, b2);                      \
OMat<output_size> LAYER_N(name, ans);                     \
OMat<hidden_size> LAYER_N(name, w1o);                     \
OMat<hidden_size> LAYER_N(name, h1o);                     \
OMat<output_size> LAYER_N(name, w2o);                     \
OMat<output_size> LAYER_N(name, out);                     \
BMat<output_size> LAYER_N(name, out_error);               \
WMat<hidden_size, output_size> LAYER_N(name, w2_error);   \
BMat<hidden_size> LAYER_N(name, h1_error);                \
WMat<input_size, hidden_size> LAYER_N(name, w1_error);    