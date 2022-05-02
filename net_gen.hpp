#pragma once
#include "third_party/Matrix/std/ml_ops.hpp"

#define LAYER_N(net_name, name) net_name##_layer_##name         

#define RUM_FF(name, input_size, hidden_size, output_size)\
mat::WMat<input_size, hidden_size> LAYER_N(name, w1);          \
mat::BMat<hidden_size> LAYER_N(name, b1);                      \
mat::WMat<hidden_size, output_size> LAYER_N(name, w2);         \
mat::BMat<output_size> LAYER_N(name, b2);                      \
mat::OMat<output_size> LAYER_N(name, ans);                     \
mat::OMat<hidden_size> LAYER_N(name, w1o);                     \
mat::OMat<hidden_size> LAYER_N(name, h1o);                     \
mat::OMat<output_size> LAYER_N(name, w2o);                     \
mat::OMat<output_size> LAYER_N(name, out);                     \
mat::BMat<output_size> LAYER_N(name, out_error);               \
mat::WMat<hidden_size, output_size> LAYER_N(name, w2_error);   \
mat::BMat<hidden_size> LAYER_N(name, h1_error);                \
mat::WMat<input_size, hidden_size> LAYER_N(name, w1_error);    

#define RUM_FF_F(name, input, h1_a, out_a)                                                     \
LAYER_N(name, w1o) = WeightForward(input, LAYER_N(name, w1), LAYER_N(name, b1));               \
LAYER_N(name, h1o) = HiddenForward(LAYER_N(name, w1o), h1_a);                                  \
LAYER_N(name, w2o) = WeightForward(LAYER_N(name, h1o), LAYER_N(name, w2), LAYER_N(name, b2));  \
LAYER_N(name, out) = HiddenForward(LAYER_N(name, w2o), out_a);                               

#define RUM_FF_B(name, input, h1_ap, out_ap)                                                                     \
LAYER_N(name, out_error) = OutputBackward(LAYER_N(name, out), LAYER_N(name, ans), LAYER_N(name, w2o), out_ap);   \
LAYER_N(name, w2_error) = WeightBackward(LAYER_N(name, out_error), LAYER_N(name, h1o));                          \
LAYER_N(name, h1_error) = HiddenBackward(LAYER_N(name, out_error), LAYER_N(name, w2), LAYER_N(name, w1o), h1_ap);\
LAYER_N(name, w1_error) = WeightBackward(LAYER_N(name, h1_error), input);                        

#define RUM_FF_LEARN(name, learning_rate)                         \
Learn(LAYER_N(name, b2), LAYER_N(name, out_error), learning_rate);\
Learn(LAYER_N(name, w2), LAYER_N(name, w2_error), learning_rate); \
Learn(LAYER_N(name, b1), LAYER_N(name, h1_error), learning_rate); \
Learn(LAYER_N(name, w1), LAYER_N(name, w1_error), learning_rate);      

#define RUM_AE(name, input_size, hidden_size)         \
mat::WMat<input_size, hidden_size> LAYER_N(name, w1);      \
mat::BMat<hidden_size> LAYER_N(name, b1);                  \
mat::WMat<hidden_size, input_size> LAYER_N(name, w2);      \
mat::BMat<input_size> LAYER_N(name, b2);                   \
mat::OMat<hidden_size> LAYER_N(name, w1o);                 \
mat::OMat<hidden_size> LAYER_N(name, h1o);                 \
mat::OMat<input_size> LAYER_N(name, w2o);                  \
mat::OMat<input_size> LAYER_N(name, out);                  \
mat::BMat<input_size> LAYER_N(name, out_error);            \
mat::WMat<hidden_size, input_size> LAYER_N(name, w2_error);\
mat::BMat<hidden_size> LAYER_N(name, h1_error);            \
mat::WMat<input_size, hidden_size> LAYER_N(name, w1_error);    

#define RUM_AE_F(name, input, h1_a, out_a) RUM_FF_F(name, input, h1_a, out_a)                              

#define RUM_AE_B(name, input, h1_ap, out_ap)                                                                     \
LAYER_N(name, out_error) = OutputBackward(LAYER_N(name, out), input, LAYER_N(name, w2o), out_ap);                \
LAYER_N(name, w2_error) = WeightBackward(LAYER_N(name, out_error), LAYER_N(name, h1o));                          \
LAYER_N(name, h1_error) = HiddenBackward(LAYER_N(name, out_error), LAYER_N(name, w2), LAYER_N(name, w1o), h1_ap);\
LAYER_N(name, w1_error) = WeightBackward(LAYER_N(name, h1_error), input);                        

#define RUM_AE_LEARN(name, learning_rate) RUM_FF_LEARN(name, learning_rate) 
