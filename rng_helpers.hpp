#pragma once
#include <cassert>
#include <cstdint>
#include "third_party/pcg32_8.h"

namespace rum
{
    thread_local pcg32_8 gen; // global generator
    
    // thread-safe
    float GenerateFloat()
    {
        thread_local uint_fast8_t counter = 0;
        thread_local float rand_cache[8];
        if(counter == 0)
        {
            gen.nextFloat(rand_cache);
            counter = 8;
        }
        return rand_cache[--counter];
    }

    class BasicGen
    {
    private:
        const float _low, _high;
    public:
        BasicGen(float low = 0.0f, float high = 1.0f) noexcept : _low(low), _high(high) { assert(this->_high > this->_low); }
        float operator()() const { return (GenerateFloat() * (this->_high - this->_low)) + this->_low; }
    }; 

    //for Sigmoid & Tanh
    template <size_t N>
    struct XavierWeight : BasicGen { XavierWeight() : BasicGen(-(1.0f/std::sqrt(N)), (1.0f/std::sqrt(N))) {}; };
    template <size_t N, size_t M>
    struct NormXavierWeight : BasicGen { NormXavierWeight() : BasicGen(-(std::sqrt(6.0f)/std::sqrt(N * M)), std::sqrt(6)/std::sqrt(N * M)) {}; };
    
    //for Relu
    template <size_t N>
    struct He : BasicGen { He() : BasicGen(0, std::sqrt(2.0f/N)) {} };
};
