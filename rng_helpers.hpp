#pragma once
#include <cassert>
#include <cstdint>
#include "third_party/pcg32_8.h"

namespace rum
{
    pcg32_8 gen; // global generator
    
    float GenerateFloat()
    {
        static uint_fast8_t counter = 0;
        static float rand_cache[8];
        if (counter == 0)
        {
            // std::cout << "generating new floats\n";
            gen.nextFloat(rand_cache);
            counter = 8;
        }
        // std::cout << counter - 1 << '\n';
        return rand_cache[--counter];
    }

    class BasicGen
    {
    private:
        const float _low, _high;
    public:
        constexpr BasicGen(float low = 0, float high = 1) noexcept : _low(low), _high(high) { assert(this->_high > this->_low); }
        float operator()() const { return (GenerateFloat() * (this->_high - this->_low)) + this->_low; }
    }; 

    //for Sigmoid & Tanh
    template <size_t N>
    struct XavierWeight : BasicGen { constexpr XavierWeight() : BasicGen(-(1.0f/std::sqrt(N)), (1.0f/std::sqrt(N))) {}; };
    template <size_t N, size_t M>
    struct NormXavierWeight : BasicGen { constexpr NormXavierWeight() : BasicGen(-(std::sqrt(6.0f)/std::sqrt(N * M)), std::sqrt(6)/std::sqrt(N * M)) {}; };
    
    //for Relu
    template <size_t N>
    struct He : BasicGen { constexpr He() : BasicGen(0, std::sqrt(2.0f/N)) {} };
};
