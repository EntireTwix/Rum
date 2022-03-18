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
};
