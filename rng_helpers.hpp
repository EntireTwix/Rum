#pragma once
#include "third_party/pcg32_8.h"

namespace rum
{
    
    // thread-safe
    float GenerateFloat()
    {
        thread_local pcg32_8 gen; // global generator
        thread_local uint_fast8_t counter = 0;
        thread_local float rand_cache[8];
        if(counter == 0)
        {
            gen.nextFloat(rand_cache);
            counter = 8;
        }
        return rand_cache[--counter];
    }

    struct BasicGen
    {
        float _low, _high;
        BasicGen(float low = 0.0f, float high = 1.0f) noexcept : _low(low), _high(high) { assert(_high > _low); }
        float Generate() const { return (GenerateFloat() * (_high - _low)) + _low; }
    };
};
