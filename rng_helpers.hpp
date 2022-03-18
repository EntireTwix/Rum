#pragma once
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

    template <float LOW = 0.0f, float HIGH = 1.0f>
    struct BasicGen
    {
        static_assert(HIGH > LOW);

        BasicGen() noexcept = default;
        float Generate() const { return (GenerateFloat() * (HIGH - LOW)) + LOW; }
    };
    template <>
    struct BasicGen<0.0f, 1.0f>
    {
        BasicGen() noexcept = default;
        float Generate() const { return GenerateFloat(); }
    };
};
