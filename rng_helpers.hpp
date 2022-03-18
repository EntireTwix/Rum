#pragma once
#include "third_party/pcg32_8.h"

namespace rum
{
    static pcg32_8 gen; // Global generator
    
    // Not thread-safe
    float GenerateFloat()
    {
        static uint_fast8_t counter = 0;
        static float rand_cache[8];
        if(counter == 0)
        {
            gen.nextFloat(rand_cache);
            counter = 8;
        }
        return rand_cache[--counter];
    }

    template <float LOW = 0, float HIGH = 1>
    class BasicGen
    {
        static_assert(HIGH > LOW);
    public:
        BasicGen() noexcept = default;
        float Generate() { return (GenerateFloat() * (HIGH - LOW)) + LOW; }
    };
    template <>
    class BasicGen<0.0f, 1.0f>
    {
    public:
        BasicGen() noexcept = default;
        float Generate() { return GenerateFloat(); }
    };
};