#pragma once
#include <type_traits>

namespace rum
{
    constexpr void SoftMaxMut(float* ptr, size_t sz)
    {
        float m, sum, constant;
        size_t i;

        m = 0;
        for (i = 0; i < sz; ++i) 
        {
            if (m < ptr[i]) 
            {
                m = ptr[i];
            }
        }

        sum = 0;
        for (i = 0; i < sz; ++i) 
        {
            sum += exp(ptr[i] - m);
        }

        constant = m + log(sum);
        for (i = 0; i < sz; ++i) 
        {
            ptr[i] = exp(ptr[i] - constant);
        }
    }
}