#pragma once

#include <curand.h>
#include <curand_kernel.h>

typedef curandStatePhilox4_32_10_t curtRNGState;
// typedef curandStateXORWOW_t curtRNGState; // bad
// typedef curandStateMRG32k3a_t curtRNGState // not tested
// typedef curandStateSobol32_t curtRNGState; // not tested
// typedef curandStateScrambledSobol32_t curtRNGState; // not tested
