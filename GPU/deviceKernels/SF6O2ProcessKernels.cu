#include <cuda.h>
#include <cuda_runtime.h>

#include <context.hpp>

#include <array>

#define rho_Si 5.02e3        // in (1e15 atoms/cm³) (rho Si)
#define k_sigma_Si 300.      // in (1e15 cm⁻²s⁻¹)
#define beta_sigma_Si 5.0e-2 // in (1e15 cm⁻²s⁻¹)
#define MASK_MAT 0

extern "C" __global__ void calculateEtchRate(const NumericType *rates,
                                             const NumericType *coverages,
                                             const std::array<NumericType, 3> *coords,
                                             const NumericType *materialIds,
                                             NumericType *etchRate,
                                             const unsigned long numPoints,
                                             const NumericType totalIonFlux,
                                             const NumericType totalEtchantFlux,
                                             const NumericType totalOxygenFlux)
{
  unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride = blockDim.x * gridDim.x;

  const NumericType *ionSputteringRate = rates;
  const NumericType *ionEnhancedRate = rates + numPoints;
  const NumericType *eCoverage = coverages;

  for (; tidx < numPoints; tidx += stride)
  {
    if ((int)materialIds[tidx] != MASK_MAT)
    {
      etchRate[tidx] = -(1.0 / rho_Si) *
                       (k_sigma_Si * eCoverage[tidx] / 4. +
                        ionSputteringRate[tidx] * totalIonFlux +
                        eCoverage[tidx] * ionEnhancedRate[tidx] * totalIonFlux);
    }
    else
    {
      etchRate[tidx] = 0.;
    }
  }
}

extern "C" __global__ void updateCoverages(const NumericType *rates,
                                           NumericType *coverages,
                                           const unsigned long numPoints,
                                           const NumericType totalIonFlux,
                                           const NumericType totalEtchantFlux,
                                           const NumericType totalOxygenFlux)
{
  unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride = blockDim.x * gridDim.x;

  const NumericType *ionEnhancedRate = rates + numPoints;
  const NumericType *oxygenSputteringRate = rates + 2 * numPoints;
  const NumericType *etchantRate = rates + 3 * numPoints;
  const NumericType *oxygenRate = rates + 4 * numPoints;

  NumericType *eCoverage = coverages;
  NumericType *oCoverage = coverages + numPoints;

  const NumericType gamma_O = 1.;
  const NumericType gamma_F = 0.7;

  for (; tidx < numPoints; tidx += stride)
  {
    if (etchantRate[tidx] < 1e-6)
    {
      eCoverage[tidx] = 0;
    }
    else
    {
      eCoverage[tidx] =
          etchantRate[tidx] * totalEtchantFlux * gamma_F /
          (etchantRate[tidx] * totalEtchantFlux * gamma_F +
           (k_sigma_Si + 2 * ionEnhancedRate[tidx] * totalIonFlux) *
               (1 + (oxygenRate[tidx] * totalOxygenFlux * gamma_O) /
                        (beta_sigma_Si +
                         oxygenSputteringRate[tidx] * totalIonFlux)));
    }

    if (oxygenRate[tidx] < 1e-6)
    {
      oCoverage[tidx] = 0;
    }
    else
    {
      oCoverage[tidx] =
          oxygenRate[tidx] * totalOxygenFlux * gamma_O /
          (oxygenRate[tidx] * totalOxygenFlux * gamma_O +
           (beta_sigma_Si + oxygenSputteringRate[tidx] * totalIonFlux) *
               (1 +
                (etchantRate[tidx] * totalEtchantFlux * gamma_F) /
                    (k_sigma_Si + 2 * ionEnhancedRate[tidx] * totalIonFlux)));
    }
  }
}