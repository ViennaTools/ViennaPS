#include <cuda.h>
#include <cuda_runtime.h>

#include <context.hpp>

#define totalIonFlux 2.e16
#define totalEtchantFlux 4.5e18
#define totalOxygenFlux 1.e18 // 5.0e16;

#define inv_rho_Si 2.0e-23 // in (atoms/cm³)⁻¹ (rho Si)
// #define inv_rho_SiO2 1. / (2.6e22)
#define k_sigma_Si 3.0e17
#define beta_sigma_Si 1.0e14

extern "C" __global__ void calculateEtchRate(const NumericType *rates,
                                             const NumericType *coverages,
                                             const NumericType *materialIds,
                                             NumericType *etchRate,
                                             const unsigned long numPoints)
{
  unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride = blockDim.x * gridDim.x;

  const NumericType *ionSputteringRate = rates;
  const NumericType *ionEnhancedRate = rates + numPoints;
  const NumericType *eCoverage = coverages;

  for (; tidx < numPoints; tidx += stride)
  {
    if (materialIds[tidx] == 1)
    {
      // 1e4 converts the rates to micrometers/s
      etchRate[tidx] = -inv_rho_Si * 1e4 *
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
                                           const unsigned long numPoints)
{
  unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride = blockDim.x * gridDim.x;

  const NumericType *ionEnhancedRate = rates + numPoints;
  const NumericType *oxygenSputteringRate = rates + 2 * numPoints;
  const NumericType *etchantRate = rates + 3 * numPoints;
  const NumericType *oxygenRate = rates + 4 * numPoints;

  NumericType *eCoverage = coverages;
  NumericType *oCoverage = coverages + numPoints;

  for (; tidx < numPoints; tidx += stride)
  {
    if (etchantRate[tidx] < 1e-6)
    {
      eCoverage[tidx] = 0;
    }
    else
    {
      eCoverage[tidx] =
          etchantRate[tidx] * totalEtchantFlux /
          (etchantRate[tidx] * totalEtchantFlux +
           (k_sigma_Si + 2 * ionEnhancedRate[tidx] * totalIonFlux) *
               (1 + (oxygenRate[tidx] * totalOxygenFlux) /
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
          oxygenRate[tidx] * totalOxygenFlux /
          (oxygenRate[tidx] * totalOxygenFlux +
           (beta_sigma_Si + oxygenSputteringRate[tidx] * totalIonFlux) *
               (1 +
                (etchantRate[tidx] * totalEtchantFlux) /
                    (k_sigma_Si + 2 * ionEnhancedRate[tidx] * totalIonFlux)));
    }
  }
}