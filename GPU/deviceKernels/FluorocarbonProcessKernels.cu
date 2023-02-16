#include <cuda.h>
#include <cuda_runtime.h>

#include <context.hpp>

// Si density 4.99e22 (atoms/cm³)
#define inv_rho_Si 2.004e-23 // in (atoms/cm³)⁻¹ (rho Si)
#define inv_rho_p 2e-15

// #define inv_rho_SiO2 1. / (2.6e22)
#define k_ie 1
#define k_ev 1
#define k_B 0.000086173324 // m² kg s⁻² K⁻¹

extern "C" __global__ void calculateEtchRate(const NumericType *rates,
                                             const NumericType *coverages,
                                             const NumericType *materialIds,
                                             NumericType *etchRate,
                                             const unsigned long numPoints,
                                             const NumericType totalIonFlux,
                                             const NumericType totalEtchantFlux,
                                             const NumericType totalPolyFlux,
                                             const NumericType temperature,
                                             const int maskId,
                                             const int depoId)
{
  unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride = blockDim.x * gridDim.x;

  const NumericType *ionSputteringRate = rates;
  const NumericType *ionEnhancedRate = rates + numPoints;
  const NumericType *ionpeRate = rates + 2 * numPoints;
  const NumericType *polyRate = rates + 4 * numPoints;

  const NumericType *eCoverage = coverages;
  const NumericType *pCoverage = coverages + numPoints;
  const NumericType *peCoverage = coverages + 2 * numPoints;

  const NumericType F_ev = 2.7 * totalEtchantFlux * std::exp(-0.168 / (k_B * temperature));

  for (; tidx < numPoints; tidx += stride)
  {
    if (pCoverage[tidx] >= 1.) // Deposition
    {
        etchRate[tidx] = inv_rho_p * (polyRate[tidx] * totalPolyFlux - ionpeRate[tidx] * totalIonFlux * peCoverage[tidx]);
    }
    else if ((int)materialIds[tidx] == depoId) // etching depolayer
    {
        etchRate[tidx] = inv_rho_p * (polyRate[tidx] * totalPolyFlux - 5 * ionpeRate[tidx] * totalIonFlux * peCoverage[tidx]);
    }
    else
    {
        etchRate[tidx] = -inv_rho_Si * 1e8 * (F_ev * eCoverage[tidx] + ionEnhancedRate[tidx] * totalIonFlux * eCoverage[tidx]
                                        + ionSputteringRate[tidx] * totalIonFlux * (1 - eCoverage[tidx]));
    }
  }
}

extern "C" __global__ void updateCoverages(const NumericType *rates,
                                           NumericType *coverages,
                                           const unsigned long numPoints,
                                           const NumericType totalIonFlux,
                                           const NumericType totalEtchantFlux,
                                           const NumericType totalPolyFlux,
                                           const NumericType temperature)
{
  unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride = blockDim.x * gridDim.x;

  const NumericType *ionEnhancedRate = rates + numPoints;
  const NumericType *ionpeRate = rates + 2 * numPoints;
  const NumericType *etchantRate = rates + 3 * numPoints;
  const NumericType *polyRate = rates + 4 * numPoints;
  const NumericType *etchantpeRate = rates + 5 * numPoints;

  NumericType *eCoverage = coverages;
  NumericType *pCoverage = coverages + numPoints;
  NumericType *peCoverage = coverages + 2 * numPoints;

  const NumericType delta_p = 0.;
  const NumericType F_ev = 2.7 * totalEtchantFlux * std::exp(-0.168 / (k_B * temperature));

    // pe coverage
    for (; tidx < numPoints; tidx += stride)
    {
        if (etchantpeRate[tidx] == 0.)
        {
            peCoverage[tidx] = 0.;
        }
        else
        {
            peCoverage[tidx] = (etchantpeRate[tidx] * totalEtchantFlux) /
                                (etchantpeRate[tidx] * totalEtchantFlux + ionpeRate[tidx] * totalIonFlux);
        }
    }
    tidx = blockIdx.x * blockDim.x + threadIdx.x;

    // polymer coverage
    for (; tidx < numPoints; tidx += stride)
    {
        if (polyRate[tidx] == 0.)
        {
            pCoverage[tidx] = 0.;
        }
        else if (peCoverage[tidx] == 0. || ionpeRate[tidx] < 1e-6)
        {
            pCoverage[tidx] = 1.;
        }
        else
        {
            pCoverage[tidx] = (polyRate[tidx] * totalPolyFlux - delta_p) /
                              (ionpeRate[tidx] * totalIonFlux * peCoverage[tidx]);
        }
    }
    tidx = blockIdx.x * blockDim.x + threadIdx.x;

    // etchant coverage
    for (; tidx < numPoints; tidx += stride)
    {
        if (pCoverage[tidx] < 1.)
        {
            if (etchantRate[tidx] < 1e-6)
            {
                eCoverage[tidx] = 0;
            }
            else
            {
                eCoverage[tidx] = (etchantRate[tidx] * totalEtchantFlux * (1 - pCoverage[tidx])) /
                                  (k_ie * ionEnhancedRate[tidx] * totalIonFlux + k_ev * F_ev + etchantRate[tidx] * totalEtchantFlux);
            }
        }
        else
        {
            eCoverage[tidx] = 0.;
        }
    }
}