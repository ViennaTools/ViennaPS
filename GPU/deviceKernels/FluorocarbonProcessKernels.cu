#include <cuda.h>
#include <cuda_runtime.h>

#include <context.hpp>

#define rho_SiO2 2.3 // in (1e22 atoms/cm³)
#define rho_SiNx 2.3 // in (1e22 atoms/cm³)
#define rho_Si 5.02  // in (1e22 atoms/cm³)
#define rho_p 2      // in (1e22 atoms/cm³)

#define k_ie 1
#define k_ev 1

#define k_B 0.000086173324 // m² kg s⁻² K⁻¹

#define Mask 0
#define Si 1
#define SiO2 2
#define Si3N4 3
#define Polymer 13

extern "C" __global__ void calculateEtchRate(const NumericType *rates,
                                             const NumericType *coverages,
                                             const NumericType *materialIds,
                                             NumericType *etchRate,
                                             const unsigned long numPoints,
                                             const NumericType totalIonFlux,
                                             const NumericType totalEtchantFlux,
                                             const NumericType totalPolyFlux,
                                             const NumericType FJ_ev)
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

    const NumericType gamma_p = 0.26;

    // calculate etch rates
    for (; tidx < numPoints; tidx += stride)
    {
        int matId = materialIds[tidx];
        if (matId == Mask)
        {
            etchRate[tidx] = 0.;
            continue;
        }
        if (pCoverage[tidx] >= 1.)
        {
            // Deposition
            etchRate[tidx] = std::max(
                (1 / rho_p) * (polyRate[tidx] * totalPolyFlux * gamma_p -
                               ionpeRate[tidx] * totalIonFlux * peCoverage[tidx]),
                (NumericType)0.);
        }
        else if (matId == Polymer)
        {
            // Etching depo layer
            etchRate[tidx] = std::min(
                (1 / rho_p) * (polyRate[tidx] * totalPolyFlux * gamma_p -
                               ionpeRate[tidx] * totalIonFlux * peCoverage[tidx]),
                (NumericType)0.);
        }
        else
        {
            NumericType mat_density = 0;
            if (matId == Si) // crystalline Si at the bottom
            {
                mat_density = rho_Si;
            }
            else if (matId == SiO2) // Etching SiO2
            {
                mat_density = rho_SiO2;
            }
            else if (matId == Si3N4) // Etching SiNx
            {
                mat_density = rho_SiNx;
            }
            etchRate[tidx] =
                (-1. / mat_density) *
                (FJ_ev * eCoverage[tidx] +
                 ionEnhancedRate[tidx] * totalIonFlux * eCoverage[tidx] +
                 ionSputteringRate[tidx] * totalIonFlux * (1 - eCoverage[tidx]));
        }

        // etch rate is in nm / s
        // etchRate[tidx] *= 1; // to convert to nm / s
    }
}

extern "C" __global__ void updateCoverages(const NumericType *rates,
                                           NumericType *coverages,
                                           const unsigned long numPoints,
                                           const NumericType totalIonFlux,
                                           const NumericType totalEtchantFlux,
                                           const NumericType totalPolyFlux,
                                           const NumericType FJ_ev)
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

    const NumericType gamma_p = 0.26;
    const NumericType gamma_e = 0.9;
    const NumericType gamma_pe = 0.6;

    // pe coverage
    for (; tidx < numPoints; tidx += stride)
    {
        if (etchantpeRate[tidx] == 0.)
        {
            peCoverage[tidx] = 0.;
        }
        else
        {
            peCoverage[tidx] = (etchantpeRate[tidx] * totalEtchantFlux * gamma_pe) /
                               (etchantpeRate[tidx] * totalEtchantFlux * gamma_pe + ionpeRate[tidx] * totalIonFlux);
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
        else if (peCoverage[tidx] < 1e-6 || ionpeRate[tidx] < 1e-6)
        {
            pCoverage[tidx] = 1.;
        }
        else
        {
            pCoverage[tidx] = (polyRate[tidx] * totalPolyFlux * gamma_p) /
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
                eCoverage[tidx] = (etchantRate[tidx] * totalEtchantFlux * (1 - pCoverage[tidx]) * gamma_e) /
                                  (k_ie * ionEnhancedRate[tidx] * totalIonFlux +
                                   k_ev * FJ_ev +
                                   etchantRate[tidx] * totalEtchantFlux * gamma_e);
            }
        }
        else
        {
            eCoverage[tidx] = 0.;
        }
    }
}