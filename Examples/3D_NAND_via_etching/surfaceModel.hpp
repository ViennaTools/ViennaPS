#pragma once

#include <psSurfaceModel.hpp>

template <typename NumericType>
class viaEtchingSurfaceModel : public psSurfaceModel<NumericType>
{
    using psSurfaceModel<NumericType>::Coverages;

public:
    std::vector<NumericType>
    calculateVelocities(SurfaceDataType &Rates, std::vector<NumericType> &materialIds) override
    {
        const auto numPoints = Rates.back().size();
        std::vector<NumericType> velocities(numPoints);
        return velocities;
    }

    void updateCoverages(SurfaceDataType &Rates) override
    {
        // update coverages based on fluxes
        const auto numPoints = ionEnhancedRate.size();

        eCoverage.clear();
        pCoverage.clear();
        peCoverage.clear();
        eCoverage.resize(numPoints);
        pCoverage.resize(numPoints);
        peCoverage.resize(numPoints);

        // pe coverage
        for (size_t i = 0; i < numPoints; ++i)
        {
            if (etchantpeRate[i] == 0.)
            {
                peCoverage[i] = 0.;
            }
            else
            {
                peCoverage[i] = (etchantpeRate[i] * totalEtchantFlux) /
                                (etchantpeRate[i] * totalEtchantFlux + ionpeRate[i] * totalIonFlux);
            }
            assert(!std::isnan(peCoverage[i]) && "peCoverage NaN");
        }

        // polymer coverage
        for (size_t i = 0; i < numPoints; ++i)
        {
            if (polyRate[i] == 0.)
            {
                pCoverage[i] = 0.;
            }
            else if (peCoverage[i] == 0. || ionpeRate[i] < 1e-6)
            {
                pCoverage[i] = 1.;
            }
            else
            {
                pCoverage[i] = (polyRate[i] * totalPolyFlux - delta_p) /
                               (ionpeRate[i] * totalIonFlux * peCoverage[i]);
            }
            assert(!std::isnan(pCoverage[i]) && "pCoverage NaN");
        }

        // etchant coverage
        for (size_t i = 0; i < numPoints; ++i)
        {
            if (pCoverage[i] < 1.)
            {
                if (etchanteRate[i] < 1e-6)
                {
                    eCoverage[i] = 0;
                }
                else
                {
                    eCoverage[i] = (etchanteRate[i] * totalEtchantFlux * (1 - pCoverage[i])) /
                                   (k_ie * ionEnhancedRate[i] * totalIonFlux + k_ev * F_ev + etchanteRate[i] * totalEtchantFlux);
                }
            }
            else
            {
                eCoverage[i] = 0.;
            }
            assert(!std::isnan(eCoverage[i]) && "eCoverage NaN");
        }
    }
};