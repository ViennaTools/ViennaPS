#pragma once

#include <psSurfaceModel.hpp>

template <typename NumericType>
class viaEtchingSurfaceModel : public psSurfaceModel<NumericType>
{
    using psSurfaceModel<NumericType>::Coverages;
    long totalEtchantFlux = 1e7;
    long totalPolyFlux = 1e7;
    long totalIonFlux = 1e7;

    int k_ie = 1;
    int k_ev = 1;

public:
    void initializeCoverages(unsigned numGeometryPoints) override
    {
        if (Coverages == nullptr)
        {
            Coverages = psSmartPointer<psPointData<NumericType>>::New();
        }
        else
        {
            Coverages->clear();
        }
        std::vector<NumericType> cov(numGeometryPoints);
        Coverages->insertNextScalarData(cov, "eCoverage");
        Coverages->insertNextScalarData(cov, "pCoverage");
        Coverages->insertNextScalarData(cov, "peCoverage");
    }

    std::vector<NumericType>
    calculateVelocities(psSmartPointer<psPointData<NumericType>> Rates, std::vector<NumericType> &materialIds) override
    {
        const auto numPoints = Rates->getScalarData(0)->size();
        updateCoverages(Rates);
        std::vector<NumericType> velocities(numPoints);
        return velocities;
    }

    void updateCoverages(psSmartPointer<psPointData<NumericType>> Rates) override
    {
        // update coverages based on fluxes
        const auto numPoints = Rates->getScalarData(0)->size();
        initializeCoverages(numPoints);

        auto ionpeRate = Rates->getScalarData("ionpeRate");
        auto etchantpeRate = Rates->getScalarData("etchantpeRate");
        auto etchanteRate = Rates->getScalarData("etchanteRate");
        auto polyRate = Rates->getScalarData("polyRate");
        auto ionEnhancedRate = Rates->getScalarData("ionEnhancedRate");

        // pe coverage
        auto peCoverage = Coverages->getScalarData("peCoverage");
        for (size_t i = 0; i < numPoints; ++i)
        {
            if (etchantpeRate->at(i) == 0.)
            {
                peCoverage->at(i) = 0.;
            }
            else
            {
                peCoverage->at(i) = (etchantpeRate->at(i) * totalEtchantFlux) /
                                (etchantpeRate->at(i) * totalEtchantFlux + ionpeRate->at(i) * totalIonFlux);
            }
            assert(!std::isnan(peCoverage->at(i)) && "peCoverage NaN");
        }

        // polymer coverage
        auto pCoverage = Coverages->getScalarData("pCoverage");
        for (size_t i = 0; i < numPoints; ++i)
        {
            if (polyRate->at(i) == 0.)
            {
                pCoverage->at(i) = 0.;
            }
            else if (peCoverage->at(i) == 0. || ionpeRate->at(i) < 1e-6)
            {
                pCoverage->at(i) = 1.;
            }
            else
            {
                pCoverage->at(i) = (polyRate->at(i) * totalPolyFlux) /
                               (ionpeRate->at(i) * totalIonFlux * peCoverage->at(i));
            }
            assert(!std::isnan(pCoverage->at(i)) && "pCoverage NaN");
        }

        // etchant coverage
        auto eCoverage = Coverages->getScalarData("eCoverage");
        for (size_t i = 0; i < numPoints; ++i)
        {
            if (pCoverage->at(i) < 1.)
            {
                if (etchanteRate->at(i) < 1e-6)
                {
                    eCoverage->at(i) = 0;
                }
                else
                {
                    eCoverage->at(i) = (etchanteRate->at(i) * totalEtchantFlux * (1 - pCoverage->at(i))) /
                                   (k_ie * ionEnhancedRate->at(i) * totalIonFlux + k_ev * F_ev + etchanteRate->at(i) * totalEtchantFlux);
                }
            }
            else
            {
                eCoverage->at(i) = 0.;
            }
            assert(!std::isnan(eCoverage->at(i)) && "eCoverage NaN");
        }
    }
};