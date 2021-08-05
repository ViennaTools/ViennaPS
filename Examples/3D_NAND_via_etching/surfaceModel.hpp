#pragma once

#include <psSurfaceModel.hpp>

template <typename NumericType>
class viaEtchingSurfaceModel : public psSurfaceModel<NumericType>
{
    using psSurfaceModel<NumericType>::Coverages;
    NumericType totalEtchantFlux = 1e7;
    NumericType totalPolyFlux = 1e7;
    NumericType totalIonFlux = 1e7;

    const NumericType ionFlux = 1e17;
    const NumericType etchantFlux = 5e17; //2.5e17;
    const NumericType polyFlux = 1e17;

    int k_ie = 1;
    int k_ev = 1;
    const int numLayers = 20;

    static constexpr double inv_rho_SiO2 = 2.2e-15;  // in (atoms/cm³)⁻¹ (rho SiO2)
    static constexpr double inv_rho_SiNx = 10.3e-15; // in (atoms/cm³)⁻¹ (rho polySi)
    static constexpr double inv_rho_Si = 2e-23;      // in (atoms/cm³)⁻¹ (rho Si)
    static constexpr double inv_rho_p = 2e-15;

    static constexpr double kB = 0.000086173324; // m² kg s⁻² K⁻¹
    static constexpr double temperature = 300.;  // K

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

    psSmartPointer<std::vector<NumericType>>
    calculateVelocities(psSmartPointer<psPointData<NumericType>> Rates, const std::vector<NumericType> &materialIds, const long numRaysPerPoint) override
    {
        updateCoverages(Rates, numRaysPerPoint);
        std::vector<NumericType> etchRate(materialIds.size(), 0.);

        auto ionSputteringRate = Rates->getScalarData("ionSputteringRate");
        auto ionpeRate = Rates->getScalarData("ionpeRate");
        auto ionEnhancedRate = Rates->getScalarData("ionEnhancedRate");
        auto polyRate = Rates->getScalarData("polyRate");

        auto peCoverage = Coverages->getScalarData("peCoverage");
        auto pCoverage = Coverages->getScalarData("pCoverage");
        auto eCoverage = Coverages->getScalarData("eCoverage");

        const NumericType F_ev = 2.7 * etchantFlux * std::exp(-0.168 / (kB * temperature));

        // calculate etch rates
        for (size_t i = 0; i < etchRate.size(); ++i)
        {
            // etchRate[i] = 0;//-inv_rho_SiO2 * (F_ev * eCoverage[i]);
            const int matId = static_cast<int>(materialIds[i]);
            if (matId != 0) // && matId != numLayers)
            {
                if (matId == numLayers + 1) // Etching depo layer
                {
                    etchRate[i] = inv_rho_p * (polyRate->at(i) * totalPolyFlux - 5 * ionpeRate->at(i) * totalIonFlux * peCoverage->at(i));
                    // assert(etchRate[i] <= 0 && "Positive etching");
                }
                else if (pCoverage->at(i) >= 1.) // Deposition
                {
                    etchRate[i] = inv_rho_p * (polyRate->at(i) * totalPolyFlux - ionpeRate->at(i) * totalIonFlux * peCoverage->at(i));
                    // assert(etchRate[i] >= 0 && "Negative deposition");
                }
                else if (matId == 1) // crystalline Si at the bottom
                {
                    etchRate[i] = -1e8 * inv_rho_Si * (F_ev * eCoverage->at(i) + ionEnhancedRate->at(i) * totalIonFlux * eCoverage->at(i) + ionSputteringRate->at(i) * totalIonFlux * (1 - eCoverage->at(i)));
                    // assert(etchRate[i] <= 0 && "Positive etching");
                }
                else if (matId % 2 == 0) // Etching SiO2
                {

                    etchRate[i] = -inv_rho_SiO2 * (F_ev * eCoverage->at(i) + ionEnhancedRate->at(i) * totalIonFlux * eCoverage->at(i) +
                                                   ionSputteringRate->at(i) * totalIonFlux * (1 - eCoverage->at(i)));
                }
                else // Etching SiNx
                {
                    etchRate[i] = -inv_rho_SiNx * (F_ev * eCoverage->at(i) + ionEnhancedRate->at(i) * totalIonFlux * eCoverage->at(i) +
                                                   ionSputteringRate->at(i) * totalIonFlux * (1 - eCoverage->at(i)));
                    // assert(etchRate[i] <= 0 && "Positive etching");
                }
            }
        }

        return psSmartPointer<std::vector<NumericType>>::New(etchRate);
    }

    void updateCoverages(psSmartPointer<psPointData<NumericType>> Rates, const long numRaysPerPoint) override
    {
        // update coverages based on fluxes
        const auto numPoints = Rates->getScalarData(0)->size();
        initializeCoverages(numPoints);

        auto ionpeRate = Rates->getScalarData("ionpeRate");
        auto etchantpeRate = Rates->getScalarData("etchantpeRate");
        auto etchanteRate = Rates->getScalarData("etchanteRate");
        auto polyRate = Rates->getScalarData("polyRate");
        auto ionEnhancedRate = Rates->getScalarData("ionEnhancedRate");

        totalIonFlux = ionFlux / numRaysPerPoint;
        totalEtchantFlux = etchantFlux / numRaysPerPoint;
        totalPolyFlux = polyFlux / numRaysPerPoint;
        const NumericType F_ev = 2.7 * etchantFlux * std::exp(-0.168 / (kB * temperature));

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