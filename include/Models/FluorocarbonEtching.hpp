#pragma once

template <typename NumericType, int D>
class FluorocarbonSurfaceModel : public psSurfaceModel<NumericType>
{
public:
    // etchModel(int layers) : numLayers(layers) {}

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

    void setTotalFluxes(const NumericType ionFlux, const NumericType etchantFlux,
                        const NumericType polyFlux, const NumericType numRaysTraced)
    {
        totalIonFlux = ionFlux / numRaysTraced;
        totalEtchantFlux = etchantFlux / numRaysTraced;
        totalPolyFlux = polyFlux / numRaysTraced;

        F_ev = 2.7 * etchantFlux * std::exp(-0.168 / (kB * temperature));
    }

    std::vector<NumericType> calculateVelocities(const std::vector<NumericType> &ionSputteringRate,
                                                 const std::vector<NumericType> &ionEnhancedRate,
                                                 const std::vector<NumericType> &ionpeRate,
                                                 const std::vector<NumericType> &polyRate,
                                                 const std::vector<NumericType> &etchanteRate,
                                                 const std::vector<NumericType> &etchantpeRate,
                                                 const std::vector<NumericType> &materialIds,
                                                 const NumericType time)
    {
        updateCoverages(ionEnhancedRate, ionpeRate, polyRate,
                        etchanteRate, etchantpeRate);
        std::vector<NumericType> etchRate(materialIds.size(), 0.);

        // calculate etch rates
        for (size_t i = 0; i < etchRate.size(); ++i)
        {
            // etchRate[i] = 0;//-inv_rho_SiO2 * (F_ev * eCoverage[i]);
            const int matId = static_cast<int>(materialIds[i]);
            if (matId != 0) // && matId != numLayers)
            {
                if (matId == numLayers + 1) // Etching depo layer
                {
                    etchRate[i] = inv_rho_p * (polyRate[i] * totalPolyFlux - 5 * ionpeRate[i] * totalIonFlux * peCoverage[i]);
                    // etchRate[i] = -inv_rho_p * (F_ev * eCoverage[i] + ionEnhancedRate[i] * totalIonFlux * eCoverage[i] +
                    //    ionSputteringRate[i] * totalIonFlux * (1 - eCoverage[i]));
                    // assert(etchRate[i] <= 0 && "Positive etching");
                }
                else if (pCoverage[i] >= 1.) // Deposition
                {
                    etchRate[i] = inv_rho_p * (polyRate[i] * totalPolyFlux - ionpeRate[i] * totalIonFlux * peCoverage[i]);
                    assert(etchRate[i] >= 0 && "Negative deposition");
                }
                else if (matId == 1) // crystalline Si at the bottom
                {
                    etchRate[i] = -1e8 * inv_rho_Si * (F_ev * eCoverage[i] + ionEnhancedRate[i] * totalIonFlux * eCoverage[i] + ionSputteringRate[i] * totalIonFlux * (1 - eCoverage[i]));
                    assert(etchRate[i] <= 0 && "Positive etching");
                    // std::cout << F_ev * eCoverage[i] << " " << ionEnhancedRate[i] * totalIonFlux * eCoverage[i] << " " << ionSputteringRate[i] * totalIonFlux * (1 - eCoverage[i]) << std::endl;
                }
                else if (matId % 2 == 0) // Etching SiO2
                {
                    if (time > 500 && matId > 15)
                    {
                        etchRate[i] = 0;
                    }
                    else
                    {
                        // std::cout << F_ev * eCoverage[i] << " " << ionEnhancedRate[i] * totalIonFlux * eCoverage[i] << " " << ionSputteringRate[i] * totalIonFlux * (1 - eCoverage[i]) << std::endl;
                        etchRate[i] = -inv_rho_SiO2 * (F_ev * eCoverage[i] + ionEnhancedRate[i] * totalIonFlux * eCoverage[i] +
                                                       ionSputteringRate[i] * totalIonFlux * (1 - eCoverage[i]));
                    }
                }
                else // Etching SiNx
                {
                    if (time > 500 && matId > 15)
                    {
                        etchRate[i] = 0;
                    }
                    else
                    {
                        etchRate[i] = -inv_rho_SiNx * (F_ev * eCoverage[i] + ionEnhancedRate[i] * totalIonFlux * eCoverage[i] +
                                                       ionSputteringRate[i] * totalIonFlux * (1 - eCoverage[i]));
                    }
                    assert(etchRate[i] <= 0 && "Positive etching");
                }
            }
        }

        return etchRate;
    }

    void updateCoverages(const std::vector<NumericType> &ionEnhancedRate,
                         const std::vector<NumericType> &ionpeRate, const std::vector<NumericType> &polyRate,
                         const std::vector<NumericType> &etchanteRate, const std::vector<NumericType> &etchantpeRate)
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

    std::vector<NumericType> &getECoverage()
    {
        return eCoverage;
    }
    std::vector<NumericType> &getPECoverage()
    {
        return peCoverage;
    }
    std::vector<NumericType> &getPCoverage()
    {
        return pCoverage;
    }

private:
    static constexpr double inv_rho_SiO2 = 2.2e-15;  // in (atoms/cm³)⁻¹ (rho SiO2)
    static constexpr double inv_rho_SiNx = 10.3e-15; // in (atoms/cm³)⁻¹ (rho polySi)
    static constexpr double inv_rho_Si = 1e-23;      // in (atoms/cm³)⁻¹ (rho Si)
    static constexpr double inv_rho_p = 2e-15;

    static constexpr double k_ie = 1.;
    static constexpr double k_ev = 1.;

    static constexpr double delta_p = 0.;

    static constexpr double kB = 0.000086173324; // m² kg s⁻² K⁻¹
    static constexpr double temperature = 300.;  // K

    NumericType totalIonFlux;
    NumericType totalEtchantFlux;
    NumericType totalPolyFlux;
    NumericType F_ev_SiO2;
    NumericType F_ev_SiNx;
    NumericType F_ev;

    std::vector<NumericType> eCoverage;
    std::vector<NumericType> pCoverage;
    std::vector<NumericType> peCoverage;

    const int numLayers;
};
