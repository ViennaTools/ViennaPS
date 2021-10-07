#ifndef SURFACEMODEL_HPP
#define SURFACEMODEL_HPP

#include <psSurfaceModel.hpp>

#define MASK 0
#define SUBSTRATE 1
#define POLYMER 2

template <typename NumericType>
class Remove : public psSurfaceModel<NumericType>
{
public:
  psSmartPointer<std::vector<NumericType>>
  calculateVelocities(psSmartPointer<psPointData<NumericType>> Rates,
                      const std::vector<NumericType> &materialIds,
                      const long numRaysTraced) override
  {
    std::vector<NumericType> etchRate(materialIds.size(), 0.);

    auto ionSputteringRate = Rates->getScalarData("ionSputteringRate");
    auto ionEnhancedRate = Rates->getScalarData("ionEnhancedRate");

    for (size_t i = 0; i < etchRate.size(); ++i)
    {
      if (materialIds[i] == POLYMER)
        etchRate[i] = -(ionEnhancedRate->at(i) + 2 * ionSputteringRate->at(i)) /
                      numRaysTraced;
      // std::cout << etchRate[i] << std::endl;
    }

    return psSmartPointer<std::vector<NumericType>>::New(etchRate);
  }
};

template <typename NumericType>
class Clear : public psSurfaceModel<NumericType>
{
public:
  psSmartPointer<std::vector<NumericType>>
  calculateVelocities(psSmartPointer<psPointData<NumericType>> Rates,
                      const std::vector<NumericType> &materialIds,
                      const long numRaysTraced) override
  {
    std::vector<NumericType> etchRate(materialIds.size(), 0.);

    for (size_t i = 0; i < etchRate.size(); ++i)
    {
      if (materialIds[i] == POLYMER)
        etchRate[i] = -1;
    }

    return psSmartPointer<std::vector<NumericType>>::New(etchRate);
  }
};

template <typename NumericType>
class Oxidize : public psSurfaceModel<NumericType>
{
public:
  psSmartPointer<std::vector<NumericType>>
  calculateVelocities(psSmartPointer<psPointData<NumericType>> Rates,
                      const std::vector<NumericType> &materialIds,
                      const long numRaysTraced) override
  {
    std::vector<NumericType> etchRate(materialIds.size(), 1.);

    return psSmartPointer<std::vector<NumericType>>::New(etchRate);
  }
};

template <typename NumericType>
class Etch : public psSurfaceModel<NumericType>
{
  using psSurfaceModel<NumericType>::Coverages;

  // static constexpr double totalIonFlux = 1e16;
  static constexpr double totalEtchantFlux = 5.5e18;
  static constexpr double inv_rho_Si = 5.0e-17; // in (atoms/cm³)⁻¹ (rho Si)
  static constexpr double k_sigma_Si = 3e17;    // 3e17

public:
  void initializeCoverages(unsigned numGeometryPoints) override
  {
    Coverages = psSmartPointer<psPointData<NumericType>>::New();
    std::vector<NumericType> cov(numGeometryPoints);
    Coverages->insertNextScalarData(cov, "eCoverage");
  }

  psSmartPointer<std::vector<NumericType>>
  calculateVelocities(psSmartPointer<psPointData<NumericType>> Rates,
                      const std::vector<NumericType> &materialIds,
                      const long numRaysTraced) override
  {
    updateCoverages(Rates, numRaysTraced);
    const auto numPoints = Rates->getScalarData(0)->size();

    std::vector<NumericType> etchRate(numPoints, 0.);

    // const NumericType ionFlux = totalIonFlux / numRaysTraced;

    // const auto ionEnhancedRate = Rates->getScalarData("ionEnhancedRate");
    // const auto ionSputteringRate = Rates->getScalarData("ionSputteringRate");
    // const auto etchantRate = Rates->getScalarData("etchantRate");
    const auto eCoverage = Coverages->getScalarData("eCoverage");

    for (size_t i = 0; i < numPoints; ++i)
    {
      if (materialIds[i] == SUBSTRATE)
      {
        // etchRate[i] = -inv_rho_Si * 1e5 *
        //               (k_sigma_Si * eCoverage->at(i) / 4. +
        //                ionSputteringRate->at(i) * ionFlux +
        //                eCoverage->at(i) * ionEnhancedRate->at(i) * ionFlux);
        etchRate[i] = -inv_rho_Si * k_sigma_Si * eCoverage->at(i);
      }
    }

    return psSmartPointer<std::vector<NumericType>>::New(etchRate);
  }

  void updateCoverages(psSmartPointer<psPointData<NumericType>> Rates,
                       const long numRaysTraced) override
  {
    // update coverages based on fluxes
    const auto numPoints = Rates->getScalarData(0)->size();

    const auto etchanteRate = Rates->getScalarData("etchantRate");
    // const auto ionEnhancedRate = Rates->getScalarData("ionEnhancedRate");

    // const NumericType ionFlux = totalIonFlux / numRaysTraced;

    // etchant flourine coverage
    auto eCoverage = Coverages->getScalarData("eCoverage");
    eCoverage->resize(numPoints);
    for (size_t i = 0; i < numPoints; ++i)
    {
      if (etchanteRate->at(i) * totalEtchantFlux < 1e-6)
      {
        eCoverage->at(i) = 0;
      }
      else
      {
        // eCoverage->at(i) = etchanteRate->at(i) * totalEtchantFlux /
        //                    (etchanteRate->at(i) * totalEtchantFlux + k_sigma_Si +
        //                     2 * ionEnhancedRate->at(i) * ionFlux);
        eCoverage->at(i) = etchanteRate->at(i) * totalEtchantFlux /
                           (etchanteRate->at(i) * totalEtchantFlux + k_sigma_Si);
      }
      assert(!std::isnan(eCoverage->at(i)) && "eCoverage NaN");
    }
  }
};

#endif