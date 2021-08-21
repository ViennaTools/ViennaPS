#ifndef SURFACEMODEL_HPP
#define SURFACEMODEL_HPP

#include <psSurfaceModel.hpp>

#define MASK 0
#define SUBSTRATE 1
#define POLYMER 2

template <typename NumericType>
class Remove : public psSurfaceModel<NumericType> {
public:
  psSmartPointer<std::vector<NumericType>>
  calculateVelocities(psSmartPointer<psPointData<NumericType>> Rates,
                      const std::vector<NumericType> &materialIds,
                      const long numRaysPerPoint) override {
    std::vector<NumericType> etchRate(materialIds.size(), 0.);

    auto ionSputteringRate = Rates->getScalarData("ionSputteringRate");
    auto ionEnhancedRate = Rates->getScalarData("ionEnhancedRate");

    for (size_t i = 0; i < materialIds.size(); ++i) {
      if (materialIds[i] != MASK)
        etchRate[i] = -(ionEnhancedRate->at(i) + ionSputteringRate->at(i)) /
                      numRaysPerPoint;
    }

    return psSmartPointer<std::vector<NumericType>>::New(etchRate);
  }
};

template <typename NumericType>
class Clear : public psSurfaceModel<NumericType> {
public:
  psSmartPointer<std::vector<NumericType>>
  calculateVelocities(psSmartPointer<psPointData<NumericType>> Rates,
                      const std::vector<NumericType> &materialIds,
                      const long numRaysPerPoint) override {
    std::vector<NumericType> etchRate(materialIds.size(), 0.);

    for (size_t i = 0; i < materialIds.size(); ++i) {
      if (materialIds[i] == POLYMER)
        etchRate[i] = -1;
    }

    return psSmartPointer<std::vector<NumericType>>::New(etchRate);
  }
};

template <typename NumericType>
class Oxidize : public psSurfaceModel<NumericType> {
public:
  psSmartPointer<std::vector<NumericType>>
  calculateVelocities(psSmartPointer<psPointData<NumericType>> Rates,
                      const std::vector<NumericType> &materialIds,
                      const long numRaysPerPoint) override {
    std::vector<NumericType> etchRate(materialIds.size(), 1.);

    return psSmartPointer<std::vector<NumericType>>::New(etchRate);
  }
};

template <typename NumericType>
class Etch : public psSurfaceModel<NumericType> {
  using psSurfaceModel<NumericType>::Coverages;

  static constexpr double totalIonFlux = 1e12;
  static constexpr double totalEtchantFlux = 1e17;
  static constexpr double inv_rho_subs = 2.0e-13; // in (atoms/cm³)⁻¹ (rho SiO2)
  static constexpr double kB = 0.000086173324; // m² kg s⁻² K⁻¹
  static constexpr double temperature = 300.;  // K
  static constexpr double k_ie = 2;
  static constexpr double k_ev = 2;

public:
  void initializeCoverages(unsigned numGeometryPoints) override {
    Coverages = psSmartPointer<psPointData<NumericType>>::New();
    std::vector<NumericType> cov(numGeometryPoints);
    Coverages->insertNextScalarData(cov, "eCoverage");
  }

  psSmartPointer<std::vector<NumericType>>
  calculateVelocities(psSmartPointer<psPointData<NumericType>> Rates,
                      const std::vector<NumericType> &materialIds,
                      const long numRaysPerPoint) override {
    updateCoverages(Rates, numRaysPerPoint);
    std::vector<NumericType> etchRate(materialIds.size(), 0.);

    const NumericType ionFlux = totalIonFlux / numRaysPerPoint;

    auto ionEnhancedRate = Rates->getScalarData("ionEnhancedRate");
    auto ionSputteringRate = Rates->getScalarData("ionSputteringRate");
    auto etchantRate = Rates->getScalarData("etchantRate");
    auto eCoverage = Coverages->getScalarData("eCoverage");

    const NumericType F_ev = 10 * 2.7 * totalEtchantFlux / numRaysPerPoint *
                             std::exp(-0.168 / (kB * temperature));
    for (size_t i = 0; i < materialIds.size(); ++i) {
      if (materialIds[i] == SUBSTRATE) {
        etchRate[i] =
            -inv_rho_subs *
            (F_ev * eCoverage->at(i) +
             ionEnhancedRate->at(i) * ionFlux * eCoverage->at(i) +
             ionSputteringRate->at(i) * ionFlux * (1 - eCoverage->at(i)));
      }
    }

    return psSmartPointer<std::vector<NumericType>>::New(etchRate);
  }

  void updateCoverages(psSmartPointer<psPointData<NumericType>> Rates,
                       const long numRaysPerPoint) override {
    // update coverages based on fluxes
    const auto numPoints = Rates->getScalarData(0)->size();

    auto etchanteRate = Rates->getScalarData("etchantRate");
    auto ionEnhancedRate = Rates->getScalarData("ionEnhancedRate");

    const NumericType ionFlux = totalIonFlux / numRaysPerPoint;
    const NumericType etchantFlux = totalEtchantFlux / numRaysPerPoint;

    const NumericType F_ev =
        2.7 * etchantFlux * std::exp(-0.168 / (kB * temperature));

    // etchant coverage
    auto eCoverage = Coverages->getScalarData("eCoverage");
    eCoverage->resize(numPoints);
    for (size_t i = 0; i < numPoints; ++i) {
      if (etchanteRate->at(i) < 1e-6) {
        eCoverage->at(i) = 0;
      } else {
        eCoverage->at(i) = etchanteRate->at(i) * etchantFlux /
                           (k_ie * ionEnhancedRate->at(i) * ionFlux +
                            k_ev * F_ev + etchanteRate->at(i) * etchantFlux);
      }
      assert(!std::isnan(eCoverage->at(i)) && "eCoverage NaN");
    }
  }
};

#endif