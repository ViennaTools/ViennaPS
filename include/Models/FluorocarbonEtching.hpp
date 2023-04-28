#pragma once

#include <psSurfaceModel.hpp>

template <typename NumericType, int D>
class FluorocarbonSurfaceModel : public psSurfaceModel<NumericType> {
  using psSurfaceModel<NumericType>::Coverages;
  const int maskId;
  const int numLayers;

public:
  FluorocarbonSurfaceModel(const NumericType ionFlux,
                           const NumericType etchantFlux,
                           const NumericType polyFlux, const int passedMask,
                           const int passedNumLayers)
      : totalIonFlux(ionFlux), totalEtchantFlux(etchantFlux),
        totalPolyFlux(polyFlux), maskId(passedMask), numLayers(passedNumLayers),
        F_ev(2.7 * etchantFlux * std::exp(-0.168 / (kB * temperature))) {}

  void initializeCoverages(unsigned numGeometryPoints) override {
    if (Coverages == nullptr) {
      Coverages = psSmartPointer<psPointData<NumericType>>::New();
    } else {
      Coverages->clear();
    }
    std::vector<NumericType> cov(numGeometryPoints);
    Coverages->insertNextScalarData(cov, "eCoverage");
    Coverages->insertNextScalarData(cov, "pCoverage");
    Coverages->insertNextScalarData(cov, "peCoverage");
  }

  psSmartPointer<std::vector<NumericType>> calculateVelocities(
      psSmartPointer<psPointData<NumericType>> Rates,
      const std::vector<std::array<NumericType, 3>> &coordinates,
      const std::vector<NumericType> &materialIds) override {
    updateCoverages(Rates);
    const auto numPoints = Rates->getScalarData(0)->size();
    std::vector<NumericType> etchRate(numPoints, 0.);

    const auto ionEnhancedRate = Rates->getScalarData("ionEnhancedRate");
    const auto ionSputteringRate = Rates->getScalarData("ionSputteringRate");
    const auto ionpeRate = Rates->getScalarData("ionpeRate");
    const auto polyRate = Coverages->getScalarData("polyRate");

    const auto eCoverage = Coverages->getScalarData("eCoverage");
    const auto pCoverage = Coverages->getScalarData("pCoverage");
    const auto peCoverage = Coverages->getScalarData("peCoverage");

    // calculate etch rates
    for (size_t i = 0; i < etchRate.size(); ++i) {
      const int matId = static_cast<int>(materialIds[i]);
      if (matId != maskId) {
        if (matId == numLayers + 1) // Etching depo layer
        {
          etchRate[i] = inv_rho_p * (polyRate->at(i) * totalPolyFlux -
                                     5 * ionpeRate->at(i) * totalIonFlux *
                                         peCoverage->at(i));
          assert(etchRate[i] <= 0 && "Positive etching");
        } else if (pCoverage[i] >= 1.) // Deposition
        {
          etchRate[i] =
              inv_rho_p * (polyRate->at(i) * totalPolyFlux -
                           ionpeRate->at(i) * totalIonFlux * peCoverage->at(i));
          assert(etchRate[i] >= 0 && "Negative deposition");
        } else if (matId == 1) // crystalline Si at the bottom
        {
          etchRate[i] =
              -1e8 * inv_rho_Si *
              (F_ev * eCoverage->at(i) +
               ionEnhancedRate->at(i) * totalIonFlux * eCoverage->at(i) +
               ionSputteringRate->at(i) * totalIonFlux *
                   (1 - eCoverage->at(i)));
          assert(etchRate[i] <= 0 && "Positive etching");
        } else if (matId % 2 == 0) // Etching SiO2
        {
          etchRate[i] =
              -inv_rho_SiO2 *
              (F_ev * eCoverage->at(i) +
               ionEnhancedRate->at(i) * totalIonFlux * eCoverage->at(i) +
               ionSputteringRate->at(i) * totalIonFlux *
                   (1 - eCoverage->at(i)));
        } else // Etching SiNx
        {
          etchRate[i] =
              -inv_rho_SiNx *
              (F_ev * eCoverage->at(i) +
               ionEnhancedRate->at(i) * totalIonFlux * eCoverage->at(i) +
               ionSputteringRate->at(i) * totalIonFlux *
                   (1 - eCoverage->at(i)));
          assert(etchRate[i] <= 0 && "Positive etching");
        }
      }
    }

    return psSmartPointer<std::vector<NumericType>>::New(etchRate);
  }

  void
  updateCoverages(psSmartPointer<psPointData<NumericType>> Rates) override {

    const auto ionEnhancedRate = Rates->getScalarData("ionEnhancedRate");
    const auto ionpeRate = Rates->getScalarData("ionpeRate");
    const auto polyRate = Coverages->getScalarData("polyRate");
    const auto etchanteRate = Rates->getScalarData("etchanteRate");
    const auto etchantpeRate = Rates->getScalarData("etchantpeRate");

    const auto eCoverage = Coverages->getScalarData("eCoverage");
    const auto pCoverage = Coverages->getScalarData("pCoverage");
    const auto peCoverage = Coverages->getScalarData("peCoverage");

    // update coverages based on fluxes
    const auto numPoints = ionEnhancedRate->size();
    eCoverage->resize(numPoints);
    pCoverage->resize(numPoints);
    peCoverage->resize(numPoints);

    // pe coverage
    for (size_t i = 0; i < numPoints; ++i) {
      if (etchantpeRate->at(i) == 0.) {
        peCoverage->at(i) = 0.;
      } else {
        peCoverage->at(i) = (etchantpeRate->at(i) * totalEtchantFlux) /
                            (etchantpeRate->at(i) * totalEtchantFlux +
                             ionpeRate->at(i) * totalIonFlux);
      }
      assert(!std::isnan(peCoverage->at(i)) && "peCoverage NaN");
    }

    // polymer coverage
    for (size_t i = 0; i < numPoints; ++i) {
      if (polyRate->at(i) == 0.) {
        pCoverage->at(i) = 0.;
      } else if (peCoverage->at(i) == 0. || ionpeRate->at(i) < 1e-6) {
        pCoverage->at(i) = 1.;
      } else {
        pCoverage->at(i) =
            (polyRate->at(i) * totalPolyFlux - delta_p) /
            (ionpeRate->at(i) * totalIonFlux * peCoverage->at(i));
      }
      assert(!std::isnan(pCoverage->at(i)) && "pCoverage NaN");
    }

    // etchant coverage
    for (size_t i = 0; i < numPoints; ++i) {
      if (pCoverage->at(i) < 1.) {
        if (etchanteRate->at(i) < 1e-6) {
          eCoverage->at(i) = 0;
        } else {
          eCoverage->at(i) =
              (etchanteRate->at(i) * totalEtchantFlux *
               (1 - pCoverage->at(i))) /
              (k_ie * ionEnhancedRate[i] * totalIonFlux + k_ev * F_ev +
               etchanteRate->at(i) * totalEtchantFlux);
        }
      } else {
        eCoverage->at(i) = 0.;
      }
      assert(!std::isnan(eCoverage->at(i)) && "eCoverage NaN");
    }
  }

private:
  static constexpr double inv_rho_SiO2 = 2.2e-15; // in (atoms/cm³)⁻¹ (rho SiO2)
  static constexpr double inv_rho_SiNx =
      10.3e-15; // in (atoms/cm³)⁻¹ (rho polySi)
  static constexpr double inv_rho_Si = 1e-23; // in (atoms/cm³)⁻¹ (rho Si)
  static constexpr double inv_rho_p = 2e-15;

  static constexpr double k_ie = 1.;
  static constexpr double k_ev = 1.;

  static constexpr double delta_p = 0.;

  static constexpr double kB = 0.000086173324; // m² kg s⁻² K⁻¹
  static constexpr double temperature = 300.;  // K

  const NumericType totalIonFlux;
  const NumericType totalEtchantFlux;
  const NumericType totalPolyFlux;
  const NumericType F_ev;
};
