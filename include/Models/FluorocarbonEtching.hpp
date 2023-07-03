#pragma once

#include <cmath>

#include <psLogger.hpp>
#include <psMaterials.hpp>
#include <psProcessModel.hpp>

#include <rayParticle.hpp>
#include <rayReflection.hpp>
#include <rayUtil.hpp>

namespace FluorocarbonImplementation {
template <typename NumericType, int D>
class SurfaceModel : public psSurfaceModel<NumericType> {
  using psSurfaceModel<NumericType>::Coverages;
  static constexpr double eps = 1e-6;

public:
  SurfaceModel(const NumericType ionFlux, const NumericType etchantFlux,
               const NumericType polyFlux, const NumericType passedDeltaP,
               const NumericType passedEtchStopDepth)
      : totalIonFlux(ionFlux), totalEtchantFlux(etchantFlux),
        totalPolyFlux(polyFlux),
        F_ev(2.7 * etchantFlux * std::exp(-0.168 / (kB * temperature))),
        delta_p(passedDeltaP), etchStopDepth(passedEtchStopDepth) {}

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
    const auto numPoints = materialIds.size();
    std::vector<NumericType> etchRate(numPoints, 0.);

    const auto ionEnhancedRate = Rates->getScalarData("ionEnhancedRate");
    const auto ionSputteringRate = Rates->getScalarData("ionSputteringRate");
    const auto ionpeRate = Rates->getScalarData("ionpeRate");
    const auto polyRate = Rates->getScalarData("polyRate");

    const auto eCoverage = Coverages->getScalarData("eCoverage");
    const auto pCoverage = Coverages->getScalarData("pCoverage");
    const auto peCoverage = Coverages->getScalarData("peCoverage");

    bool etchStop = false;

    // calculate etch rates
    for (size_t i = 0; i < numPoints; ++i) {
      if (coordinates[i][D - 1] <= etchStopDepth) {
        etchStop = true;
        break;
      }

      auto matId = psMaterialMap::mapToMaterial(materialIds[i]);
      assert(matId == psMaterial::Mask || matId == psMaterial::Polymer ||
             matId == psMaterial::Si || matId == psMaterial::SiO2 ||
             matId == psMaterial::Si3N4 && "Unexptected material");
      if (matId == psMaterial::Mask) {
        etchRate[i] =
            (-1. / rho_mask) * ionSputteringRate->at(i) * totalIonFlux;
      } else if (pCoverage->at(i) >= 1.) {
        assert(pCoverage->at(i) == 1. && "Correctness assumption");
        // Deposition
        etchRate[i] =
            (1 / rho_p) * (polyRate->at(i) * totalPolyFlux -
                           ionpeRate->at(i) * totalIonFlux * peCoverage->at(i));
        assert(etchRate[i] >= 0 && "Negative deposition");
      } else if (matId == psMaterial::Polymer) {
        // Etching depo layer
        etchRate[i] = std::min(
            (1 / rho_p) * (polyRate->at(i) * totalPolyFlux -
                           ionpeRate->at(i) * totalIonFlux * peCoverage->at(i)),
            0.);
      } else {
        NumericType mat_density = 0;
        if (matId == psMaterial::Si) // Etching Si
        {
          mat_density = -rho_Si;
        } else if (matId == psMaterial::SiO2) // Etching SiO2
        {
          mat_density = -rho_SiO2;
        } else if (matId == psMaterial::Si3N4) // Etching SiNx
        {
          mat_density = -rho_SiNx;
        }
        etchRate[i] =
            (1 / mat_density) *
            (F_ev * eCoverage->at(i) +
             ionEnhancedRate->at(i) * totalIonFlux * eCoverage->at(i) +
             ionSputteringRate->at(i) * totalIonFlux * (1 - eCoverage->at(i)));
      }

      // etch rate is in nm / s
      // etchRate[i] *= 1e4; // to convert to um / s

      assert(!std::isnan(etchRate[i]) && "etchRate NaN");
    }

    if (etchStop) {
      std::fill(etchRate.begin(), etchRate.end(), 0.);
      psLogger::getInstance().addInfo("Etch stop depth reached.").print();
    }

    return psSmartPointer<std::vector<NumericType>>::New(etchRate);
  }

  void
  updateCoverages(psSmartPointer<psPointData<NumericType>> Rates) override {

    const auto ionEnhancedRate = Rates->getScalarData("ionEnhancedRate");
    const auto ionpeRate = Rates->getScalarData("ionpeRate");
    const auto polyRate = Rates->getScalarData("polyRate");
    const auto etchantRate = Rates->getScalarData("etchantRate");

    const auto eCoverage = Coverages->getScalarData("eCoverage");
    const auto pCoverage = Coverages->getScalarData("pCoverage");
    const auto peCoverage = Coverages->getScalarData("peCoverage");

    // update coverages based on fluxes
    const auto numPoints = ionEnhancedRate->size();
    eCoverage->resize(numPoints);
    pCoverage->resize(numPoints);
    peCoverage->resize(numPoints);

    // pe coverage
    for (std::size_t i = 0; i < numPoints; ++i) {
      if (etchantRate->at(i) == 0.) {
        peCoverage->at(i) = 0.;
      } else {
        peCoverage->at(i) = (etchantRate->at(i) * totalEtchantFlux * gamma_pe) /
                            (etchantRate->at(i) * totalEtchantFlux * gamma_pe +
                             ionpeRate->at(i) * totalIonFlux);
      }
      assert(!std::isnan(peCoverage->at(i)) && "peCoverage NaN");
    }

    // polymer coverage
    for (std::size_t i = 0; i < numPoints; ++i) {
      if (polyRate->at(i) < eps) {
        pCoverage->at(i) = 0.;
      } else if (peCoverage->at(i) < eps || ionpeRate->at(i) < eps) {
        pCoverage->at(i) = 1.;
      } else {
        pCoverage->at(i) =
            std::max((polyRate->at(i) * totalPolyFlux * gamma_p - delta_p) /
                         (ionpeRate->at(i) * totalIonFlux * peCoverage->at(i)),
                     0.);
      }
      assert(!std::isnan(pCoverage->at(i)) && "pCoverage NaN");
    }

    // etchant coverage
    for (std::size_t i = 0; i < numPoints; ++i) {
      if (pCoverage->at(i) < 1.) {
        if (etchantRate->at(i) == 0.) {
          eCoverage->at(i) = 0;
        } else {
          eCoverage->at(i) =
              (etchantRate->at(i) * totalEtchantFlux * gamma_e *
               (1 - pCoverage->at(i))) /
              (k_ie * ionEnhancedRate->at(i) * totalIonFlux + k_ev * F_ev +
               etchantRate->at(i) * totalEtchantFlux * gamma_e);
        }
      } else {
        eCoverage->at(i) = 0.;
      }
      assert(!std::isnan(eCoverage->at(i)) && "eCoverage NaN");
    }
  }

private:
  static constexpr double rho_SiO2 = 2.3; // in (1e22 atoms/cm³)
  static constexpr double rho_SiNx = 2.3; // in (1e22 atoms/cm³)
  static constexpr double rho_Si = 5.02;  // in (1e22 atoms/cm³)
  static constexpr double rho_p = 2;      // in (1e22 atoms/cm³)
  static constexpr double rho_mask = 500; // in (1e22 atoms/cm³)

  static constexpr double k_ie = 2.;
  static constexpr double k_ev = 2.;

  // sticking probabilities
  static constexpr NumericType gamma_e = 0.9;
  static constexpr NumericType gamma_p = 0.26;
  static constexpr NumericType gamma_pe = 0.6;

  static constexpr double kB = 0.000086173324; // m² kg s⁻² K⁻¹
  static constexpr double temperature = 300.;  // K

  // fluxes in (1e15 /cm²)
  const NumericType totalIonFlux;
  const NumericType totalEtchantFlux;
  const NumericType totalPolyFlux;
  const NumericType F_ev;
  const NumericType delta_p;

  const NumericType etchStopDepth = 0.;
};

// Parameters from:
// A. LaMagna and G. Garozzo "Factors affecting profile evolution in plasma
// etching of SiO2: Modeling and experimental verification" Journal of the
// Electrochemical Society 150(10) 2003 pp. 1896-1902

template <typename NumericType>
class Ion : public rayParticle<Ion<NumericType>, NumericType> {
public:
  Ion(const NumericType passedMeanEnergy, const NumericType passedSigmaEnergy)
      : meanEnergy(passedMeanEnergy), sigmaEnergy(passedSigmaEnergy) {}
  void surfaceCollision(NumericType rayWeight,
                        const rayTriple<NumericType> &rayDir,
                        const rayTriple<NumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        rayTracingData<NumericType> &localData,
                        const rayTracingData<NumericType> *globalData,
                        rayRNG &Rng) override final {
    // collect data for this hit
    assert(primID < localData.getVectorData(0).size() && "id out of bounds");
    assert(E >= 0 && "Negative energy ion");

    const auto cosTheta = -rayInternal::DotProduct(rayDir, geomNormal);

    assert(cosTheta >= 0 && "Hit backside of disc");
    assert(cosTheta <= 1 + 4 && "Error in calculating cos theta");

    const auto sqrtE = std::sqrt(E);
    const auto f_e_sp = (1 + B_sp * (1 - cosTheta * cosTheta)) * cosTheta;
    const auto Y_s = Ae_sp * std::max(sqrtE - sqrtE_th_sp, 0.) * f_e_sp;
    const auto Y_ie = Ae_ie * std::max(sqrtE - sqrtE_th_ie, 0.) * cosTheta;
    const auto Y_p = Ap_ie * std::max(sqrtE - sqrtE_th_p, 0.) * cosTheta;

    // sputtering yield Y_s
    localData.getVectorData(0)[primID] += rayWeight * Y_s;

    // ion enhanced etching yield Y_ie
    localData.getVectorData(1)[primID] += rayWeight * Y_ie;

    // polymer yield Y_p
    localData.getVectorData(2)[primID] += rayWeight * Y_p;
  }
  std::pair<NumericType, rayTriple<NumericType>>
  surfaceReflection(NumericType rayWeight, const rayTriple<NumericType> &rayDir,
                    const rayTriple<NumericType> &geomNormal,
                    const unsigned int primId, const int materialId,
                    const rayTracingData<NumericType> *globalData,
                    rayRNG &Rng) override final {
    const auto cosTheta = -rayInternal::DotProduct(rayDir, geomNormal);
    const double Phi = std::acos(cosTheta);
    std::uniform_real_distribution<NumericType> uniDist;

    double Eref_peak;
    if (Phi >= Phi_inflect) {
      Eref_peak =
          1 - (1 - A) * std::pow((M_PI_2 - Phi) / (M_PI_2 - Phi_inflect), n_r);
    } else {
      Eref_peak = A * std::pow(Phi / Phi_inflect, n_l);
    }

    const double TempEnergy = Eref_peak * E;
    double NewEnergy;
    do {
      NewEnergy =
          TempEnergy + (std::min((E - TempEnergy), TempEnergy) + 0.05 * E) *
                           (1 - 2. * uniDist(Rng)) *
                           std::sqrt(std::fabs(std::log(uniDist(Rng))));
    } while (NewEnergy > E || NewEnergy < 0.);

    if (NewEnergy > minEnergy) {
      E = NewEnergy;
      auto direction = rayReflectionSpecular<NumericType>(rayDir, geomNormal);
      return std::pair<NumericType, rayTriple<NumericType>>{1 - Eref_peak,
                                                            direction};
    } else {
      return std::pair<NumericType, rayTriple<NumericType>>{
          1., rayTriple<NumericType>{0., 0., 0.}};
    }
  }
  void initNew(rayRNG &RNG) override final {
    std::normal_distribution<NumericType> normalDist{meanEnergy, sigmaEnergy};
    do {
      E = normalDist(RNG);
    } while (E < minEnergy);
  }

  int getRequiredLocalDataSize() const override final { return 3; }
  NumericType getSourceDistributionPower() const override final { return 100.; }
  std::vector<std::string> getLocalDataLabels() const override final {
    return std::vector<std::string>{"ionSputteringRate", "ionEnhancedRate",
                                    "ionpeRate"};
  }

private:
  static constexpr double sqrtE_th_sp = 4.2426406871;
  static constexpr double sqrtE_th_ie = 2.;
  static constexpr double sqrtE_th_p = 2.;

  static constexpr double Ae_sp = 0.0139;
  static constexpr double Ae_ie = 0.0361;
  static constexpr double Ap_ie = 4 * 0.0361;

  static constexpr double B_sp = 9.3;

  static constexpr double Phi_inflect = 1.55334303;
  static constexpr double n_r = 1.;
  static constexpr double n_l = 10.;

  const double A = 1. / (1. + (n_l / n_r) * (M_PI_2 / Phi_inflect - 1.));

  static constexpr NumericType minEnergy = 4.;
  const NumericType meanEnergy;
  const NumericType sigmaEnergy;
  NumericType E;
};

template <typename NumericType, int D>
class Polymer : public rayParticle<Polymer<NumericType, D>, NumericType> {
public:
  void surfaceCollision(NumericType rayWeight,
                        const rayTriple<NumericType> &rayDir,
                        const rayTriple<NumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        rayTracingData<NumericType> &localData,
                        const rayTracingData<NumericType> *globalData,
                        rayRNG &Rng) override final {
    // collect data for this hit
    localData.getVectorData(0)[primID] += rayWeight;
  }
  std::pair<NumericType, rayTriple<NumericType>>
  surfaceReflection(NumericType rayWeight, const rayTriple<NumericType> &rayDir,
                    const rayTriple<NumericType> &geomNormal,
                    const unsigned int primId, const int materialId,
                    const rayTracingData<NumericType> *globalData,
                    rayRNG &Rng) override final {
    auto direction = rayReflectionDiffuse<NumericType, D>(geomNormal, Rng);
    NumericType stick;
    if (psMaterialMap::isMaterial(materialId, psMaterial::Mask)) {
      stick = gamma_p_mask;
    } else {
      stick = gamma_p;
    }
    return std::pair<NumericType, rayTriple<NumericType>>{stick, direction};
  }
  int getRequiredLocalDataSize() const override final { return 1; }
  NumericType getSourceDistributionPower() const override final { return 1.; }
  std::vector<std::string> getLocalDataLabels() const override final {
    return std::vector<std::string>{"polyRate"};
  }

private:
  static constexpr NumericType gamma_p = 0.26;
  static constexpr NumericType gamma_p_mask = 0.01;
};

template <typename NumericType, int D>
class Etchant : public rayParticle<Etchant<NumericType, D>, NumericType> {
public:
  void surfaceCollision(NumericType rayWeight,
                        const rayTriple<NumericType> &rayDir,
                        const rayTriple<NumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        rayTracingData<NumericType> &localData,
                        const rayTracingData<NumericType> *globalData,
                        rayRNG &Rng) override final {
    // collect data for this hit
    localData.getVectorData(0)[primID] += rayWeight;
  }
  std::pair<NumericType, rayTriple<NumericType>>
  surfaceReflection(NumericType rayWeight, const rayTriple<NumericType> &rayDir,
                    const rayTriple<NumericType> &geomNormal,
                    const unsigned int primID, const int materialId,
                    const rayTracingData<NumericType> *globalData,
                    rayRNG &Rng) override final {
    auto direction = rayReflectionDiffuse<NumericType, D>(geomNormal, Rng);

    const auto &phi_e = globalData->getVectorData(0)[primID];
    const auto &phi_p = globalData->getVectorData(1)[primID];
    const auto &phi_pe = globalData->getVectorData(2)[primID];

    NumericType Seff;
    if (psMaterialMap::isMaterial(materialId, psMaterial::Mask)) {
      Seff = gamma_e_mask * std::max(1 - phi_e - phi_p, 0.);
    } else if (psMaterialMap::isMaterial(materialId, psMaterial::Polymer)) {
      Seff = gamma_pe * std::max(1. - phi_pe, 0.);
    } else {
      Seff = gamma_e * std::max(1 - phi_e - phi_p, 0.);
    }

    return std::pair<NumericType, rayTriple<NumericType>>{Seff, direction};
  }
  int getRequiredLocalDataSize() const override final { return 1; }
  NumericType getSourceDistributionPower() const override final { return 1.; }
  std::vector<std::string> getLocalDataLabels() const override final {
    return std::vector<std::string>{"etchantRate"};
  }

private:
  static constexpr NumericType gamma_e = 0.9;
  static constexpr NumericType gamma_pe = 0.6;
  static constexpr NumericType gamma_e_mask = 0.1;
};
} // namespace FluorocarbonImplementation

template <typename NumericType, int D>
class FluorocarbonEtching : public psProcessModel<NumericType, D> {
public:
  FluorocarbonEtching(const double ionFlux, const double etchantFlux,
                      const double polyFlux, const NumericType meanEnergy,
                      const NumericType sigmaEnergy,
                      const NumericType deltaP = 0.,
                      const NumericType etchStopDepth =
                          std::numeric_limits<NumericType>::lowest()) {
    // particles
    auto ion = std::make_unique<FluorocarbonImplementation::Ion<NumericType>>(
        meanEnergy, sigmaEnergy);
    auto etchant =
        std::make_unique<FluorocarbonImplementation::Etchant<NumericType, D>>();
    auto poly =
        std::make_unique<FluorocarbonImplementation::Polymer<NumericType, D>>();

    // surface model
    auto surfModel = psSmartPointer<FluorocarbonImplementation::SurfaceModel<
        NumericType, D>>::New(ionFlux, etchantFlux, polyFlux, deltaP,
                              etchStopDepth);

    // velocity field
    auto velField = psSmartPointer<psDefaultVelocityField<NumericType>>::New(2);

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("FluorocarbonEtching");
    this->insertNextParticleType(ion);
    this->insertNextParticleType(etchant);
    this->insertNextParticleType(poly);
  }
};