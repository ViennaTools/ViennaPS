#pragma once

#include <cmath>

#include <ModelParameters.hpp>

#include <psLogger.hpp>
#include <psMaterials.hpp>
#include <psProcessModel.hpp>

#include <rayParticle.hpp>
#include <rayReflection.hpp>
#include <rayUtil.hpp>

namespace FluorocarbonImplementation {

// sticking probabilities
static constexpr double beta_e = 0.9;
static constexpr double beta_e_mask = 0.1;
static constexpr double beta_p = 0.26;
static constexpr double beta_p_mask = 0.01;
static constexpr double beta_pe = 0.6;

template <typename NumericType, int D>
class SurfaceModel : public psSurfaceModel<NumericType> {
  using psSurfaceModel<NumericType>::Coverages;
  static constexpr double eps = 1e-6;

public:
  SurfaceModel(const NumericType ionFlux, const NumericType etchantFlux,
               const NumericType polyFlux, const NumericType passedDeltaP,
               const NumericType passedEtchStopDepth)
      : totalIonFlux(ionFlux), totalEtchantFlux(etchantFlux),
        totalPolyFlux(polyFlux), delta_p(passedDeltaP),
        etchStopDepth(passedEtchStopDepth) {}

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
    updateCoverages(Rates, materialIds);
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
        etchRate[i] = (-1. / psParameters::Mask::rho) *
                      ionSputteringRate->at(i) * totalIonFlux;
      } else if (pCoverage->at(i) >= 1.) {
        // Deposition
        etchRate[i] = (1 / psParameters::Polymer::rho) *
                      (polyRate->at(i) * totalPolyFlux -
                       ionpeRate->at(i) * totalIonFlux * peCoverage->at(i));
        assert(etchRate[i] >= 0 && "Negative deposition");
      } else if (matId == psMaterial::Polymer) {
        // Etching depo layer
        etchRate[i] =
            std::min((1 / psParameters::Polymer::rho) *
                         (polyRate->at(i) * totalPolyFlux -
                          ionpeRate->at(i) * totalIonFlux * peCoverage->at(i)),
                     0.);
      } else {
        NumericType mat_density = 1.;
        NumericType F_ev = 0.;
        if (matId == psMaterial::Si) // Etching Si
        {
          mat_density = -psParameters::Si::rho;
          F_ev = psParameters::Si::K * totalEtchantFlux *
                 std::exp(-psParameters::Si::E_a /
                          (psParameters::kB * psParameters::roomTemperature));
        } else if (matId == psMaterial::SiO2) // Etching SiO2
        {
          F_ev = psParameters::SiO2::K * totalEtchantFlux *
                 std::exp(-psParameters::SiO2::E_a /
                          (psParameters::kB * psParameters::roomTemperature));
          mat_density = -psParameters::SiO2::rho;
        } else if (matId == psMaterial::Si3N4) // Etching Si3N4
        {
          F_ev = psParameters::SiO2::K * totalEtchantFlux *
                 std::exp(-psParameters::SiO2::E_a /
                          (psParameters::kB * psParameters::roomTemperature));
          mat_density = -psParameters::Si3N4::rho;
        }
        etchRate[i] =
            (1 / mat_density) *
            (F_ev * eCoverage->at(i) +
             ionEnhancedRate->at(i) * totalIonFlux * eCoverage->at(i) +
             ionSputteringRate->at(i) * totalIonFlux * (1 - eCoverage->at(i)));
      }

      // etch rate is in nm / s

      assert(!std::isnan(etchRate[i]) && "etchRate NaN");
    }

    if (etchStop) {
      std::fill(etchRate.begin(), etchRate.end(), 0.);
      psLogger::getInstance().addInfo("Etch stop depth reached.").print();
    }

    return psSmartPointer<std::vector<NumericType>>::New(etchRate);
  }

  void updateCoverages(psSmartPointer<psPointData<NumericType>> Rates,
                       const std::vector<NumericType> &materialIds) override {

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
        peCoverage->at(i) = (etchantRate->at(i) * totalEtchantFlux * beta_pe) /
                            (etchantRate->at(i) * totalEtchantFlux * beta_pe +
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
            std::max((polyRate->at(i) * totalPolyFlux * beta_p - delta_p) /
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
          NumericType F_ev;
          if (psMaterialMap::isMaterial(materialIds[i], psMaterial::Si)) {
            F_ev = psParameters::Si::K * totalEtchantFlux *
                   std::exp(-psParameters::Si::E_a /
                            (psParameters::kB * psParameters::roomTemperature));
          } else {
            F_ev = psParameters::SiO2::K * totalEtchantFlux *
                   std::exp(-psParameters::SiO2::E_a /
                            (psParameters::kB * psParameters::roomTemperature));
          }
          eCoverage->at(i) =
              (etchantRate->at(i) * totalEtchantFlux * beta_e *
               (1 - pCoverage->at(i))) /
              (k_ie * ionEnhancedRate->at(i) * totalIonFlux + k_ev * F_ev +
               etchantRate->at(i) * totalEtchantFlux * beta_e);
        }
      } else {
        eCoverage->at(i) = 0.;
      }
      assert(!std::isnan(eCoverage->at(i)) && "eCoverage NaN");
    }
  }

private:
  static constexpr double k_ie = 2.;
  static constexpr double k_ev = 2.;

  // fluxes in (1e15 /cm²)
  const NumericType totalIonFlux;
  const NumericType totalEtchantFlux;
  const NumericType totalPolyFlux;
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
  Ion(const NumericType passedMeanEnergy, const NumericType passedSigmaEnergy,
      const NumericType passedPower)
      : meanEnergy(passedMeanEnergy), sigmaEnergy(passedSigmaEnergy),
        power(passedPower) {}
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
    const double angle = std::acos(std::max(std::min(cosTheta, 1.), 0.));

    assert(cosTheta >= 0 && "Hit backside of disc");
    assert(cosTheta <= 1 + 4 && "Error in calculating cos theta");

    NumericType f_theta;
    if (cosTheta > 0.5) {
      f_theta = 1.;
    } else {
      f_theta = 3. - 6. * angle / M_PI;
    }

    const auto sqrtE = std::sqrt(E);
    auto Y_s = psParameters::Si::A_sp *
               std::max(sqrtE - psParameters::Si::Eth_sp_Ar_sqrt, 0.) *
               (1 + psParameters::Si::B_sp * (1 - cosTheta * cosTheta)) *
               cosTheta;
    auto Y_ie = Ae_ie * std::max(sqrtE - sqrtE_th_ie, 0.) * f_theta;
    auto Y_p = Ap_ie * std::max(sqrtE - sqrtE_th_p, 0.) * f_theta;

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

    // Small incident angles are reflected with the energy fraction centered at
    // 0
    double incAngle = std::acos(-rayInternal::DotProduct(rayDir, geomNormal));
    double Eref_peak;
    if (incAngle >= psParameters::Ion::inflectAngle) {
      Eref_peak = (1 - (1 - psParameters::Ion::A) * (M_PI_2 - incAngle) /
                           (M_PI_2 - psParameters::Ion::inflectAngle));
    } else {
      Eref_peak = psParameters::Ion::A *
                  std::pow(incAngle / psParameters::Ion::inflectAngle,
                           psParameters::Ion::n_l);
    }
    // Gaussian distribution around the Eref_peak scaled by the particle energy
    NumericType NewEnergy;
    std::normal_distribution<NumericType> normalDist{Eref_peak * E, 0.1 * E};
    do {
      NewEnergy = normalDist(Rng);
    } while (NewEnergy > E || NewEnergy < 0.);

    if (NewEnergy > minEnergy) {
      E = NewEnergy;
      auto direction = rayReflectionSpecular<NumericType>(rayDir, geomNormal);
      return std::pair<NumericType, rayTriple<NumericType>>{1. - Eref_peak,
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
  NumericType getSourceDistributionPower() const override final {
    return power;
  }
  std::vector<std::string> getLocalDataLabels() const override final {
    return {"ionSputteringRate", "ionEnhancedRate", "ionpeRate"};
  }

private:
  static constexpr double sqrtE_th_ie = 2.;
  static constexpr double sqrtE_th_p = 2.;

  static constexpr double Ae_ie = 0.0361;
  static constexpr double Ap_ie = 4 * 0.0361;

  static constexpr NumericType minEnergy = 4.;
  const NumericType meanEnergy;
  const NumericType sigmaEnergy;
  const NumericType power;
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
      stick = beta_p_mask;
    } else {
      stick = beta_p;
    }
    return std::pair<NumericType, rayTriple<NumericType>>{stick, direction};
  }
  int getRequiredLocalDataSize() const override final { return 1; }
  NumericType getSourceDistributionPower() const override final { return 1.; }
  std::vector<std::string> getLocalDataLabels() const override final {
    return {"polyRate"};
  }
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
      Seff = beta_e_mask * std::max(1 - phi_e - phi_p, 0.);
    } else if (psMaterialMap::isMaterial(materialId, psMaterial::Polymer)) {
      Seff = beta_pe * std::max(1. - phi_pe, 0.);
    } else {
      Seff = beta_e * std::max(1 - phi_e - phi_p, 0.);
    }

    return std::pair<NumericType, rayTriple<NumericType>>{Seff, direction};
  }
  int getRequiredLocalDataSize() const override final { return 1; }
  NumericType getSourceDistributionPower() const override final { return 1.; }
  std::vector<std::string> getLocalDataLabels() const override final {
    return {"etchantRate"};
  }
};
} // namespace FluorocarbonImplementation

template <typename NumericType, int D>
class FluorocarbonEtching : public psProcessModel<NumericType, D> {
public:
  FluorocarbonEtching(const double ionFlux, const double etchantFlux,
                      const double polyFlux, const NumericType meanEnergy,
                      const NumericType sigmaEnergy,
                      const NumericType ionExponent = 100.,
                      const NumericType deltaP = 0.,
                      const NumericType etchStopDepth =
                          std::numeric_limits<NumericType>::lowest()) {
    // particles
    auto ion = std::make_unique<FluorocarbonImplementation::Ion<NumericType>>(
        meanEnergy, sigmaEnergy, ionExponent);
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