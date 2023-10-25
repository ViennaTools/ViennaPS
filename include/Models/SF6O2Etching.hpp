#pragma once

#include <ModelParameters.hpp>

#include <rayParticle.hpp>
#include <rayReflection.hpp>
#include <rayUtil.hpp>

#include <psLogger.hpp>
#include <psProcessModel.hpp>
#include <psSurfaceModel.hpp>
#include <psVelocityField.hpp>

namespace SF6O2Implementation {

// sticking probabilities
constexpr double beta_F = 0.7;
constexpr double beta_O = 1.;

template <typename NumericType, int D>
class SurfaceModel : public psSurfaceModel<NumericType> {
  using psSurfaceModel<NumericType>::Coverages;

  // fluxes in (1e15 /cm²)
  const NumericType totalIonFlux = 12.;
  const NumericType totalEtchantFlux = 1.8e3;
  const NumericType totalOxygenFlux = 1.0e2;
  static constexpr NumericType k_sigma_Si = 3.0e2; // in (1e15 cm⁻²s⁻¹)
  static constexpr NumericType beta_sigma_Si = 5.0e-2; // in (1e15 cm⁻²s⁻¹)

  const NumericType etchStop;

public:
  SurfaceModel(const double ionFlux, const double etchantFlux,
               const double oxygenFlux, const NumericType etchStopDepth)
      : totalIonFlux(ionFlux), totalEtchantFlux(etchantFlux),
        totalOxygenFlux(oxygenFlux), etchStop(etchStopDepth) {}

  void initializeCoverages(unsigned numGeometryPoints) override {
    if (Coverages == nullptr) {
      Coverages = psSmartPointer<psPointData<NumericType>>::New();
    } else {
      Coverages->clear();
    }
    std::vector<NumericType> cov(numGeometryPoints);
    Coverages->insertNextScalarData(cov, "eCoverage");
    Coverages->insertNextScalarData(cov, "oCoverage");
  }

  psSmartPointer<std::vector<NumericType>> calculateVelocities(
      psSmartPointer<psPointData<NumericType>> Rates,
      const std::vector<std::array<NumericType, 3>> &coordinates,
      const std::vector<NumericType> &materialIds) override {
    updateCoverages(Rates, materialIds);
    const auto numPoints = Rates->getScalarData(0)->size();
    std::vector<NumericType> etchRate(numPoints, 0.);

    const auto ionEnhancedRate = Rates->getScalarData("ionEnhancedRate");
    const auto ionSputteringRate = Rates->getScalarData("ionSputteringRate");
    const auto etchantRate = Rates->getScalarData("etchantRate");
    const auto eCoverage = Coverages->getScalarData("eCoverage");
    const auto oCoverage = Coverages->getScalarData("oCoverage");

    bool stop = false;

    for (size_t i = 0; i < numPoints; ++i) {
      if (coordinates[i][D - 1] < etchStop) {
        stop = true;
        break;
      }

      if (!psMaterialMap::isMaterial(materialIds[i], psMaterial::Mask)) {

        etchRate[i] = -(1 / psParameters::Si::rho) *
                      (k_sigma_Si * eCoverage->at(i) / 4. +
                       ionSputteringRate->at(i) * totalIonFlux +
                       eCoverage->at(i) * ionEnhancedRate->at(i) *
                           totalIonFlux); // in nm / s
      }
    }

    if (stop) {
      std::fill(etchRate.begin(), etchRate.end(), 0.);
      psLogger::getInstance().addInfo("Etch stop depth reached.").print();
    }

    return psSmartPointer<std::vector<NumericType>>::New(std::move(etchRate));
  }

  void updateCoverages(psSmartPointer<psPointData<NumericType>> Rates,
                       const std::vector<NumericType> &materialIds) override {
    // update coverages based on fluxes
    const auto numPoints = Rates->getScalarData(0)->size();

    const auto etchantRate = Rates->getScalarData("etchantRate");
    const auto ionEnhancedRate = Rates->getScalarData("ionEnhancedRate");
    const auto oxygenRate = Rates->getScalarData("oxygenRate");
    const auto oxygenSputteringRate =
        Rates->getScalarData("oxygenSputteringRate");

    // etchant flourine coverage
    auto eCoverage = Coverages->getScalarData("eCoverage");
    eCoverage->resize(numPoints);
    // oxygen coverage
    auto oCoverage = Coverages->getScalarData("oCoverage");
    oCoverage->resize(numPoints);
    for (size_t i = 0; i < numPoints; ++i) {
      if (etchantRate->at(i) < 1e-6) {
        eCoverage->at(i) = 0;
      } else {
        eCoverage->at(i) =
            etchantRate->at(i) * totalEtchantFlux * beta_F /
            (etchantRate->at(i) * totalEtchantFlux * beta_F +
             (k_sigma_Si + 2 * ionEnhancedRate->at(i) * totalIonFlux) *
                 (1 + (oxygenRate->at(i) * totalOxygenFlux * beta_O) /
                          (beta_sigma_Si +
                           oxygenSputteringRate->at(i) * totalIonFlux)));
      }

      if (oxygenRate->at(i) < 1e-6) {
        oCoverage->at(i) = 0;
      } else {
        oCoverage->at(i) =
            oxygenRate->at(i) * totalOxygenFlux * beta_O /
            (oxygenRate->at(i) * totalOxygenFlux * beta_O +
             (beta_sigma_Si + oxygenSputteringRate->at(i) * totalIonFlux) *
                 (1 + (etchantRate->at(i) * totalEtchantFlux * beta_F) /
                          (k_sigma_Si +
                           2 * ionEnhancedRate->at(i) * totalIonFlux)));
      }
    }
  }
};

template <typename NumericType, int D>
class Ion : public rayParticle<Ion<NumericType, D>, NumericType> {
public:
  Ion(const NumericType passedMeanEnergy, const NumericType passedSigmaEnergy,
      const NumericType passedPower, const NumericType oxySputterYield)
      : meanEnergy(passedMeanEnergy), sigmaEnergy(passedSigmaEnergy),
        power(passedPower), A_O(oxySputterYield) {}

  void surfaceCollision(NumericType rayWeight,
                        const rayTriple<NumericType> &rayDir,
                        const rayTriple<NumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        rayTracingData<NumericType> &localData,
                        const rayTracingData<NumericType> *globalData,
                        rayRNG &Rng) override final {

    // collect data for this hit
    assert(primID < localData.getVectorData(0).size() && "id out of bounds");

    const double cosTheta = -rayInternal::DotProduct(rayDir, geomNormal);

    assert(cosTheta >= 0 && "Hit backside of disc");
    assert(cosTheta <= 1 + 1e6 && "Error in calculating cos theta");

    const double angle = std::acos(std::max(std::min(cosTheta, 1.), 0.));

    NumericType f_Si_theta;
    if (cosTheta > 0.5) {
      f_Si_theta = 1.;
    } else {
      f_Si_theta = 3. - 6. * angle / rayInternal::PI;
    }
    NumericType f_O_theta = f_Si_theta;
    NumericType f_sp_theta =
        (1 + psParameters::Si::B_sp * (1 - cosTheta * cosTheta)) * cosTheta;

    double sqrtE = std::sqrt(E);
    NumericType Y_sp = psParameters::Si::A_sp *
                       std::max(sqrtE - psParameters::Si::Eth_sp_Ar_sqrt, 0.) *
                       f_sp_theta;
    NumericType Y_Si =
        A_Si * std::max(sqrtE - std::sqrt(Eth_Si), 0.) * f_Si_theta;
    NumericType Y_O = A_O * std::max(sqrtE - std::sqrt(Eth_O), 0.) * f_O_theta;

    // sputtering yield Y_sp ionSputteringRate
    localData.getVectorData(0)[primID] += rayWeight * Y_sp;

    // ion enhanced etching yield Y_Si ionEnhancedRate
    localData.getVectorData(1)[primID] += rayWeight * Y_Si;

    // ion enhanced O sputtering yield Y_O oxygenSputteringRate
    localData.getVectorData(2)[primID] += rayWeight * Y_O;
  }

  std::pair<NumericType, rayTriple<NumericType>>
  surfaceReflection(NumericType rayWeight, const rayTriple<NumericType> &rayDir,
                    const rayTriple<NumericType> &geomNormal,
                    const unsigned int primId, const int materialId,
                    const rayTracingData<NumericType> *globalData,
                    rayRNG &Rng) override final {
    auto cosTheta = -rayInternal::DotProduct(rayDir, geomNormal);

    assert(cosTheta >= 0 && "Hit backside of disc");
    assert(cosTheta <= 1 + 1e-6 && "Error in calculating cos theta");

    NumericType incAngle =
        std::acos(std::max(std::min(cosTheta, static_cast<NumericType>(1.)),
                           static_cast<NumericType>(0.)));

    // Small incident angles are reflected with the energy fraction centered at
    // 0
    NumericType Eref_peak;
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
    std::normal_distribution<NumericType> normalDist(E * Eref_peak, 0.1 * E);
    do {
      NewEnergy = normalDist(Rng);
    } while (NewEnergy > E || NewEnergy < 0.);

    // Set the flag to stop tracing if the energy is below the threshold
    if (NewEnergy > minEnergy) {
      E = NewEnergy;
      // auto direction = rayReflectionConedCosine<NumericType, D>(
      //     M_PI_2 - std::min(incAngle, minAngle), rayDir, geomNormal, Rng);
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
  NumericType getSourceDistributionPower() const override final {
    return power;
  }
  std::vector<std::string> getLocalDataLabels() const override final {
    return {"ionSputteringRate", "ionEnhancedRate", "oxygenSputteringRate"};
  }

private:
  static constexpr NumericType A_Si = 7.;
  static constexpr NumericType B_sp = 9.3;
  const NumericType A_O;

  static constexpr NumericType Eth_Si = 15.;
  static constexpr NumericType Eth_O = 10.;

  // ion energy
  static constexpr NumericType minEnergy =
      4.; // Discard particles with energy < 1eV
  const NumericType meanEnergy;
  const NumericType sigmaEnergy;
  const NumericType power;
  NumericType E;
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
    localData.getVectorData(0)[primID] += rayWeight;
  }
  std::pair<NumericType, rayTriple<NumericType>>
  surfaceReflection(NumericType rayWeight, const rayTriple<NumericType> &rayDir,
                    const rayTriple<NumericType> &geomNormal,
                    const unsigned int primID, const int materialId,
                    const rayTracingData<NumericType> *globalData,
                    rayRNG &Rng) override final {

    // F surface coverage
    const auto &phi_F = globalData->getVectorData(0)[primID];
    // O surface coverage
    const auto &phi_O = globalData->getVectorData(1)[primID];
    // Obtain the sticking probability
    NumericType Seff = beta_F * std::max(1. - phi_F - phi_O, 0.);

    auto direction = rayReflectionDiffuse<NumericType, D>(geomNormal, Rng);
    return std::pair<NumericType, rayTriple<NumericType>>{Seff, direction};
  }
  NumericType getSourceDistributionPower() const override final { return 1.; }
  std::vector<std::string> getLocalDataLabels() const override final {
    return {"etchantRate"};
  }
};

template <typename NumericType, int D>
class Oxygen : public rayParticle<Oxygen<NumericType, D>, NumericType> {
public:
  void surfaceCollision(NumericType rayWeight,
                        const rayTriple<NumericType> &rayDir,
                        const rayTriple<NumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        rayTracingData<NumericType> &localData,
                        const rayTracingData<NumericType> *globalData,
                        rayRNG &Rng) override final {
    // Rate is normalized by dividing with the local sticking coefficient
    localData.getVectorData(0)[primID] += rayWeight;
  }
  std::pair<NumericType, rayTriple<NumericType>>
  surfaceReflection(NumericType rayWeight, const rayTriple<NumericType> &rayDir,
                    const rayTriple<NumericType> &geomNormal,
                    const unsigned int primID, const int materialId,
                    const rayTracingData<NumericType> *globalData,
                    rayRNG &Rng) override final {

    NumericType Seff;
    const auto &phi_F = globalData->getVectorData(0)[primID];
    const auto &phi_O = globalData->getVectorData(1)[primID];
    Seff = beta_O * std::max(1. - phi_O - phi_F, 0.);

    auto direction = rayReflectionDiffuse<NumericType, D>(geomNormal, Rng);
    return std::pair<NumericType, rayTriple<NumericType>>{Seff, direction};
  }
  NumericType getSourceDistributionPower() const override final { return 1.; }
  std::vector<std::string> getLocalDataLabels() const override final {
    return {"oxygenRate"};
  }
};
} // namespace SF6O2Implementation

template <typename NumericType, int D>
class SF6O2Etching : public psProcessModel<NumericType, D> {
public:
  SF6O2Etching(const double ionFlux, const double etchantFlux,
               const double oxygenFlux, const NumericType meanEnergy,
               const NumericType sigmaEnergy,
               const NumericType ionExponent = 100.,
               const NumericType oxySputterYield = 2.,
               const NumericType etchStopDepth =
                   std::numeric_limits<NumericType>::lowest()) {
    // particles
    auto ion = std::make_unique<SF6O2Implementation::Ion<NumericType, D>>(
        meanEnergy, sigmaEnergy, ionExponent, oxySputterYield);
    auto etchant =
        std::make_unique<SF6O2Implementation::Etchant<NumericType, D>>();
    auto oxygen =
        std::make_unique<SF6O2Implementation::Oxygen<NumericType, D>>();

    // surface model
    auto surfModel =
        psSmartPointer<SF6O2Implementation::SurfaceModel<NumericType, D>>::New(
            ionFlux, etchantFlux, oxygenFlux, etchStopDepth);

    // velocity field
    auto velField = psSmartPointer<psDefaultVelocityField<NumericType>>::New(2);

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("SF6O2Etching");
    this->insertNextParticleType(ion);
    this->insertNextParticleType(etchant);
    this->insertNextParticleType(oxygen);
  }
};