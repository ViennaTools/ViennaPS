#pragma once

#include <csTracingParticle.hpp>

#include <rayParticle.hpp>
#include <rayReflection.hpp>
#include <rayUtil.hpp>

#include <psSmartPointer.hpp>
#include <psSurfaceModel.hpp>
#include <psVelocityField.hpp>

template <typename NumericType, int D>
class SF6O2SurfaceModel : public psSurfaceModel<NumericType> {
  using psSurfaceModel<NumericType>::Coverages;

  // Best match for yO2 = 0.44
  const double totalIonFlux = 2.e16;
  const double totalEtchantFlux = 4.5e18;
  const double totalOxygenFlux = 1.e18;         // 5.0e16;
  static constexpr double inv_rho_Si = 2.0e-23; // in (atoms/cm³)⁻¹ (rho Si)
  static constexpr double inv_rho_SiO2 = 1. / (2.6e22);
  static constexpr double k_sigma_Si = 3.0e17;
  static constexpr double beta_sigma_Si = 1.0e14;

public:
  SF6O2SurfaceModel(const double ionFlux, const double etchantFlux,
                    const double oxygenFlux)
      : totalIonFlux(ionFlux), totalEtchantFlux(etchantFlux),
        totalOxygenFlux(oxygenFlux) {}

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

  psSmartPointer<std::vector<NumericType>>
  calculateVelocities(psSmartPointer<psPointData<NumericType>> Rates,
                      const std::vector<NumericType> &materialIds) override {

    std::vector<NumericType> velocities(materialIds.size(), 0.);
    updateCoverages(Rates);
    const auto numPoints = Rates->getScalarData(0)->size();
    std::vector<NumericType> etchRate(numPoints, 0.);

    const auto ionEnhancedRate = Rates->getScalarData("ionEnhancedRate");
    const auto ionSputteringRate = Rates->getScalarData("ionSputteringRate");
    const auto etchantRate = Rates->getScalarData("etchantRate");
    const auto oxideEtchRate = Rates->getScalarData("oxideEtchRate");
    const auto eCoverage = Coverages->getScalarData("eCoverage");
    const auto oCoverage = Coverages->getScalarData("oCoverage");

    for (size_t i = 0; i < numPoints; ++i) {
      if (materialIds[i] == 1) {
        // 1e4 converts the rates to micrometers/s
        etchRate[i] =
            -inv_rho_Si * 1e4 *
            (k_sigma_Si * eCoverage->at(i) / 4. +
             ionSputteringRate->at(i) * totalIonFlux +
             eCoverage->at(i) * ionEnhancedRate->at(i) * totalIonFlux);
      }
    }

    return psSmartPointer<std::vector<NumericType>>::New(etchRate);
  }

  void
  updateCoverages(psSmartPointer<psPointData<NumericType>> Rates) override {
    // update coverages based on fluxes
    const auto numPoints = Rates->getScalarData(0)->size();

    const auto etchantRate = Rates->getScalarData("etchantRate");
    const auto ionEnhancedRate = Rates->getScalarData("ionEnhancedRate");
    const auto oxygenRate = Rates->getScalarData("oxygenRate");
    const auto oxygenSputteringRate =
        Rates->getScalarData("oxygenSputteringRate");
    const auto oxideEtchRate = Rates->getScalarData("oxideEtchRate");

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
            etchantRate->at(i) * totalEtchantFlux /
            (etchantRate->at(i) * totalEtchantFlux +
             (k_sigma_Si + 2 * ionEnhancedRate->at(i) * totalIonFlux) *
                 (1 + (oxygenRate->at(i) * totalOxygenFlux) /
                          (beta_sigma_Si +
                           oxygenSputteringRate->at(i) * totalIonFlux)));
      }

      if (oxygenRate->at(i) < 1e-6) {
        oCoverage->at(i) = 0;
      } else {
        oCoverage->at(i) =
            oxygenRate->at(i) * totalOxygenFlux /
            (oxygenRate->at(i) * totalOxygenFlux +
             (beta_sigma_Si + oxygenSputteringRate->at(i) * totalIonFlux) *
                 (1 + (etchantRate->at(i) * totalEtchantFlux) /
                          (k_sigma_Si +
                           2 * ionEnhancedRate->at(i) * totalIonFlux)));
      }
    }
  }
};

template <class T> class SF6O2VelocityField : public psVelocityField<T> {
public:
  SF6O2VelocityField(const int maskId) : maskMaterial(maskId) {}

  T getScalarVelocity(const std::array<T, 3> & /*coordinate*/, int material,
                      const std::array<T, 3> & /*normalVector*/,
                      unsigned long pointID) override {
    if (material != maskMaterial)
      return velocities->at(pointID);
    else
      return 0.;
  }

  void setVelocities(psSmartPointer<std::vector<T>> passedVelocities) override {
    velocities = passedVelocities;
  }

private:
  psSmartPointer<std::vector<T>> velocities = nullptr;
  const int maskMaterial = 0;
};

template <class T>
class SFO2VolumeIon : public csParticle<SFO2VolumeIon<T>, T> {
public:
  SFO2VolumeIon(const T passedMeanEnergy = 100.,
                const T passedMeanFreePath = 1.)
      : meanIonEnergy(passedMeanEnergy), meanFreePath(passedMeanFreePath) {}

  void initNew(rayRNG &RNG) override final {
    do {
      const auto rand1 = uniDist(RNG);
      const auto rand2 = uniDist(RNG);
      E = std::cos(rayInternal::PI * 2 * rand1) *
              std::sqrt(-2. * std::log(rand2)) * deltaIonEnergy +
          meanIonEnergy;
    } while (E < minEnergy);
  }

  std::pair<T, rayTriple<T>> surfaceHit(const rayTriple<T> &rayDir,
                                        const rayTriple<T> &geomNormal,
                                        bool &reflect,
                                        rayRNG &Rng) override final {
    auto cosTheta = -rayInternal::DotProduct(rayDir, geomNormal);
    const T incAngle = std::acos(std::max(std::min(cosTheta, 1.), 0.));

    T Eref_peak = 0;

    // Small incident angles are reflected with the energy fraction centered at
    // 0
    if (incAngle >= inflectAngle) {
      Eref_peak =
          Eref_max *
          (1 - (1 - A) * std::pow((rayInternal::PI / 2. - incAngle) /
                                      (rayInternal::PI / 2. - inflectAngle),
                                  n_r));
    } else {
      Eref_peak = Eref_max * A * std::pow(incAngle / inflectAngle, n_l);
    }
    // Gaussian distribution around the Eref_peak scaled by the particle energy
    T tempEnergy = Eref_peak * E;

    T NewEnergy;
    do {
      const auto rand1 = uniDist(Rng);
      const auto rand2 = uniDist(Rng);
      NewEnergy = tempEnergy +
                  (std::min((E - tempEnergy), tempEnergy) + E * 0.05) *
                      (1 - 2. * rand1) * std::sqrt(std::fabs(std::log(rand2)));
    } while (NewEnergy > E || NewEnergy <= 0.);

    auto impactEnergy = E - NewEnergy;

    if (NewEnergy > minEnergy) {
      reflect = true;
      auto direction = rayReflectionConedCosine<T>(
          rayInternal::PI / 2. - std::min(incAngle, minAngle), rayDir,
          geomNormal, Rng);
      E = NewEnergy;
      return std::pair<T, rayTriple<T>>{impactEnergy, direction};
    } else {
      reflect = false;
      return std::pair<T, rayTriple<T>>{impactEnergy, rayTriple<T>{0., 0., 0.}};
    }
  }

  T getSourceDistributionPower() const override final { return 1000.; }
  csPair<T> getMeanFreePath() const override final {
    return {meanFreePath, meanFreePath / 2.};
  }

private:
  std::uniform_real_distribution<T> uniDist;

  static constexpr T A_p = 0.0337;
  static constexpr T A_Si = 7.;
  static constexpr T A_O = 2;
  static constexpr T A_SiO2 = 0.3;

  static constexpr T sqrt_Eth_p = 0.;
  static constexpr T sqrt_Eth_Si = 3.8729833462;
  static constexpr T sqrt_Eth_O = 3.8729833462;
  static constexpr T sqrt_Eth_SiO2 = 3.8729833462;
  static constexpr T Eref_max = 1.;

  const T meanFreePath = 1.;
  const T meanIonEnergy = 100.;
  const T deltaIonEnergy = meanIonEnergy / 10.;
  static constexpr T minEnergy = 1.; // Discard particles with energy < 1eV

  static constexpr T inflectAngle = 1.55334;
  static constexpr T minAngle = 1.3962634;
  static constexpr T n_l = 10.;
  static constexpr T n_r = 1.;

  static constexpr T A =
      1. / (1. + (n_l / n_r) * (rayInternal::PI / (2 * inflectAngle) - 1.));

  // T f_p_theta;
  // T f_Si_theta;
  // T f_O_theta;
  // T f_SiO2_theta;
  T E;
};

template <typename NumericType, int D>
class SF6O2Ion : public rayParticle<SF6O2Ion<NumericType, D>, NumericType> {
public:
  SF6O2Ion(NumericType passedEnergy = 100.) : meanIonEnergy(passedEnergy) {}

  void surfaceCollision(NumericType rayWeight,
                        const rayTriple<NumericType> &rayDir,
                        const rayTriple<NumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        rayTracingData<NumericType> &localData,
                        const rayTracingData<NumericType> *globalData,
                        rayRNG &Rng) override final {

    // collect data for this hit
    assert(primID < localData.getVectorData(0).size() && "id out of bounds");

    auto cosTheta = -rayInternal::DotProduct(rayDir, geomNormal);

    assert(cosTheta >= 0 && "Hit backside of disc");
    assert(cosTheta <= 1 + 1e6 && "Error in calculating cos theta");

    const double angle = std::acos(std::max(std::min(cosTheta, 1.), 0.));

    if (cosTheta > 0.5) {
      f_Si_theta = 1.;
      f_O_theta = 1.;
    } else {
      f_Si_theta = std::max(3. - 6. * angle / rayInternal::PI, 0.);
      f_O_theta = std::max(3. - 6. * angle / rayInternal::PI, 0.);
    }

    f_p_theta = 1.;
    f_SiO2_theta = 1.;

    const double sqrtE = std::sqrt(E);
    const double Y_p = A_p * std::max(sqrtE - std::sqrt(Eth_p), 0.) * f_p_theta;
    const double Y_Si =
        A_Si * std::max(sqrtE - std::sqrt(Eth_Si), 0.) * f_Si_theta;
    const double Y_O = A_O * std::max(sqrtE - std::sqrt(Eth_O), 0.) * f_O_theta;
    const double Y_SiO2 =
        A_SiO2 * std::max(sqrtE - std::sqrt(Eth_SiO2), 0.) * f_SiO2_theta;

    // sputtering yield Y_p ionSputteringRate
    localData.getVectorData(0)[primID] += rayWeight * Y_p;

    // ion enhanced etching yield Y_Si ionEnhancedRate
    localData.getVectorData(1)[primID] += rayWeight * Y_Si;

    // ion enhanced O sputtering yield Y_O oxygenSputteringRate
    localData.getVectorData(2)[primID] += rayWeight * Y_O;

    // ion enhanced SiO2 sputtering yield Y_SiO2 oxideEtchRate
    localData.getVectorData(3)[primID] += rayWeight * Y_SiO2;
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

    const NumericType incAngle =
        std::acos(std::max(std::min(cosTheta, 1.), 0.));

    NumericType Eref_peak = 0;

    // Small incident angles are reflected with the energy fraction centered at
    // 0
    if (incAngle >= inflectAngle) {
      Eref_peak =
          Eref_max *
          (1 - (1 - A) * std::pow((rayInternal::PI / 2 - incAngle) /
                                      (rayInternal::PI / 2 - inflectAngle),
                                  n_r));
    } else {
      Eref_peak = Eref_max * A * std::pow(incAngle / inflectAngle, n_l);
    }
    // Gaussian distribution around the Eref_peak scaled by the particle energy
    NumericType tempEnergy = Eref_peak * E;

    NumericType NewEnergy;

    std::uniform_real_distribution<NumericType> uniDist;

    do {
      const auto rand1 = uniDist(Rng);
      const auto rand2 = uniDist(Rng);
      // NewEnergy = std::cos(2 * PI * rand1) * std::sqrt(-2. * std::log(rand2))
      // *
      //         (std::min((E - tempEnergy), tempEnergy) + E * 0.05) +
      //         tempEnergy;
      NewEnergy = tempEnergy +
                  (std::min((E - tempEnergy), tempEnergy) + E * 0.05) *
                      (1 - 2. * rand1) * std::sqrt(std::fabs(std::log(rand2)));

    } while (NewEnergy > E || NewEnergy <= 0.);

    // Set the flag to stop tracing if the energy is below the threshold
    if (NewEnergy > minEnergy) {
      E = NewEnergy;
      auto direction = rayReflectionConedCosine<NumericType>(
          rayInternal::PI / 2. - std::min(incAngle, minAngle), rayDir,
          geomNormal, Rng);
      return std::pair<NumericType, rayTriple<NumericType>>{0., direction};
    } else {
      return std::pair<NumericType, rayTriple<NumericType>>{
          1., rayTriple<NumericType>{0., 0., 0.}};
    }
  }
  void initNew(rayRNG &RNG) override final {
    std::uniform_real_distribution<NumericType> uniDist;
    do {
      const auto rand1 = uniDist(RNG);
      const auto rand2 = uniDist(RNG);
      E = std::cos(rayInternal::PI * 2 * rand1) *
              std::sqrt(-2. * std::log(rand2)) * deltaIonEnergy +
          meanIonEnergy;
    } while (E < minEnergy);
  }

  int getRequiredLocalDataSize() const override final { return 4; }
  NumericType getSourceDistributionPower() const override final {
    return 1000.;
  }
  std::vector<std::string> getLocalDataLabels() const override final {
    return std::vector<std::string>{"ionSputteringRate", "ionEnhancedRate",
                                    "oxygenSputteringRate", "oxideEtchRate"};
  }

private:
  static constexpr NumericType A_p = 0.0337;
  static constexpr NumericType A_Si = 7.;
  static constexpr NumericType A_O = 2;
  static constexpr NumericType A_SiO2 = 0.3;

  static constexpr NumericType Eth_p = 0.;
  static constexpr NumericType Eth_Si = 15.;
  static constexpr NumericType Eth_O = 15.;
  static constexpr NumericType Eth_SiO2 = 15.;

  static constexpr NumericType Eref_max = 1.;

  const NumericType meanIonEnergy = 100.;
  const NumericType deltaIonEnergy = meanIonEnergy / 10.;
  static constexpr NumericType minEnergy =
      1.; // Discard particles with energy < 1eV

  static constexpr NumericType inflectAngle = 1.55334;
  static constexpr NumericType minAngle = 1.3962634;
  static constexpr NumericType n_l = 10.;
  static constexpr NumericType n_r = 1.;

  static constexpr NumericType A =
      1. / (1. + (n_l / n_r) * (rayInternal::PI / (2 * inflectAngle) - 1.));

  NumericType f_p_theta;
  NumericType f_Si_theta;
  NumericType f_O_theta;
  NumericType f_SiO2_theta;
  NumericType E;
};

template <typename NumericType, int D>
class SF6O2Etchant
    : public rayParticle<SF6O2Etchant<NumericType, D>, NumericType> {
public:
  void surfaceCollision(NumericType rayWeight,
                        const rayTriple<NumericType> &rayDir,
                        const rayTriple<NumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        rayTracingData<NumericType> &localData,
                        const rayTracingData<NumericType> *globalData,
                        rayRNG &Rng) override final {

    NumericType Seff;
    // F surface coverage
    const auto &phi_F = globalData->getVectorData(0)[primID];
    // O surface coverage
    const auto &phi_O = globalData->getVectorData(1)[primID];

    Seff = gamma_F * std::max(1. - phi_F - phi_O, 0.);

    localData.getVectorData(0)[primID] += rayWeight * Seff;
  }
  std::pair<NumericType, rayTriple<NumericType>>
  surfaceReflection(NumericType rayWeight, const rayTriple<NumericType> &rayDir,
                    const rayTriple<NumericType> &geomNormal,
                    const unsigned int primID, const int materialId,
                    const rayTracingData<NumericType> *globalData,
                    rayRNG &Rng) override final {
    auto direction = rayReflectionDiffuse<NumericType, D>(geomNormal, Rng);

    NumericType Seff;
    // F surface coverage
    const auto &phi_F = globalData->getVectorData(0)[primID];
    // O surface coverage
    const auto &phi_O = globalData->getVectorData(1)[primID];
    // Obtain the sticking probability
    Seff = gamma_F * std::max(1. - phi_F - phi_O, 0.);

    return std::pair<NumericType, rayTriple<NumericType>>{Seff, direction};
  }
  void initNew(rayRNG &RNG) override final {}
  int getRequiredLocalDataSize() const override final { return 1; }
  NumericType getSourceDistributionPower() const override final { return 1.; }
  std::vector<std::string> getLocalDataLabels() const override final {
    return std::vector<std::string>{"etchantRate"};
  }

private:
  static constexpr NumericType gamma_F = 0.7;
};

template <typename NumericType, int D>
class SF6O2Oxygen
    : public rayParticle<SF6O2Oxygen<NumericType, D>, NumericType> {
public:
  void surfaceCollision(NumericType rayWeight,
                        const rayTriple<NumericType> &rayDir,
                        const rayTriple<NumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        rayTracingData<NumericType> &localData,
                        const rayTracingData<NumericType> *globalData,
                        rayRNG &Rng) override final {
    NumericType Seff;
    // F surface coverage
    const auto &phi_F = globalData->getVectorData(0)[primID];
    // O surface coverage
    const auto &phi_O = globalData->getVectorData(1)[primID];
    Seff = gamma_O * std::max(1. - phi_O - phi_F, 0.);

    localData.getVectorData(0)[primID] += rayWeight * Seff;
  }
  std::pair<NumericType, rayTriple<NumericType>>
  surfaceReflection(NumericType rayWeight, const rayTriple<NumericType> &rayDir,
                    const rayTriple<NumericType> &geomNormal,
                    const unsigned int primID, const int materialId,
                    const rayTracingData<NumericType> *globalData,
                    rayRNG &Rng) override final {
    auto direction = rayReflectionDiffuse<NumericType, D>(geomNormal, Rng);

    NumericType Seff;
    const auto &phi_F = globalData->getVectorData(0)[primID];
    // O surface coverage
    const auto &phi_O = globalData->getVectorData(1)[primID];
    Seff = gamma_O * std::max(1. - phi_O - phi_F, 0.);

    return std::pair<NumericType, rayTriple<NumericType>>{Seff, direction};
  }
  void initNew(rayRNG &RNG) override final {}
  int getRequiredLocalDataSize() const override final { return 1; }
  NumericType getSourceDistributionPower() const override final { return 1.; }
  std::vector<std::string> getLocalDataLabels() const override final {
    return std::vector<std::string>{"oxygenRate"};
  }

private:
  static constexpr NumericType gamma_O = 1.;
};