#pragma once

#include <csTracingParticle.hpp>

#include <rayParticle.hpp>
#include <rayReflection.hpp>
#include <rayUtil.hpp>

#include <psLogger.hpp>
#include <psProcessModel.hpp>
#include <psSmartPointer.hpp>
#include <psSurfaceModel.hpp>
#include <psVelocityField.hpp>

template <typename NumericType, int D>
class SF6O2SurfaceModel : public psSurfaceModel<NumericType> {
  using psSurfaceModel<NumericType>::Coverages;

  // fluxes in (1e15 /cm²)
  const NumericType totalIonFlux = 12.;
  const NumericType totalEtchantFlux = 1.8e3;
  const NumericType totalOxygenFlux = 1.0e2;
  static constexpr double rho_Si = 5.02 * 1e7;     // in (1e15 atoms/cm³)
  static constexpr NumericType k_sigma_Si = 3.0e2; // in (1e15 cm⁻²s⁻¹)
  static constexpr NumericType beta_sigma_Si = 5.0e-2; // in (1e15 cm⁻²s⁻¹)

  const NumericType etchStop = 0.;

public:
  SF6O2SurfaceModel(const double ionFlux, const double etchantFlux,
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
    updateCoverages(Rates);
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

      if (psMaterialMap::mapToMaterial(static_cast<int>(materialIds[i])) ==
          psMaterial::Si) {

        etchRate[i] =
            -(1 / rho_Si) *
            (k_sigma_Si * eCoverage->at(i) / 4. +
             ionSputteringRate->at(i) * totalIonFlux +
             eCoverage->at(i) * ionEnhancedRate->at(i) * totalIonFlux);

        etchRate[i] *= 1e4; // to convert to micrometers / s
      }
    }

    if (stop) {
      std::fill(etchRate.begin(), etchRate.end(), 0.);
      psLogger::getInstance().addInfo("Etch stop depth reached.").print();
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

template <typename NumericType, int D>
class SF6O2Ion : public rayParticle<SF6O2Ion<NumericType, D>, NumericType> {
public:
  SF6O2Ion(NumericType passedPower = 100., NumericType oxySputterYield = 3)
      : power(passedPower), A_O(oxySputterYield) {}

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

    const double angle =
        std::acos(std::max(std::min(cosTheta, static_cast<NumericType>(1.)),
                           static_cast<NumericType>(0.)));

    if (cosTheta > 0.5) {
      f_Si_theta = 1.;
      f_O_theta = 1.;
    } else {
      f_Si_theta = std::max(3. - 6. * angle / rayInternal::PI, 0.);
      f_O_theta = std::max(3. - 6. * angle / rayInternal::PI, 0.);
    }

    f_p_theta = (1 + B_sp * (1 - cosTheta * cosTheta)) * cosTheta;

    const double sqrtE = std::sqrt(E);
    const double Y_sp =
        A_sp * std::max(sqrtE - std::sqrt(Eth_sp), 0.) * f_p_theta;
    const double Y_Si =
        A_Si * std::max(sqrtE - std::sqrt(Eth_Si), 0.) * f_Si_theta;
    const double Y_O = A_O * std::max(sqrtE - std::sqrt(Eth_O), 0.) * f_O_theta;

    // sputtering yield Y_sp ionSputteringRate
    localData.getVectorData(0)[primID] += Y_sp;

    // ion enhanced etching yield Y_Si ionEnhancedRate
    localData.getVectorData(1)[primID] += Y_Si;

    // ion enhanced O sputtering yield Y_O oxygenSputteringRate
    localData.getVectorData(2)[primID] += Y_O;
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
        std::acos(std::max(std::min(cosTheta, static_cast<NumericType>(1.)),
                           static_cast<NumericType>(0.)));

    // Small incident angles are reflected with the energy fraction centered at
    // 0
    NumericType Eref_peak;
    if (incAngle >= inflectAngle) {
      Eref_peak =
          Eref_max *
          (1 - (1 - A) * std::pow((halfPI - incAngle) / (halfPI - inflectAngle),
                                  n_r));
    } else {
      Eref_peak = Eref_max * A * std::pow(incAngle / inflectAngle, n_l);
    }
    // Gaussian distribution around the Eref_peak scaled by the particle energy
    NumericType tempEnergy = Eref_peak * E;

    NumericType NewEnergy;
    do {
      const auto rand1 = uniDist(Rng);
      const auto rand2 = uniDist(Rng);
      NewEnergy = tempEnergy +
                  (std::min((E - tempEnergy), tempEnergy) + E * 0.05) *
                      (1 - 2. * rand1) * std::sqrt(std::fabs(std::log(rand2)));

    } while (NewEnergy > E || NewEnergy < 0.);

    // Set the flag to stop tracing if the energy is below the threshold
    if (NewEnergy > minEnergy) {
      E = NewEnergy;

      auto direction = rayReflectionConedCosine<NumericType, D>(
          halfPI - std::min(incAngle, minAngle), rayDir, geomNormal, Rng);

      return std::pair<NumericType, rayTriple<NumericType>>{0., direction};
    } else {
      return std::pair<NumericType, rayTriple<NumericType>>{
          1., rayTriple<NumericType>{0., 0., 0.}};
    }
  }
  void initNew(rayRNG &RNG) override final {
    do {
      auto rand1 = uniDist(RNG) * (twoPI - 2 * peak) + peak;
      E = (1 + std::cos(rand1)) * (power / 2 * 0.75 + 10);
    } while (E < minEnergy);
  }

  int getRequiredLocalDataSize() const override final { return 3; }

  NumericType getSourceDistributionPower() const override final {
    return 1000.;
  }

  std::vector<std::string> getLocalDataLabels() const override final {
    return std::vector<std::string>{"ionSputteringRate", "ionEnhancedRate",
                                    "oxygenSputteringRate"};
  }

  void logData(rayDataLog<NumericType> &dataLog) override final {
    NumericType max = 0.75 * power + 20 + 1e-6;
    int idx = static_cast<int>(50 * E / max);
    assert(idx < 50 && idx >= 0);
    dataLog.data[0][idx] += 1.;
  }

private:
  std::uniform_real_distribution<NumericType> uniDist;

  static constexpr NumericType A_sp = 0.00339;
  static constexpr NumericType A_Si = 7.;
  const NumericType A_O = 2.;

  static constexpr NumericType Eth_sp = 18.;
  static constexpr NumericType Eth_Si = 15.;
  static constexpr NumericType Eth_O = 10.;
  static constexpr NumericType B_sp = 9.3;

  static constexpr NumericType Eref_max = 1.;

  static constexpr NumericType inflectAngle = 1.55334;
  static constexpr NumericType minAngle = 1.3962634;
  static constexpr NumericType n_l = 10.;
  static constexpr NumericType n_r = 1.;

  static constexpr NumericType twoPI = 6.283185307179586;
  static constexpr NumericType halfPI = 6.283185307179586 / 4;

  static constexpr NumericType A =
      1. / (1. + (n_l / n_r) * (halfPI / inflectAngle - 1.));

  NumericType f_p_theta;
  NumericType f_Si_theta;
  NumericType f_O_theta;
  NumericType f_SiO2_theta;

  // ion energy
  static constexpr NumericType minEnergy =
      4.; // Discard particles with energy < 1eV
  const NumericType power;
  static constexpr NumericType peak = 0.2;
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

    // NumericType Seff;
    // // F surface coverage
    // const auto &phi_F = globalData->getVectorData(0)[primID];
    // // O surface coverage
    // const auto &phi_O = globalData->getVectorData(1)[primID];

    // Seff = gamma_F * std::max(1. - phi_F - phi_O, 0.);

    // Rate is normalized by dividing with the local sticking coefficient
    localData.getVectorData(0)[primID] += rayWeight; // * Seff;
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
    // NumericType Seff;
    // // F surface coverage
    // const auto &phi_F = globalData->getVectorData(0)[primID];
    // // O surface coverage
    // const auto &phi_O = globalData->getVectorData(1)[primID];
    // Seff = gamma_O * std::max(1. - phi_O - phi_F, 0.);

    // Rate is normalized by dividing with the local sticking coefficient
    localData.getVectorData(0)[primID] += rayWeight; // * Seff;
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

template <typename NumericType, int D>
class SF6O2Etching : public psProcessModel<NumericType, D> {
public:
  SF6O2Etching(const double ionFlux, const double etchantFlux,
               const double oxygenFlux, const NumericType rfBias,
               const NumericType oxySputterYield = 2.,
               const NumericType etchStopDepth = 0.) {
    // particles
    auto ion =
        std::make_unique<SF6O2Ion<NumericType, D>>(rfBias, oxySputterYield);
    auto etchant = std::make_unique<SF6O2Etchant<NumericType, D>>();
    auto oxygen = std::make_unique<SF6O2Oxygen<NumericType, D>>();

    // surface model
    auto surfModel = psSmartPointer<SF6O2SurfaceModel<NumericType, D>>::New(
        ionFlux, etchantFlux, oxygenFlux, etchStopDepth);

    // velocity field
    auto velField = psSmartPointer<psDefaultVelocityField<NumericType>>::New();

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("SF6O2Etching");
    this->insertNextParticleType(ion, 50 /* log particle energies */);
    this->insertNextParticleType(etchant);
    this->insertNextParticleType(oxygen);
  }
};