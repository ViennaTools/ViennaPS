#pragma once

#include <rayParticle.hpp>
#include <rayReflection.hpp>
#include <rayUtil.hpp>

#include "../psProcessModel.hpp"
#include "../psSurfaceModel.hpp"
#include "../psVelocityField.hpp"

namespace viennaps {

using namespace viennacore;

template <typename NumericType> struct SF6O2Parameters {
  // fluxes in (1e15 /cm² /s)
  NumericType ionFlux = 12.;
  NumericType etchantFlux = 1.8e3;
  NumericType oxygenFlux = 1.0e2;

  NumericType etchStopDepth = std::numeric_limits<NumericType>::lowest();

  // sticking probabilities
  NumericType beta_F = 0.7;
  NumericType beta_O = 1.;

  // Mask
  struct MaskType {
    NumericType rho = 500.; // 1e22 atoms/cm³
    NumericType beta_F = 0.01;
    NumericType beta_O = 0.1;

    NumericType Eth_sp = 20.; // eV
    NumericType A_sp = 0.0139;
    NumericType B_sp = 9.3;
  } Mask;

  // Si
  struct SiType {
    // density
    NumericType rho = 5.02; // 1e22 atoms/cm³

    // sputtering coefficients
    NumericType Eth_sp = 20.; // eV
    NumericType Eth_ie = 4.;  // eV
    NumericType A_sp = 0.0337;
    NumericType B_sp = 9.3;
    NumericType A_ie = 0.0361;

    // chemical etching
    NumericType k_sigma = 3.0e2;     // in (1e15 cm⁻²s⁻¹)
    NumericType beta_sigma = 5.0e-2; // in (1e15 cm⁻²s⁻¹)
  } Si;

  // Passivation
  struct PassivationType {
    // sputtering coefficients
    NumericType Eth_ie = 4.; // eV
    NumericType A_ie = 0.0361;
  } Passivation;

  struct IonType {
    NumericType meanEnergy = 100.; // eV
    NumericType sigmaEnergy = 10.; // eV
    NumericType exponent = 500.;

    NumericType inflectAngle = 1.55334303;
    NumericType n_l = 10.;
    NumericType minAngle = 1.3962634;
  } Ions;
};

namespace impl {

template <typename NumericType, int D>
class SF6O2SurfaceModel : public SurfaceModel<NumericType> {
public:
  using SurfaceModel<NumericType>::coverages;
  const SF6O2Parameters<NumericType> &params;

  SF6O2SurfaceModel(const SF6O2Parameters<NumericType> &pParams)
      : params(pParams) {}

  void initializeCoverages(unsigned numGeometryPoints) override {
    if (coverages == nullptr) {
      coverages = SmartPointer<viennals::PointData<NumericType>>::New();
    } else {
      coverages->clear();
    }
    std::vector<NumericType> cov(numGeometryPoints, 0.);
    coverages->insertNextScalarData(cov, "eCoverage");
    coverages->insertNextScalarData(cov, "oCoverage");
  }

  SmartPointer<std::vector<NumericType>>
  calculateVelocities(SmartPointer<viennals::PointData<NumericType>> rates,
                      const std::vector<Vec3D<NumericType>> &coordinates,
                      const std::vector<NumericType> &materialIds) override {
    updateCoverages(rates, materialIds);
    const auto numPoints = rates->getScalarData(0)->size();
    std::vector<NumericType> etchRate(numPoints, 0.);

    const auto ionEnhancedRate = rates->getScalarData("ionEnhancedRate");
    const auto ionSputteringRate = rates->getScalarData("ionSputteringRate");
    const auto etchantRate = rates->getScalarData("etchantRate");
    const auto eCoverage = coverages->getScalarData("eCoverage");

    bool stop = false;

    for (size_t i = 0; i < numPoints; ++i) {
      if (coordinates[i][D - 1] < params.etchStopDepth) {
        stop = true;
        break;
      }

      if (MaterialMap::isMaterial(materialIds[i], Material::Mask)) {
        etchRate[i] =
            -(1 / params.Mask.rho) * ionSputteringRate->at(i) * params.ionFlux;
      } else {
        etchRate[i] =
            -(1 / params.Si.rho) * (params.Si.k_sigma * eCoverage->at(i) / 4. +
                                    ionSputteringRate->at(i) * params.ionFlux +
                                    eCoverage->at(i) * ionEnhancedRate->at(i) *
                                        params.ionFlux); // in um / s
      }
    }

    if (stop) {
      std::fill(etchRate.begin(), etchRate.end(), 0.);
      Logger::getInstance().addInfo("Etch stop depth reached.").print();
    }

    return SmartPointer<std::vector<NumericType>>::New(std::move(etchRate));
  }

  void updateCoverages(SmartPointer<viennals::PointData<NumericType>> rates,
                       const std::vector<NumericType> &materialIds) override {
    // update coverages based on fluxes
    const auto numPoints = rates->getScalarData(0)->size();

    const auto etchantRate = rates->getScalarData("etchantRate");
    const auto ionEnhancedRate = rates->getScalarData("ionEnhancedRate");
    const auto oxygenRate = rates->getScalarData("oxygenRate");
    const auto oxygenSputteringRate =
        rates->getScalarData("oxygenSputteringRate");

    // etchant fluorine coverage
    auto eCoverage = coverages->getScalarData("eCoverage");
    eCoverage->resize(numPoints);
    // oxygen coverage
    auto oCoverage = coverages->getScalarData("oCoverage");
    oCoverage->resize(numPoints);
    for (size_t i = 0; i < numPoints; ++i) {
      auto Gb_F = etchantRate->at(i) * params.etchantFlux * params.beta_F;
      auto Gb_O = oxygenRate->at(i) * params.oxygenFlux * params.beta_O;
      auto GY_ie = ionEnhancedRate->at(i) * params.ionFlux;
      auto GY_o = oxygenSputteringRate->at(i) * params.ionFlux;

      if (etchantRate->at(i) < 1e-6) {
        eCoverage->at(i) = 0;
      } else {
        double tmp = 1 + ((params.Si.k_sigma + 2 * GY_ie) / Gb_F) *
                             (1 + Gb_O / (params.Si.beta_sigma + GY_o));
        eCoverage->at(i) = 1 / tmp;
      }

      if (oxygenRate->at(i) < 1e-6) {
        oCoverage->at(i) = 0;
      } else {
        double tmp = 1 + ((params.Si.beta_sigma + GY_ie) / Gb_O) *
                             (1 + Gb_F / (params.Si.k_sigma + 2 * GY_ie));
        oCoverage->at(i) = 1 / tmp;
      }
    }
  }
};

template <typename NumericType, int D>
class SF6SurfaceModel : public SF6O2SurfaceModel<NumericType, D> {
public:
  using SurfaceModel<NumericType>::coverages;
  using SF6O2SurfaceModel<NumericType, D>::params;

  SF6SurfaceModel(const SF6O2Parameters<NumericType> &pParams)
      : SF6O2SurfaceModel<NumericType, D>(pParams) {}

  void updateCoverages(SmartPointer<viennals::PointData<NumericType>> rates,
                       const std::vector<NumericType> &materialIds) override {
    // update coverages based on fluxes
    const auto numPoints = rates->getScalarData(0)->size();

    const auto etchantRate = rates->getScalarData("etchantRate");
    const auto ionEnhancedRate = rates->getScalarData("ionEnhancedRate");

    // etchant fluorine coverage
    auto eCoverage = coverages->getScalarData("eCoverage");
    eCoverage->resize(numPoints);
    for (size_t i = 0; i < numPoints; ++i) {
      if (etchantRate->at(i) < 1e-6) {
        eCoverage->at(i) = 0;
      } else {
        eCoverage->at(i) =
            etchantRate->at(i) * params.etchantFlux * params.beta_F /
            (etchantRate->at(i) * params.etchantFlux * params.beta_F +
             params.Si.k_sigma + 2 * ionEnhancedRate->at(i) * params.ionFlux);
      }
    }
  }
};

template <typename NumericType, int D>
class SF6O2Ion
    : public viennaray::Particle<SF6O2Ion<NumericType, D>, NumericType> {
public:
  SF6O2Ion(const SF6O2Parameters<NumericType> &pParams)
      : params(pParams),
        A(1. /
          (1. + params.Ions.n_l * (M_PI_2 / params.Ions.inflectAngle - 1.))) {}

  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                        const Vec3D<NumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        viennaray::TracingData<NumericType> &localData,
                        const viennaray::TracingData<NumericType> *globalData,
                        RNG &) override final {

    // collect data for this hit
    assert(primID < localData.getVectorData(0).size() && "id out of bounds");

    const double cosTheta = -DotProduct(rayDir, geomNormal);

    assert(cosTheta >= 0 && "Hit backside of disc");
    assert(cosTheta <= 1 + 1e6 && "Error in calculating cos theta");
    assert(rayWeight > 0. && "Invalid ray weight");

    const double angle = std::acos(std::max(std::min(cosTheta, 1.), 0.));

    NumericType f_ie_theta;
    if (cosTheta > 0.5) {
      f_ie_theta = 1.;
    } else {
      f_ie_theta = 3. - 6. * angle / M_PI;
    }
    NumericType A_sp = params.Si.A_sp;
    NumericType B_sp = params.Si.B_sp;
    NumericType Eth_sp = params.Si.Eth_sp;
    if (MaterialMap::isMaterial(materialId, Material::Mask)) {
      A_sp = params.Mask.A_sp;
      B_sp = params.Mask.B_sp;
      Eth_sp = params.Mask.Eth_sp;
    }

    NumericType f_sp_theta = (1 + B_sp * (1 - cosTheta * cosTheta)) * cosTheta;

    double sqrtE = std::sqrt(E);
    NumericType Y_sp =
        params.Si.A_sp * std::max(sqrtE - std::sqrt(Eth_sp), 0.) * f_sp_theta;
    NumericType Y_Si = params.Si.A_ie *
                       std::max(sqrtE - std::sqrt(params.Si.Eth_ie), 0.) *
                       f_ie_theta;
    NumericType Y_O =
        params.Passivation.A_ie *
        std::max(sqrtE - std::sqrt(params.Passivation.Eth_ie), 0.) * f_ie_theta;

    assert(Y_sp >= 0. && "Invalid yield");
    assert(Y_Si >= 0. && "Invalid yield");
    assert(Y_O >= 0. && "Invalid yield");

    // sputtering yield Y_sp ionSputteringRate
    localData.getVectorData(0)[primID] += Y_sp;

    // ion enhanced etching yield Y_Si ionEnhancedRate
    localData.getVectorData(1)[primID] += Y_Si;

    // ion enhanced O sputtering yield Y_O oxygenSputteringRate
    localData.getVectorData(2)[primID] += Y_O;
  }

  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                    const Vec3D<NumericType> &geomNormal,
                    const unsigned int primId, const int materialId,
                    const viennaray::TracingData<NumericType> *globalData,
                    RNG &Rng) override final {
    auto cosTheta = -DotProduct(rayDir, geomNormal);

    assert(cosTheta >= 0 && "Hit backside of disc");
    assert(cosTheta <= 1 + 1e-6 && "Error in calculating cos theta");

    NumericType incAngle =
        std::acos(std::max(std::min(cosTheta, static_cast<NumericType>(1.)),
                           static_cast<NumericType>(0.)));

    // Small incident angles are reflected with the energy fraction centered at
    // 0
    NumericType Eref_peak;
    if (incAngle >= params.Ions.inflectAngle) {
      Eref_peak = (1 - (1 - A) * (M_PI_2 - incAngle) /
                           (M_PI_2 - params.Ions.inflectAngle));
    } else {
      Eref_peak =
          A * std::pow(incAngle / params.Ions.inflectAngle, params.Ions.n_l);
    }
    // Gaussian distribution around the Eref_peak scaled by the particle energy
    NumericType NewEnergy;
    std::normal_distribution<NumericType> normalDist(E * Eref_peak, 0.1 * E);
    do {
      NewEnergy = normalDist(Rng);
    } while (NewEnergy > E || NewEnergy < 0.);

    // Set the flag to stop tracing if the energy is below the threshold
    if (NewEnergy > params.Si.Eth_ie) {
      E = NewEnergy;
      auto direction = viennaray::ReflectionConedCosine<NumericType, D>(
          rayDir, geomNormal, Rng, std::max(incAngle, params.Ions.minAngle));
      return std::pair<NumericType, Vec3D<NumericType>>{0., direction};
    } else {
      return std::pair<NumericType, Vec3D<NumericType>>{
          1., Vec3D<NumericType>{0., 0., 0.}};
    }
  }
  void initNew(RNG &rngState) override final {
    std::normal_distribution<NumericType> normalDist{params.Ions.meanEnergy,
                                                     params.Ions.sigmaEnergy};
    do {
      E = normalDist(rngState);
    } while (E <= 0.);
  }
  NumericType getSourceDistributionPower() const override final {
    return params.Ions.exponent;
  }
  std::vector<std::string> getLocalDataLabels() const override final {
    return {"ionSputteringRate", "ionEnhancedRate", "oxygenSputteringRate"};
  }

private:
  const SF6O2Parameters<NumericType> &params;
  const NumericType A;
  NumericType E;
};

template <typename NumericType, int D>
class SF6O2Etchant
    : public viennaray::Particle<SF6O2Etchant<NumericType, D>, NumericType> {
  const SF6O2Parameters<NumericType> &params;

public:
  SF6O2Etchant(const SF6O2Parameters<NumericType> &pParams) : params(pParams) {}

  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &,
                        const Vec3D<NumericType> &, const unsigned int primID,
                        const int,
                        viennaray::TracingData<NumericType> &localData,
                        const viennaray::TracingData<NumericType> *,
                        RNG &) override final {
    localData.getVectorData(0)[primID] += rayWeight;
  }
  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                    const Vec3D<NumericType> &geomNormal,
                    const unsigned int primID, const int materialId,
                    const viennaray::TracingData<NumericType> *globalData,
                    RNG &rngState) override final {

    // F surface coverage
    const auto &phi_F = globalData->getVectorData(0)[primID];
    // O surface coverage
    const auto &phi_O = globalData->getVectorData(1)[primID];
    // Obtain the sticking probability
    NumericType beta = params.beta_F;
    if (MaterialMap::isMaterial(materialId, Material::Mask))
      beta = params.Mask.beta_F;
    NumericType S_eff = beta * std::max(1. - phi_F - phi_O, 0.);

    auto direction =
        viennaray::ReflectionDiffuse<NumericType, D>(geomNormal, rngState);
    return std::pair<NumericType, Vec3D<NumericType>>{S_eff, direction};
  }
  NumericType getSourceDistributionPower() const override final { return 1.; }
  std::vector<std::string> getLocalDataLabels() const override final {
    return {"etchantRate"};
  }
};

template <typename NumericType, int D>
class SF6Etchant
    : public viennaray::Particle<SF6Etchant<NumericType, D>, NumericType> {
  const SF6O2Parameters<NumericType> &params;

public:
  SF6Etchant(const SF6O2Parameters<NumericType> &pParams) : params(pParams) {}

  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &,
                        const Vec3D<NumericType> &, const unsigned int primID,
                        const int,
                        viennaray::TracingData<NumericType> &localData,
                        const viennaray::TracingData<NumericType> *,
                        RNG &) override final {
    localData.getVectorData(0)[primID] += rayWeight;
  }
  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                    const Vec3D<NumericType> &geomNormal,
                    const unsigned int primID, const int materialId,
                    const viennaray::TracingData<NumericType> *globalData,
                    RNG &rngState) override final {

    // F surface coverage
    const auto &phi_F = globalData->getVectorData(0)[primID];
    // Obtain the sticking probability
    NumericType beta = params.beta_F;
    if (MaterialMap::isMaterial(materialId, Material::Mask))
      beta = params.Mask.beta_F;
    NumericType S_eff = beta * std::max(1. - phi_F, 0.);

    auto direction =
        viennaray::ReflectionDiffuse<NumericType, D>(geomNormal, rngState);
    return std::pair<NumericType, Vec3D<NumericType>>{S_eff, direction};
  }
  NumericType getSourceDistributionPower() const override final { return 1.; }
  std::vector<std::string> getLocalDataLabels() const override final {
    return {"etchantRate"};
  }
};

template <typename NumericType, int D>
class SF6O2Oxygen
    : public viennaray::Particle<SF6O2Oxygen<NumericType, D>, NumericType> {
  const SF6O2Parameters<NumericType> &params;

public:
  SF6O2Oxygen(const SF6O2Parameters<NumericType> &pParams) : params(pParams) {}

  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &,
                        const Vec3D<NumericType> &, const unsigned int primID,
                        const int,
                        viennaray::TracingData<NumericType> &localData,
                        const viennaray::TracingData<NumericType> *,
                        RNG &) override final {
    // Rate is normalized by dividing with the local sticking coefficient
    localData.getVectorData(0)[primID] += rayWeight;
  }
  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                    const Vec3D<NumericType> &geomNormal,
                    const unsigned int primID, const int materialId,
                    const viennaray::TracingData<NumericType> *globalData,
                    RNG &rngState) override final {

    NumericType S_eff;
    const auto &phi_F = globalData->getVectorData(0)[primID];
    const auto &phi_O = globalData->getVectorData(1)[primID];
    NumericType beta = params.beta_O;
    if (MaterialMap::isMaterial(materialId, Material::Mask))
      beta = params.Mask.beta_O;
    S_eff = beta * std::max(1. - phi_O - phi_F, 0.);

    auto direction =
        viennaray::ReflectionDiffuse<NumericType, D>(geomNormal, rngState);
    return std::pair<NumericType, Vec3D<NumericType>>{S_eff, direction};
  }
  NumericType getSourceDistributionPower() const override final { return 1.; }
  std::vector<std::string> getLocalDataLabels() const override final {
    return {"oxygenRate"};
  }
};
} // namespace impl

/// Model for etching Si in a SF6/O2 plasma. The model is based on the paper by
/// Belen et al., Vac. Sci. Technol. A 23, 99–113 (2005),
/// DOI: https://doi.org/10.1116/1.1830495
/// The resulting rate is in units of um / s.
template <typename NumericType, int D>
class SF6O2Etching : public ProcessModel<NumericType, D> {
public:
  SF6O2Etching() { initializeModel(); }

  // All flux values are in units 1e16 / cm²
  SF6O2Etching(const double ionFlux, const double etchantFlux,
               const double oxygenFlux, const NumericType meanEnergy /* eV */,
               const NumericType sigmaEnergy /* eV */, // 5 parameters
               const NumericType ionExponent = 300.,
               const NumericType oxySputterYield = 2.,
               const NumericType etchStopDepth =
                   std::numeric_limits<NumericType>::lowest()) {
    params.ionFlux = ionFlux;
    params.etchantFlux = etchantFlux;
    params.oxygenFlux = oxygenFlux;
    params.Ions.meanEnergy = meanEnergy;
    params.Ions.sigmaEnergy = sigmaEnergy;
    params.Ions.exponent = ionExponent;
    params.Passivation.A_ie = oxySputterYield;
    params.etchStopDepth = etchStopDepth;
    initializeModel();
  }

  SF6O2Etching(const SF6O2Parameters<NumericType> &pParams) : params(pParams) {
    initializeModel();
  }

  void setParameters(const SF6O2Parameters<NumericType> &pParams) {
    params = pParams;
  }

  SF6O2Parameters<NumericType> &getParameters() { return params; }

private:
  void initializeModel() {
    // particles
    auto ion = std::make_unique<impl::SF6O2Ion<NumericType, D>>(params);
    auto etchant = std::make_unique<impl::SF6O2Etchant<NumericType, D>>(params);
    auto oxygen = std::make_unique<impl::SF6O2Oxygen<NumericType, D>>(params);

    // surface model
    auto surfModel =
        SmartPointer<impl::SF6O2SurfaceModel<NumericType, D>>::New(params);

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New(2);

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("SF6O2Etching");
    this->insertNextParticleType(ion);
    this->insertNextParticleType(etchant);
    this->insertNextParticleType(oxygen);
  }

  SF6O2Parameters<NumericType> params;
};

template <typename NumericType, int D>
class SF6Etching : public ProcessModel<NumericType, D> {
public:
  SF6Etching() { initializeModel(); }

  // All flux values are in units 1e16 / cm²
  SF6Etching(const double ionFlux, const double etchantFlux,
             const NumericType meanEnergy /* eV */,
             const NumericType sigmaEnergy /* eV */, // 5 parameters
             const NumericType ionExponent = 300.,
             const NumericType etchStopDepth =
                 std::numeric_limits<NumericType>::lowest()) {
    params.ionFlux = ionFlux;
    params.etchantFlux = etchantFlux;
    params.Ions.meanEnergy = meanEnergy;
    params.Ions.sigmaEnergy = sigmaEnergy;
    params.Ions.exponent = ionExponent;
    params.etchStopDepth = etchStopDepth;
    initializeModel();
  }

  SF6Etching(const SF6O2Parameters<NumericType> &pParams) : params(pParams) {
    initializeModel();
  }

  void setParameters(const SF6O2Parameters<NumericType> &pParams) {
    params = pParams;
  }

  SF6O2Parameters<NumericType> &getParameters() { return params; }

private:
  void initializeModel() {
    // particles
    auto ion = std::make_unique<impl::SF6O2Ion<NumericType, D>>(params);
    auto etchant = std::make_unique<impl::SF6Etchant<NumericType, D>>(params);

    // surface model
    auto surfModel =
        SmartPointer<impl::SF6SurfaceModel<NumericType, D>>::New(params);

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New(2);

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("SF6Etching");
    this->insertNextParticleType(ion);
    this->insertNextParticleType(etchant);
  }

  SF6O2Parameters<NumericType> params;
};

} // namespace viennaps
