#pragma once

#include <rayParticle.hpp>
#include <rayReflection.hpp>
#include <rayUtil.hpp>

#include "../psConstants.hpp"
#include "../psProcessModel.hpp"
#include "../psSurfaceModel.hpp"
#include "../psUnits.hpp"
#include "../psVelocityField.hpp"
#include "psSF6O2Parameters.hpp"

#include "psSF6O2Parameters.hpp"

namespace viennaps {

using namespace viennacore;

namespace impl {

template <typename NumericType, int D>
class SF6O2SurfaceModel : public SurfaceModel<NumericType> {
public:
  using SurfaceModel<NumericType>::coverages;
  using SurfaceModel<NumericType>::surfaceData;
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

  void initializeSurfaceData(unsigned numGeometryPoints) override {
    if (Logger::getLogLevel() > 3) {
      if (surfaceData == nullptr) {
        surfaceData = SmartPointer<viennals::PointData<NumericType>>::New();
      } else {
        surfaceData->clear();
      }

      std::vector<NumericType> data(numGeometryPoints, 0.);
      surfaceData->insertNextScalarData(data, "ionEnhancedRate");
      surfaceData->insertNextScalarData(data, "sputterRate");
      surfaceData->insertNextScalarData(data, "chemicalRate");
    }
  }

  SmartPointer<std::vector<NumericType>>
  calculateVelocities(SmartPointer<viennals::PointData<NumericType>> fluxes,
                      const std::vector<Vec3D<NumericType>> &coordinates,
                      const std::vector<NumericType> &materialIds) override {
<<<<<<< HEAD
    updateCoverages(fluxes, materialIds);
=======
>>>>>>> master
    const auto numPoints = fluxes->getScalarData(0)->size();
    std::vector<NumericType> etchRate(numPoints, 0.);

    const auto ionEnhancedFlux = fluxes->getScalarData("ionEnhancedFlux");
    const auto ionSputterFlux = fluxes->getScalarData("ionSputterFlux");

    const auto etchantFlux = fluxes->getScalarData("etchantFlux");
    const auto eCoverage = coverages->getScalarData("eCoverage");

    // save the etch rate components for visualization
    std::vector<NumericType> *ieRate = nullptr, *spRate = nullptr,
                             *chRate = nullptr;
    if (Logger::getLogLevel() > 3) {
      ieRate = surfaceData->getScalarData("ionEnhancedRate");
      spRate = surfaceData->getScalarData("sputterRate");
      chRate = surfaceData->getScalarData("chemicalRate");
      ieRate->resize(numPoints);
      spRate->resize(numPoints);
      chRate->resize(numPoints);
    }
    bool stop = false;

    for (size_t i = 0; i < numPoints; ++i) {
      if (coordinates[i][D - 1] < params.etchStopDepth) {
        stop = true;
        break;
      }

      const auto sputterRate = ionSputterFlux->at(i) * params.ionFlux;
      const auto ionEnhancedRate =
          eCoverage->at(i) * ionEnhancedFlux->at(i) * params.ionFlux;
      const auto chemicalRate = params.Si.k_sigma * eCoverage->at(i) / 4.;

      // The etch rate is calculated in nm/s
      const double unitConversion =
          units::Time::getInstance().convertSecond() /
          units::Length::getInstance().convertNanometer();

      if (MaterialMap::isMaterial(materialIds[i], Material::Mask)) {
        etchRate[i] = -(1 / params.Mask.rho) * sputterRate * unitConversion;
        if (Logger::getLogLevel() > 3) {
          spRate->at(i) = sputterRate;
          ieRate->at(i) = 0.;
          chRate->at(i) = 0.;
        }
      } else {
        etchRate[i] = -(1 / params.Si.rho) *
                      (chemicalRate + sputterRate + ionEnhancedRate) *
                      unitConversion;
        if (Logger::getLogLevel() > 3) {
          spRate->at(i) = sputterRate;
          ieRate->at(i) = ionEnhancedRate;
          chRate->at(i) = chemicalRate;
        }
      }
    }

    if (stop) {
      std::fill(etchRate.begin(), etchRate.end(), 0.);
      Logger::getInstance().addInfo("Etch stop depth reached.").print();
    }

    return SmartPointer<std::vector<NumericType>>::New(std::move(etchRate));
  }

  void updateCoverages(SmartPointer<viennals::PointData<NumericType>> fluxes,
                       const std::vector<NumericType> &materialIds) override {
    // update coverages based on fluxes
    const auto numPoints = fluxes->getScalarData(0)->size();

    const auto etchantFlux = fluxes->getScalarData("etchantFlux");
    std::vector<NumericType> oxygenFlux(numPoints, 0.);
    if (params.oxygenFlux > 0)
      oxygenFlux = *fluxes->getScalarData("oxygenFlux");

    const auto ionEnhancedFlux = fluxes->getScalarData("ionEnhancedFlux");
    const auto ionEnhancedPassivationFlux =
        fluxes->getScalarData("ionEnhancedPassivationFlux");

    // etchant fluorine coverage
    auto eCoverage = coverages->getScalarData("eCoverage");
    eCoverage->resize(numPoints);
    // oxygen coverage
    auto oCoverage = coverages->getScalarData("oCoverage");
    oCoverage->resize(numPoints);

    for (size_t i = 0; i < numPoints; ++i) {
      auto Gb_F = etchantFlux->at(i) * params.etchantFlux;
      auto Gb_O = oxygenFlux.at(i) * params.oxygenFlux;
      auto GY_ie = ionEnhancedFlux->at(i) * params.ionFlux;
      auto GY_o = ionEnhancedPassivationFlux->at(i) * params.ionFlux;

      auto a = (params.Si.k_sigma + 2 * GY_ie) / Gb_F;
      auto b = (params.Si.beta_sigma + GY_o) / Gb_O;

      eCoverage->at(i) = Gb_F < 1e-6 ? 0. : 1 / (1 + (a * (1 + 1 / b)));
      oCoverage->at(i) = Gb_O < 1e-6 ? 0. : 1 / (1 + (b * (1 + 1 / a)));
    }
  }
};

template <typename NumericType, int D>
class SF6O2Ion
    : public viennaray::Particle<SF6O2Ion<NumericType, D>, NumericType> {
public:
  SF6O2Ion(const SF6O2Parameters<NumericType> &pParams)
      : params(pParams),
        A_energy(1. / (1. + params.Ions.n_l *
                                (M_PI_2 / params.Ions.inflectAngle - 1.))),
        sqrt_E_th_ie_O(std::sqrt(params.Passivation.Eth_ie)),
        sqrt_E_th_ie_Si(std::sqrt(params.Si.Eth_ie)) {}

  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                        const Vec3D<NumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        viennaray::TracingData<NumericType> &localData,
                        const viennaray::TracingData<NumericType> *globalData,
                        RNG &) override final {
    // collect data for this hit
    auto cosTheta = std::clamp(-DotProduct(rayDir, geomNormal), NumericType(0),
                               NumericType(1));
    NumericType angle = std::acos(cosTheta);

    assert(cosTheta >= 0 && "Hit backside of disc");
    assert(cosTheta <= 1 + 1e6 && "Error in calculating cos theta");
    assert(rayWeight > 0. && "Invalid ray weight");

    NumericType A_sp = params.Si.A_sp;
    NumericType B_sp = params.Si.B_sp;
    NumericType Eth_sp = params.Si.Eth_sp;
    if (MaterialMap::isMaterial(materialId, Material::Mask)) {
      A_sp = params.Mask.A_sp;
      B_sp = params.Mask.B_sp;
      Eth_sp = params.Mask.Eth_sp;
    }

    // NumericType f_sp_theta =
    //     std::max((1 + B_sp * (1 - cosTheta * cosTheta)) *
    //                  std::cos(angle / params.Si.theta_g_sp * M_PI_2),
    //              0.);
    // NumericType f_ie_theta =
    //     std::max((1 + params.Si.B_ie * (1 - cosTheta * cosTheta)) *
    //                  std::cos(angle / params.Si.theta_g_ie * M_PI_2),
    //              0.);
    NumericType f_sp_theta = 1.;
    NumericType f_ie_theta = 1.;
    if (cosTheta < 0.5) {
      f_ie_theta = std::max(3 - 6 * angle / M_PI, 0.);
    }

    const double sqrtE = std::sqrt(E);
    NumericType Y_sp =
        params.Si.A_sp * std::max(sqrtE - std::sqrt(Eth_sp), 0.) * f_sp_theta;
    NumericType Y_Si =
        params.Si.A_ie * std::max(sqrtE - sqrt_E_th_ie_Si, 0.) * f_ie_theta;
    NumericType Y_O = params.Passivation.A_ie *
                      std::max(sqrtE - sqrt_E_th_ie_O, 0.) * f_ie_theta;

    assert(Y_sp >= 0. && "Invalid yield");
    assert(Y_Si >= 0. && "Invalid yield");
    assert(Y_O >= 0. && "Invalid yield");

    // sputtering yield Y_sp ionSputterFlux
    localData.getVectorData(0)[primID] += Y_sp * rayWeight;

    // ion enhanced etching yield Y_Si ionEnhancedFlux
    localData.getVectorData(1)[primID] += Y_Si * rayWeight;

    // ion enhanced O sputtering yield Y_O ionEnhancedPassivationFlux
    localData.getVectorData(2)[primID] += Y_O * rayWeight;
  }

  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                    const Vec3D<NumericType> &geomNormal,
                    const unsigned int primId, const int materialId,
                    const viennaray::TracingData<NumericType> *globalData,
                    RNG &Rng) override final {
    auto cosTheta = std::clamp(-DotProduct(rayDir, geomNormal), NumericType(0),
                               NumericType(1));
    NumericType incAngle = std::acos(cosTheta);

    // Small incident angles are reflected with the energy fraction centered at
    // 0
    NumericType Eref_peak;
    if (incAngle >= params.Ions.inflectAngle) {
      Eref_peak = (1 - (1 - A_energy) * (M_PI_2 - incAngle) /
                           (M_PI_2 - params.Ions.inflectAngle));
    } else {
      Eref_peak = A_energy * std::pow(incAngle / params.Ions.inflectAngle,
                                      params.Ions.n_l);
    }
    // Gaussian distribution around the Eref_peak scaled by the particle energy
    NumericType newEnergy;
    std::normal_distribution<NumericType> normalDist(E * Eref_peak, 0.1 * E);
    do {
      newEnergy = normalDist(Rng);
    } while (newEnergy > E || newEnergy < 0.);

    NumericType sticking = 1.;
    if (incAngle > params.Ions.thetaRMin) {
      sticking =
          1. - std::clamp((incAngle - params.Ions.thetaRMin) /
                              (params.Ions.thetaRMax - params.Ions.thetaRMin),
                          NumericType(0.), NumericType(1.));
    }

    // Set the flag to stop tracing if the energy is below the threshold
    if (newEnergy > params.Si.Eth_ie) {
      E = newEnergy;
      auto direction = viennaray::ReflectionConedCosine<NumericType, D>(
          rayDir, geomNormal, Rng,
          M_PI_2 - std::min(incAngle, params.Ions.minAngle));
      return std::pair<NumericType, Vec3D<NumericType>>{sticking, direction};
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
    return {"ionSputterFlux", "ionEnhancedFlux", "ionEnhancedPassivationFlux"};
  }

private:
  const SF6O2Parameters<NumericType> &params;
  const NumericType A_energy;
  // save precomputed square roots
  const NumericType sqrt_E_th_ie_O;
  const NumericType sqrt_E_th_ie_Si;

  NumericType E;
};

template <typename NumericType, int D>
class SF6O2Neutral
    : public viennaray::Particle<SF6O2Neutral<NumericType, D>, NumericType> {
  const SF6O2Parameters<NumericType> &params;
  const std::string fluxLabel;
  const std::unordered_map<int, NumericType> &beta_map;

public:
  SF6O2Neutral(const SF6O2Parameters<NumericType> &pParams,
               const std::string pFluxLabel,
               std::unordered_map<int, NumericType> &pBetaMap)
      : params(pParams), fluxLabel(pFluxLabel), beta_map(pBetaMap) {}

  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &,
                        const Vec3D<NumericType> &, const unsigned int primID,
                        const int materialId,
                        viennaray::TracingData<NumericType> &localData,
                        const viennaray::TracingData<NumericType> *globalData,
                        RNG &) override final {
    NumericType S_eff = 1.;
    if (params.fluxIncludeSticking) {
      const auto &phi_F = globalData->getVectorData(0)[primID];
      const auto &phi_O = globalData->getVectorData(1)[primID];
      NumericType beta = sticking(materialId);
      S_eff = beta * std::max(1. - phi_F - phi_O, 0.);
    }

    localData.getVectorData(0)[primID] += rayWeight * S_eff;
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
    NumericType beta = sticking(materialId);
    NumericType S_eff = beta * std::max(1. - phi_F - phi_O, 0.);

    auto direction =
        viennaray::ReflectionDiffuse<NumericType, D>(geomNormal, rngState);
    return std::pair<NumericType, Vec3D<NumericType>>{S_eff, direction};
  }
  NumericType getSourceDistributionPower() const override final { return 1.; }
  std::vector<std::string> getLocalDataLabels() const override final {
    return {fluxLabel};
  }

private:
  NumericType sticking(const int matieralId) const {
    auto beta = beta_map.find(matieralId);
    if (beta != beta_map.end())
      return beta->second;

    // default value
    return 1.0;
  }
};
} // namespace impl

/// Model for etching Si in a SF6/O2 plasma. The model is based on the paper by
/// Belen et al., Vac. Sci. Technol. A 23, 99–113 (2005),
/// DOI: https://doi.org/10.1116/1.1830495
template <typename NumericType, int D>
class SF6O2Etching : public ProcessModel<NumericType, D> {
public:
  SF6O2Etching() { initializeModel(); }

  // All flux values are in units 1e15 / cm²
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
    initializeModel();
  }

  SF6O2Parameters<NumericType> &getParameters() { return params; }

private:
  void initializeModel() {
    // check if units have been set
    if (units::Length::getInstance().getUnit() == units::Length::UNDEFINED ||
        units::Time::getInstance().getUnit() == units::Time::UNDEFINED) {
      Logger::getInstance().addError("Units have not been set.").print();
    }

    // particles
    this->particles.clear();
    auto ion = std::make_unique<impl::SF6O2Ion<NumericType, D>>(params);
    this->insertNextParticleType(ion);
    auto etchant = std::make_unique<impl::SF6O2Neutral<NumericType, D>>(
        params, "etchantFlux", params.beta_F);
    this->insertNextParticleType(etchant);
    if (params.oxygenFlux > 0) {
      auto oxygen = std::make_unique<impl::SF6O2Neutral<NumericType, D>>(
          params, "oxygenFlux", params.beta_O);
      this->insertNextParticleType(oxygen);
    }

    // surface model
    auto surfModel =
        SmartPointer<impl::SF6O2SurfaceModel<NumericType, D>>::New(params);
    this->setSurfaceModel(surfModel);

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New(2);
    this->setVelocityField(velField);

    this->setProcessName("SF6O2Etching");
  }

  SF6O2Parameters<NumericType> params;
};

} // namespace viennaps
