#pragma once

#include <rayParticle.hpp>
#include <rayReflection.hpp>
#include <rayUtil.hpp>

#include "../psConstants.hpp"
#include "../psProcessModel.hpp"
#include "../psSurfaceModel.hpp"
#include "../psUnits.hpp"
#include "../psVelocityField.hpp"

#include "psPlasmaEtchingParameters.hpp"

namespace viennaps::impl {

using namespace viennacore;

template <typename NumericType, int D>
class PlasmaEtchingSurfaceModel : public SurfaceModel<NumericType> {
public:
  using SurfaceModel<NumericType>::coverages;
  using SurfaceModel<NumericType>::surfaceData;
  const PlasmaEtchingParameters<NumericType> &params;

  PlasmaEtchingSurfaceModel(const PlasmaEtchingParameters<NumericType> &pParams)
      : params(pParams) {}

  void initializeCoverages(unsigned numGeometryPoints) override {
    if (coverages == nullptr) {
      coverages = viennals::PointData<NumericType>::New();
    } else {
      coverages->clear();
    }
    std::vector<NumericType> cov(numGeometryPoints, 0.);
    coverages->insertNextScalarData(cov, "eCoverage");
    coverages->insertNextScalarData(cov, "pCoverage");
  }

  void initializeSurfaceData(unsigned numGeometryPoints) override {
    if (Logger::getLogLevel() > 3) {
      if (surfaceData == nullptr) {
        surfaceData = viennals::PointData<NumericType>::New();
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
    const auto numPoints = materialIds.size();
    std::vector<NumericType> etchRate(numPoints, 0.);

    std::vector<NumericType> ionEnhancedFlux, ionSputterFlux;
    if (params.ionFlux > 0) {
      ionEnhancedFlux = *fluxes->getScalarData("ionEnhancedFlux");
      ionSputterFlux = *fluxes->getScalarData("ionSputterFlux");
    } else {
      ionEnhancedFlux.resize(numPoints, 0.);
      ionSputterFlux.resize(numPoints, 0.);
    }

    const auto eCoverage = coverages->getScalarData("eCoverage");
    assert(eCoverage != nullptr && "eCoverage not initialized");

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

      const auto sputterRate = ionSputterFlux[i] * params.ionFlux;
      const auto ionEnhancedRate =
          eCoverage->at(i) * ionEnhancedFlux[i] * params.ionFlux;
      const auto chemicalRate =
          params.Substrate.k_sigma * eCoverage->at(i) / 4.;

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
        etchRate[i] = -(1 / params.Substrate.rho) *
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
    const auto numPoints = materialIds.size();
    std::vector<NumericType> zero(numPoints, 0.);
    std::vector<NumericType> *etchantFlux = nullptr, *passivationFlux = nullptr,
                             *ionEnhancedFlux = nullptr,
                             *ionEnhancedPassivationFlux = nullptr;

    if (params.etchantFlux > 0) {
      etchantFlux = fluxes->getScalarData("etchantFlux");
    } else {
      etchantFlux = &zero;
    }

    if (params.passivationFlux > 0) {
      passivationFlux = fluxes->getScalarData("passivationFlux");
    } else {
      passivationFlux = &zero;
    }

    if (params.ionFlux > 0) {
      ionEnhancedFlux = fluxes->getScalarData("ionEnhancedFlux");
      ionEnhancedPassivationFlux =
          fluxes->getScalarData("ionEnhancedPassivationFlux");
    } else {
      ionEnhancedFlux = &zero;
      ionEnhancedPassivationFlux = &zero;
    }

    // etchant coverage
    auto eCoverage = coverages->getScalarData("eCoverage");
    eCoverage->resize(numPoints);
    // passivation coverage
    auto pCoverage = coverages->getScalarData("pCoverage");
    pCoverage->resize(numPoints);

    for (size_t i = 0; i < numPoints; ++i) {
      auto Gb_E = etchantFlux->at(i) * params.etchantFlux;
      auto Gb_P = passivationFlux->at(i) * params.passivationFlux;
      auto GY_ie = ionEnhancedFlux->at(i) * params.ionFlux;
      auto GY_p = ionEnhancedPassivationFlux->at(i) * params.ionFlux;

      auto a = (params.Substrate.k_sigma + 2 * GY_ie) / Gb_E;
      auto b = (params.Substrate.beta_sigma + GY_p) / Gb_P;

      eCoverage->at(i) = Gb_E < 1e-6 ? 0. : 1 / (1 + (a * (1 + 1 / b)));
      pCoverage->at(i) = Gb_P < 1e-6 ? 0. : 1 / (1 + (b * (1 + 1 / a)));
    }
  }
};

template <typename NumericType, int D>
class PlasmaEtchingIon
    : public viennaray::Particle<PlasmaEtchingIon<NumericType, D>,
                                 NumericType> {
public:
  PlasmaEtchingIon(const PlasmaEtchingParameters<NumericType> &pParams)
      : params(pParams),
        A_energy(1. / (1. + params.Ions.n_l *
                                (M_PI_2 / params.Ions.inflectAngle - 1.))),
        sqrt_E_th_ie_P(std::sqrt(params.Passivation.Eth_ie)),
        sqrt_E_th_ie_Sub(std::sqrt(params.Substrate.Eth_ie)) {}

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

    NumericType A_sp = params.Substrate.A_sp;
    NumericType B_sp = params.Substrate.B_sp;
    NumericType Eth_sp = params.Substrate.Eth_sp;
    if (MaterialMap::isMaterial(materialId, Material::Mask)) {
      A_sp = params.Mask.A_sp;
      B_sp = params.Mask.B_sp;
      Eth_sp = params.Mask.Eth_sp;
    }

    // NumericType f_sp_theta = 1.;
    NumericType f_sp_theta =
        std::max((1. + B_sp * (1. - cosTheta * cosTheta)) * cosTheta, 0.);

    NumericType f_ie_theta = 1.;
    if (cosTheta < 0.5) {
      f_ie_theta = std::max(3. - 6. * angle / M_PI, 0.);
    }
    // NumericType f_ie_theta =
    //     std::max((1 + params.Substrate.B_ie * (1 - cosTheta * cosTheta)) *
    //                  std::cos(angle / params.Substrate.theta_g_ie * M_PI_2),
    //              0.);

    const double sqrtE = std::sqrt(E);
    NumericType Y_sp =
        A_sp * std::max(sqrtE - std::sqrt(Eth_sp), 0.) * f_sp_theta;
    NumericType Y_Si = params.Substrate.A_ie *
                       std::max(sqrtE - sqrt_E_th_ie_Sub, 0.) * f_ie_theta;
    NumericType Y_P = params.Passivation.A_ie *
                      std::max(sqrtE - sqrt_E_th_ie_P, 0.) * f_ie_theta;

    assert(Y_sp >= 0. && "Invalid yield");
    assert(Y_Si >= 0. && "Invalid yield");
    assert(Y_P >= 0. && "Invalid yield");

    // sputtering yield Y_sp ionSputterFlux
    localData.getVectorData(0)[primID] += Y_sp * rayWeight;

    // ion enhanced etching yield Y_Si ionEnhancedFlux
    localData.getVectorData(1)[primID] += Y_Si * rayWeight;

    // ion enhanced passivation sputtering yield Y_O ionEnhancedPassivationFlux
    localData.getVectorData(2)[primID] += Y_P * rayWeight;
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
    NumericType minEnergy =
        std::min(params.Substrate.Eth_ie, params.Substrate.Eth_sp);
    if (newEnergy > minEnergy) {
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
  const PlasmaEtchingParameters<NumericType> &params;
  const NumericType A_energy;
  // save precomputed square roots
  const NumericType sqrt_E_th_ie_P;
  const NumericType sqrt_E_th_ie_Sub;

  NumericType E;
};

template <typename NumericType, int D>
class PlasmaEtchingNeutral
    : public viennaray::Particle<PlasmaEtchingNeutral<NumericType, D>,
                                 NumericType> {
  const std::string fluxLabel;
  const std::unordered_map<int, NumericType> &beta_map;
  const int numCoverages;

public:
  PlasmaEtchingNeutral(const std::string &pFluxLabel,
                       std::unordered_map<int, NumericType> &pBetaMap,
                       const int pNumCoverages)
      : fluxLabel(pFluxLabel), beta_map(pBetaMap), numCoverages(pNumCoverages) {
  }

  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &,
                        const Vec3D<NumericType> &, const unsigned int primID,
                        const int materialId,
                        viennaray::TracingData<NumericType> &localData,
                        const viennaray::TracingData<NumericType> *globalData,
                        RNG &) override final {
    localData.getVectorData(0)[primID] += rayWeight;
  }
  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                    const Vec3D<NumericType> &geomNormal,
                    const unsigned int primID, const int materialId,
                    const viennaray::TracingData<NumericType> *globalData,
                    RNG &rngState) override final {

    NumericType S_eff = 1.;
    for (int i = 0; i < numCoverages; ++i) {
      S_eff -= globalData->getVectorData(i)[primID];
    }
    if (S_eff < 0.) {
      S_eff = 0.;
    } else {
      S_eff *= sticking(materialId);
    }

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
} // namespace viennaps::impl
