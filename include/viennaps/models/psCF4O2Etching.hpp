#pragma once

#include <rayParticle.hpp>
#include <rayReflection.hpp>
#include <rayUtil.hpp>

#include "../psProcessModel.hpp"
#include "../psUnits.hpp"

#include "psCF4O2Parameters.hpp"

namespace viennaps {

using namespace viennacore;

namespace impl {

template <typename NumericType, int D>
class CF4O2SurfaceModel : public SurfaceModel<NumericType> {
public:
  using SurfaceModel<NumericType>::coverages;
  using SurfaceModel<NumericType>::surfaceData;
  const CF4O2Parameters<NumericType> &params;

  explicit CF4O2SurfaceModel(const CF4O2Parameters<NumericType> &pParams)
      : params(pParams) {}

  void initializeCoverages(unsigned numGeometryPoints) override {
    if (coverages == nullptr) {
      coverages = viennals::PointData<NumericType>::New();
    } else {
      coverages->clear();
    }
    std::vector<NumericType> cov(numGeometryPoints, 0.);
    coverages->insertNextScalarData(cov, "eCoverage");
    coverages->insertNextScalarData(cov, "oCoverage");
    coverages->insertNextScalarData(cov, "cCoverage");
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
  calculateVelocities(SmartPointer<viennals::PointData<NumericType>> rates,
                      const std::vector<Vec3D<NumericType>> &coordinates,
                      const std::vector<NumericType> &materialIds) override {
    updateCoverages(rates, materialIds);
    const auto numPoints = rates->getScalarData(0)->size();
    std::vector<NumericType> etchRate(numPoints, 0.);

    const auto ionEnhancedFlux = rates->getScalarData("ionEnhancedFlux");
    const auto ionSputterFlux = rates->getScalarData("ionSputterFlux");
    const auto etchantFlux = rates->getScalarData("etchantFlux");

    const auto eCoverage = coverages->getScalarData("eCoverage");
    const auto oCoverage = coverages->getScalarData("oCoverage");
    const auto cCoverage = coverages->getScalarData("cCoverage");

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
      // const auto chemicalRate = params.Si.k_sigma * eCoverage->at(i) / 4.;

      // The etch rate is calculated in nm/s
      const double unitConversion =
          units::Time::convertSecond() / units::Length::convertNanometer();

      if (MaterialMap::isMaterial(materialIds[i], Material::SiGe)) {
        const auto k_sigma =
            oCoverage->at(i) > 0.5 * eCoverage->at(i)
                ? params.SiGe.k_sigma_SiGe(1.)
                : (eCoverage->at(i) - oCoverage->at(i)) * params.SiGe.k_sigma +
                      oCoverage->at(i) * params.SiGe.k_sigma_SiGe(1.);

        const auto chemicalRate = k_sigma * eCoverage->at(i) / 4.;

        etchRate[i] = -(1 / params.SiGe.rho) *
                      (chemicalRate + sputterRate + ionEnhancedRate) *
                      unitConversion;
        if (Logger::getLogLevel() > 3) {
          spRate->at(i) = sputterRate;
          ieRate->at(i) = ionEnhancedRate;
          chRate->at(i) = chemicalRate;
        }
      } else if (MaterialMap::isMaterial(materialIds[i], Material::Si)) {
        // Standard etching for Si
        const auto chemicalRate = params.Si.k_sigma * eCoverage->at(i) / 4.;

        etchRate[i] = -(1 / params.Si.rho) *
                      (chemicalRate + sputterRate + ionEnhancedRate) *
                      unitConversion;
        if (Logger::getLogLevel() > 3) {
          spRate->at(i) = sputterRate;
          ieRate->at(i) = ionEnhancedRate;
          chRate->at(i) = chemicalRate;
        }
      } else { // Mask
               // if (MaterialMap::isMaterial(materialIds[i], Material::Mask)) {
        etchRate[i] = -(1 / params.Mask.rho) * sputterRate * unitConversion;
        if (Logger::getLogLevel() > 3) {
          spRate->at(i) = sputterRate;
          ieRate->at(i) = 0.;
          chRate->at(i) = 0.;
        }
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

    const auto etchantFlux = rates->getScalarData("etchantFlux");
    std::vector<NumericType> oxygenFlux(numPoints, 0.);
    std::vector<NumericType> polymerFlux(numPoints, 0.);
    if (params.oxygenFlux > 0)
      oxygenFlux = *rates->getScalarData("oxygenFlux");
    if (params.polymerFlux > 0)
      polymerFlux = *rates->getScalarData("polymerFlux");

    const auto ionEnhancedFlux = rates->getScalarData("ionEnhancedFlux");
    const auto ionEnhancedOxidationFlux =
        rates->getScalarData("ionEnhancedOxidationFlux");
    const auto ionEnhancedPassivationFlux =
        rates->getScalarData("ionEnhancedPassivationFlux");

    // etchant fluorine coverage
    auto eCoverage = coverages->getScalarData("eCoverage");
    eCoverage->resize(numPoints);
    // oxygen coverage
    auto oCoverage = coverages->getScalarData("oCoverage");
    oCoverage->resize(numPoints);
    // polymer coverage
    auto cCoverage = coverages->getScalarData("cCoverage");
    cCoverage->resize(numPoints);

    for (size_t i = 0; i < numPoints; ++i) {
      auto Gb_e = etchantFlux->at(i) * params.etchantFlux;
      auto Gb_o = oxygenFlux.at(i) * params.oxygenFlux;
      auto Gb_c = polymerFlux.at(i) * params.polymerFlux;
      auto GY_ie = ionEnhancedFlux->at(i) * params.ionFlux;
      auto GY_o = ionEnhancedOxidationFlux->at(i) * params.ionFlux;
      auto GY_c = ionEnhancedPassivationFlux->at(i) * params.ionFlux;

      auto a = Gb_o;
      auto b = Gb_e;
      auto c = Gb_c;
      auto d = GY_o; // need to add beta_sigma and Gb_e (material dependent)
      auto e = 2 * GY_ie; // need to add k_sigma
      auto f =
          GY_c + Gb_o; // need to add beta_sigma and Gb_e (material dependent)

      if (MaterialMap::isMaterial(materialIds[i], Material::SiGe)) {
        d += params.SiGe.beta_sigma +
             Gb_e; // Gb_e represents removal of G-G bonds
        e += params.SiGe.k_sigma;
      } else { // F cannot remove O from Si-O bonds, so Gb_e = 0 for Si
        d += params.Si.beta_sigma;
        e += params.Si.k_sigma;
      }
      eCoverage->at(i) = std::max(
          0., std::min(1., std::abs(Gb_e) < 1e-6
                               ? 0.
                               : (b * d * f) / (e * f * (a + d) + b * d * f +
                                                c * d * e)));
      oCoverage->at(i) = std::max(
          0., std::min(1., std::abs(Gb_o) < 1e-6
                               ? 0.
                               : (a * e * f) / (e * f * (a + d) + b * d * f +
                                                c * d * e)));
      cCoverage->at(i) = std::max(
          0., std::min(1., std::abs(Gb_c) < 1e-6
                               ? 0.
                               : (c * d * e) / (e * f * (a + d) + b * d * f +
                                                c * d * e)));
    }
  }
};

template <typename NumericType, int D>
class CF4O2Ion final
    : public viennaray::Particle<CF4O2Ion<NumericType, D>, NumericType> {
public:
  explicit CF4O2Ion(const CF4O2Parameters<NumericType> &pParams)
      : params(pParams),
        A_energy(1. / (1. + params.Ions.n_l *
                                (M_PI_2 / params.Ions.inflectAngle - 1.))),
        sqrt_E_th_ie_O(std::sqrt(params.Passivation.Eth_O_ie)),
        sqrt_E_th_ie_C(std::sqrt(params.Passivation.Eth_C_ie)),
        sqrt_E_th_ie_Si(std::sqrt(params.Si.Eth_ie)) {}

  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                        const Vec3D<NumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        viennaray::TracingData<NumericType> &localData,
                        const viennaray::TracingData<NumericType> *globalData,
                        RNG &) override {
    // collect data for this hit
    const double cosTheta = -DotProduct(rayDir, geomNormal);
    NumericType angle = std::acos(std::max(std::min(cosTheta, 1.), 0.));

    assert(cosTheta >= 0 && "Hit backside of disc");
    assert(cosTheta <= 1 + 1e6 && "Error in calculating cos theta");
    assert(rayWeight > 0. && "Invalid ray weight");

    NumericType A_sp = params.Si.A_sp;
    NumericType Eth_sp = params.Si.Eth_sp;
    if (MaterialMap::isMaterial(materialId, Material::Mask)) {
      A_sp = params.Mask.A_sp;
      Eth_sp = params.Mask.Eth_sp;
    }

    NumericType f_sp_theta = 1.;
    NumericType f_ie_theta = 1.;
    if (cosTheta < 0.5) {
      f_ie_theta = std::max(3 - 6 * angle / M_PI, 0.);
    }

    const double sqrtE = std::sqrt(E);
    NumericType Y_sp =
        A_sp * std::max(sqrtE - std::sqrt(Eth_sp), 0.) * f_sp_theta;
    NumericType Y_Si =
        params.Si.A_ie * std::max(sqrtE - sqrt_E_th_ie_Si, 0.) * f_ie_theta;
    NumericType Y_O = params.Passivation.A_O_ie *
                      std::max(sqrtE - sqrt_E_th_ie_O, 0.) * f_ie_theta;
    NumericType Y_C = params.Passivation.A_C_ie *
                      std::max(sqrtE - sqrt_E_th_ie_C, 0.) * f_ie_theta;

    assert(Y_sp >= 0. && "Invalid yield");
    assert(Y_Si >= 0. && "Invalid yield");
    assert(Y_O >= 0. && "Invalid yield");
    assert(Y_C >= 0. && "Invalid yield");

    // sputtering yield Y_sp ionSputterFlux
    localData.getVectorData(0)[primID] += Y_sp * rayWeight;

    // ion enhanced etching yield Y_Si ionEnhancedFlux
    localData.getVectorData(1)[primID] += Y_Si * rayWeight;

    // ion enhanced O sputtering yield Y_O ionEnhancedOxidationFlux
    localData.getVectorData(2)[primID] += Y_O * rayWeight;

    // ion enhanced C sputtering yield Y_C ionEnhancedPassivationFlux
    localData.getVectorData(3)[primID] += Y_C * rayWeight;
  }

  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                    const Vec3D<NumericType> &geomNormal,
                    const unsigned int primId, const int materialId,
                    const viennaray::TracingData<NumericType> *globalData,
                    RNG &Rng) override {
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
      Eref_peak = (1 - (1 - A_energy) * (M_PI_2 - incAngle) /
                           (M_PI_2 - params.Ions.inflectAngle));
    } else {
      Eref_peak = A_energy * std::pow(incAngle / params.Ions.inflectAngle,
                                      params.Ions.n_l);
    }
    // Gaussian distribution around the Eref_peak scaled by the particle energy
    NumericType NewEnergy;
    std::normal_distribution<NumericType> normalDist(E * Eref_peak, 0.1 * E);
    do {
      NewEnergy = normalDist(Rng);
    } while (NewEnergy > E || NewEnergy < 0.);

    NumericType sticking = 0.;
    // NumericType sticking = 1.;
    // if (incAngle > params.Ions.thetaRMin) {
    //   sticking =
    //       1. - std::min((incAngle - params.Ions.thetaRMin) /
    //                         (params.Ions.thetaRMax - params.Ions.thetaRMin),
    //                     NumericType(1.));
    // }

    // Set the flag to stop tracing if the energy is below the threshold
    if (NewEnergy > params.Si.Eth_ie) {
      E = NewEnergy;
      auto direction = viennaray::ReflectionConedCosine<NumericType, D>(
          rayDir, geomNormal, Rng,
          M_PI_2 - std::min(incAngle, params.Ions.minAngle));
      return std::pair<NumericType, Vec3D<NumericType>>{sticking, direction};
    } else {
      return std::pair<NumericType, Vec3D<NumericType>>{
          1., Vec3D<NumericType>{0., 0., 0.}};
    }
  }
  void initNew(RNG &rngState) override {
    std::normal_distribution<NumericType> normalDist{params.Ions.meanEnergy,
                                                     params.Ions.sigmaEnergy};
    do {
      E = normalDist(rngState);
    } while (E <= 0.);
  }
  NumericType getSourceDistributionPower() const override {
    return params.Ions.exponent;
  }
  [[nodiscard]] std::vector<std::string> getLocalDataLabels() const override {
    return {"ionSputterFlux", "ionEnhancedFlux", "ionEnhancedOxidationFlux",
            "ionEnhancedPassivationFlux"};
  }

private:
  const CF4O2Parameters<NumericType> &params;
  const NumericType A_energy;
  // save precomputed square roots
  const NumericType sqrt_E_th_ie_O;
  const NumericType sqrt_E_th_ie_C;
  const NumericType sqrt_E_th_ie_Si;

  NumericType E;
};

template <typename NumericType, int D>
class CF4O2Etchant final
    : public viennaray::Particle<CF4O2Etchant<NumericType, D>, NumericType> {
  const CF4O2Parameters<NumericType> &params;

public:
  explicit CF4O2Etchant(const CF4O2Parameters<NumericType> &pParams)
      : params(pParams) {}

  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &,
                        const Vec3D<NumericType> &, const unsigned int primID,
                        const int materialId,
                        viennaray::TracingData<NumericType> &localData,
                        const viennaray::TracingData<NumericType> *globalData,
                        RNG &) override {
    NumericType S_eff = 1.;
    if (params.fluxIncludeSticking) {
      // F surface coverage
      const auto &phi_F = globalData->getVectorData(0)[primID];
      // O surface coverage
      const auto &phi_O = globalData->getVectorData(1)[primID];
      // F surface coverage on oxidized SiGe
      const auto &phi_C = globalData->getVectorData(2)[primID];
      NumericType gamma_F = sticking(materialId);
      NumericType gamma_FO = stickingOxidized(materialId);
      S_eff =
          gamma_F * std::max(1. - phi_F - phi_O - phi_C, 0.) + gamma_FO * phi_O;
    }

    localData.getVectorData(0)[primID] += rayWeight * S_eff;
  }
  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                    const Vec3D<NumericType> &geomNormal,
                    const unsigned int primID, const int materialId,
                    const viennaray::TracingData<NumericType> *globalData,
                    RNG &rngState) override {

    // F surface coverage
    const auto &phi_F = globalData->getVectorData(0)[primID];
    // O surface coverage
    const auto &phi_O = globalData->getVectorData(1)[primID];
    // F surface coverage on oxidized SiGe
    const auto &phi_C = globalData->getVectorData(2)[primID];
    // Obtain the sticking probability
    NumericType gamma_F = sticking(materialId);
    NumericType gamma_FO = stickingOxidized(materialId);
    NumericType S_eff =
        gamma_F * std::max(1. - phi_F - phi_O - phi_C, 0.) + gamma_FO * phi_O;

    auto direction =
        viennaray::ReflectionDiffuse<NumericType, D>(geomNormal, rngState);
    return std::pair<NumericType, Vec3D<NumericType>>{S_eff, direction};
  }
  NumericType getSourceDistributionPower() const override { return 1.; }
  [[nodiscard]] std::vector<std::string> getLocalDataLabels() const override {
    return {"etchantFlux"};
  }

private:
  NumericType sticking(const int materialId) const {
    auto material = static_cast<Material>(materialId);
    auto gamma = params.gamma_F.find(material);
    if (gamma != params.gamma_F.end())
      return gamma->second;

    // default value
    return 1.0;
  }

  // Sticking probability for oxidized SiGe or Si
  NumericType stickingOxidized(const int materialId) const {
    auto material = static_cast<Material>(materialId);
    auto gamma = params.gamma_F_oxidized.find(material);
    if (gamma != params.gamma_F_oxidized.end())
      return gamma->second;

    return 1.0; // Default
  }
};

template <typename NumericType, int D>
class CF4O2Oxygen final
    : public viennaray::Particle<CF4O2Oxygen<NumericType, D>, NumericType> {
  const CF4O2Parameters<NumericType> &params;

public:
  explicit CF4O2Oxygen(const CF4O2Parameters<NumericType> &pParams)
      : params(pParams) {}

  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &,
                        const Vec3D<NumericType> &, const unsigned int primID,
                        const int materialId,
                        viennaray::TracingData<NumericType> &localData,
                        const viennaray::TracingData<NumericType> *globalData,
                        RNG &) override {
    NumericType S_eff = 1.;
    if (params.fluxIncludeSticking) {
      const auto &phi_F = globalData->getVectorData(0)[primID];
      const auto &phi_O = globalData->getVectorData(1)[primID];
      const auto &phi_C = globalData->getVectorData(2)[primID];
      NumericType gamma_O = sticking(materialId);
      NumericType gamma_OC = stickingPassivated(materialId);
      S_eff =
          gamma_O * std::max(1. - phi_O - phi_F - phi_C, 0.) + gamma_OC * phi_C;
    }

    localData.getVectorData(0)[primID] += rayWeight * S_eff;
  }
  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                    const Vec3D<NumericType> &geomNormal,
                    const unsigned int primID, const int materialId,
                    const viennaray::TracingData<NumericType> *globalData,
                    RNG &rngState) override {

    const auto &phi_F = globalData->getVectorData(0)[primID];
    const auto &phi_O = globalData->getVectorData(1)[primID];
    const auto &phi_C = globalData->getVectorData(2)[primID];
    NumericType gamma_O = sticking(materialId);
    NumericType gamma_OC = stickingPassivated(materialId);
    NumericType S_eff =
        gamma_O * std::max(1. - phi_O - phi_F - phi_C, 0.) + gamma_OC * phi_C;

    auto direction =
        viennaray::ReflectionDiffuse<NumericType, D>(geomNormal, rngState);
    return std::pair<NumericType, Vec3D<NumericType>>{S_eff, direction};
  }
  NumericType getSourceDistributionPower() const override { return 1.; }
  [[nodiscard]] std::vector<std::string> getLocalDataLabels() const override {
    return {"oxygenFlux"};
  }

private:
  NumericType sticking(const int materialId) const {
    auto material = static_cast<Material>(materialId);
    auto gamma = params.gamma_O.find(material);
    if (gamma != params.gamma_O.end())
      return gamma->second;

    // default value
    return 1.0;
  }

  NumericType stickingPassivated(const int materialId) const {
    auto material = static_cast<Material>(materialId);
    auto gamma = params.gamma_O_passivated.find(material);
    if (gamma != params.gamma_O_passivated.end())
      return gamma->second;

    // default value
    return 1.0;
  }
};

template <typename NumericType, int D>
class CF4O2Polymer final
    : public viennaray::Particle<CF4O2Polymer<NumericType, D>, NumericType> {
  const CF4O2Parameters<NumericType> &params;

public:
  explicit CF4O2Polymer(const CF4O2Parameters<NumericType> &pParams)
      : params(pParams) {}

  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &,
                        const Vec3D<NumericType> &, const unsigned int primID,
                        const int materialId,
                        viennaray::TracingData<NumericType> &localData,
                        const viennaray::TracingData<NumericType> *globalData,
                        RNG &) override {
    NumericType S_eff = 1.;
    if (params.fluxIncludeSticking) {
      const auto &phi_F = globalData->getVectorData(0)[primID];
      const auto &phi_O = globalData->getVectorData(1)[primID];
      const auto &phi_C = globalData->getVectorData(2)[primID];
      NumericType gamma_C = sticking(materialId);
      NumericType gamma_CO = stickingOxidized(materialId);
      S_eff =
          gamma_C * std::max(1. - phi_O - phi_F - phi_C, 0.) + gamma_CO * phi_O;
    }

    localData.getVectorData(0)[primID] += rayWeight * S_eff;
  }
  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                    const Vec3D<NumericType> &geomNormal,
                    const unsigned int primID, const int materialId,
                    const viennaray::TracingData<NumericType> *globalData,
                    RNG &rngState) override {

    const auto &phi_F = globalData->getVectorData(0)[primID];
    const auto &phi_O = globalData->getVectorData(1)[primID];
    const auto &phi_C = globalData->getVectorData(2)[primID];
    NumericType gamma_C = sticking(materialId);
    NumericType gamma_CO = stickingOxidized(materialId);
    NumericType S_eff =
        gamma_C * std::max(1. - phi_O - phi_F - phi_C, 0.) + gamma_CO * phi_O;

    auto direction =
        viennaray::ReflectionDiffuse<NumericType, D>(geomNormal, rngState);
    return std::pair<NumericType, Vec3D<NumericType>>{S_eff, direction};
  }
  NumericType getSourceDistributionPower() const override { return 1.; }
  [[nodiscard]] std::vector<std::string> getLocalDataLabels() const override {
    return {"polymerFlux"};
  }

private:
  NumericType sticking(const int materialId) const {
    auto material = static_cast<Material>(materialId);
    auto gamma = params.gamma_C.find(material);
    if (gamma != params.gamma_C.end())
      return gamma->second;

    // default value
    return 1.0;
  }

  NumericType stickingOxidized(const int materialId) const {
    auto material = static_cast<Material>(materialId);
    auto gamma = params.gamma_C_oxidized.find(material);
    if (gamma != params.gamma_C_oxidized.end())
      return gamma->second;

    // default value
    return 1.0;
  }
};
} // namespace impl

template <typename NumericType, int D>
class CF4O2Etching final : public ProcessModel<NumericType, D> {
public:
  CF4O2Etching() { initializeModel(); }

  // All flux values are in units 1e15 / cm²
  CF4O2Etching(const double ionFlux, const double etchantFlux,
               const double oxygenFlux, const double polymerFlux,
               const NumericType meanEnergy /* eV */,
               const NumericType sigmaEnergy /* eV */, // 5 parameters
               const NumericType ionExponent = 300.,
               const NumericType oxySputterYield = 2.,
               const NumericType polySputterYield = 2.,
               const NumericType etchStopDepth =
                   std::numeric_limits<NumericType>::lowest()) {
    params.ionFlux = ionFlux;
    params.etchantFlux = etchantFlux;
    params.oxygenFlux = oxygenFlux;
    params.polymerFlux = polymerFlux;
    params.Ions.meanEnergy = meanEnergy;
    params.Ions.sigmaEnergy = sigmaEnergy;
    params.Ions.exponent = ionExponent;
    params.Passivation.A_O_ie = oxySputterYield;
    params.Passivation.A_C_ie = polySputterYield;
    params.etchStopDepth = etchStopDepth;
    initializeModel();
  }

  explicit CF4O2Etching(const CF4O2Parameters<NumericType> &pParams)
      : params(pParams) {
    initializeModel();
  }

  void setParameters(const CF4O2Parameters<NumericType> &pParams) {
    params = pParams;
    initializeModel();
  }

  CF4O2Parameters<NumericType> &getParameters() { return params; }

private:
  void initializeModel() {
    // check if units have been set
    if (units::Length::getUnit() == units::Length::UNDEFINED ||
        units::Time::getUnit() == units::Time::UNDEFINED) {
      Logger::getInstance().addError("Units have not been set.").print();
    }

    // particles
    auto ion = std::make_unique<impl::CF4O2Ion<NumericType, D>>(params);
    auto etchant = std::make_unique<impl::CF4O2Etchant<NumericType, D>>(params);
    auto oxygen = std::make_unique<impl::CF4O2Oxygen<NumericType, D>>(params);
    auto polymer = std::make_unique<impl::CF4O2Polymer<NumericType, D>>(params);

    // surface model
    auto surfModel =
        SmartPointer<impl::CF4O2SurfaceModel<NumericType, D>>::New(params);

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New(2);

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("CF4O2Etching");
    this->particles.clear();
    this->insertNextParticleType(ion);
    this->insertNextParticleType(etchant);
    if (params.oxygenFlux > 0)
      this->insertNextParticleType(oxygen);
    if (params.polymerFlux > 0)
      this->insertNextParticleType(polymer);
  }

  CF4O2Parameters<NumericType> params;
};

} // namespace viennaps
