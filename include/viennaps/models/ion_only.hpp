#pragma once

#include <rayParticle.hpp>
#include <rayReflection.hpp>
#include <rayUtil.hpp>

#include "../psConstants.hpp"
#include "../psProcessModel.hpp"
#include "../psSurfaceModel.hpp"
#include "../psUnits.hpp"
#include "../psVelocityField.hpp"

#include "psSF6O2Etching.hpp"

namespace viennaps {

using namespace viennacore;

namespace impl {

template <typename NumericType, int D>
class IonOnlySurfModel : public SurfaceModel<NumericType> {
public:
  const SF6O2Parameters<NumericType> &params;

  IonOnlySurfModel(const SF6O2Parameters<NumericType> &pParams)
      : params(pParams) {}

  SmartPointer<std::vector<NumericType>>
  calculateVelocities(SmartPointer<viennals::PointData<NumericType>> rates,
                      const std::vector<Vec3D<NumericType>> &coordinates,
                      const std::vector<NumericType> &materialIds) override {
    const auto numPoints = rates->getScalarData(0)->size();
    std::vector<NumericType> etchRate(numPoints, 0.);

    const auto flux = rates->getScalarData("ionFlux");

    for (size_t i = 0; i < numPoints; ++i) {

      if (MaterialMap::isMaterial(materialIds[i], Material::Mask)) {
        etchRate[i] = -params.Mask.rho * flux->at(i);
      } else {
        etchRate[i] = -params.Si.rho * flux->at(i);
      }
    }

    return SmartPointer<std::vector<NumericType>>::New(std::move(etchRate));
  }
};

template <typename NumericType, int D>
class IonOnlyIon
    : public viennaray::Particle<IonOnlyIon<NumericType, D>, NumericType> {
public:
  IonOnlyIon(const SF6O2Parameters<NumericType> &pParams) : params(pParams) {}

  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                        const Vec3D<NumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        viennaray::TracingData<NumericType> &localData,
                        const viennaray::TracingData<NumericType> *globalData,
                        RNG &) override final {
    // collect data for this hit
    const double cosTheta = -DotProduct(rayDir, geomNormal);
    NumericType angle = std::acos(std::max(std::min(cosTheta, 1.), 0.));

    NumericType f_ie_theta = 1.;
    if (cosTheta < 0.5) {
      f_ie_theta = std::max(3 - 6 * angle / M_PI, 0.);
    }

    NumericType Y_Si =
        params.Si.A_ie *
        std::max(std::sqrt(E) - std::sqrt(params.Si.Eth_ie), 0.) * f_ie_theta;

    localData.getVectorData(0)[primID] += Y_Si * rayWeight;
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
    NumericType A_energy =
        1. / (1. + params.Ions.n_l * (M_PI_2 / params.Ions.inflectAngle - 1.));
    if (incAngle >= params.Ions.inflectAngle) {
      Eref_peak = (1 - (1 - A_energy) * (M_PI_2 - incAngle) /
                           (M_PI_2 - params.Ions.inflectAngle));
    } else {
      Eref_peak = A_energy * std::pow(incAngle / params.Ions.inflectAngle,
                                      params.Ions.n_l);
    }
    // Gaussian distribution around the Eref_peak scaled by the particle energy
    NumericType NewEnergy = Eref_peak * E;
    // std::normal_distribution<NumericType> normalDist(E * Eref_peak, 0.1 * E);
    // do {
    //   NewEnergy = normalDist(Rng);
    // } while (NewEnergy > E || NewEnergy < 0.);

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
      // auto direction = viennaray::ReflectionConedCosine<NumericType, D>(
      //     rayDir, geomNormal, Rng,
      //     M_PI_2 - std::min(incAngle, params.Ions.minAngle));
      auto direction =
          viennaray::ReflectionSpecular<NumericType>(rayDir, geomNormal);
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
    return {"ionFlux"};
  }

private:
  const SF6O2Parameters<NumericType> &params;

  NumericType E;
};
} // namespace impl

template <typename NumericType, int D>
class IonOnlyEtching : public ProcessModel<NumericType, D> {
public:
  IonOnlyEtching() { initializeModel(); }

  // All flux values are in units 1e15 / cm²
  IonOnlyEtching(const double ionFlux, const double etchantFlux,
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

  IonOnlyEtching(const SF6O2Parameters<NumericType> &pParams)
      : params(pParams) {
    initializeModel();
  }

  void setParameters(const SF6O2Parameters<NumericType> &pParams) {
    params = pParams;
  }

  SF6O2Parameters<NumericType> &getParameters() { return params; }

private:
  void initializeModel() {
    // particles
    auto ion = std::make_unique<impl::IonOnlyIon<NumericType, D>>(params);

    // surface model
    auto surfModel =
        SmartPointer<impl::IonOnlySurfModel<NumericType, D>>::New(params);

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New(2);

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("ionOnly");
    this->particles.clear();
    this->insertNextParticleType(ion);
  }

  SF6O2Parameters<NumericType> params;
};

} // namespace viennaps
