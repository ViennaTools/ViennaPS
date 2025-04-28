#pragma once

#include <rayParticle.hpp>
#include <rayReflection.hpp>
#include <rayUtil.hpp>

#include "../psConstants.hpp"
#include "../psProcessModel.hpp"
#include "../psSurfaceModel.hpp"
#include "../psUnits.hpp"
#include "../psVelocityField.hpp"

#include "psHBrO2Parameters.hpp"
#include "psSF6O2Etching.hpp"

namespace viennaps {

using namespace viennacore;

/// Model for etching Si in a HBr/O2 plasma.
template <typename NumericType, int D>
class HBrO2Etching : public ProcessModel<NumericType, D> {
public:
  HBrO2Etching() { initializeModel(); }

  // All flux values are in units 1e15 / cm²
  HBrO2Etching(const double ionFlux, const double etchantFlux,
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

  HBrO2Etching(const HBrO2Parameters<NumericType> &pParams) : params(pParams) {
    initializeModel();
  }

  void setParameters(const HBrO2Parameters<NumericType> &pParams) {
    params = pParams;
    initializeModel();
  }

  HBrO2Parameters<NumericType> &getParameters() { return params; }

private:
  void initializeModel() {
    // check if units have been set
    if (units::Length::getInstance().getUnit() == units::Length::UNDEFINED ||
        units::Time::getInstance().getUnit() == units::Time::UNDEFINED) {
      Logger::getInstance().addError("Units have not been set.").print();
    }

    // particles
    this->particles.clear();
    auto ion = std::make_unique<
        impl::SF6O2Ion<HBrO2Parameters<NumericType>, NumericType, D>>(params);
    this->insertNextParticleType(ion);
    auto etchant = std::make_unique<
        impl::SF6O2Neutral<HBrO2Parameters<NumericType>, NumericType, D>>(
        params, "etchantFlux", params.beta_Br);
    this->insertNextParticleType(etchant);
    if (params.oxygenFlux > 0) {
      auto oxygen = std::make_unique<
          impl::SF6O2Neutral<HBrO2Parameters<NumericType>, NumericType, D>>(
          params, "oxygenFlux", params.beta_O);
      this->insertNextParticleType(oxygen);
    }

    // surface model
    auto surfModel =
        SmartPointer<impl::SF6O2SurfaceModel<HBrO2Parameters<NumericType>,
                                             NumericType, D>>::New(params);
    this->setSurfaceModel(surfModel);

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New(2);
    this->setVelocityField(velField);

    this->setProcessName("HBrO2Etching");
  }

  HBrO2Parameters<NumericType> params;
};

} // namespace viennaps
