#pragma once

// psOxidation — ViennaPS process model for thermal silicon oxidation.
//
// Wraps the ViennaLS coupled diffusion+deformation oxidation engine as a
// psProcess model, exposing process-level inputs (temperature, time, oxidant
// type, pressure, crystal orientation) instead of raw OxidationParameters.
//
// Rate constants follow the Deal-Grove model (Deal & Grove, J. Appl. Phys.
// 36, 3770 (1965)). The built-in Arrhenius table is the common 1 atm
// low-doping silicon table used in process texts and calculators: dry/wet
// B and B/A for <100>/<111>, with B/A(111)/B/A(100) = 1.68.
//
// Usage:
//   auto model = SmartPointer<Oxidation<T, D>>::New();
//   model->setTemperature(1000.);               // °C
//   model->setTime(0.5);                         // hours
//   model->setOxidant(OxidantType::Wet);
//   model->setOrientation(SiliconOrientation::Si100);
//   psProcess<T, D>(domain, model, 0.).apply();  // duration=0: one-shot
//
// The domain must contain a level set mapped to Material::Si (or BulkSi).
// If no SiO2 level set is present, a thin native-oxide layer is created
// automatically and appended to the top of the level-set stack.

#include <lsAdvect.hpp>
#include <lsGeometricAdvect.hpp>
#include <lsOxidationDeformation.hpp>
#include <lsOxidationDiffusion.hpp>
#include <lsOxidationMaterials.hpp>
#include <lsOxidationModel.hpp>

#include "../process/psAdvectionCallback.hpp"
#include "../process/psProcessModel.hpp"
#include "../materials/psMaterial.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace viennaps {

using namespace viennacore;

enum class OxidantType { Dry, Wet };
enum class SiliconOrientation { Si100, Si111, PolySi };

template <class NumericType, int D>
class Oxidation : public ProcessModelBase<NumericType, D> {

  // -----------------------------------------------------------------------
  // Callback: owns all settings, drives the ViennaLS time-stepping loop.
  // -----------------------------------------------------------------------
  class OxidationCallback : public AdvectionCallback<NumericType, D> {

    // Deal-Grove Arrhenius table.
    // B  in µm²/hr  (parabolic rate constant; orientation-independent)
    // B/A in µm/hr  (linear rate constant; orientation-dependent)
    // Ea in eV
    // Source: standard Deal-Grove table for 1 atm low-doping silicon.
    // Units are chosen to match ViennaLS oxidation examples (µm and hours).
    // <100> dry/wet B/A values are <111> values divided by 1.68.
    struct DealGroveRow {
      NumericType B0, EB;       // B  = B0  * exp(-EB  / kT)
      NumericType BoA0, EBoA;   // B/A = BoA0 * exp(-EBoA / kT)
    };

    static constexpr NumericType kB_ = NumericType(8.617333e-5); // eV/K

    // settings
    NumericType temperature_ = 1000.;
    NumericType time_ = 1.;
    OxidantType oxidant_ = OxidantType::Dry;
    NumericType pressure_ = 1.;
    SiliconOrientation orientation_ = SiliconOrientation::Si100;
    NumericType timeStep_ = 0.;
    NumericType initialOxideThickness_ = NumericType(0.002); // µm (2 nm)
    NumericType transferCoefficient_ = NumericType(100);
    NumericType reactionActivationVolume_ = NumericType(1.76e-35);
    NumericType diffusionActivationVolume_ = NumericType(0);
    std::size_t maxGridPoints_ = 5000000;
    unsigned couplingIterations_ = 8;
    NumericType couplingTolerance_ = NumericType(1e-6);
    bool useSolveBounds_ = false;
    viennahrle::Index<D> solveMinIndex_{};
    viennahrle::Index<D> solveMaxIndex_{};
    Material siliconMaterial_ = Material::Si;
    Material oxideMaterial_ = Material::SiO2;

  public:
    // --- setters ---
    void setTemperature(NumericType t) { temperature_ = t; }
    void setTime(NumericType t) { time_ = t; }
    void setOxidant(OxidantType o) { oxidant_ = o; }
    void setPressure(NumericType p) { pressure_ = p; }
    void setOrientation(SiliconOrientation o) { orientation_ = o; }
    void setTimeStep(NumericType dt) { timeStep_ = dt; }
    void setInitialOxideThickness(NumericType t) { initialOxideThickness_ = t; }
    void setTransferCoefficient(NumericType h) { transferCoefficient_ = h; }
    void setReactionActivationVolume(NumericType v) {
      reactionActivationVolume_ = v;
    }
    void setDiffusionActivationVolume(NumericType v) {
      diffusionActivationVolume_ = v;
    }
    void setMaxGridPoints(std::size_t n) { maxGridPoints_ = n; }
    void setCouplingIterations(unsigned n) {
      couplingIterations_ = std::max(1u, n);
    }
    void setCouplingTolerance(NumericType tolerance) {
      couplingTolerance_ = std::max(tolerance, NumericType(1e-12));
    }
    void setSolveBounds(const viennahrle::Index<D> &minIndex,
                        const viennahrle::Index<D> &maxIndex) {
      solveMinIndex_ = minIndex;
      solveMaxIndex_ = maxIndex;
      useSolveBounds_ = true;
    }
    void clearSolveBounds() { useSolveBounds_ = false; }
    void setSiliconMaterial(Material m) { siliconMaterial_ = m; }
    void setOxideMaterial(Material m) { oxideMaterial_ = m; }

    NumericType estimatePlanarOxideThickness(
        NumericType initialOxideThickness = NumericType(0)) const {
      const auto rates = computeDealGroveRates();
      if (rates.B <= NumericType(0) || rates.BoA <= NumericType(0))
        return std::max(initialOxideThickness, NumericType(0));

      const NumericType initial =
          std::max(initialOxideThickness, NumericType(0));
      const NumericType A = rates.B / rates.BoA;
      const NumericType tau = (initial * initial + A * initial) / rates.B;
      const NumericType effectiveTime = std::max(time_, NumericType(0)) + tau;
      return (std::sqrt(A * A + NumericType(4) * rates.B * effectiveTime) -
              A) /
             NumericType(2);
    }

    bool applyPreAdvect(NumericType /*processTime*/) override {
      namespace ls = viennals;

      auto &domain = *this->domain;
      auto &levelSets = domain.getLevelSets();
      auto matMap = domain.getMaterialMap();

      if (levelSets.empty() || !matMap) {
        Logger::getInstance()
            .addWarning("Oxidation: domain has no level sets or material map.")
            .print();
        return false;
      }

      // Find Si (reaction) and SiO2 (ambient) level-set indices.
      // Scan top-down so we pick the topmost Si and SiO2 layers.
      int siIdx = -1, sio2Idx = -1;
      for (int i = static_cast<int>(levelSets.size()) - 1; i >= 0; --i) {
        const auto mat = matMap->getMaterialAtIdx(i);
        if (siIdx < 0 &&
            (mat == siliconMaterial_ || mat == Material::BulkSi ||
             mat == Material::PolySi || mat == Material::aSi))
          siIdx = i;
        if (sio2Idx < 0 && mat == oxideMaterial_)
          sio2Idx = i;
      }

      if (siIdx < 0) {
        Logger::getInstance()
            .addWarning("Oxidation: no silicon layer found in domain.")
            .print();
        return false;
      }

      auto reactionInterface = levelSets[siIdx];
      auto &grid = reactionInterface->getGrid();
      const NumericType gridDelta = grid.getGridDelta();

      // If no oxide layer exists, create a thin native oxide on top of Si.
      SmartPointer<ls::Domain<NumericType, D>> ambientInterface;
      if (sio2Idx >= 0) {
        ambientInterface = levelSets[sio2Idx];
      } else {
        ambientInterface =
            SmartPointer<ls::Domain<NumericType, D>>::New(reactionInterface);
        if (initialOxideThickness_ > NumericType(0)) {
          auto sphere = SmartPointer<
              ls::SphereDistribution<viennahrle::CoordType, D>>::New(
              initialOxideThickness_);
          ls::GeometricAdvect<NumericType, D>(ambientInterface, sphere).apply();
        }
        domain.insertNextLevelSetAsMaterial(ambientInterface, oxideMaterial_,
                                            false);
      }

      // Optional user override for the Cartesian solve bounds. By default the
      // ViennaLS oxidation model derives finite bounds from the level-set
      // narrow bands, which is important for infinite boundary conditions.
      viennahrle::Index<D> minIndex, maxIndex;
      if (useSolveBounds_) {
        minIndex = solveMinIndex_;
        maxIndex = solveMaxIndex_;
      }

      // Compute time step: user override or default to time/20.
      NumericType dt = (timeStep_ > NumericType(0))
                           ? timeStep_
                           : std::max(time_ / NumericType(20),
                                      std::numeric_limits<NumericType>::min());

      // Build ViennaLS velocity fields.
      auto oxParams = computeOxParams(gridDelta);
      auto defParams = computeDefParams(dt);

      auto diffField = ls::OxidationDiffusion<NumericType, D>::New(
          reactionInterface, ambientInterface, oxParams);
      if (useSolveBounds_)
        diffField->setSolveBounds(minIndex, maxIndex);

      auto defField = ls::OxidationDeformation<NumericType, D>::New(
          reactionInterface, ambientInterface, diffField, oxParams, defParams);
      if (useSolveBounds_)
        defField->setSolveBounds(minIndex, maxIndex);

      ls::OxidationCouplingParameters<NumericType> coupling;
      coupling.maxIterations = couplingIterations_;
      coupling.tolerance = couplingTolerance_;
      auto coupledModel = ls::OxidationModel<NumericType, D>::New(
          diffField, defField, coupling);
      if (useSolveBounds_)
        coupledModel->setSolveBounds(minIndex, maxIndex);

      // Advection objects (level sets fixed; only time and dt change per step).
      ls::Advect<NumericType, D> ambientAdvect;
      ambientAdvect.insertNextLevelSet(ambientInterface);
      ambientAdvect.setVelocityField(defField);
      ambientAdvect.setSpatialScheme(
          ls::SpatialSchemeEnum::ENGQUIST_OSHER_1ST_ORDER);
      ambientAdvect.setTemporalScheme(
          ls::TemporalSchemeEnum::RUNGE_KUTTA_2ND_ORDER);

      ls::Advect<NumericType, D> reactionAdvect;
      reactionAdvect.insertNextLevelSet(reactionInterface);
      reactionAdvect.setVelocityField(diffField);
      reactionAdvect.setSpatialScheme(
          ls::SpatialSchemeEnum::ENGQUIST_OSHER_1ST_ORDER);
      reactionAdvect.setTemporalScheme(
          ls::TemporalSchemeEnum::RUNGE_KUTTA_2ND_ORDER);

      // Time-stepping loop.
      NumericType time = 0.;
      while (time < time_) {
        if (time + dt > time_)
          dt = time_ - time;

        // Update the deformation time step for this (possibly truncated) step.
        defParams.stressTimeStep = dt;
        defField->setDeformationParameters(defParams);

        coupledModel->apply();

        ambientAdvect.setAdvectionTime(dt);
        ambientAdvect.apply();

        reactionAdvect.setAdvectionTime(dt);
        reactionAdvect.apply();

        diffField->markGeometryChanged();
        defField->markGeometryChanged();

        time += dt;
      }

      return true;
    }

  private:
    // Map process conditions → OxidationParameters.
    // With C* = N = 1 (normalised): B = 2D, so diffusionCoefficient = B/2.
    viennals::OxidationParameters<NumericType>
    computeOxParams(NumericType /*gridDelta*/) const {
      const NumericType T_K = temperature_ + NumericType(273.15);
      const auto rates = computeDealGroveRates();

      viennals::OxidationParameters<NumericType> p;
      p.diffusionCoefficient = rates.B / NumericType(2); // B = 2D
      p.reactionRate = rates.BoA;
      p.transferCoefficient = transferCoefficient_;
      p.equilibriumConcentration = NumericType(1);
      p.oxidantMoleculeDensity = NumericType(1);
      p.expansionCoefficient = NumericType(2.27);  // V_SiO2 / V_Si
      p.velocitySign = NumericType(-1);             // reaction interface moves inward
      p.temperature = T_K;
      p.reactionActivationVolume = reactionActivationVolume_;
      p.diffusionActivationVolume = diffusionActivationVolume_;
      p.reactionRateRatio111 = NumericType(1);     // orientation encoded in B/A choice
      p.maxGridPoints = maxGridPoints_;
      p.maxIterations = 10000;
      p.tolerance = NumericType(1e-7);
      return p;
    }

    // Oxide mechanics parameters (SiO2 at ~1000 °C).
    // Source: Plummer, Deal & Griffin, Silicon VLSI Technology (2000), Table 6.2.
    viennals::OxidationDeformationParameters<NumericType>
    computeDefParams(NumericType dt) const {
      viennals::OxidationDeformationParameters<NumericType> p;
      p.viscosity = NumericType(1e10);        // Pa·hr
      p.bulkModulus = NumericType(7.5e8);     // Pa
      p.shearModulus = NumericType(3e10);     // Pa
      p.stressTimeStep = dt;
      p.mechanicsIterations = 2;
      p.mechanicsTolerance = NumericType(1e-7);
      p.pressureIterations = 500;
      p.stokesIterations = 100;
      p.pressureTolerance = NumericType(1e-6);
      p.stokesTolerance = NumericType(1e-7);
      p.tolerance = NumericType(1e-7);
      p.maxGridPoints = maxGridPoints_;
      return p;
    }

    // Select the appropriate Deal-Grove row.
    // PolySi: isotropic, use <100> rates as a conservative baseline.
    struct DealGroveRates {
      NumericType B = 0.;   // µm²/hr
      NumericType BoA = 0.; // µm/hr
    };

    DealGroveRates computeDealGroveRates() const {
      const NumericType T_K = temperature_ + NumericType(273.15);
      const auto row = dealGroveRow();
      const NumericType pressure = std::max(pressure_, NumericType(0));
      return {pressure * row.B0 * std::exp(-row.EB / (kB_ * T_K)),
              pressure * row.BoA0 * std::exp(-row.EBoA / (kB_ * T_K))};
    }

    DealGroveRow dealGroveRow() const {
      if (oxidant_ == OxidantType::Dry) {
        if (orientation_ == SiliconOrientation::Si111)
          return {NumericType(772), NumericType(1.23),
                  NumericType(6.23e6), NumericType(2.00)};
        return   {NumericType(772), NumericType(1.23),
                  NumericType(3.71e6), NumericType(2.00)};
      }
      // Wet
      if (orientation_ == SiliconOrientation::Si111)
        return {NumericType(386), NumericType(0.78),
                NumericType(1.63e8), NumericType(2.05)};
      return   {NumericType(386), NumericType(0.78),
                NumericType(9.70e7), NumericType(2.05)};
    }
  };

  SmartPointer<OxidationCallback> callback_;

public:
  Oxidation() {
    this->setProcessName("Oxidation");
    callback_ = SmartPointer<OxidationCallback>::New();
    this->setAdvectionCallback(callback_);
  }

  // Temperature in °C (valid range: 800–1200 °C)
  void setTemperature(NumericType temperatureC) {
    callback_->setTemperature(temperatureC);
  }

  // Total oxidation time in hours
  void setTime(NumericType timeHr) { callback_->setTime(timeHr); }

  // Oxidant species: OxidantType::Dry (O2) or OxidantType::Wet (H2O)
  void setOxidant(OxidantType oxidant) { callback_->setOxidant(oxidant); }

  // Ambient pressure in atm (scales both B and B/A linearly; default 1.0)
  void setPressure(NumericType pressureAtm) {
    callback_->setPressure(pressureAtm);
  }

  // Crystal orientation: Si100, Si111, or PolySi (isotropic, uses <100> rates)
  void setOrientation(SiliconOrientation orientation) {
    callback_->setOrientation(orientation);
  }

  // Set the duration of each explicit process time step in hours.
  // In each step, velocities are computed on the current geometry, and then the
  // interfaces are advected for this duration. Default: total time / 20.
  void setTimeStep(NumericType dtHr) { callback_->setTimeStep(dtHr); }

  // Thickness of the auto-created native oxide if no SiO2 layer exists (µm)
  void setInitialOxideThickness(NumericType thicknessUm) {
    callback_->setInitialOxideThickness(thicknessUm);
  }

  // Gas-transfer coefficient in µm/hr; large values approximate C_s = C*.
  void setTransferCoefficient(NumericType coefficient) {
    callback_->setTransferCoefficient(coefficient);
  }

  // Stress-coupling activation volume for interface reaction rate (m^3).
  void setReactionActivationVolume(NumericType volume) {
    callback_->setReactionActivationVolume(volume);
  }

  // Stress-coupling activation volume for oxide diffusivity (m^3).
  void setDiffusionActivationVolume(NumericType volume) {
    callback_->setDiffusionActivationVolume(volume);
  }

  // Maximum Cartesian grid points in the ViennaLS diffusion/mechanics solve.
  void setMaxGridPoints(std::size_t maxGridPoints) {
    callback_->setMaxGridPoints(maxGridPoints);
  }

  void setCouplingIterations(unsigned iterations) {
    callback_->setCouplingIterations(iterations);
  }

  void setCouplingTolerance(NumericType tolerance) {
    callback_->setCouplingTolerance(tolerance);
  }

  void setSolveBounds(const viennahrle::Index<D> &minIndex,
                      const viennahrle::Index<D> &maxIndex) {
    callback_->setSolveBounds(minIndex, maxIndex);
  }

  void clearSolveBounds() { callback_->clearSolveBounds(); }

  // Deal-Grove planar oxide thickness estimate in µm for the current settings.
  NumericType
  estimatePlanarOxideThickness(NumericType initialOxideThickness = 0.) const {
    return callback_->estimatePlanarOxideThickness(initialOxideThickness);
  }

  // Override which material is treated as silicon (default: Material::Si)
  void setSiliconMaterial(Material mat) { callback_->setSiliconMaterial(mat); }

  // Override which material is treated as oxide (default: Material::SiO2)
  void setOxideMaterial(Material mat) { callback_->setOxideMaterial(mat); }
};

} // namespace viennaps
