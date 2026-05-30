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
#include <lsBooleanOperation.hpp>
#include <lsGeometricAdvect.hpp>
#include <lsLOCOSOxidation.hpp>
#include <lsOxidationDeformation.hpp>
#include <lsOxidationDiffusion.hpp>
#include <lsOxidationMask.hpp>
#include <lsOxidationMaterials.hpp>
#include <lsOxidationModel.hpp>

#include <hrleSparseIterator.hpp>

#include "../process/psAdvectionCallback.hpp"
#include "../process/psProcessModel.hpp"
#include "../materials/psMaterial.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>

namespace viennaps {

using namespace viennacore;

enum class OxidantType { Dry, Wet };
enum class SiliconOrientation { Si100, Si111, PolySi };

template <class NumericType, int D>
class Oxidation : public ProcessModelBase<NumericType, D> {

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
  NumericType timeStep_ = 0.;   // 0 = no user cap; internal steps remain CFL-limited
  NumericType cflFactor_ = NumericType(0.499);
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
  Material maskMaterial_ = Material::Si3N4;
  viennals::OxidationMaskParameters<NumericType> maskParams_ =
      viennals::OxidationMaterials<NumericType>::siliconNitrideMask1000C();
  bool useMaskBendingBounds_ = false;
  viennahrle::Index<D> maskBendingMinIndex_{};
  viennahrle::Index<D> maskBendingMaxIndex_{};
  unsigned maskCouplingIterations_ = 8;
  NumericType maskCouplingTolerance_ = NumericType(2e-2);
  unsigned mechanicsIterations_ = 10;
  unsigned pressureIterations_ = 500;
  unsigned stokesIterations_ = 200;

  // Thin wrapper: forwards applyPreAdvect to Oxidation::doApplyPreAdvect.
  // Nested classes in C++ have full access to enclosing-class private members.
  class OxidationCallback : public AdvectionCallback<NumericType, D> {
    Oxidation *owner_;
  public:
    explicit OxidationCallback(Oxidation *o) : owner_(o) {}
    bool applyPreAdvect(NumericType processTime) override {
      return owner_->doApplyPreAdvect(processTime, this->domain);
    }
  };

public:
  Oxidation() {
    this->setProcessName("Oxidation");
    this->setAdvectionCallback(SmartPointer<OxidationCallback>::New(this));
  }

  // Temperature in °C (valid range: 800–1200 °C)
  void setTemperature(NumericType temperatureC) { temperature_ = temperatureC; }

  // Total oxidation time in hours
  void setTime(NumericType timeHr) { time_ = timeHr; }

  // Oxidant species: OxidantType::Dry (O2) or OxidantType::Wet (H2O)
  void setOxidant(OxidantType oxidant) { oxidant_ = oxidant; }

  // Ambient pressure in atm (scales both B and B/A linearly; default 1.0)
  void setPressure(NumericType pressureAtm) { pressure_ = pressureAtm; }

  // Crystal orientation: Si100, Si111, or PolySi (isotropic, uses <100> rates)
  void setOrientation(SiliconOrientation orientation) {
    orientation_ = orientation;
  }

  // Set the maximum duration of an internal oxidation step in hours.
  // The model always recomputes diffusion/mechanics/contact on CFL-limited
  // internal substeps. A positive value caps those internal steps; it does not
  // force the interfaces to move by this amount if the CFL limit is smaller.
  // Default (0): no user cap, only the CFL limit and remaining process time.
  void setTimeStep(NumericType dtHr) { timeStep_ = dtHr; }

  // Courant number used for CFL-limited internal stepping (default 0.499).
  // The actual internal step is min(user timeStep cap, CFL step, remaining).
  void setCFLFactor(NumericType factor) {
    cflFactor_ = std::clamp(factor, NumericType(1e-3), NumericType(0.499));
  }

  // Thickness of the auto-created native oxide if no SiO2 layer exists (µm)
  void setInitialOxideThickness(NumericType thicknessUm) {
    initialOxideThickness_ = thicknessUm;
  }

  // Gas-transfer coefficient in µm/hr; large values approximate C_s = C*.
  void setTransferCoefficient(NumericType coefficient) {
    transferCoefficient_ = coefficient;
  }

  // Stress-coupling activation volume for interface reaction rate (m^3).
  void setReactionActivationVolume(NumericType volume) {
    reactionActivationVolume_ = volume;
  }

  // Stress-coupling activation volume for oxide diffusivity (m^3).
  void setDiffusionActivationVolume(NumericType volume) {
    diffusionActivationVolume_ = volume;
  }

  // Maximum Cartesian grid points in the ViennaLS diffusion/mechanics solve.
  void setMaxGridPoints(std::size_t maxGridPoints) {
    maxGridPoints_ = maxGridPoints;
  }

  void setCouplingIterations(unsigned iterations) {
    couplingIterations_ = std::max(1u, iterations);
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

  // Deal-Grove planar oxide thickness estimate in µm for the current settings.
  NumericType
  estimatePlanarOxideThickness(NumericType initialOxideThickness = 0.) const {
    const auto rates = computeDealGroveRates();
    if (rates.B <= NumericType(0) || rates.BoA <= NumericType(0))
      return std::max(initialOxideThickness, NumericType(0));

    const NumericType initial = std::max(initialOxideThickness, NumericType(0));
    const NumericType A = rates.B / rates.BoA;
    const NumericType tau = (initial * initial + A * initial) / rates.B;
    const NumericType effectiveTime = std::max(time_, NumericType(0)) + tau;
    return (std::sqrt(A * A + NumericType(4) * rates.B * effectiveTime) - A) /
           NumericType(2);
  }

  // Override which material is treated as silicon (default: Material::Si)
  void setSiliconMaterial(Material mat) { siliconMaterial_ = mat; }

  // Override which material is treated as oxide (default: Material::SiO2)
  void setOxideMaterial(Material mat) { oxideMaterial_ = mat; }

  // Material treated as the oxidation mask (default: Material::Si3N4).
  // If a level set with this material is present in the domain, LOCOS physics
  // are activated: mask bending + constrained-ambient advection.
  void setMaskMaterial(Material mat) { maskMaterial_ = mat; }

  // Viscous-elasticity parameters for the mask layer (default: SiN at 1000 °C).
  void setMaskParameters(viennals::OxidationMaskParameters<NumericType> params) {
    maskParams_ = std::move(params);
  }

  // Explicit Cartesian index bounds for the mask bending solve.
  // When not set, bounds are auto-computed from the mask level-set narrow band.
  void setMaskBendingBounds(const viennahrle::Index<D> &minIdx,
                             const viennahrle::Index<D> &maxIdx) {
    maskBendingMinIndex_ = minIdx;
    maskBendingMaxIndex_ = maxIdx;
    useMaskBendingBounds_ = true;
  }

  void clearMaskBendingBounds() { useMaskBendingBounds_ = false; }

  // Number of outer mask-bending / deformation coupling iterations per step.
  void setMaskCouplingIterations(unsigned iterations) {
    maskCouplingIterations_ = std::max(1u, iterations);
  }

  // Convergence tolerance for the mask-deformation coupling loop.
  void setMaskCouplingTolerance(NumericType tolerance) {
    maskCouplingTolerance_ = tolerance;
  }

  void setMechanicsIterations(unsigned iterations) {
    mechanicsIterations_ = std::max(1u, iterations);
  }

  void setPressureIterations(unsigned iterations) {
    pressureIterations_ = std::max(1u, iterations);
  }

  void setStokesIterations(unsigned iterations) {
    stokesIterations_ = std::max(1u, iterations);
  }

  // Extracts and saves a mathematically wrapped surface mesh.
  // The mechanics solvers require the level sets to be independent
  // (wrapLowerLevelSet = false), but visualization extractors require
  // layered materials to explicitly enclose all lower layers. These methods
  // perform the necessary deep copies and boolean union operations to safely
  // output the layered meshes without mutating the active simulation state.
  void saveSurfaceMesh(SmartPointer<Domain<NumericType, D>> domain,
                       const std::string &fileName) const {
    namespace ls = viennals;

    auto &levelSets = domain->getLevelSets();
    auto matMap = domain->getMaterialMap();

    if (levelSets.empty() || !matMap) {
      Logger::getInstance()
          .addWarning("Oxidation: domain has no level sets or material map.")
          .print();
      return;
    }

    // Find Si, SiO2, and optional mask level-set indices.
    int siIdx = -1, sio2Idx = -1, maskIdx = -1;
    for (int i = static_cast<int>(levelSets.size()) - 1; i >= 0; --i) {
      const auto mat = matMap->getMaterialAtIdx(i);
      if (siIdx < 0 &&
          (mat == siliconMaterial_ || mat == Material::BulkSi ||
           mat == Material::PolySi || mat == Material::aSi))
        siIdx = i;
      if (sio2Idx < 0 && mat == oxideMaterial_)
        sio2Idx = i;
      if (maskIdx < 0 && mat == maskMaterial_)
        maskIdx = i;
    }

    if (siIdx < 0 || sio2Idx < 0) {
      Logger::getInstance()
          .addWarning("Oxidation: missing Silicon or Oxide layer for surface mesh extraction.")
          .print();
      return;
    }

    // Deep copies of the base simulation level sets
    auto siCopy = ls::Domain<NumericType, D>::New(levelSets[siIdx]);
    auto oxSurf = ls::Domain<NumericType, D>::New(levelSets[sio2Idx]);

    if (maskIdx >= 0) {
      auto mskSurf = ls::Domain<NumericType, D>::New(levelSets[maskIdx]);
      ls::BooleanOperation<NumericType, D>(oxSurf, mskSurf,
                                           ls::BooleanOperationEnum::UNION).apply();
      auto surfDomain = SmartPointer<Domain<NumericType, D>>::New();
      surfDomain->insertNextLevelSetAsMaterial(siCopy, matMap->getMaterialAtIdx(siIdx), false);
      surfDomain->insertNextLevelSetAsMaterial(oxSurf, matMap->getMaterialAtIdx(sio2Idx), false);
      surfDomain->insertNextLevelSetAsMaterial(mskSurf, matMap->getMaterialAtIdx(maskIdx), false);
      surfDomain->saveSurfaceMesh(fileName);
    } else {
      // Standard geometry (no mask)
      auto surfDomain = SmartPointer<Domain<NumericType, D>>::New();
      surfDomain->insertNextLevelSetAsMaterial(siCopy, matMap->getMaterialAtIdx(siIdx), false);
      surfDomain->insertNextLevelSetAsMaterial(oxSurf, matMap->getMaterialAtIdx(sio2Idx), false);
      surfDomain->saveSurfaceMesh(fileName);
    }
  }

  // Extracts and saves a mathematically wrapped volume mesh.
  void saveVolumeMesh(SmartPointer<Domain<NumericType, D>> domain,
                      const std::string &baseName) const {
    namespace ls = viennals;

    auto &levelSets = domain->getLevelSets();
    auto matMap = domain->getMaterialMap();

    if (levelSets.empty() || !matMap) {
      Logger::getInstance()
          .addWarning("Oxidation: domain has no level sets or material map.")
          .print();
      return;
    }

    int siIdx = -1, sio2Idx = -1, maskIdx = -1;
    for (int i = static_cast<int>(levelSets.size()) - 1; i >= 0; --i) {
      const auto mat = matMap->getMaterialAtIdx(i);
      if (siIdx < 0 &&
          (mat == siliconMaterial_ || mat == Material::BulkSi ||
           mat == Material::PolySi || mat == Material::aSi))
        siIdx = i;
      if (sio2Idx < 0 && mat == oxideMaterial_)
        sio2Idx = i;
      if (maskIdx < 0 && mat == maskMaterial_)
        maskIdx = i;
    }

    if (siIdx < 0 || sio2Idx < 0) {
      Logger::getInstance()
          .addWarning("Oxidation: missing Silicon or Oxide layer for volume mesh extraction.")
          .print();
      return;
    }

    auto siCopy = ls::Domain<NumericType, D>::New(levelSets[siIdx]);
    auto oxCopy = ls::Domain<NumericType, D>::New(levelSets[sio2Idx]);

    if (maskIdx >= 0) {
      auto mskCopy = ls::Domain<NumericType, D>::New(levelSets[maskIdx]);
      ls::BooleanOperation<NumericType, D>(oxCopy, siCopy,
                                           ls::BooleanOperationEnum::UNION).apply();
      ls::BooleanOperation<NumericType, D>(mskCopy, oxCopy,
                                           ls::BooleanOperationEnum::UNION).apply();

      auto volDomain = SmartPointer<Domain<NumericType, D>>::New();
      volDomain->insertNextLevelSetAsMaterial(siCopy, matMap->getMaterialAtIdx(siIdx), false);
      volDomain->insertNextLevelSetAsMaterial(oxCopy, matMap->getMaterialAtIdx(sio2Idx), false);
      volDomain->insertNextLevelSetAsMaterial(mskCopy, matMap->getMaterialAtIdx(maskIdx), false);
      volDomain->saveVolumeMesh(baseName);
    } else {
      ls::BooleanOperation<NumericType, D>(oxCopy, siCopy,
                                           ls::BooleanOperationEnum::UNION).apply();
      auto volDomain = SmartPointer<Domain<NumericType, D>>::New();
      volDomain->insertNextLevelSetAsMaterial(siCopy, matMap->getMaterialAtIdx(siIdx), false);
      volDomain->insertNextLevelSetAsMaterial(oxCopy, matMap->getMaterialAtIdx(sio2Idx), false);
      volDomain->saveVolumeMesh(baseName);
    }
  }

private:
  bool doApplyPreAdvect(NumericType /*processTime*/,
                        SmartPointer<Domain<NumericType, D>> domainPtr) {
    namespace ls = viennals;

    auto &domain = *domainPtr;
    auto &levelSets = domain.getLevelSets();
    auto matMap = domain.getMaterialMap();

    if (levelSets.empty() || !matMap) {
      Logger::getInstance()
          .addWarning("Oxidation: domain has no level sets or material map.")
          .print();
      return false;
    }

    // Find Si (reaction), SiO2 (ambient), and optional mask level-set indices.
    // Scan top-down so we pick the topmost layer of each material.
    int siIdx = -1, sio2Idx = -1, maskIdx = -1;
    for (int i = static_cast<int>(levelSets.size()) - 1; i >= 0; --i) {
      const auto mat = matMap->getMaterialAtIdx(i);
      if (siIdx < 0 &&
          (mat == siliconMaterial_ || mat == Material::BulkSi ||
           mat == Material::PolySi || mat == Material::aSi))
        siIdx = i;
      if (sio2Idx < 0 && mat == oxideMaterial_)
        sio2Idx = i;
      if (maskIdx < 0 && mat == maskMaterial_)
        maskIdx = i;
    }

    if (siIdx < 0) {
      Logger::getInstance()
          .addWarning("Oxidation: no silicon layer found in domain.")
          .print();
      return false;
    }

    if (temperature_ < NumericType(700) || temperature_ > NumericType(1200))
      Logger::getInstance()
          .addWarning("Oxidation: temperature " +
                      std::to_string(temperature_) +
                      " °C is outside the calibrated Deal-Grove range "
                      "[700, 1200] °C — rate constants may be inaccurate.")
          .print();

    auto reactionInterface = levelSets[siIdx];

    // If no oxide layer exists, create a thin native oxide on top of Si.
    SmartPointer<ls::Domain<NumericType, D>> ambientInterface;
    if (sio2Idx >= 0) {
      ambientInterface = levelSets[sio2Idx];
    } else {
      ambientInterface =
          SmartPointer<ls::Domain<NumericType, D>>::New(reactionInterface);
      if (initialOxideThickness_ > NumericType(0)) {
        Logger::getInstance()
            .addInfo("Oxidation: no SiO₂ layer found; seeding " +
                     std::to_string(initialOxideThickness_) +
                     " µm native oxide.")
            .print();
        auto sphere = SmartPointer<
            ls::SphereDistribution<viennahrle::CoordType, D>>::New(
            initialOxideThickness_);
        ls::GeometricAdvect<NumericType, D>(ambientInterface, sphere).apply();
      } else {
        Logger::getInstance()
            .addInfo("Oxidation: no SiO₂ layer found; inserting "
                     "zero-thickness oxide seed.")
            .print();
      }
      domain.insertNextLevelSetAsMaterial(ambientInterface, oxideMaterial_,
                                          false);
    }

    // Optional user override for the Cartesian solve bounds.
    viennahrle::Index<D> minIndex, maxIndex;
    if (useSolveBounds_) {
      minIndex = solveMinIndex_;
      maxIndex = solveMaxIndex_;
    }

    const NumericType gridDelta =
        reactionInterface->getGrid().getGridDelta();

    auto cflStep = [&](NumericType maxVel) -> NumericType {
      if (maxVel <= NumericType(0))
        return std::max(time_ / NumericType(20),
                        std::numeric_limits<NumericType>::min());
      return cflFactor_ * gridDelta / maxVel;
    };

    const auto rates = computeDealGroveRates();
    const NumericType initialVelocityEstimate =
        rates.BoA > NumericType(0) ? rates.BoA : NumericType(0);
    const NumericType seedStep = cflStep(initialVelocityEstimate);
    const NumericType userStepCap =
        timeStep_ > NumericType(0) ? timeStep_
                                   : std::numeric_limits<NumericType>::max();

    auto oxParams = computeOxParams();

    ls::OxidationCouplingParameters<NumericType> coupling;
    coupling.maxIterations = couplingIterations_;
    coupling.tolerance = couplingTolerance_;

    if (Logger::hasInfo()) {
      const std::string mode = (maskIdx >= 0) ? "LOCOS" : "standard";
      const std::string oxStr = (oxidant_ == OxidantType::Wet) ? "wet" : "dry";
      const NumericType initDt = std::min(userStepCap, seedStep);
      Logger::getInstance()
          .addInfo("Oxidation: starting " + mode + " simulation"
                   ", T=" + std::to_string(temperature_) + " °C"
                   ", " + oxStr + " oxidation"
                   ", B=" + std::to_string(rates.B) + " µm²/hr"
                   ", B/A=" + std::to_string(rates.BoA) + " µm/hr"
                   ", Δx=" + std::to_string(gridDelta) + " µm"
                   ", total=" + std::to_string(time_) + " hr"
                   ", initial_dt≤" + std::to_string(initDt) + " hr")
          .print();
    }

    if (maskIdx >= 0) {
      // ── LOCOS path ─────────────────────────────────────────────────────────
      auto maskInterface = levelSets[maskIdx];

      auto locos = ls::LOCOSOxidation<NumericType, D>::New(
          reactionInterface, ambientInterface, maskInterface);
      locos->setOxidationParameters(oxParams);
      locos->setCouplingParameters(coupling);
      locos->setMaskParameters(maskParams_);
      if (useSolveBounds_)
        locos->setSolveBounds(solveMinIndex_, solveMaxIndex_);
      {
        viennahrle::Index<D> mbMin{}, mbMax{};
        if (useMaskBendingBounds_) {
          mbMin = maskBendingMinIndex_;
          mbMax = maskBendingMaxIndex_;
        } else {
          std::tie(mbMin, mbMax) = computeLevelSetBounds(maskInterface, 4);
        }
        locos->setMaskBendingBounds(mbMin, mbMax);
      }
      locos->setMaskCouplingIterations(maskCouplingIterations_);
      locos->setMaskCouplingTolerance(maskCouplingTolerance_);

      NumericType time = 0.;
      unsigned substep = 0;
      NumericType nextStepEstimate = seedStep;
      const NumericType timeEps = NumericType(1e-9) * time_;
      while (time_ - time > timeEps) {
        NumericType requestedDt =
            std::min({userStepCap, nextStepEstimate, time_ - time});
        if (requestedDt <= NumericType(0))
          break;

        locos->setDeformationParameters(computeDefParams(requestedDt));

        logCFLStepStart("LOCOS", substep + 1, time, requestedDt);

        const NumericType actualDt =
            locos->applyCFLLimited(requestedDt, cflFactor_);
        if (actualDt <= NumericType(0))
          break;

        logCFLStep("LOCOS", ++substep, time, requestedDt, actualDt);

        time += actualDt;
        nextStepEstimate = std::min(userStepCap, actualDt);
      }
      if (Logger::hasInfo())
        Logger::getInstance()
            .addInfo("Oxidation: LOCOS complete — " +
                     std::to_string(substep) + " substep(s), " +
                     std::to_string(time) + " hr simulated.")
            .print();
    } else {
      // ── Standard (non-LOCOS) path ──────────────────────────────────────────

      auto diffField = ls::OxidationDiffusion<NumericType, D>::New(
          reactionInterface, ambientInterface, oxParams);
      if (useSolveBounds_)
        diffField->setSolveBounds(minIndex, maxIndex);

      auto defParams = computeDefParams(std::min(userStepCap, seedStep));
      auto defField = ls::OxidationDeformation<NumericType, D>::New(
          reactionInterface, ambientInterface, diffField, oxParams, defParams);
      if (useSolveBounds_)
        defField->setSolveBounds(minIndex, maxIndex);

      auto coupledModel = ls::OxidationModel<NumericType, D>::New(
          diffField, defField, coupling);
      if (useSolveBounds_)
        coupledModel->setSolveBounds(minIndex, maxIndex);

      ls::Advect<NumericType, D> ambientAdvect;
      ambientAdvect.insertNextLevelSet(ambientInterface);
      ambientAdvect.setVelocityField(defField);
      ambientAdvect.setSpatialScheme(
          ls::SpatialSchemeEnum::ENGQUIST_OSHER_1ST_ORDER);
      ambientAdvect.setTemporalScheme(
          ls::TemporalSchemeEnum::FORWARD_EULER);

      ls::Advect<NumericType, D> reactionAdvect;
      reactionAdvect.insertNextLevelSet(reactionInterface);
      reactionAdvect.setVelocityField(diffField);
      reactionAdvect.setSpatialScheme(
          ls::SpatialSchemeEnum::ENGQUIST_OSHER_1ST_ORDER);
      reactionAdvect.setTemporalScheme(
          ls::TemporalSchemeEnum::FORWARD_EULER);

      NumericType time = 0.;
      unsigned substep = 0;
      NumericType nextStepEstimate = seedStep;
      const NumericType timeEps = NumericType(1e-9) * time_;
      while (time_ - time > timeEps) {
        NumericType requestedDt =
            std::min({userStepCap, nextStepEstimate, time_ - time});
        if (requestedDt <= NumericType(0))
          break;

        defParams.stressTimeStep = requestedDt;
        defField->setDeformationParameters(defParams);

        logCFLStepStart("Oxidation", substep + 1, time, requestedDt);

        coupledModel->apply();

        NumericType maxVelocity = maxCoupledVelocity(diffField, defField);
        NumericType actualDt = std::min(requestedDt, cflStep(maxVelocity));
        if (actualDt < requestedDt * (NumericType(1) - NumericType(1e-8))) {
          defParams.stressTimeStep = actualDt;
          defField->setDeformationParameters(defParams);
          coupledModel->apply();
        }

        logCFLStep("Oxidation", ++substep, time, requestedDt, actualDt,
                   maxVelocity);

        ambientAdvect.setAdvectionTime(actualDt);
        ambientAdvect.apply();

        reactionAdvect.setAdvectionTime(actualDt);
        reactionAdvect.apply();

        diffField->markGeometryChanged();
        defField->markGeometryChanged();

        time += actualDt;
        nextStepEstimate = std::min(userStepCap, actualDt);
      }
      if (Logger::hasInfo())
        Logger::getInstance()
            .addInfo("Oxidation: simulation complete — " +
                     std::to_string(substep) + " substep(s), " +
                     std::to_string(time) + " hr simulated.")
            .print();
    }

    return true;
  }

  NumericType maxCoupledVelocity(
      SmartPointer<viennals::OxidationDiffusion<NumericType, D>> diffField,
      SmartPointer<viennals::OxidationDeformation<NumericType, D>> defField)
      const {
    NumericType maxVelocity = diffField->getDissipationAlpha(0, -1, {});
    for (unsigned d = 0; d < D; ++d)
      maxVelocity =
          std::max(maxVelocity, defField->getDissipationAlpha(d, -1, {}));
    return maxVelocity;
  }

  void logCFLStep(const std::string &label, unsigned substep,
                  NumericType elapsed, NumericType requestedDt,
                  NumericType actualDt,
                  NumericType maxVelocity = NumericType(-1)) const {
    if (!Logger::hasInfo())
      return;

    std::string message = label + " CFL substep " + std::to_string(substep) +
                          ": t=" + std::to_string(elapsed) +
                          " hr, requested_dt=" +
                          std::to_string(requestedDt) +
                          " hr, actual_dt=" + std::to_string(actualDt) +
                          " hr";
    if (maxVelocity >= NumericType(0))
      message += ", max_velocity=" + std::to_string(maxVelocity) + " um/hr";
    if (actualDt < requestedDt * (NumericType(1) - NumericType(1e-8)))
      message += " (CFL-limited)";
    Logger::getInstance().addInfo(message).print();
  }

  void logCFLStepStart(const std::string &label, unsigned substep,
                       NumericType elapsed, NumericType requestedDt) const {
    if (!Logger::hasInfo())
      return;

    Logger::getInstance()
        .addInfo(label + " CFL substep " + std::to_string(substep) +
                 ": starting solve at t=" + std::to_string(elapsed) +
                 " hr, requested_dt=" + std::to_string(requestedDt) + " hr")
        .print();
  }

  // Map process conditions → OxidationParameters.
  // With C* = N = 1 (normalised): B = 2D, so diffusionCoefficient = B/2.
  viennals::OxidationParameters<NumericType> computeOxParams() const {
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
    p.mechanicsIterations = mechanicsIterations_;
    p.mechanicsTolerance = NumericType(1e-7);
    p.pressureIterations = pressureIterations_;
    p.stokesIterations = stokesIterations_;
    p.pressureTolerance = NumericType(1e-6);
    p.stokesTolerance = NumericType(1e-7);
    p.tolerance = NumericType(1e-7);
    p.maxGridPoints = maxGridPoints_;
    return p;
  }

  // Compute Cartesian index bounding box of the defined (active) points in a
  // level-set narrow band, padded by `padding` cells.  Same pattern as
  // OxidationSolverBase::definedPointBounds().
  static std::pair<viennahrle::Index<D>, viennahrle::Index<D>>
  computeLevelSetBounds(
      SmartPointer<viennals::Domain<NumericType, D>> ls, int padding) {
    using DomType = typename viennals::Domain<NumericType, D>::DomainType;
    viennahrle::Index<D> lo{}, hi{};
    bool found = false;
    for (viennahrle::ConstSparseIterator<DomType> it(ls->getDomain());
         !it.isFinished(); ++it) {
      if (!it.isDefined())
        continue;
      const auto &idx = it.getStartIndices();
      if (!found) {
        lo = hi = idx;
        found = true;
      } else {
        for (int d = 0; d < D; ++d) {
          lo[d] = std::min(lo[d], idx[d]);
          hi[d] = std::max(hi[d], idx[d]);
        }
      }
    }
    if (found)
      for (int d = 0; d < D; ++d) {
        lo[d] -= padding;
        hi[d] += padding;
      }
    return {lo, hi};
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

} // namespace viennaps
