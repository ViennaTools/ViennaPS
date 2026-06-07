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

#include <lsBooleanOperation.hpp>
#include <lsGeometricAdvect.hpp>
#include <lsOxidation.hpp>
#include <lsToMesh.hpp>
#include <lsVTKWriter.hpp>
#include <lsOxidationDeformation.hpp>
#include <lsOxidationDiffusion.hpp>
#include <lsOxidationPresets.hpp>

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
enum class SiliconOrientation { Si100, Si110, Si111, PolySi };
using GpuMode = viennals::GpuMode;
using GpuPreconditioner = viennals::GpuPreconditioner;

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
      viennals::OxidationPresets<NumericType>::siliconNitrideMask1000C();
  bool useMaskBendingBounds_ = false;
  viennahrle::Index<D> maskBendingMinIndex_{};
  viennahrle::Index<D> maskBendingMaxIndex_{};
  unsigned maskCouplingIterations_ = 8;
  NumericType maskCouplingTolerance_ = NumericType(2e-2);
  GpuMode gpuMode_ = GpuMode::Auto;
  GpuPreconditioner gpuPreconditioner_ = GpuPreconditioner::Jacobi;
  unsigned mechanicsIterations_ = 200;
  unsigned pressureIterations_ = 500;
  unsigned stokesIterations_ = 200;
  NumericType pressureTolerance_ = NumericType(1e-6);
  NumericType stokesTolerance_ = NumericType(1e-7);
  NumericType mechanicsTolerance_ = NumericType(1e-7);
  bool toleranceWarningEmitted_ = false;

  void validateTolerances() {
    if (toleranceWarningEmitted_)
      return;
    toleranceWarningEmitted_ = true;

    // Each outer loop can only converge as tightly as the loop beneath it.
    // The mechanics coupling residual is contaminated by solver noise at the
    // level of max(pressureTolerance, stokesTolerance).  mechanicsTolerance
    // must be at least 5× looser to avoid the solver stagnating at the noise
    // floor (which causes hundreds of wasted iterations with no progress).
    const NumericType solverFloor =
        std::max(pressureTolerance_, stokesTolerance_);
    if (mechanicsTolerance_ < NumericType(5) * solverFloor)
      Logger::getInstance()
          .addWarning("Oxidation: mechanicsTolerance (" +
                      std::to_string(mechanicsTolerance_) +
                      ") is less than 5× max(pressureTolerance, stokesTolerance) ("
                      + std::to_string(solverFloor) +
                      "). The mechanics coupling residual cannot go below the "
                      "solver noise floor — solveMechanics will stall without "
                      "converging. Set mechanicsTolerance >= " +
                      std::to_string(NumericType(5) * solverFloor) + ".")
          .print();

    if (couplingTolerance_ < mechanicsTolerance_)
      Logger::getInstance()
          .addWarning("Oxidation: couplingTolerance (" +
                      std::to_string(couplingTolerance_) +
                      ") is tighter than mechanicsTolerance (" +
                      std::to_string(mechanicsTolerance_) +
                      "). The diffusion–deformation coupling cannot converge "
                      "more precisely than the mechanics solve — couplingTolerance "
                      "will never be reached.")
          .print();
  }

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

  // Inner traction-mask solve controls.  These update the stored mask
  // parameters directly; call them after setMaskParameters() when overriding a
  // complete parameter object.
  void setMaskTractionIterations(unsigned iterations) {
    maskParams_.maxIterations = std::max(1u, iterations);
  }

  void setMaskTractionTolerance(NumericType tolerance) {
    maskParams_.tolerance = std::max(tolerance, NumericType(1e-12));
  }

  void setMaskTractionRelaxation(NumericType relaxation) {
    maskParams_.relaxation =
        std::clamp(relaxation, NumericType(0.01), NumericType(1));
  }

  void setMaskContactLoadRelaxation(NumericType relaxation) {
    maskParams_.contactLoadRelaxation =
        std::clamp(relaxation, NumericType(0.02), NumericType(1));
  }

  void setMaskContactReleaseFraction(NumericType fraction) {
    maskParams_.contactReleaseFraction =
        std::clamp(fraction, NumericType(0), NumericType(0.25));
  }

  void setMaskUnilateralContact(bool enabled) {
    maskParams_.unilateralContact = enabled;
  }

  /// SOR omega for the multigrid V-cycle smoother used by the traction mask
  /// solve (contactMode=1).  1.0 = Gauss-Seidel; values in (1, 1.4] add
  /// over-relaxation.  Separate from the outer Aitken relaxation set by
  /// setMaskTractionRelaxation().
  void setMaskSmootherOmega(NumericType omega) {
    maskParams_.multigridSmootherOmega =
        std::clamp(omega, NumericType(0.2), NumericType(1.4));
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

  /// Select the BiCGSTAB solver back-end.
  ///   GpuMode::Auto — CPU below threshold, GPU above threshold (default)
  ///   GpuMode::Gpu  — always GPU (throws if unavailable or unsuccessful)
  ///   GpuMode::Cpu  — always CPU
  void setGpuMode(GpuMode mode) { gpuMode_ = mode; }
  /// Select the GPU BiCGSTAB preconditioner. Jacobi matches the CPU solver.
  void setGpuPreconditioner(GpuPreconditioner preconditioner) {
    gpuPreconditioner_ = preconditioner;
  }

  void setMechanicsIterations(unsigned iterations) {
    mechanicsIterations_ = std::max(1u, iterations);
  }

  void setPressureIterations(unsigned iterations) {
    pressureIterations_ = std::max(1u, iterations);
  }

  void setPressureTolerance(NumericType tol) { pressureTolerance_ = tol; }
  void setStokesTolerance(NumericType tol) { stokesTolerance_ = tol; }
  void setMechanicsTolerance(NumericType tol) { mechanicsTolerance_ = tol; }

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

    // Write warm-restart fields (OxVelocity, OxPressure, OxConcentration,
    // OxStressR0-R2) from the original level sets as companion point-cloud
    // files alongside the volume mesh.  Uses the pre-Boolean-op level sets so
    // the pointData arrays are intact; lsToMesh + lsVTKWriter propagate them
    // faithfully, unlike lsWriteVisualizationMesh which strips all point data.
    const auto &oxLS = levelSets[sio2Idx];
    if (oxLS->getPointData().getScalarDataSize() > 0 ||
        oxLS->getPointData().getVectorDataSize() > 0) {
      auto oxFieldMesh = ls::SmartPointer<ls::Mesh<NumericType>>::New();
      ls::ToMesh<NumericType, D>(oxLS, oxFieldMesh).apply();
      ls::VTKWriter<NumericType>(oxFieldMesh,
                                 baseName + "_oxide_fields.vtp").apply();
    }
    if (maskIdx >= 0) {
      const auto &maskLS = levelSets[maskIdx];
      if (maskLS->getPointData().getVectorDataSize() > 0) {
        auto maskFieldMesh = ls::SmartPointer<ls::Mesh<NumericType>>::New();
        ls::ToMesh<NumericType, D>(maskLS, maskFieldMesh).apply();
        ls::VTKWriter<NumericType>(maskFieldMesh,
                                   baseName + "_mask_fields.vtp").apply();
      }
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

    validateTolerances();

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

    const NumericType gridDelta =
        reactionInterface->getGrid().getGridDelta();

    auto cflStep = [&](NumericType maxVel) -> NumericType {
      if (maxVel <= NumericType(0))
        return std::max(time_ / NumericType(20),
                        std::numeric_limits<NumericType>::min());
      return cflFactor_ * gridDelta / maxVel;
    };

    const auto rates = computeDealGroveRates();
    auto oxParams = computeOxParams();
    // B/A is the total oxide growth rate, but the CFL limit is set by the
    // faster of the two surfaces: the ambient interface moves at
    // (β−1)/β × B/A while the reaction interface moves at B/A/β.
    // Using B/A directly over-constrains the seed step by ~β/(β−1) ≈ 1.8×.
    const NumericType beta = oxParams.expansionCoefficient;
    const NumericType maxSurfaceVelocity =
        (rates.BoA > NumericType(0) && beta > NumericType(1))
            ? rates.BoA * (beta - NumericType(1)) / beta
            : (rates.BoA > NumericType(0) ? rates.BoA : NumericType(0));
    const NumericType seedStep = cflStep(maxSurfaceVelocity);
    const NumericType userStepCap =
        timeStep_ > NumericType(0) ? timeStep_
                                   : std::numeric_limits<NumericType>::max();

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

    const std::string modeLabel = (maskIdx >= 0) ? "LOCOS" : "Oxidation";

    auto locos = ls::Oxidation<NumericType, D>::New(
        reactionInterface, ambientInterface);
    locos->setGpuMode(gpuMode_);
    locos->setGpuPreconditioner(gpuPreconditioner_);
    locos->setOxidationParameters(oxParams);
    locos->setCouplingParameters(coupling);
    if (useSolveBounds_)
      locos->setSolveBounds(solveMinIndex_, solveMaxIndex_);

    if (maskIdx >= 0) {
      auto maskLS = levelSets[maskIdx];
      locos->setMaskInterface(maskLS);
      // Sync simulation temperature so OxidationMaskBending's Arrhenius
      // viscosity scaling uses the correct temperature, not the preset's default.
      auto activeMaskParams = maskParams_;
      activeMaskParams.temperature = temperature_ + NumericType(273.15);
      locos->setMaskParameters(activeMaskParams);
      {
        viennahrle::Index<D> mbMin{}, mbMax{};
        if (useMaskBendingBounds_) {
          mbMin = maskBendingMinIndex_;
          mbMax = maskBendingMaxIndex_;
        } else {
          std::tie(mbMin, mbMax) = computeLevelSetBounds(maskLS, 4);
        }
        locos->setMaskBendingBounds(mbMin, mbMax);
      }
      locos->setMaskCouplingIterations(maskCouplingIterations_);
      locos->setMaskCouplingTolerance(maskCouplingTolerance_);
    }

    NumericType time = 0.;
    unsigned substep = 0;
    NumericType nextStepEstimate = seedStep;
    NumericType lastAcceptedDt = seedStep;
    constexpr NumericType maxStepGrowth = NumericType(2);
    const NumericType timeEps = NumericType(1e-9) * time_;
    while (time_ - time > timeEps) {
      const NumericType growthLimitedStep =
          (substep == 0) ? nextStepEstimate
                         : std::min(nextStepEstimate,
                                    maxStepGrowth * lastAcceptedDt);
      NumericType requestedDt =
          std::min({userStepCap, growthLimitedStep, time_ - time});
      if (requestedDt <= NumericType(0))
        break;

      locos->setDeformationParameters(computeDefParams(requestedDt));

      logCFLStepStart(modeLabel, substep + 1, time, requestedDt);

      const NumericType actualDt =
          locos->applyCFLLimited(requestedDt, cflFactor_);
      if (actualDt <= NumericType(0))
        break;

      logCFLStep(modeLabel, ++substep, time, requestedDt, actualDt);

      time += actualDt;
      lastAcceptedDt = actualDt;
      const NumericType maxV = locos->getLastMaxVelocity();
      nextStepEstimate = (maxV > NumericType(0))
                             ? std::min(userStepCap, cflStep(maxV))
                             : std::min(userStepCap, actualDt);
    }
    if (Logger::hasInfo())
      Logger::getInstance()
          .addInfo("Oxidation: " + modeLabel + " complete — " +
                   std::to_string(substep) + " substep(s), " +
                   std::to_string(time) + " hr simulated.")
          .print();

    return true;
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
    // Continuous crystal-axis correction applied per-face by OxidationDiffusion:
    //   k(n) = k_base × [1 + (ratio − 1) × (1 − (n · axis)²)]
    // k_base is the B/A for the chosen bulk orientation; ratio is the B/A of
    // faces perpendicular to the wafer normal divided by k_base. crystalAxis
    // points along the wafer normal (y = surface-normal convention in 2D).
    // Ratios derived from the (100):(110):(111) = 1 : 1.45 : 1.68 ladder.
    p.crystalAxis = {NumericType(0), NumericType(1), NumericType(0)};
    switch (orientation_) {
      case SiliconOrientation::Si100:
        // Perpendicular faces are (110)-like: 1.45× faster than (100).
        p.reactionRateRatio111 = NumericType(1.45);
        break;
      case SiliconOrientation::Si110:
        // Perpendicular faces are (100)-like: 1/1.45 ≈ 0.690× of (110).
        p.reactionRateRatio111 = NumericType(1) / NumericType(1.45);
        break;
      case SiliconOrientation::Si111:
        // Perpendicular faces are (100)-like: 1/1.68 ≈ 0.595× of (111).
        p.reactionRateRatio111 = NumericType(1) / NumericType(1.68);
        break;
      default: // PolySi — isotropic
        p.reactionRateRatio111 = NumericType(1);
        break;
    }
    p.maxGridPoints = maxGridPoints_;
    p.maxIterations = 10000;
    p.tolerance = NumericType(1e-7);
    return p;
  }

  // Oxide mechanics: viscosity follows Arrhenius (Irene, J. Electrochem. Soc.
  // 125, 1708 (1978)); bulk and shear moduli treated as temperature-independent.
  viennals::OxidationDeformationParameters<NumericType>
  computeDefParams(NumericType dt) const {
    const NumericType T_K = temperature_ + NumericType(273.15);
    // η(T) = η_ref × exp(Ea/kB × (1/T − 1/T_ref))
    // η_ref = 1×10¹⁰ Pa·hr at T_ref = 1000 °C, Ea = 1.5 eV (Irene 1978)
    static constexpr NumericType etaRef  = NumericType(1e10);
    static constexpr NumericType etaEa   = NumericType(1.5);     // eV
    static constexpr NumericType etaTref = NumericType(1273.15); // K
    const NumericType viscosity =
        etaRef * std::exp(etaEa / kB_ *
                          (NumericType(1) / T_K - NumericType(1) / etaTref));

    viennals::OxidationDeformationParameters<NumericType> p;
    p.viscosity = viscosity;
    p.bulkModulus = NumericType(7.5e8);     // Pa
    p.shearModulus = NumericType(3e10);     // Pa
    p.stressTimeStep = dt;
    p.mechanicsIterations = mechanicsIterations_;
    p.mechanicsTolerance = mechanicsTolerance_;
    p.pressureIterations = pressureIterations_;
    p.stokesIterations = stokesIterations_;
    p.pressureTolerance = pressureTolerance_;
    p.stokesTolerance = stokesTolerance_;
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

  // Deal-Grove Arrhenius table.
  // B (parabolic) is orientation-independent. B/A (linear) pre-exponentials
  // differ by orientation; activation energies are the same for all.
  //
  // Dry B/A has a well-established two-regime Arrhenius break near 950 °C:
  //   T > 950 °C: E_a = 2.00 eV  (Massoud, Plummer & Irene, J. Electrochem.
  //               Soc. 132, 2685 (1985), Table III — Si(100) and Si(111))
  //   T < 950 °C: E_a = 2.30 eV  (ibid., low-T fit; BoA0 scaled to match
  //               orientation via the same 1.68 / 1.45 ratios)
  // Wet B/A: single-regime suffices — the two-regime effect is less pronounced
  //   and less well-established for H₂O (Deal, J. Electrochem. Soc. 125, 576
  //   (1978), Table I).
  // Si(110): BoA0 = 1.45 × BoA0(100) in both regimes (Massoud 1985).
  // PolySi: isotropic; (100) rates used as conservative baseline.
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
      if (temperature_ < NumericType(950)) {
        // Low-T regime (<950 °C): higher activation energy 2.30 eV.
        if (orientation_ == SiliconOrientation::Si111)
          return {NumericType(772), NumericType(1.23),
                  NumericType(5.82e7), NumericType(2.30)};
        if (orientation_ == SiliconOrientation::Si110)
          return {NumericType(772), NumericType(1.23),
                  NumericType(5.02e7), NumericType(2.30)}; // 3.46e7 × 1.45
        return   {NumericType(772), NumericType(1.23),
                  NumericType(3.46e7), NumericType(2.30)}; // Si100 / PolySi
      }
      // High-T regime (≥950 °C): standard activation energy 2.00 eV.
      if (orientation_ == SiliconOrientation::Si111)
        return {NumericType(772), NumericType(1.23),
                NumericType(6.23e6), NumericType(2.00)};
      if (orientation_ == SiliconOrientation::Si110)
        return {NumericType(772), NumericType(1.23),
                NumericType(5.38e6), NumericType(2.00)}; // 3.71e6 × 1.45
      return   {NumericType(772), NumericType(1.23),
                NumericType(3.71e6), NumericType(2.00)}; // Si100 / PolySi
    }
    // Wet: single-regime for all orientations.
    if (orientation_ == SiliconOrientation::Si111)
      return {NumericType(386), NumericType(0.78),
              NumericType(1.63e8), NumericType(2.05)};
    if (orientation_ == SiliconOrientation::Si110)
      return {NumericType(386), NumericType(0.78),
              NumericType(1.41e8), NumericType(2.05)}; // 9.70e7 × 1.45
    return   {NumericType(386), NumericType(0.78),
              NumericType(9.70e7), NumericType(2.05)}; // Si100 / PolySi
  }
};

} // namespace viennaps
