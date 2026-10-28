#pragma once

/// psOxidation — ViennaPS process model for thermal silicon oxidation.
///
/// Wraps the ViennaLS coupled diffusion+deformation oxidation engine as a
/// psProcess model, exposing process-level inputs (temperature, time, oxidant
/// type, pressure, crystal orientation) instead of raw OxidationParameters.
///
/// Rate constants follow the Deal-Grove model (Deal & Grove, J. Appl. Phys.
/// 36, 3770 (1965)). The built-in Arrhenius table is the common 1 atm
/// low-doping silicon table used in process texts and calculators: dry/wet
/// B and B/A for <100>/<111>, with B/A(111)/B/A(100) = 1.68.
///
/// Usage:
///   auto model = SmartPointer<Oxidation<T, D>>::New();
///   model->setTemperature(1000.);               // °C
///   model->setTime(0.5);                         // hours
///   model->setOxidant(OxidantType::WET);
///   model->setOrientation(SiliconOrientation::Si100);
///   psProcess<T, D>(domain, model).apply();
///
/// The domain must contain a level set mapped to Material::Si (or BulkSi).
/// If no SiO2 level set is present, the native oxide (setInitialOxideThickness)
/// is created and appended to the top of the level-set stack (only where no
/// mask covers the silicon).  An oxide thinner than three grid cells first
/// grows by the 1-D Deal-Grove law until the grid resolves it (the time is
/// charged to the process; an impermeable mask on the silicon blocks it);
/// the 2-D solve continues from there.

#include <lsAdvect.hpp>
#include <lsBooleanOperation.hpp>
#include <lsGeometricAdvect.hpp>
#include <lsOxidation.hpp>
#include <lsOxidationDeformation.hpp>
#include <lsOxidationDiffusion.hpp>
#include <lsOxidationPresets.hpp>
#include <lsResample.hpp>
#include <lsToMesh.hpp>
#include <lsVTKWriter.hpp>
#include <lsVelocityField.hpp>

#include <hrleSparseIterator.hpp>

#include "../materials/psMaterial.hpp"
#include "../process/psProcessModel.hpp"
#include "../psUtil.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <unordered_map>

namespace viennaps {

using namespace viennacore;

using GpuMode = viennals::GpuMode;

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
    NumericType B0, EB;     // B  = B0  * exp(-EB  / kT)
    NumericType BoA0, EBoA; // B/A = BoA0 * exp(-EBoA / kT)
  };

  static constexpr NumericType kB_ = NumericType(8.617333e-5); // eV/K

  // settings
  NumericType temperature_ = 1000.;
  NumericType time_ = 1.;
  OxidantType oxidant_ = OxidantType::DRY;
  NumericType pressure_ = 1.;
  SiliconOrientation orientation_ = SiliconOrientation::Si100;
  NumericType timeStep_ =
      0.; // 0 = no user cap; internal steps remain CFL-limited
  // First internal substep cap (hr).  0 = model default (0.004): start where
  // the cold coupled solve accepts, ramp recovers full step size.
  NumericType initialTimeStep_ = NumericType(0);
  // One-shot persistent regrid (TODO.md unified design, Mode 1): when the
  // thinnest oxide layer clears regridFactor_ * regridDelta_, resample the
  // ENTIRE level-set stack once onto regridDelta_ and continue coarse.
  // 0 = disabled.  Fine grid is spent only while the thin seed needs it.
  NumericType regridDelta_ = NumericType(0);
  NumericType regridFactor_ = NumericType(2);
  NumericType cflFactor_ = NumericType(0.499);
  NumericType initialOxideThickness_ = NumericType(0.002); // µm (2 nm)
  // The 2-D solve needs the oxide to span this many grid cells: diffusion and
  // mechanics live on the grid points between the Si and the oxide surface.
  // Every validated run started from three (Drob 2025: 1.8 nm on 0.6 nm,
  // Senez 1996: 15 nm on 5 nm, Kao 1987: 30 nm on 10 nm).
  static constexpr NumericType resolvedOxideCells_ = NumericType(3);
  NumericType transferCoefficient_ = NumericType(100);
  // Meshed mask diffusion: gamma = D_SiO2/D_mask.  The mask body joins the
  // oxidant diffusion solve with D/gamma; its resistance follows the actual
  // level-set geometry.  0 = impermeable mask (upstream default).
  NumericType maskBarrierPermeabilityRatio_ = NumericType(0);
  bool maskBarrierOnly_ = false;
  NumericType rateB_override_ = NumericType(0);   // 0 = use Arrhenius table
  NumericType rateBoA_override_ = NumericType(0); // 0 = use Arrhenius table
  // Stress-dependent oxidation kinetics (Kao 1988, Senez 1996): the
  // interface normal stress retards the reaction (V_k), the pressure retards
  // oxidant diffusion (V_D).  Calibrated 2026-10-22 on the Kao et al. 1987
  // cylinder data together with the Arrhenius viscosity below (rms 0.065 in
  // t_curved/t_flat over 81 points, wet 800-1200 °C and dry 900-1200 °C):
  // V_k = 25 Å³ (= Ω_SiO2 − Ω_Si, Kao's value), V_D = 40 Å³.
  bool stressDependentOxidation_ = true;
  NumericType reactionActivationVolume_ = NumericType(2.5e-29);
  // Shear activation volume of the stress-dependent oxide viscosity (m^3);
  // < 0 = preset (300 A^3), 0 = linear viscosity.
  NumericType shearActivationVolume_ = NumericType(-1);
  NumericType diffusionActivationVolume_ = NumericType(4.0e-29);
  std::size_t maxGridPoints_ = 5000000;
  unsigned couplingIterations_ = 30;
  NumericType couplingTolerance_ = NumericType(5e-2);
  bool useSolveBounds_ = false;
  viennahrle::Index<D> solveMinIndex_{};
  viennahrle::Index<D> solveMaxIndex_{};
  Material siliconMaterial_ = Material::Si;
  Material oxideMaterial_ = Material::SiO2;
  Material maskMaterial_ = Material::Si3N4;
  // Mask parameters; unset: Si3N4 at the process temperature
  // (OxidationPresets::siliconNitrideMask).
  viennals::OxidationMaskParameters maskParams_{};
  bool maskParamsSet_ = false;
  GpuMode gpuMode_ = GpuMode::Cpu;
  unsigned mechanicsIterations_ = 200;
  NumericType mechanicsTolerance_ = NumericType(1e-4);
  bool toleranceWarningEmitted_ = false;

  void validateTolerances() {
    if (toleranceWarningEmitted_)
      return;
    toleranceWarningEmitted_ = true;

    // Each outer loop can only converge as tightly as the loop beneath it.
    if (couplingTolerance_ < mechanicsTolerance_)
      Logger::getInstance()
          .addWarning(
              "Oxidation: couplingTolerance (" +
              std::to_string(couplingTolerance_) +
              ") is tighter than mechanicsTolerance (" +
              std::to_string(mechanicsTolerance_) +
              "). The diffusion–deformation coupling cannot converge "
              "more precisely than the mechanics solve — couplingTolerance "
              "will never be reached.")
          .print();
  }

public:
  Oxidation() { this->setProcessName("Oxidation"); }

  bool managesOwnPhysics() const override { return true; }

  void applyModel(SmartPointer<Domain<NumericType, D>> domain) override {
    run(domain);
  }

  // Temperature in °C (valid range: 800–1200 °C)
  void setTemperature(NumericType temperatureC) { temperature_ = temperatureC; }

  // Total oxidation time in hours
  void setTime(NumericType timeHr) { time_ = timeHr; }

  // Oxidant species: OxidantType::DRY (O2) or OxidantType::WET (H2O)
  void setOxidant(OxidantType oxidant) { oxidant_ = oxidant; }

  void setOxidant(const std::string &oxidantStr) {
    oxidant_ = util::convert<OxidantType>(oxidantStr);
  }

  // Ambient pressure in atm (scales both B and B/A linearly; default 1.0)
  void setPressure(NumericType pressureAtm) { pressure_ = pressureAtm; }

  // Crystal orientation: Si100, Si111, or PolySi (isotropic, uses <100> rates)
  void setOrientation(SiliconOrientation orientation) {
    orientation_ = orientation;
  }

  void setOrientation(const std::string &orientationStr) {
    orientation_ = util::convert<SiliconOrientation>(orientationStr);
  }

  // Set the maximum duration of an internal oxidation step in hours.
  // The model always recomputes diffusion/mechanics/contact on CFL-limited
  // internal substeps. A positive value caps those internal steps; it does not
  // force the interfaces to move by this amount if the CFL limit is smaller.
  // Default (0): no user cap, only the CFL limit and remaining process time.
  void setTimeStep(NumericType dtHr) { timeStep_ = dtHr; }

  // Cap only the FIRST internal substep (hr); 0 = model default (0.004 hr).
  void setInitialTimeStep(NumericType dtHr) { initialTimeStep_ = dtHr; }

  // One-shot persistent regrid: coarsen the whole level-set stack to
  // coarseDelta (um) once the thinnest oxide layer exceeds factor*coarseDelta.
  // 0 disables (default).  See TODO.md "unified design", Mode 1.
  void setRegrid(NumericType coarseDelta, NumericType factor = NumericType(2)) {
    regridDelta_ = coarseDelta;
    regridFactor_ = std::max(factor, NumericType(1));
  }

  // Courant number used for CFL-limited internal stepping (default 0.499).
  // The actual internal step is min(user timeStep cap, CFL step, remaining).
  void setCFLFactor(NumericType factor) {
    cflFactor_ = std::clamp(factor, NumericType(1e-3), NumericType(0.499));
  }

  // Oxide on the silicon at t = 0 (µm; default 2 nm native oxide), created
  // when the domain has no SiO2 layer, not under a mask; 0 = bare silicon.
  // An oxide thinner than three grid cells grows by the 1-D Deal-Grove law
  // until the 2-D solve resolves it (growResolvedOxide).
  void setInitialOxideThickness(NumericType thicknessUm) {
    initialOxideThickness_ = thicknessUm;
  }

  // Gas-transfer coefficient in µm/hr; large values approximate C_s = C*.
  void setTransferCoefficient(NumericType coefficient) {
    transferCoefficient_ = coefficient;
  }

  // Diffusion THROUGH the mask: the mask body is meshed into the oxidant
  // diffusion solve with D_mask = D_SiO2/gamma, so the oxidant enters the
  // film's outer surface, diffuses across it, and continues through the oxide
  // to react at the Si interface.  The film's resistance follows its actual
  // geometry — the level-set mask thickness IS the barrier thickness e_c, and
  // it must be resolved by the grid (>= ~3 cells).  For a uniform flat film
  // this reproduces Deal-Grove A_coated = A + 2*gamma*e_c (Drob et al. 2025).
  //   permeabilityRatio: gamma = D_SiO2/D_mask (dimensionless).
  // 0 (default) = impermeable mask (upstream zero-flux behaviour).
  void setMaskPermeabilityRatio(NumericType permeabilityRatio) {
    maskBarrierPermeabilityRatio_ = permeabilityRatio;
  }

  // Mask as diffusion barrier only: mechanically transparent, carried by the
  // oxide.  Exact for a full-width mask (uniform growth = pure translation).
  void setMaskBarrierOnly(bool enable) { maskBarrierOnly_ = enable; }

  // Override Deal-Grove parabolic rate B (um^2/hr) and linear rate B/A
  // (um/hr), e.g. to match measured kinetics (Drob 2025 p-type substrate).
  // Pass 0 for either to revert to the built-in Arrhenius table.
  void setDealGroveRates(NumericType B, NumericType BoA) {
    rateB_override_ = std::max(B, NumericType(0));
    rateBoA_override_ = std::max(BoA, NumericType(0));
  }

  // Stress-dependent oxidation kinetics on (default) or off.  Off = the
  // stress-free Deal-Grove kinetics (both activation volumes zero); the oxide
  // mechanics, including the stress-dependent viscosity, are unaffected.
  void setStressDependentOxidation(bool enable) {
    stressDependentOxidation_ = enable;
  }

  // Stress-coupling activation volume for interface reaction rate (m^3),
  // acting on the interface normal stress.  Default 2.5e-29 (25 Å³).
  void setReactionActivationVolume(NumericType volume) {
    reactionActivationVolume_ = volume;
  }

  // Stress-dependent oxide viscosity eta = eta0 x / sinh(x),
  // x = tau Vc / 2kT (Sutardja & Oldham 1989): shear activation volume Vc in
  // m^3; 0 = linear viscosity.  Default: the preset's 300 A^3.
  void setShearActivationVolume(NumericType volume) {
    shearActivationVolume_ = std::max(volume, NumericType(0));
  }

  // Stress-coupling activation volume for oxide diffusivity (m^3), acting on
  // the pressure.  Default 4e-29 (40 Å³).
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
  // are activated: the mask joins the oxide mechanics solve as an elastic
  // body with Maxwell creep (or, with setMaskBarrierOnly, is a diffusion
  // barrier only).
  void setMaskMaterial(Material mat) { maskMaterial_ = mat; }

  // Elastic constants and creep of the mask (default: Si3N4 at the process
  // temperature, OxidationPresets::siliconNitrideMask: Senez et al. 1996).
  void setMaskParameters(viennals::OxidationMaskParameters params) {
    maskParams_ = params;
    maskParamsSet_ = true;
  }

  /// GPU for the diffusion solve and the 3-D mechanics solve (large systems).
  ///   GpuMode::Cpu  — always CPU (default)
  ///   GpuMode::Gpu  — always GPU (throws if unavailable or unsuccessful)
  ///   GpuMode::Auto — GPU when available, else CPU
  void setGpuMode(GpuMode mode) { gpuMode_ = mode; }

  void setGpuMode(std::string mode) {
    mode = util::detail::lower(mode);
    if (mode == "cpu")
      gpuMode_ = GpuMode::Cpu;
    else if (mode == "gpu")
      gpuMode_ = GpuMode::Gpu;
    else if (mode == "auto")
      gpuMode_ = GpuMode::Auto;
    else
      throw std::invalid_argument("Oxidation: invalid gpuMode string '" + mode +
                                  "'; must be 'cpu', 'gpu', or 'auto'.");
  }

  void setMechanicsIterations(unsigned iterations) {
    mechanicsIterations_ = std::max(1u, iterations);
  }

  void setMechanicsTolerance(NumericType tol) {
    mechanicsTolerance_ = tol;
    toleranceWarningEmitted_ = false;
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

    const auto idx = findMaterialIndices(*domain);
    const int siIdx = idx.si, sio2Idx = idx.sio2, maskIdx = idx.mask;

    if (!idx.valid()) {
      Logger::getInstance()
          .addWarning("Oxidation: missing Silicon or Oxide layer for surface "
                      "mesh extraction.")
          .print();
      return;
    }

    // Deep copies of the base simulation level sets
    auto siCopy = ls::Domain<NumericType, D>::New(levelSets[siIdx]);
    auto oxSurf = ls::Domain<NumericType, D>::New(levelSets[sio2Idx]);

    if (maskIdx >= 0) {
      auto mskSurf = ls::Domain<NumericType, D>::New(levelSets[maskIdx]);
      ls::BooleanOperation<NumericType, D>(oxSurf, mskSurf,
                                           ls::BooleanOperationEnum::UNION)
          .apply();
      auto surfDomain = SmartPointer<Domain<NumericType, D>>::New();
      surfDomain->insertNextLevelSetAsMaterial(
          siCopy, matMap->getMaterialAtIdx(siIdx), false);
      surfDomain->insertNextLevelSetAsMaterial(
          oxSurf, matMap->getMaterialAtIdx(sio2Idx), false);
      surfDomain->insertNextLevelSetAsMaterial(
          mskSurf, matMap->getMaterialAtIdx(maskIdx), false);
      surfDomain->saveSurfaceMesh(fileName);
    } else {
      // Standard geometry (no mask)
      auto surfDomain = SmartPointer<Domain<NumericType, D>>::New();
      surfDomain->insertNextLevelSetAsMaterial(
          siCopy, matMap->getMaterialAtIdx(siIdx), false);
      surfDomain->insertNextLevelSetAsMaterial(
          oxSurf, matMap->getMaterialAtIdx(sio2Idx), false);
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

    const auto idx = findMaterialIndices(*domain);
    const int siIdx = idx.si, sio2Idx = idx.sio2, maskIdx = idx.mask;

    if (!idx.valid()) {
      Logger::getInstance()
          .addWarning("Oxidation: missing Silicon or Oxide layer for volume "
                      "mesh extraction.")
          .print();
      return;
    }

    auto siCopy = ls::Domain<NumericType, D>::New(levelSets[siIdx]);
    auto oxCopy = ls::Domain<NumericType, D>::New(levelSets[sio2Idx]);

    if (maskIdx >= 0) {
      auto mskCopy = ls::Domain<NumericType, D>::New(levelSets[maskIdx]);
      ls::BooleanOperation<NumericType, D>(oxCopy, siCopy,
                                           ls::BooleanOperationEnum::UNION)
          .apply();
      ls::BooleanOperation<NumericType, D>(mskCopy, oxCopy,
                                           ls::BooleanOperationEnum::UNION)
          .apply();

      auto volDomain = SmartPointer<Domain<NumericType, D>>::New();
      volDomain->insertNextLevelSetAsMaterial(
          siCopy, matMap->getMaterialAtIdx(siIdx), false);
      volDomain->insertNextLevelSetAsMaterial(
          oxCopy, matMap->getMaterialAtIdx(sio2Idx), false);
      volDomain->insertNextLevelSetAsMaterial(
          mskCopy, matMap->getMaterialAtIdx(maskIdx), false);
      volDomain->saveVolumeMesh(baseName);
    } else {
      ls::BooleanOperation<NumericType, D>(oxCopy, siCopy,
                                           ls::BooleanOperationEnum::UNION)
          .apply();
      auto volDomain = SmartPointer<Domain<NumericType, D>>::New();
      volDomain->insertNextLevelSetAsMaterial(
          siCopy, matMap->getMaterialAtIdx(siIdx), false);
      volDomain->insertNextLevelSetAsMaterial(
          oxCopy, matMap->getMaterialAtIdx(sio2Idx), false);
      volDomain->saveVolumeMesh(baseName);
    }

    // Write warm-restart fields (OxVelocity, OxPressure, OxConcentration,
    // OxStressR0-R2) from the original level sets as companion point-cloud
    // files alongside the volume mesh.  Uses the pre-Boolean-op level sets so
    // the pointData arrays are intact; lsToMesh + lsVTKWriter propagate them
    // faithfully, unlike lsWriteVisualizationMesh which strips all point data.
    // Written unconditionally: at step 0 (before any solve) the meshes carry
    // geometry + LSValues only; solver fields appear from the first solve on.
    {
      const auto &oxLS = levelSets[sio2Idx];
      auto oxFieldMesh = ls::SmartPointer<ls::Mesh<NumericType>>::New();
      ls::ToMesh<NumericType, D>(oxLS, oxFieldMesh).apply();
      ls::VTKWriter<NumericType>(oxFieldMesh, baseName + "_oxide_fields.vtp")
          .apply();
    }
    if (maskIdx >= 0) {
      const auto &maskLS = levelSets[maskIdx];
      auto maskFieldMesh = ls::SmartPointer<ls::Mesh<NumericType>>::New();
      ls::ToMesh<NumericType, D>(maskLS, maskFieldMesh).apply();
      ls::VTKWriter<NumericType>(maskFieldMesh, baseName + "_mask_fields.vtp")
          .apply();
    }
  }

private:
  struct MaterialIndices {
    int si = -1, sio2 = -1, mask = -1;
    bool valid() const { return si >= 0 && sio2 >= 0; }
  };

  MaterialIndices
  findMaterialIndices(const Domain<NumericType, D> &domain) const {
    MaterialIndices idx;
    const auto &levelSets = domain.getLevelSets();
    const auto matMap = domain.getMaterialMap();
    if (!matMap)
      return idx;
    for (int i = static_cast<int>(levelSets.size()) - 1; i >= 0; --i) {
      const auto mat = matMap->getMaterialAtIdx(i);
      if (idx.si < 0 && (mat == siliconMaterial_ || mat == Material::BulkSi ||
                         mat == Material::PolySi || mat == Material::aSi))
        idx.si = i;
      if (idx.sio2 < 0 && mat == oxideMaterial_)
        idx.sio2 = i;
      if (idx.mask < 0 && mat == maskMaterial_)
        idx.mask = i;
    }
    return idx;
  }

  bool run(SmartPointer<Domain<NumericType, D>> domainPtr) {
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
    const auto idx = findMaterialIndices(domain);
    const int siIdx = idx.si, sio2Idx = idx.sio2, maskIdx = idx.mask;

    if (siIdx < 0) {
      Logger::getInstance()
          .addWarning("Oxidation: no silicon layer found in domain.")
          .print();
      return false;
    }

    if (temperature_ < NumericType(700) || temperature_ > NumericType(1200))
      Logger::getInstance()
          .addWarning("Oxidation: temperature " + std::to_string(temperature_) +
                      " °C is outside the calibrated Deal-Grove range "
                      "[700, 1200] °C — rate constants may be inaccurate.")
          .print();

    validateTolerances();

    auto reactionInterface = levelSets[siIdx];

    // If no oxide layer exists, start from the native oxide on top of the
    // open Si and grow it in 1-D until the grid resolves it.
    SmartPointer<ls::Domain<NumericType, D>> ambientInterface;
    NumericType startupTime = NumericType(0);
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
        auto sphere =
            SmartPointer<ls::SphereDistribution<viennahrle::CoordType, D>>::New(
                initialOxideThickness_);
        ls::GeometricAdvect<NumericType, D>(ambientInterface, sphere).apply();
      } else {
        Logger::getInstance()
            .addInfo("Oxidation: no SiO₂ layer found; starting from bare "
                     "silicon.")
            .print();
      }
      SmartPointer<ls::Domain<NumericType, D>> mask = nullptr;
      if (maskIdx >= 0) {
        // The mask rests on the silicon: no oxide under it.
        mask = levelSets[maskIdx];
        ls::BooleanOperation<NumericType, D>(
            ambientInterface, mask,
            ls::BooleanOperationEnum::RELATIVE_COMPLEMENT)
            .apply();
        if (maskBarrierPermeabilityRatio_ > NumericType(0))
          Logger::getInstance()
              .addWarning("Oxidation: a permeable mask on bare silicon.  The "
                          "1-D start-up treats it as impermeable, and the 2-D "
                          "solve has no reaction where the film rests on the "
                          "silicon; give the film an oxide to rest on.")
              .print();
      }
      startupTime =
          growResolvedOxide(reactionInterface, ambientInterface, mask);
      domain.insertNextLevelSetAsMaterial(ambientInterface, oxideMaterial_,
                                          false);
    }

    NumericType gridDelta = reactionInterface->getGrid().getGridDelta();

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
    // Conservative cold start: the first coupled solve is the most fragile
    // (no warm start), and each rejected attempt burns the full coupling
    // iteration budget before halving dt.  Start at a step the coupling
    // accepts and let the x2 growth ramp recover full size within a few
    // substeps.  initialTimeStep_ > 0 overrides the default cap.
    const NumericType coldStartCap = initialTimeStep_ > NumericType(0)
                                         ? initialTimeStep_
                                         : NumericType(0.004);
    const NumericType seedStep =
        std::min(cflStep(maxSurfaceVelocity), coldStartCap);
    const NumericType userStepCap =
        timeStep_ > NumericType(0) ? timeStep_
                                   : std::numeric_limits<NumericType>::max();

    ls::OxidationCouplingParameters coupling;
    coupling.maxIterations = couplingIterations_;
    coupling.tolerance = couplingTolerance_;

    if (Logger::hasInfo()) {
      const std::string mode = (maskIdx >= 0) ? "LOCOS" : "standard";
      const std::string oxStr =
          util::toString(oxidant_) + " (" + util::toString(orientation_) + ")";
      const NumericType initDt = std::min(userStepCap, seedStep);
      Logger::getInstance()
          .addInfo("Oxidation: starting " + mode +
                   " simulation"
                   ", T=" +
                   std::to_string(temperature_) +
                   " °C"
                   ", " +
                   oxStr +
                   " oxidation"
                   ", B=" +
                   std::to_string(rates.B) +
                   " µm²/hr"
                   ", B/A=" +
                   std::to_string(rates.BoA) +
                   " µm/hr"
                   ", Δx=" +
                   std::to_string(gridDelta) +
                   " µm"
                   ", total=" +
                   std::to_string(time_) +
                   " hr"
                   ", initial_dt≤" +
                   std::to_string(initDt) + " hr")
          .print();
    }

    const std::string modeLabel = (maskIdx >= 0) ? "LOCOS" : "Oxidation";

    auto locos =
        ls::Oxidation<NumericType, D>::New(reactionInterface, ambientInterface);
    locos->setGpuMode(gpuMode_);
    locos->setOxidationParameters(oxParams);
    locos->setCouplingParameters(coupling);
    if (useSolveBounds_)
      locos->setSolveBounds(solveMinIndex_, solveMaxIndex_);

    if (maskIdx >= 0) {
      auto maskLS = levelSets[maskIdx];
      locos->setMaskInterface(maskLS);
      locos->setMaskParameters(
          maskParamsSet_
              ? maskParams_
              : viennals::OxidationPresets::siliconNitrideMask(temperature_));
      locos->setMaskBarrierOnly(maskBarrierOnly_);
    }

    NumericType time = startupTime;
    unsigned substep = 0;
    NumericType nextStepEstimate = seedStep;
    NumericType lastAcceptedDt = seedStep;
    constexpr NumericType maxStepGrowth = NumericType(2);
    const NumericType timeEps = NumericType(1e-9) * time_;
    while (time_ - time > timeEps) {
      const NumericType growthLimitedStep =
          (substep == 0)
              ? nextStepEstimate
              : std::min(nextStepEstimate, maxStepGrowth * lastAcceptedDt);
      NumericType requestedDt =
          std::min({userStepCap, growthLimitedStep, time_ - time});
      if (requestedDt <= NumericType(0))
        break;

      // One-shot persistent regrid (Mode 1): once the thinnest oxide layer
      // is comfortably resolvable at the coarse target, resample the whole
      // stack in place and continue on the coarse grid.  In-place deepCopy
      // keeps every held SmartPointer valid; the solvers rebuild from the
      // current grids each substep; the index-keyed concentration warm-start
      // cannot survive a grid change and is cleared (one cold-started
      // substep, recovered by the quasi-steady solve).
      if (regridDelta_ > gridDelta) {
        const NumericType minOxide =
            minOxideLayerUm(reactionInterface, ambientInterface, gridDelta);
        if (minOxide < std::numeric_limits<NumericType>::max() &&
            minOxide >= regridFactor_ * regridDelta_) {
          Logger::getInstance()
              .addInfo(
                  modeLabel + ": REGRID event at t=" + std::to_string(time) +
                  " hr — min oxide layer " + std::to_string(minOxide * 1e3) +
                  " nm; resampling " + std::to_string(levelSets.size()) +
                  " level sets from delta=" + std::to_string(gridDelta) +
                  " to " + std::to_string(regridDelta_) + " um")
              .print();
          for (auto &ls : levelSets) {
            ls::Resample<NumericType, D> resampler(ls, regridDelta_);
            resampler.apply();
            if (resampler.getResult())
              ls->deepCopy(resampler.getResult());
          }
          locos->setConcentrationCache({});
          gridDelta = regridDelta_;
          regridDelta_ = NumericType(0); // one-shot: disarm
        }
      }

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
                  NumericType actualDt) const {
    if (!Logger::hasInfo())
      return;

    std::string message = label + " CFL substep " + std::to_string(substep) +
                          ": t=" + std::to_string(elapsed) +
                          " hr, requested_dt=" + std::to_string(requestedDt) +
                          " hr, actual_dt=" + std::to_string(actualDt) + " hr";
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

  // Normal speed of one surface during the 1-D start-up, by position and
  // normal.
  using StartupSpeed = std::function<NumericType(const Vec3D<NumericType> &,
                                                 const Vec3D<NumericType> &)>;
  class StartupVelocity : public viennals::VelocityField<NumericType> {
    StartupSpeed speed_;

  public:
    explicit StartupVelocity(StartupSpeed speed) : speed_(std::move(speed)) {}

    NumericType getScalarVelocity(const Vec3D<NumericType> &coordinate, int,
                                  const Vec3D<NumericType> &normal,
                                  unsigned long) override {
      return speed_(coordinate, normal);
    }
  };

  // 1-D start-up from bare silicon.  Oxide thinner than resolvedOxideCells_
  // has no grid points of its own, and on this grid it carries no lateral
  // diffusion or stress anyway.  Until it is resolved, every point of the Si
  // surface grows by the 1-D Deal-Grove law with its own linear rate k(n)
  // (the per-face rate of OxidationDiffusion), from the native oxide x_n:
  //   x^2 + A(n) x = B (t + tau(n)),   A(n) = B/k(n) + B/h,
  //   tau(n) = (x_n^2 + A(n) x_n) / B,
  // with h the gas transfer coefficient.  The Si surface recedes by 1/beta
  // of the grown oxide and the oxide surface rises by the rest.  A point
  // whose next grid point along its normal lies inside the mask is covered
  // (the Si under the mask, the oxide surface against its bottom) and does
  // not grow: no oxidant passes an impermeable mask.  The stage ends when
  // the slowest face, k = k_base min(1, ratio), reaches the resolved
  // thickness, or with the process.  On a flat surface this is the solution
  // the 2-D solve continues.  Returns the process time spent.
  NumericType
  growResolvedOxide(SmartPointer<viennals::Domain<NumericType, D>> reaction,
                    SmartPointer<viennals::Domain<NumericType, D>> ambient,
                    SmartPointer<viennals::Domain<NumericType, D>> mask) const {
    namespace ls = viennals;
    const auto rates = computeDealGroveRates();
    const auto params = computeOxParams();
    const NumericType gridDelta = reaction->getGrid().getGridDelta();
    const NumericType native = std::max(initialOxideThickness_, NumericType(0));
    if (rates.B <= NumericType(0) || rates.BoA <= NumericType(0) ||
        time_ <= NumericType(0) ||
        native >= (resolvedOxideCells_ - NumericType(1e-6)) * gridDelta)
      return NumericType(0);

    const NumericType B = rates.B;
    const NumericType ratio = params.reactionRateRatio111;
    auto dealGroveA = [&](NumericType k) {
      return B / k + B / params.transferCoefficient;
    };
    auto tau = [&](NumericType A, NumericType x) {
      return (x * x + A * x) / B;
    };
    const NumericType resolved = resolvedOxideCells_ * gridDelta;
    const NumericType aSlowest =
        dealGroveA(rates.BoA * std::min(NumericType(1), ratio));
    const NumericType startup =
        std::min(time_, tau(aSlowest, resolved) - tau(aSlowest, native));

    auto grown = [&, startup](const Vec3D<NumericType> &normal) {
      NumericType c = NumericType(0);
      for (unsigned d = 0; d < D; ++d)
        c += normal[d] * NumericType(params.crystalAxis[d]);
      const NumericType A =
          dealGroveA(rates.BoA *
                     (NumericType(1) + (ratio - 1) * (NumericType(1) - c * c)));
      return (std::sqrt(A * A + 4 * B * (startup + tau(A, native))) - A) / 2 -
             native;
    };
    // Read-only lookups with an iterator of their own: Advect evaluates the
    // velocity in parallel.
    auto covered = [&](const Vec3D<NumericType> &x,
                       const Vec3D<NumericType> &n) {
      if (mask == nullptr)
        return false;
      viennahrle::Index<D> next;
      for (unsigned d = 0; d < D; ++d)
        next[d] = static_cast<viennahrle::IndexType>(
            std::llround(x[d] / gridDelta + n[d]));
      viennahrle::ConstSparseIterator<
          typename ls::Domain<NumericType, D>::DomainType>
          it(mask->getDomain());
      it.goToIndices(next);
      return it.getValue() < NumericType(0);
    };
    const NumericType beta = params.expansionCoefficient;
    auto advect = [](SmartPointer<ls::Domain<NumericType, D>> levelSet,
                     StartupSpeed distance) {
      ls::Advect<NumericType, D> advection;
      advection.insertNextLevelSet(levelSet);
      advection.setVelocityField(
          SmartPointer<StartupVelocity>::New(std::move(distance)));
      advection.setAdvectionTime(1.);
      advection.apply();
    };
    advect(
        ambient, [&](const Vec3D<NumericType> &x, const Vec3D<NumericType> &n) {
          return covered(x, n) ? NumericType(0) : grown(n) * (beta - 1) / beta;
        });
    advect(reaction,
           [&](const Vec3D<NumericType> &x, const Vec3D<NumericType> &n) {
             return covered(x, n) ? NumericType(0) : -grown(n) / beta;
           });
    if (mask != nullptr)
      ls::BooleanOperation<NumericType, D>(
          ambient, mask, ls::BooleanOperationEnum::RELATIVE_COMPLEMENT)
          .apply();

    const NumericType reached =
        (std::sqrt(aSlowest * aSlowest +
                   4 * B * (startup + tau(aSlowest, native))) -
         aSlowest) /
        2;
    Logger::getInstance()
        .addInfo("Oxidation: 1-D start-up from " +
                 std::to_string(native * 1e3) + " to " +
                 std::to_string(reached * 1e3) +
                 " nm on the slowest face (resolved: three cells, " +
                 std::to_string(resolved * 1e3) + " nm) in " +
                 std::to_string(startup) + " hr.")
        .print();
    return startup;
  }

  // Map process conditions → OxidationParameters.
  // With C* = N = 1 (normalised): B = 2D, so diffusionCoefficient = B/2.
  viennals::OxidationParameters computeOxParams() const {
    const NumericType T_K = temperature_ + NumericType(273.15);
    const auto rates = computeDealGroveRates();

    viennals::OxidationParameters p;
    p.diffusionCoefficient = rates.B / 2.; // B = 2D
    p.reactionRate = rates.BoA;
    p.transferCoefficient = transferCoefficient_;
    p.equilibriumConcentration = 1.;
    // Meshed mask diffusion: the mask body joins the diffusion domain with
    // D_mask = D/gamma.  Oxidant enters the film's outer surface at the
    // ambient Robin BC, diffuses through it, crosses into the oxide, and
    // reacts at the Si interface.  The film's resistance comes from its
    // actual level-set geometry, so the LS mask thickness IS the physical
    // barrier thickness e_c.
    if (maskBarrierPermeabilityRatio_ > NumericType(0))
      p.maskPermeabilityRatio = maskBarrierPermeabilityRatio_;
    p.oxidantMoleculeDensity = 1.;
    p.expansionCoefficient = 2.27; // V_SiO2 / V_Si
    p.velocitySign = -1.;          // reaction interface moves inward
    p.temperature = T_K;
    p.reactionActivationVolume =
        stressDependentOxidation_ ? reactionActivationVolume_ : NumericType(0);
    p.diffusionActivationVolume =
        stressDependentOxidation_ ? diffusionActivationVolume_ : NumericType(0);
    // Continuous crystal-axis correction applied per-face by
    // OxidationDiffusion:
    //   k(n) = k_base × [1 + (ratio − 1) × (1 − (n · axis)²)]
    // k_base is the B/A for the chosen bulk orientation; ratio is the B/A of
    // faces perpendicular to the wafer normal divided by k_base. crystalAxis
    // points along the wafer normal, the last axis (y in 2-D, z in 3-D).
    // Ratios derived from the (100):(110):(111) = 1 : 1.45 : 1.68 ladder.
    p.crystalAxis = {0., 0., 0.};
    p.crystalAxis[D - 1] = 1.;
    switch (orientation_) {
    case SiliconOrientation::Si100:
      // Perpendicular faces are (110)-like: 1.45× faster than (100).
      p.reactionRateRatio111 = 1.45;
      break;
    case SiliconOrientation::Si110:
      // Perpendicular faces are (100)-like: 1/1.45 ≈ 0.690× of (110).
      p.reactionRateRatio111 = 1. / 1.45;
      break;
    case SiliconOrientation::Si111:
      // Perpendicular faces are (100)-like: 1/1.68 ≈ 0.595× of (111).
      p.reactionRateRatio111 = 1. / 1.68;
      break;
    default: // PolySi — isotropic
      p.reactionRateRatio111 = 1.;
      break;
    }
    p.maxGridPoints = maxGridPoints_;
    p.maxIterations = 10000;
    p.tolerance = 1e-7;
    return p;
  }

  // Oxide mechanics: Arrhenius low-stress viscosity per oxidant; the shear
  // modulus is treated as temperature-independent.
  viennals::OxidationDeformationParameters
  computeDefParams(NumericType dt) const {
    const NumericType T_K = temperature_ + NumericType(273.15);
    // η(T) = η_ref × exp(Ea/kB × (1/T − 1/T_ref)), T_ref = 1000 °C.
    // Calibrated 2026-10-22 on the Kao et al. cylinder data (IEEE TED 34,
    // 1008 (1987): wet Fig. 7/8 800-1200 °C, dry Fig. 10 900-1200 °C) with
    // this solver's stress model (normal-stress reaction law V_k = 25 Å³,
    // diffusivity V_D = 40 Å³, Eyring V_c = 300 Å³), 1-D cylinder analysis,
    // rms 0.065 in t_curved/t_flat over 81 points:
    //   wet: η_ref = 3.0e9 Pa·hr, Ea = 3.46 eV
    //   dry: η_ref = 6.5e11 Pa·hr, Ea = 4.56 eV
    // Wet agrees within ×2.5 with Senez et al., IEEE TED 43, 720 (1996),
    // Table II (calibrated independently on Kao, 800-1100 °C).  The previous
    // single law (Irene 1978: 1e10 Pa·hr, 1.5 eV) was too flat in T: it
    // predicted strong retardation at 1200 °C wet, where Kao measured none.
    const bool wet = oxidant_ == OxidantType::WET;
    const NumericType etaRef = wet ? NumericType(3.0e9) : NumericType(6.5e11);
    const NumericType etaEa = wet ? NumericType(3.46) : NumericType(4.56); // eV
    static constexpr NumericType etaTref = NumericType(1273.15);           // K
    const NumericType viscosity =
        etaRef * std::exp(etaEa / kB_ *
                          (NumericType(1) / T_K - NumericType(1) / etaTref));

    // Pull temperature-independent elastic constants from the canonical preset
    // so there is a single source of truth for the SiO2 moduli.
    static const auto basePreset =
        viennals::OxidationPresets::oxideMechanics1000C(1.0);
    viennals::OxidationDeformationParameters p;
    p.viscosity = viscosity;
    p.shearModulus = basePreset.shearModulus;
    p.shearActivationVolume = (shearActivationVolume_ < NumericType(0))
                                  ? basePreset.shearActivationVolume
                                  : shearActivationVolume_;
    p.stressTimeStep = dt;
    p.mechanicsIterations = mechanicsIterations_;
    p.mechanicsTolerance = mechanicsTolerance_;
    p.maxGridPoints = maxGridPoints_;
    return p;
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

  /// Minimum oxide-layer thickness (um): per-column difference between the
  /// ambient (oxide top) and reaction (Si) interface heights along the
  /// growth axis.  Column heights come from each level set's near-surface
  /// defined points (y_surf = (i_y - phi) * delta), so the measure is not
  /// limited by narrow-band width.  Returns max() when no column has both
  /// interfaces measurable — callers must treat that as "unknown", never as
  /// "thick".
  NumericType
  minOxideLayerUm(SmartPointer<viennals::Domain<NumericType, D>> reaction,
                  SmartPointer<viennals::Domain<NumericType, D>> ambient,
                  NumericType delta) const {
    using It = viennahrle::ConstSparseIterator<
        typename viennals::Domain<NumericType, D>::DomainType>;
    auto columnKey = [](const viennahrle::Index<D> &idx) -> long long {
      if constexpr (D == 2)
        return static_cast<long long>(idx[0]);
      else
        return (static_cast<long long>(idx[0]) << 32) ^
               static_cast<long long>(static_cast<unsigned int>(idx[1]));
    };
    // column -> (best |phi|, surface height) from the tightest defined point
    auto columnHeights = [&](SmartPointer<viennals::Domain<NumericType, D>>
                                 ls) {
      std::unordered_map<long long, std::pair<NumericType, NumericType>> cols;
      for (It it(ls->getDomain()); !it.isFinished(); ++it) {
        if (!it.isDefined())
          continue;
        const NumericType v = it.getValue();
        if (std::abs(v) > NumericType(0.9))
          continue;
        const auto idx = it.getStartIndices();
        const NumericType ySurf =
            (NumericType(idx[D - 1]) - v) * delta; // phi positive above
        auto [entry, inserted] =
            cols.try_emplace(columnKey(idx), std::abs(v), ySurf);
        if (!inserted && std::abs(v) < entry->second.first)
          entry->second = {std::abs(v), ySurf};
      }
      return cols;
    };
    const auto reacCols = columnHeights(reaction);
    const auto ambCols = columnHeights(ambient);
    NumericType minTh = std::numeric_limits<NumericType>::max();
    for (const auto &[key, rc] : reacCols) {
      const auto found = ambCols.find(key);
      if (found == ambCols.end())
        continue;
      const NumericType th = found->second.second - rc.second;
      if (th > NumericType(0))
        minTh = std::min(minTh, th);
    }
    return minTh;
  }

  DealGroveRates computeDealGroveRates() const {
    const NumericType T_K = temperature_ + NumericType(273.15);
    const auto row = dealGroveRow();
    const NumericType pressure = std::max(pressure_, NumericType(0));
    const NumericType B =
        (rateB_override_ > NumericType(0))
            ? rateB_override_
            : pressure * row.B0 * std::exp(-row.EB / (kB_ * T_K));
    const NumericType BoA =
        (rateBoA_override_ > NumericType(0))
            ? rateBoA_override_
            : pressure * row.BoA0 * std::exp(-row.EBoA / (kB_ * T_K));
    return {B, BoA};
  }

  DealGroveRow dealGroveRow() const {
    if (oxidant_ == OxidantType::DRY) {
      if (temperature_ < NumericType(950)) {
        // Low-T regime (<950 °C): higher activation energy 2.30 eV.
        if (orientation_ == SiliconOrientation::Si111)
          return {NumericType(772), NumericType(1.23), NumericType(5.82e7),
                  NumericType(2.30)};
        if (orientation_ == SiliconOrientation::Si110)
          return {NumericType(772), NumericType(1.23), NumericType(5.02e7),
                  NumericType(2.30)}; // 3.46e7 × 1.45
        return {NumericType(772), NumericType(1.23), NumericType(3.46e7),
                NumericType(2.30)}; // Si100 / PolySi
      }
      // High-T regime (≥950 °C): standard activation energy 2.00 eV.
      if (orientation_ == SiliconOrientation::Si111)
        return {NumericType(772), NumericType(1.23), NumericType(6.23e6),
                NumericType(2.00)};
      if (orientation_ == SiliconOrientation::Si110)
        return {NumericType(772), NumericType(1.23), NumericType(5.38e6),
                NumericType(2.00)}; // 3.71e6 × 1.45
      return {NumericType(772), NumericType(1.23), NumericType(3.71e6),
              NumericType(2.00)}; // Si100 / PolySi
    }
    // Wet: single-regime for all orientations.
    if (orientation_ == SiliconOrientation::Si111)
      return {NumericType(386), NumericType(0.78), NumericType(1.63e8),
              NumericType(2.05)};
    if (orientation_ == SiliconOrientation::Si110)
      return {NumericType(386), NumericType(0.78), NumericType(1.41e8),
              NumericType(2.05)}; // 9.70e7 × 1.45
    return {NumericType(386), NumericType(0.78), NumericType(9.70e7),
            NumericType(2.05)}; // Si100 / PolySi
  }
};

} // namespace viennaps
