#pragma once

// psAnneal — ViennaPS process model for analytical dopant annealing.
//
// Wraps viennacs::Anneal (temperature schedule + diffusion + optional defect
// coupling / TED / solid activation) as a ProcessModelBase so it can be driven
// by psProcess with processDuration = 0:
//
//   auto model = SmartPointer<Anneal<T, D>>::New();
//   model->setTemperature(1050.);                 // K
//   model->setArrheniusParameters(D0, Ea_eV);
//   model->setDuration(60.);                       // seconds
//   model->setDiffusionMaterials({Material::Si});
//   model->setBlockingMaterials({Material::Air});
//   psProcess<T, D>(domain, model, /*duration=*/0.).apply();
//
// The process evolves scalar fields (dopant concentration, optional active 
// concentration, optional I/V defect fields) inside the domain's cell set.
//
// Prerequisites: the domain must have a cell set with a dopant concentration field
// initialised before apply().

#include "../materials/psMaterial.hpp"
#include "../process/psAdvectionCallback.hpp"
#include "../process/psProcessModel.hpp"

#include <csAnneal.hpp>

#include <string>
#include <vector>

namespace viennaps {

using namespace viennacore;

// Re-export the solver mode enum and diagnostic struct.
using AnnealMode = viennacs::AnnealMode;

template <class NumericType, int D>
class Anneal : public ProcessModelBase<NumericType, D> {

  // -----------------------------------------------------------------------
  // Adapter callback: runs the configured ViennaCS anneal solver once the
  // process domain exposes its cell set.
  // -----------------------------------------------------------------------
  class AnnealCallback : public AdvectionCallback<NumericType, D> {
    viennacs::Anneal<NumericType, D> anneal_;

  public:
    viennacs::Anneal<NumericType, D> &anneal() { return anneal_; }

    bool applyPreAdvect(NumericType /*processTime*/) override {
      auto &cs = this->domain->getCellSet();
      if (!cs) {
        Logger::getInstance()
            .addWarning("Anneal: domain has no cell set — call "
                        "domain->generateCellSet() before applying.")
            .print();
        return false;
      }
      anneal_.setCellSet(cs);
      anneal_.apply();
      return true;
    }
  };

  SmartPointer<AnnealCallback> callback_;

  // Convert ViennaPS Material objects to the integer IDs used by cell-set
  // material fields.
  static std::vector<int> toIntIds(const std::vector<Material> &materials) {
    std::vector<int> ids;
    ids.reserve(materials.size());
    for (const auto &m : materials)
      ids.push_back(static_cast<int>(m));
    return ids;
  }

public:
  // Expose the diagnostic row type so callers don't need to include csAnneal.
  using DefectDiagnosticsRow =
      typename viennacs::Anneal<NumericType, D>::DefectDiagnosticsRow;

  Anneal() {
    this->setProcessName("Anneal");
    callback_ = SmartPointer<AnnealCallback>::New();
    this->setAdvectionCallback(callback_);
  }

  // ── Temperature ──────────────────────────────────────────────────────────

  // Isothermal temperature (K); used together with setDuration().
  void setTemperature(NumericType temperatureK) {
    callback_->anneal().setTemperature(temperatureK);
  }

  // Total anneal time (seconds); used with setTemperature() for isothermal.
  void setDuration(NumericType duration) {
    callback_->anneal().setDuration(duration);
  }

  // ── Temperature schedule (replaces setTemperature + setDuration) ─────────

  void clearTemperatureSchedule() {
    callback_->anneal().clearTemperatureSchedule();
  }

  void addIsothermalStep(NumericType duration, NumericType temperatureK) {
    callback_->anneal().addIsothermalStep(duration, temperatureK);
  }

  void addRampStep(NumericType duration, NumericType startT, NumericType endT) {
    callback_->anneal().addRampStep(duration, startT, endT);
  }

  // Convenience: N durations with N isothermal temperatures, or N+1 ramp
  // endpoint temperatures.
  void setTemperatureSchedule(const std::vector<NumericType> &durations,
                               const std::vector<NumericType> &temperatures) {
    callback_->anneal().setTemperatureSchedule(durations, temperatures);
  }

  // ── Time stepping ────────────────────────────────────────────────────────

  // Explicit time step (seconds). If ≤ 0, auto-computed for stability.
  void setTimeStep(NumericType dt) { callback_->anneal().setTimeStep(dt); }

  // Scale factor on the CFL-stable dt (clamped to [1e-6, 1]; default 0.45).
  void setStabilityFactor(NumericType factor) {
    callback_->anneal().setStabilityFactor(factor);
  }

  // ── Diffusivity ──────────────────────────────────────────────────────────

  // Constant diffusion coefficient (length_unit²/s). Disables Arrhenius.
  void setDiffusionCoefficient(NumericType diffCoeff) {
    callback_->anneal().setDiffusionCoefficient(diffCoeff);
  }

  // Arrhenius diffusivity D(T) = D0 · exp(-Ea / (kB · T)).
  void setArrheniusParameters(NumericType D0, NumericType Ea_eV) {
    callback_->anneal().setArrheniusParameters(D0, Ea_eV);
  }

  // ── Solver mode ──────────────────────────────────────────────────────────

  void setMode(AnnealMode mode) { callback_->anneal().setMode(mode); }

  // GaussSeidel-mode iteration options.
  void setImplicitSolverOptions(int maxIterations, NumericType relativeTolerance,
                                NumericType relaxation = NumericType(1)) {
    callback_->anneal().setImplicitSolverOptions(maxIterations,
                                                  relativeTolerance,
                                                  relaxation);
  }

  // ── Material roles ───────────────────────────────────────────────────────

  // Materials in which the dopant is allowed to diffuse (empty = all).
  void setDiffusionMaterials(const std::vector<Material> &materials) {
    callback_->anneal().setDiffusionMaterials(toIntIds(materials));
  }

  // Materials that completely block diffusion (hard walls).
  void setBlockingMaterials(const std::vector<Material> &materials) {
    callback_->anneal().setBlockingMaterials(toIntIds(materials));
  }

  // ── Field names ──────────────────────────────────────────────────────────

  // Cell-set field name for the dopant concentration.
  void setSpeciesLabel(const std::string &label) {
    callback_->anneal().setSpeciesLabel(label);
  }

  // Cell-set field name for the material IDs (default "Material").
  void setMaterialLabel(const std::string &label) {
    callback_->anneal().setMaterialLabel(label);
  }

  // Cell-set field name for the active (electrically active) concentration.
  void setActiveLabel(const std::string &label) {
    callback_->anneal().setActiveLabel(label);
  }

  // ── Solid activation / solid solubility ──────────────────────────────────

  void enableSolidActivation(bool enable = true) {
    callback_->anneal().enableSolidActivation(enable);
  }

  // Arrhenius solid solubility: C_SS(T) = C0 · exp(-Ea/(kB·T)).
  void setSolidSolubilityArrhenius(NumericType C0, NumericType Ea_eV) {
    callback_->anneal().setSolidSolubilityArrhenius(C0, Ea_eV);
  }

  // ── Defect coupling (interstitials / vacancies) ──────────────────────────

  void enableDefectCoupling(bool enable = true) {
    callback_->anneal().enableDefectCoupling(enable);
  }

  void resetDefectInitialization() {
    callback_->anneal().resetDefectInitialization();
  }

  void setDamageLabels(const std::string &damageLabel,
                       const std::string &lastDamageLabel) {
    callback_->anneal().setDamageLabels(damageLabel, lastDamageLabel);
  }

  void setDefectLabels(const std::string &interstitialLabel,
                        const std::string &vacancyLabel) {
    callback_->anneal().setDefectLabels(interstitialLabel, vacancyLabel);
  }

  void setDefectSourceWeights(NumericType historyWeight,
                               NumericType lastImpWeight) {
    callback_->anneal().setDefectSourceWeights(historyWeight, lastImpWeight);
  }

  void setDefectPartition(NumericType interstitialFraction,
                           NumericType vacancyFraction) {
    callback_->anneal().setDefectPartition(interstitialFraction,
                                            vacancyFraction);
  }

  void setDefectPartitionFromDamageFactors(NumericType interstitialFactor,
                                           NumericType vacancyFactor) {
    callback_->anneal().setDefectPartitionFromDamageFactors(interstitialFactor,
                                                            vacancyFactor);
  }

  void setDefectDiffusivities(NumericType Di, NumericType Dv) {
    callback_->anneal().setDefectDiffusivities(Di, Dv);
  }

  void setDefectReactionRates(NumericType kRecombination,
                               NumericType kInterstitialSink,
                               NumericType kVacancySink) {
    callback_->anneal().setDefectReactionRates(kRecombination,
                                                kInterstitialSink,
                                                kVacancySink);
  }

  void enableDefectEquilibrium(bool enable = true) {
    callback_->anneal().enableDefectEquilibrium(enable);
  }

  void setDefectEquilibrium(NumericType Ieq, NumericType Veq) {
    callback_->anneal().setDefectEquilibrium(Ieq, Veq);
  }

  void setDefectEquilibriumArrhenius(NumericType interstitialC0,
                                      NumericType interstitialEa_eV,
                                      NumericType vacancyC0,
                                      NumericType vacancyEa_eV) {
    callback_->anneal().setDefectEquilibriumArrhenius(
        interstitialC0, interstitialEa_eV, vacancyC0, vacancyEa_eV);
  }

  void clearDefectEquilibriumArrhenius() {
    callback_->anneal().clearDefectEquilibriumArrhenius();
  }

  // TED: dopant diffusivity multiplier driven by (I - V) supersaturation.
  void setDefectEnhancedDiffusion(NumericType tedCoefficient,
                                   NumericType normalization) {
    callback_->anneal().setDefectEnhancedDiffusion(tedCoefficient,
                                                    normalization);
  }

  void setDefectEnhancedDiffusionFromDamageFactor(
      NumericType damageFactor, NumericType coefficientScale = NumericType(0.5),
      NumericType normalization = NumericType(1e20)) {
    callback_->anneal().setDefectEnhancedDiffusionFromDamageFactor(
        damageFactor, coefficientScale, normalization);
  }

  // ── Defect clustering ────────────────────────────────────────────────────

  void enableDefectClustering(bool enable = true) {
    callback_->anneal().enableDefectClustering(enable);
  }

  void setDefectClusterLabel(const std::string &label) {
    callback_->anneal().setDefectClusterLabel(label);
  }

  void setDefectClusterKinetics(NumericType kfi, NumericType kfc,
                                  NumericType kr) {
    callback_->anneal().setDefectClusterKinetics(kfi, kfc, kr);
  }

  void setDefectClusterInitFraction(NumericType fraction) {
    callback_->anneal().setDefectClusterInitFraction(fraction);
  }

  // ── Diagnostics ──────────────────────────────────────────────────────────

  void enableDiagnostics(bool enable = true) {
    callback_->anneal().enableDiagnostics(enable);
  }

  void setDiagnosticsMaterialFilter(int materialId) {
    callback_->anneal().setDiagnosticsMaterialFilter(materialId);
  }

  void clearDefectDiagnostics() {
    callback_->anneal().clearDefectDiagnostics();
  }

  const std::vector<DefectDiagnosticsRow> &getDefectDiagnostics() const {
    return callback_->anneal().getDefectDiagnostics();
  }

  // ── Clamping / misc ──────────────────────────────────────────────────────

  void setClampNonNegative(bool enable = true) {
    callback_->anneal().setClampNonNegative(enable);
  }
};

} // namespace viennaps
