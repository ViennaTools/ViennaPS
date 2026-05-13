#pragma once

// psIonImplantation — ViennaPS process model for analytical ion implantation.
//
// Wraps viennacs::Implant (beam sweep + dose control) as a ProcessModelBase so
// it can be driven by psProcess with processDuration = 0:
//
//   auto model = SmartPointer<IonImplantation<T, D>>::New();
//   model->setImplantModel(...);
//   model->setDose(1e14);
//   psProcess<T, D>(domain, model, /*duration=*/0.).apply();
//
// The process does not move any level sets; it only stamps the profile into
// the domain's cell set (dopant concentration + optional damage fields).
//
// Prerequisites: the domain must have a cell set initialised before apply().

#include "psImplantProfile.hpp"
#include "../process/psAdvectionCallback.hpp"
#include "../process/psProcessModel.hpp"

#include <csImplant.hpp>

#include <string>
#include <vector>

namespace viennaps {

using namespace viennacore;

// Re-export dose-control mode so users only need one header.
using ImplantDoseControl = viennacs::ImplantDoseControl;

template <class NumericType, int D>
class IonImplantation : public ProcessModelBase<NumericType, D> {

  // -----------------------------------------------------------------------
  // Inner callback: holds the configured csImplant and executes it once the
  // domain's cell set is known.
  // -----------------------------------------------------------------------
  class ImplantCallback : public AdvectionCallback<NumericType, D> {
    viennacs::Implant<NumericType, D> implant_;

  public:
    viennacs::Implant<NumericType, D> &implant() { return implant_; }

    bool applyPreAdvect(NumericType /*processTime*/) override {
      auto &cs = this->domain->getCellSet();
      if (!cs) {
        Logger::getInstance()
            .addWarning("IonImplantation: domain has no cell set — call "
                        "domain->generateCellSet() before applying.")
            .print();
        return false;
      }
      // Treat the cell set's cover material (air/vacuum region) as void so
      // the beam sweep skips it correctly regardless of ViennaPS material IDs.
      const int coverMat = cs->getCoverMaterial();
      if (coverMat >= 0)
        implant_.setVoidMaterial(coverMat);
      implant_.setCellSet(cs);
      implant_.apply();
      return true;
    }
  };

  SmartPointer<ImplantCallback> callback_;

  // Helper: convert ViennaPS Material values to cell-set material IDs.
  static std::vector<int> toIntIds(const std::vector<Material> &materials) {
    std::vector<int> ids;
    ids.reserve(materials.size());
    for (const auto &m : materials)
      ids.push_back(static_cast<int>(m));
    return ids;
  }

public:
  IonImplantation() {
    this->setProcessName("IonImplantation");
    callback_ = SmartPointer<ImplantCallback>::New();
    this->setAdvectionCallback(callback_);
  }

  // Dopant concentration profile (required)
  void setImplantModel(
      SmartPointer<ImplantProfileModel<NumericType, D>> model) {
    callback_->implant().setImplantModel(model);
  }

  // Optional damage profile — if set, a damage field is also written
  void setDamageModel(
      SmartPointer<ImplantProfileModel<NumericType, D>> damageModel) {
    callback_->implant().setDamageModel(damageModel);
  }

  // Implant dose in ions/cm²
  void setDose(NumericType dosePerCm2) {
    callback_->implant().setDose(dosePerCm2);
  }

  // Beam tilt angle in degrees (0 = normal incidence)
  void setTiltAngle(NumericType angleDeg) {
    callback_->implant().setImplantAngle(angleDeg);
  }

  // Length unit expressed in centimetres (default 1e-7 = nanometres)
  void setLengthUnit(NumericType lengthUnitInCm) {
    callback_->implant().setLengthUnitInCm(lengthUnitInCm);
  }

  // Dose control mode: Off, WaferDose (project onto wafer plane), BeamDose
  void setDoseControl(ImplantDoseControl mode) {
    callback_->implant().setDoseControl(mode);
  }

  // Materials that completely block the beam (e.g. hard masks)
  void setMaskMaterials(const std::vector<Material> &materials) {
    callback_->implant().setMaskMaterials(toIntIds(materials));
  }

  // Materials the beam passes through without counting as the implant surface
  // (e.g. thin screen oxides)
  void setScreenMaterials(const std::vector<Material> &materials) {
    callback_->implant().setScreenMaterials(toIntIds(materials));
  }

  // Cell-set field name for the deposited concentration.
  void setConcentrationLabel(const std::string &label) {
    callback_->implant().setConcentrationLabel(label);
  }

  // Cell-set field name for accumulated damage.
  void setDamageLabel(const std::string &label) {
    callback_->implant().setDamageLabel(label);
  }

  // Cell-set field name for damage from the last implant step.
  void setLastDamageLabel(const std::string &label) {
    callback_->implant().setLastDamageLabel(label);
  }

  // Cell-set field name for optional beam-hit diagnostics.
  void setBeamHitsLabel(const std::string &label) {
    callback_->implant().setBeamHitsLabel(label);
  }

  // When true, convert the stored concentration from length-unit⁻³ to cm⁻³
  void setOutputConcentrationInCm3(bool enable = true) {
    callback_->implant().setOutputConcentrationInCm3(enable);
  }

  // Scale factor applied when accumulating damage across multiple implants
  void setDamageFactor(NumericType factor) {
    callback_->implant().setDamageFactor(factor);
  }

  // Write a binary beam-hit map into the cell set (useful for debugging)
  void enableBeamHits(bool enable = true) {
    callback_->implant().enableBeamHits(enable);
  }

  // Override which material IDs are treated as vacuum (default: auto-detected
  // from the cell set's cover material). Rarely needed.
  void setVoidMaterials(const std::vector<Material> &materials) {
    callback_->implant().setVoidMaterials(toIntIds(materials));
  }
};

} // namespace viennaps
