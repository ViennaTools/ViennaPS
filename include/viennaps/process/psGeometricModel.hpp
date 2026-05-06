#pragma once

#include <lsGeometricAdvect.hpp>

#include "../psPreCompileMacros.hpp"

#include "../materials/psMaterial.hpp"

namespace viennaps {

using namespace viennacore;

VIENNAPS_TEMPLATE_ND(NumericType, D) class GeometricModel {
  SmartPointer<viennals::GeometricAdvectDistribution<NumericType, D>> dist =
      nullptr;
  SmartPointer<viennals::Domain<NumericType, D>> mask = nullptr;
  std::vector<Material> maskMaterials;
  bool isDepo = false;
  bool applySingleMaterial = false;

public:
  GeometricModel() = default;

  GeometricModel(
      SmartPointer<viennals::GeometricAdvectDistribution<NumericType, D>>
          passedDist,
      SmartPointer<viennals::Domain<NumericType, D>> passedMask = nullptr)
      : dist(passedDist), mask(passedMask) {}

  void setDistribution(
      SmartPointer<viennals::GeometricAdvectDistribution<NumericType, D>>
          passedDist) {
    dist = passedDist;
  }

  void setMask(SmartPointer<viennals::Domain<NumericType, D>> passedMask) {
    mask = passedMask;
  }

  void addMaskMaterial(const Material material) {
    maskMaterials.push_back(material);
  }

  void setMaskMaterials(const std::vector<Material> &materials) {
    maskMaterials = materials;
  }

  auto &getDistribution() const { return dist; }
  auto &getMask() const { return mask; }
  auto &getMaskMaterials() const { return maskMaterials; }
  auto isDeposition() const { return isDepo; }
  void setDeposition(bool deposition) { isDepo = deposition; }
  auto isSingleMaterial() const { return applySingleMaterial; }
  void setSingleMaterial(bool singleMaterial) {
    applySingleMaterial = singleMaterial;
  }
};

} // namespace viennaps
