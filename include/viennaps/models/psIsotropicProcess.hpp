#pragma once

#include "../materials/psMaterialValueMap.hpp"
#include "../process/psProcessModel.hpp"
#include "../process/psSurfaceModel.hpp"
#include "../process/psVelocityField.hpp"

#include <vcSmartPointer.hpp>

namespace viennaps {

using namespace viennacore;

namespace impl {

template <class NumericType, int D>
class IsotropicVelocityField : public VelocityField<NumericType, D> {
  MaterialValueMap<NumericType> &materialRates_;

public:
  IsotropicVelocityField(MaterialValueMap<NumericType> &materialRates)
      : materialRates_{materialRates} {}

  NumericType getScalarVelocity(const Vec3D<NumericType> &, int material,
                                const Vec3D<NumericType> &,
                                unsigned long) override {
    return materialRates_.get(MaterialMap::mapToMaterial(material));
  }
};
} // namespace impl

/// Isotropic etching with one masking material.
template <typename NumericType, int D>
class IsotropicProcess : public ProcessModelCPU<NumericType, D> {
public:
  IsotropicProcess(NumericType isotropicRate,
                   Material maskMaterial = Material::Undefined) {
    materialRates.setDefault(isotropicRate);
    materialRates.set(maskMaterial, 0.);
    setup();
  }

  IsotropicProcess(NumericType isotropicRate,
                   const std::vector<Material> &maskMaterials) {
    materialRates.setDefault(isotropicRate);
    for (const auto &mat : maskMaterials) {
      materialRates.set(mat, 0.);
    }
    setup();
  }

  IsotropicProcess(std::unordered_map<Material, NumericType> pMaterialRates,
                   NumericType defaultRate = 0.)
      : materialRates(pMaterialRates) {
    materialRates.setDefault(defaultRate);
    setup();
  }

  void setIsotropicRate(NumericType isotropicRate) {
    materialRates.setDefault(isotropicRate);
    addMetaData("Isotropic Rate", isotropicRate);
  }

  void setMaterialRate(Material material, NumericType rate) {
    materialRates.set(material, rate);
    addMetaData(MaterialMap::toString(material) + " Rate", rate);
  }

private:
  void setup() {
    // default surface model
    auto surfModel = SmartPointer<SurfaceModel<NumericType>>::New();

    // store process data (before moving materialRates)
    addMetaData("Isotropic Rate", materialRates.getDefault());
    for (const auto &materialRate : materialRates) {
      addMetaData("Rate " + MaterialMap::toString(materialRate.getMaterial()),
                  materialRate.getValue());
    }

    // velocity field
    auto velField =
        SmartPointer<impl::IsotropicVelocityField<NumericType, D>>::New(
            materialRates);

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("IsotropicProcess");
  }

  using ProcessModelCPU<NumericType, D>::processMetaData;

  inline void addMetaData(const std::string &key, double value) {
    processMetaData[key] = std::vector<double>{value};
  }

  MaterialValueMap<NumericType> materialRates;
};

PS_PRECOMPILE_PRECISION_DIMENSION(IsotropicProcess)

} // namespace viennaps
