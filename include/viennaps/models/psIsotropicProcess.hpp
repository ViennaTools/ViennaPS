#pragma once

#include "../psMaterials.hpp"
#include "../psProcessModel.hpp"
#include "../psSurfaceModel.hpp"
#include "../psVelocityField.hpp"

#include <vcSmartPointer.hpp>

namespace viennaps {

using namespace viennacore;

namespace impl {

template <class NumericType, int D>
class IsotropicVelocityField : public VelocityField<NumericType, D> {
  const NumericType rate_ = 1.;
  const std::vector<std::pair<int, NumericType>> materialRates_;

public:
  IsotropicVelocityField(NumericType rate,
                         std::vector<std::pair<int, NumericType>> &&matRates)
      : rate_{rate}, materialRates_{std::move(matRates)} {}

  NumericType getScalarVelocity(const Vec3D<NumericType> &, int material,
                                const Vec3D<NumericType> &,
                                unsigned long) override {
    for (const auto &materialRate : materialRates_) {
      if (material == materialRate.first) {
        return materialRate.second;
      }
    }

    return rate_;
  }

  // the translation field should be disabled when using a surface model
  // which only depends on an analytic velocity field
  int getTranslationFieldOptions() const override { return 0; }
};
} // namespace impl

/// Isotropic etching with one masking material.
template <typename NumericType, int D>
class IsotropicProcess : public ProcessModel<NumericType, D> {
public:
  IsotropicProcess(const NumericType isotropicRate,
                   const Material maskMaterial = Material::Undefined) {
    std::vector<std::pair<int, NumericType>> materialRates;
    if (maskMaterial != Material::Undefined) {
      materialRates.emplace_back(static_cast<int>(maskMaterial), 0.);
    }
    setup(isotropicRate, std::move(materialRates));
  }

  IsotropicProcess(const NumericType isotropicRate,
                   const std::vector<Material> &maskMaterials) {
    std::vector<std::pair<int, NumericType>> materialRates;
    for (const auto &mat : maskMaterials) {
      materialRates.emplace_back(static_cast<int>(mat), 0.);
    }
    setup(isotropicRate, std::move(materialRates));
  }

  IsotropicProcess(std::unordered_map<Material, NumericType> materialRates,
                   const NumericType defaultRate = 0.) {
    std::vector<std::pair<int, NumericType>> rates;
    for (const auto &[mat, rate] : materialRates) {
      rates.emplace_back(static_cast<int>(mat), rate);
    }
    setup(defaultRate, std::move(rates));
  }

private:
  void setup(NumericType rate,
             std::vector<std::pair<int, NumericType>> &&materialRates) {
    // default surface model
    auto surfModel = SmartPointer<SurfaceModel<NumericType>>::New();

    // velocity field
    auto velField =
        SmartPointer<impl::IsotropicVelocityField<NumericType, D>>::New(
            rate, std::move(materialRates));

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("IsotropicProcess");

    // store process data
    processMetaData["IsotropicRate"] = std::vector<NumericType>{rate};
    if (!materialRates.empty()) {
      processMetaData["MaterialRates"] = std::vector<NumericType>{};
      for (const auto &materialRate : materialRates) {
        processMetaData["Material"].push_back(
            static_cast<NumericType>(materialRate.first));
        processMetaData["MaterialRates"].push_back(materialRate.second);
      }
    }
  }

  using ProcessModel<NumericType, D>::processMetaData;
};

PS_PRECOMPILE_PRECISION_DIMENSION(IsotropicProcess)

} // namespace viennaps
