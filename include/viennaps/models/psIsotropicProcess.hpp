#pragma once

#include "../psMaterials.hpp"
#include "../psProcessModel.hpp"
#include "../psSurfaceModel.hpp"
#include "../psVelocityField.hpp"

namespace IsotropicImplementation {
template <class NumericType, int D>
class IsotropicVelocityField : public psVelocityField<NumericType> {
  const NumericType vel = 1.;
  const psMaterial maskMaterial;

public:
  IsotropicVelocityField(const NumericType rate,
                         const psMaterial mask = psMaterial::Mask)
      : vel(rate), maskMaterial(mask) {}

  NumericType
  getScalarVelocity(const std::array<NumericType, 3> & /*coordinate*/,
                    int material,
                    const std::array<NumericType, 3> & /* normalVector */,
                    unsigned long /*pointID*/) override {
    if (psMaterialMap::isMaterial(material, maskMaterial)) {
      return 0.;
    } else {
      return vel;
    }
  }

  // the translation field should be disabled when using a surface model
  // which only depends on an analytic velocity field
  int getTranslationFieldOptions() const override { return 0; }
};
} // namespace IsotropicImplementation

/// Isotropic etching with one masking material.
template <typename NumericType, int D>
class psIsotropicProcess : public psProcessModel<NumericType, D> {
public:
  psIsotropicProcess(const NumericType isotropicRate,
                     const psMaterial maskMaterial = psMaterial::None) {
    // default surface model
    auto surfModel = psSmartPointer<psSurfaceModel<NumericType>>::New();

    // velocity field
    auto velField =
        psSmartPointer<IsotropicImplementation::IsotropicVelocityField<
            NumericType, D>>::New(isotropicRate, maskMaterial);

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("IsotropicProcess");
  }
};
