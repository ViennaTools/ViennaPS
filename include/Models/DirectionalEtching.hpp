#pragma once

#include <psMaterials.hpp>
#include <psProcessModel.hpp>
#include <psSurfaceModel.hpp>
#include <psVelocityField.hpp>

namespace DirectionalEtchingImplementation {
template <class NumericType, int D>
class DirectionalEtchVelocityField : public psVelocityField<NumericType> {
  const std::array<NumericType, 3> direction;
  const NumericType dirVel;
  const NumericType isoVel;
  const psMaterial maskMaterial;

public:
  DirectionalEtchVelocityField(std::array<NumericType, 3> dir,
                               const NumericType dVel, const NumericType iVel,
                               const psMaterial mask)
      : direction(dir), dirVel(dVel), isoVel(iVel), maskMaterial(mask) {}

  std::array<NumericType, 3>
  getVectorVelocity(const std::array<NumericType, 3> &coordinate, int material,
                    const std::array<NumericType, 3> &normalVector,
                    unsigned long) override {
    if (!psMaterialMap::isMaterial(material, maskMaterial)) {
      std::array<NumericType, 3> dir(direction);
      for (unsigned i = 0; i < D; ++i) {
        if (dir[i] == 0.) {
          dir[i] -= isoVel * (normalVector[i] < 0 ? -1 : 1);
        } else {
          dir[i] *= dirVel;
        }
      }
      return dir;
    } else {
      return {0.};
    }
  }

  // the translation field should be disabled when using a surface model
  // which only depends on an analytic velocity field
  int getTranslationFieldOptions() const override { return 0; }
};
} // namespace DirectionalEtchingImplementation

/// Directional etching with one masking material.
template <typename NumericType, int D>
class DirectionalEtching : public psProcessModel<NumericType, D> {
public:
  explicit DirectionalEtching(const std::array<NumericType, 3> direction,
                              const NumericType directionalVelocity = 1.,
                              const NumericType isotropicVelocity = 0.,
                              const psMaterial mask = psMaterial::Mask) {
    // default surface model
    auto surfModel = psSmartPointer<psSurfaceModel<NumericType>>::New();

    // velocity field
    auto velField = psSmartPointer<
        DirectionalEtchingImplementation::DirectionalEtchVelocityField<
            NumericType, D>>::New(direction, directionalVelocity,
                                  isotropicVelocity, mask);

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("DirectionalEtching");
  }
};