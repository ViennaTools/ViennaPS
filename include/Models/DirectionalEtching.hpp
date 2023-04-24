#pragma once

#include <psProcessModel.hpp>
#include <psSurfaceModel.hpp>
#include <psVelocityField.hpp>

// Directional etch for one material
template <class NumericType, int D>
class DirectionalEtchVelocityField : public psVelocityField<NumericType> {
  const std::array<NumericType, 3> direction;
  const NumericType dirVel = 1.;
  const NumericType isoVel = 0.;
  const int maskId;

public:
  DirectionalEtchVelocityField(std::array<NumericType, 3> dir,
                               const NumericType dVel, const NumericType iVel,
                               const int mask = 0)
      : direction(dir), dirVel(dVel), isoVel(iVel), maskId(mask) {}

  std::array<NumericType, 3>
  getVectorVelocity(const std::array<NumericType, 3> &coordinate, int material,
                    const std::array<NumericType, 3> &normalVector,
                    unsigned long) override {
    if (material != maskId) {
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
      return {0};
    }
  }

  // this option should be disabled (return false) when using a surface model
  // which only depends on an analytic velocity field
  bool useTranslationField() const override { return false; }
};

template <typename NumericType, int D>
class DirectionalEtchingSurfaceModel : public psSurfaceModel<NumericType> {
public:
  psSmartPointer<std::vector<NumericType>> calculateVelocities(
      psSmartPointer<psPointData<NumericType>> Rates,
      const std::vector<std::array<NumericType, 3>> &coordinates,
      const std::vector<NumericType> &materialIds) override {
    return nullptr;
  }
};

template <typename NumericType, int D>
class DirectionalEtching : public psProcessModel<NumericType, D> {
public:
  DirectionalEtching(const std::array<NumericType, 3> direction,
                     const NumericType directionalVelocity = 1.,
                     const NumericType isotropicVelocity = 0.,
                     const int maskId = 0) {
    // surface model
    auto surfModel =
        psSmartPointer<DirectionalEtchingSurfaceModel<NumericType, D>>::New();

    // velocity field
    auto velField =
        psSmartPointer<DirectionalEtchVelocityField<NumericType, D>>::New(
            direction, directionalVelocity, isotropicVelocity, maskId);

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("DirectionalEtching");
  }
};