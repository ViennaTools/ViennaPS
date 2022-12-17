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

  NumericType getDissipationAlpha(
      int /*direction*/, int /*material*/,
      const std::array<NumericType, 3> & /*centralDifferences*/) {
    return -1;
  }
};

template <typename NumericType, int D>
class DirectionalEtchingSurfaceModel : public psSurfaceModel<NumericType> {
public:
  psSmartPointer<std::vector<NumericType>>
  calculateVelocities(psSmartPointer<psPointData<NumericType>> Rates,
                      const std::vector<NumericType> &materialIds) override {
    return nullptr;
  }
};

template <typename NumericType, int D> class DirectionalEtching {
  psSmartPointer<psProcessModel<NumericType, D>> processModel = nullptr;

public:
  DirectionalEtching(const std::array<NumericType, 3> direction,
                     const NumericType directionalVelocity = 1.,
                     const NumericType isotropicVelocity = 0.) {
    processModel = psSmartPointer<psProcessModel<NumericType, D>>::New();

    // surface model
    auto surfModel =
        psSmartPointer<DirectionalEtchingSurfaceModel<NumericType, D>>::New();

    // velocity field
    auto velField =
        psSmartPointer<DirectionalEtchVelocityField<NumericType, D>>::New(
            direction, directionalVelocity, isotropicVelocity, 0);

    processModel->setSurfaceModel(surfModel);
    processModel->setVelocityField(velField);
    processModel->setProcessName("DirectionalEtching");
  }

  psSmartPointer<psProcessModel<NumericType, D>> getProcessModel() {
    return processModel;
  }
};