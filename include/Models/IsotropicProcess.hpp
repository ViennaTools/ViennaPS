#pragma once

#include <psProcessModel.hpp>
#include <psSurfaceModel.hpp>
#include <psVelocityField.hpp>

// Isotropic etch for one material
template <class NumericType, int D>
class IsotropicVelocityField : public psVelocityField<NumericType> {
  const NumericType vel = 1.;
  const int maskId;

public:
  IsotropicVelocityField(const NumericType rate, const int mask = 0)
      : vel(rate), maskId(mask) {}

  NumericType
  getScalarVelocity(const std::array<NumericType, 3> & /*coordinate*/,
                    int material,
                    const std::array<NumericType, 3> & /* normalVector */,
                    unsigned long /*pointID*/) override {
    if (material != maskId) {
      return vel;
    } else {
      return 0;
    }
  }

  // this option should be disabled (return false) when using a surface model
  // which only depends on an analytic velocity field
  bool useTranslationField() const override { return false; }
};

template <typename NumericType, int D>
class IsotropicSurfaceModel : public psSurfaceModel<NumericType> {
public:
  psSmartPointer<std::vector<NumericType>> calculateVelocities(
      psSmartPointer<psPointData<NumericType>> Rates,
      const std::vector<std::array<NumericType, 3>> &coordinates,
      const std::vector<NumericType> &materialIds) override {
    return nullptr;
  }
};

template <typename NumericType, int D>
class IsotropicProcess : public psProcessModel<NumericType, D> {
public:
  IsotropicProcess(const NumericType isotropicRate = 0., const int maskId = 0) {
    // surface model
    auto surfModel =
        psSmartPointer<IsotropicSurfaceModel<NumericType, D>>::New();

    // velocity field
    auto velField = psSmartPointer<IsotropicVelocityField<NumericType, D>>::New(
        isotropicRate, maskId);

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("IsotropicProcess");
  }
};