#pragma once

#include <psProcessModel.hpp>
#include <psSurfaceModel.hpp>
#include <psVelocityField.hpp>

#include <rayUtil.hpp>

// Wet etch for one material
template <class NumericType, int D>
class WetEtchingVelocityField : public psVelocityField<NumericType> {
  const std::array<NumericType, 3> direction100 = {0.707106781187,
                                                   0.707106781187, 0.};
  const std::array<NumericType, 3> direction010 = {-0.707106781187,
                                                   0.707106781187, 0.};
  std::array<std::array<NumericType, 3>, 3> directions;
  const NumericType r100 = 0.797 / 60.;
  const NumericType r110 = 1.455 / 60.;
  const NumericType r111 = 0.005 / 60.;
  const NumericType r311 = 1.436 / 60.;
  const int maskId = 0;

public:
  WetEtchingVelocityField(const int mask = 0) : maskId(mask) {
    directions[0] = direction100;
    directions[1] = rayInternal::Diff(
        direction010,
        rayInternal::Scale(rayInternal::DotProduct(direction010, direction100),
                           direction100));
    directions[2] = rayInternal::CrossProduct(direction100, directions[1]);
  }

  WetEtchingVelocityField(const std::array<NumericType, 3> passedDir100,
                          const std::array<NumericType, 3> passedDir010,
                          const NumericType passedR100,
                          const NumericType passedR110,
                          const NumericType passedR111,
                          const NumericType passedR311,
                          const int passedMaskId = 0)
      : direction100(passedDir100), direction010(passedDir010),
        r100(passedR100), r110(passedR110), r111(passedR111), r311(passedR311),
        maskId(passedMaskId) {
    rayInternal::Normalize(direction100);
    rayInternal::Normalize(direction010);

    directions[0] = direction100;
    directions[1] = rayInternal::Diff(
        direction010,
        rayInternal::Scale(rayInternal::DotProduct(direction010, direction100),
                           direction100));
    directions[2] = rayInternal::CrossProduct(direction100, directions[1]);
  }

  NumericType
  getScalarVelocity(const std::array<NumericType, 3> & /*coordinate*/,
                    int material,
                    const std::array<NumericType, 3> &normalVector,
                    unsigned long /*pointID*/) override {
    if (material == maskId)
      return 0.;

    if (std::abs(rayInternal::Norm(normalVector) - 1.) > 1e-4)
      return 0.;

    std::array<NumericType, 3> N;
    for (int i = 0; i < 3; i++) {
      N[i] = std::fabs(rayInternal::DotProduct(directions[i], normalVector));
    }
    std::sort(N.begin(), N.end(), std::greater<NumericType>());

    NumericType velocity;
    if (rayInternal::DotProduct(N, std::array<NumericType, 3>{-1., 1., 2.}) <
        0) {
      velocity = -((r100 * (N[0] - N[1] - 2 * N[2]) + r110 * (N[1] - N[2]) +
                    3 * r311 * N[2]) /
                   N[0]);
    } else {
      velocity = -((r111 * ((N[1] - N[0]) * 0.5 + N[2]) + r110 * (N[1] - N[2]) +
                    1.5 * r311 * (N[0] - N[1])) /
                   N[0]);
    }

    return velocity;
  }

  // the translation field should be disabled when using a surface model
  // which only depends on an analytic velocity field
  int getTranslationFieldOptions() const override { return 0; }
};

template <typename NumericType, int D>
class WetEtchingSurfaceModel : public psSurfaceModel<NumericType> {
public:
  psSmartPointer<std::vector<NumericType>> calculateVelocities(
      psSmartPointer<psPointData<NumericType>> Rates,
      const std::vector<std::array<NumericType, 3>> &coordinates,
      const std::vector<NumericType> &materialIds) override {
    return nullptr;
  }
};

// The wet etching model should be used in combination with the
// STENCIL_LOCAL_LAX_FRIEDRIECH integration scheme.
template <typename NumericType, int D>
class WetEtching : public psProcessModel<NumericType, D> {
public:
  WetEtching(const int passedMaskId = 0) : maskId(passedMaskId) {
    static_assert(D == 3 && "Wet etch model is only implemented in 3D.");
    initialize();
  }
  WetEtching(const std::array<NumericType, 3> passedDir100,
             const std::array<NumericType, 3> passedDir010,
             const NumericType passedR100, const NumericType passedR110,
             const NumericType passedR111, const NumericType passedR311,
             const int passedMaskId = 0)
      : direction100(passedDir100), direction010(passedDir010),
        r100(passedR100), r110(passedR110), r111(passedR111), r311(passedR311),
        maskId(passedMaskId) {
    static_assert(D == 3 && "Wet etch model is only implemented in 3D.");
    initialize();
  }

private:
  void initialize() {
    // surface model
    auto surfModel =
        psSmartPointer<WetEtchingSurfaceModel<NumericType, D>>::New();

    // velocity field
    auto velField =
        psSmartPointer<WetEtchingVelocityField<NumericType, D>>::New(
            direction100, direction010, r100, r110, r111, r311, maskId);

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("WetEtching");
  }

  // crystal surface direction
  const std::array<NumericType, 3> direction100 = {0.707106781187,
                                                   0.707106781187, 0.};
  const std::array<NumericType, 3> direction010 = {-0.707106781187,
                                                   0.707106781187, 0.};
  // etch rates for crystal directions in um / s
  // 30 % KOH at 70°C
  // https://doi.org/10.1016/S0924-4247(97)01658-0
  const NumericType r100 = 0.797 / 60.;
  const NumericType r110 = 1.455 / 60.;
  const NumericType r111 = 0.005 / 60.;
  const NumericType r311 = 1.436 / 60.;

  const int maskId = 0;
  // from the ViennaTS web version example
  // const NumericType r100 = 0.0166666666667;
  // const NumericType r110 = 0.0309166666667;
  // const NumericType r111 = 0.000121666666667;
  // const NumericType r311 = 0.0300166666667;
};