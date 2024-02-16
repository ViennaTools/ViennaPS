#pragma once

#include <psMaterials.hpp>
#include <psProcessModel.hpp>

#include <rayUtil.hpp>

namespace AnisotropicProcessImplementation {

template <class NumericType, int D>
class VelocityField : public psVelocityField<NumericType> {
  std::array<NumericType, 3> Scale(const NumericType pF,
                                   const std::array<NumericType, 3> &pT) {
    return {pF * pT[0], pF * pT[1], pF * pT[2]};
  }

  const std::array<NumericType, 3> direction100;
  const std::array<NumericType, 3> direction010;
  std::array<std::array<NumericType, 3>, 3> directions;
  const NumericType r100;
  const NumericType r110;
  const NumericType r111;
  const NumericType r311;
  const std::vector<std::pair<psMaterial, NumericType>> &materials;

public:
  VelocityField(
      const std::array<NumericType, 3> passedDir100,
      const std::array<NumericType, 3> passedDir010,
      const NumericType passedR100, const NumericType passedR110,
      const NumericType passedR111, const NumericType passedR311,
      const std::vector<std::pair<psMaterial, NumericType>> &passedmaterials)
      : direction100(passedDir100), direction010(passedDir010),
        r100(passedR100), r110(passedR110), r111(passedR111), r311(passedR311),
        materials(passedmaterials) {

    rayInternal::Normalize(direction100);
    rayInternal::Normalize(direction010);

    directions[0] = direction100;
    directions[1] = direction010;

    directions[1] = rayInternal::Diff(
        directions[1],
        Scale(rayInternal::DotProduct(directions[0], directions[1]),
              directions[0]));
    directions[2] = rayInternal::CrossProduct(directions[0], directions[1]);
  }

  NumericType
  getScalarVelocity(const std::array<NumericType, 3> & /*coordinate*/,
                    int material, const std::array<NumericType, 3> &nv,
                    unsigned long /*pointID*/) override {
    for (auto epitaxyMaterial : materials) {
      if (psMaterialMap::isMaterial(material, epitaxyMaterial.first)) {
        if (std::abs(rayInternal::Norm(nv) - 1.) > 1e-4)
          return 0.;

        std::array<NumericType, 3> normalVector;
        normalVector[0] = nv[0];
        normalVector[1] = nv[1];
        if (D == 3) {
          normalVector[2] = nv[2];
        } else {
          normalVector[2] = 0;
        }
        rayInternal::Normalize(normalVector);

        std::array<NumericType, 3> N;
        for (int i = 0; i < 3; i++) {
          N[i] =
              std::fabs(rayInternal::DotProduct(directions[i], normalVector));
        }
        std::sort(N.begin(), N.end(), std::greater<NumericType>());

        NumericType velocity;
        if (rayInternal::DotProduct(
                N, std::array<NumericType, 3>{-1., 1., 2.}) < 0) {
          velocity = (r100 * (N[0] - N[1] - 2 * N[2]) + r110 * (N[1] - N[2]) +
                      3 * r311 * N[2]) /
                     N[0];
        } else {
          velocity = (r111 * ((N[1] - N[0]) * 0.5 + N[2]) +
                      r110 * (N[1] - N[2]) + 1.5 * r311 * (N[0] - N[1])) /
                     N[0];
        }

        return velocity * epitaxyMaterial.second;
      }
    }

    // not an epitaxy material
    return 0.;
  }

  // the translation field should be disabled when using a surface model
  // which only depends on an analytic velocity field
  int getTranslationFieldOptions() const override { return 0; }
};
} // namespace AnisotropicProcessImplementation

// Model for an anisotropic process, like selective epitaxy or wet etching.
template <typename NumericType, int D>
class psAnisotropicProcess : public psProcessModel<NumericType, D> {
public:
  // The constructor expects the materials where epitaxy is allowed including
  // the corresponding rates.
  psAnisotropicProcess(
      const std::vector<std::pair<psMaterial, NumericType>> pMaterials)
      : materials(pMaterials) {
    if constexpr (D == 2) {
      direction100 = {0., 1., 0.};
      direction010 = {1., 0., -1.};
    } else {
      direction100 = {0.707106781187, 0.707106781187, 0};
      direction010 = {-0.707106781187, 0.707106781187, 0.};
    }
    initialize();
  }

  psAnisotropicProcess(
      const std::array<NumericType, 3> passedDir100,
      const std::array<NumericType, 3> passedDir010,
      const NumericType passedR100, const NumericType passedR110,
      const NumericType passedR111, const NumericType passedR311,
      const std::vector<std::pair<psMaterial, NumericType>> pMaterials)
      : direction100(passedDir100), direction010(passedDir010),
        r100(passedR100), r110(passedR110), r111(passedR111), r311(passedR311),
        materials(pMaterials) {
    initialize();
  }

private:
  void initialize() {
    // default surface model
    auto surfModel = psSmartPointer<psSurfaceModel<NumericType>>::New();

    // velocity field
    auto velField =
        psSmartPointer<AnisotropicProcessImplementation::VelocityField<
            NumericType, D>>::New(direction100, direction010, r100, r110, r111,
                                  r311, materials);

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("AnisotropicProcess");
  }

  // crystal surface direction
  std::array<NumericType, 3> direction100;
  std::array<NumericType, 3> direction010;

  // rates for crystal directions in um / s
  NumericType r100 = 0.0166666666667;
  NumericType r110 = 0.0309166666667;
  NumericType r111 = 0.000121666666667;
  NumericType r311 = 0.0300166666667;

  std::vector<std::pair<psMaterial, NumericType>> materials;
};