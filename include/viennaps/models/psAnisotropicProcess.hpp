#pragma once

#include "../psMaterials.hpp"
#include "../psProcessModel.hpp"

#include <vcVectorType.hpp>

namespace viennaps {

using namespace viennacore;

namespace impl {

template <class NumericType, int D>
class AnisotropicVelocityField : public VelocityField<NumericType, D> {
  static Vec3D<NumericType> ScaleImpl(const NumericType pF,
                                      const Vec3D<NumericType> &pT) {
    return Vec3D<NumericType>{pF * pT[0], pF * pT[1], pF * pT[2]};
  }

  Vec3D<Vec3D<NumericType>> directions;
  const NumericType r100;
  const NumericType r110;
  const NumericType r111;
  const NumericType r311;
  const std::vector<std::pair<Material, NumericType>> &materials;

public:
  AnisotropicVelocityField(
      const Vec3D<NumericType> &direction100,
      const Vec3D<NumericType> &direction010, const NumericType passedR100,
      const NumericType passedR110, const NumericType passedR111,
      const NumericType passedR311,
      const std::vector<std::pair<Material, NumericType>> &passedmaterials)
      : r100(passedR100), r110(passedR110), r111(passedR111), r311(passedR311),
        materials(passedmaterials) {

    directions[0] = Normalize(direction100);
    directions[1] = Normalize(direction010);

    directions[1] =
        directions[1] -
        ScaleImpl(DotProduct(directions[0], directions[1]), directions[0]);
    directions[2] = CrossProduct(directions[0], directions[1]);
  }

  NumericType getScalarVelocity(const Vec3D<NumericType> & /*coordinate*/,
                                int material, const Vec3D<NumericType> &nv,
                                unsigned long /*pointID*/) override {
    for (auto epitaxyMaterial : materials) {
      if (MaterialMap::isMaterial(material, epitaxyMaterial.first)) {
        if (std::abs(Norm(nv) - 1.) > 1e-4)
          return 0.;

        Vec3D<NumericType> normalVector;
        normalVector[0] = nv[0];
        normalVector[1] = nv[1];
        if (D == 3) {
          normalVector[2] = nv[2];
        } else {
          normalVector[2] = 0;
        }
        Normalize(normalVector);

        Vec3D<NumericType> N;
        for (int i = 0; i < 3; i++) {
          N[i] = std::fabs(DotProduct(directions[i], normalVector));
        }
        std::sort(N.begin(), N.end(), std::greater<NumericType>());

        NumericType velocity;
        if (DotProduct(N, Vec3D<NumericType>{-1., 1., 2.}) < 0) {
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
} // namespace impl

// Model for an anisotropic process, like selective epitaxy or wet etching.
template <typename NumericType, int D>
class AnisotropicProcess : public ProcessModel<NumericType, D> {
public:
  // The constructor expects the materials where epitaxy is allowed including
  // the corresponding rates.
  AnisotropicProcess(
      const std::vector<std::pair<Material, NumericType>> pMaterials)
      : materials(pMaterials) {
    if constexpr (D == 2) {
      direction100 = Vec3D<NumericType>{0., 1., 0.};
      direction010 = Vec3D<NumericType>{1., 0., -1.};
    } else {
      direction100 = Vec3D<NumericType>{0.707106781187, 0.707106781187, 0};
      direction010 = Vec3D<NumericType>{-0.707106781187, 0.707106781187, 0.};
    }
    initialize();
  }

  AnisotropicProcess(
      const Vec3D<NumericType> &passedDir100,
      const Vec3D<NumericType> &passedDir010, const NumericType passedR100,
      const NumericType passedR110, const NumericType passedR111,
      const NumericType passedR311,
      const std::vector<std::pair<Material, NumericType>> pMaterials)
      : direction100(passedDir100), direction010(passedDir010),
        r100(passedR100), r110(passedR110), r111(passedR111), r311(passedR311),
        materials(pMaterials) {
    initialize();
  }

private:
  void initialize() {
    // default surface model
    auto surfModel = SmartPointer<SurfaceModel<NumericType>>::New();

    // velocity field
    auto velField =
        SmartPointer<impl::AnisotropicVelocityField<NumericType, D>>::New(
            direction100, direction010, r100, r110, r111, r311, materials);

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("AnisotropicProcess");

    // store process data
    processData["r100"] = {r100};
    processData["r110"] = {r110};
    processData["r111"] = {r111};
    processData["r311"] = {r311};
    processData["Direction100"] = {direction100[0], direction100[1],
                                   direction100[2]};
    processData["Direction010"] = {direction010[0], direction010[1],
                                   direction010[2]};
    for (const auto &material : Rates) {
      processData[MaterialMap::getMaterialName(pair.first) + " Rate"] =
          std::vector<NumericType>{material.second};
    }
  }

  // crystal surface direction
  Vec3D<NumericType> direction100;
  Vec3D<NumericType> direction010;

  // rates for crystal directions in um / s
  NumericType r100 = 0.0166666666667;
  NumericType r110 = 0.0309166666667;
  NumericType r111 = 0.000121666666667;
  NumericType r311 = 0.0300166666667;

  std::vector<std::pair<Material, NumericType>> materials;
  using ProcessModel<NumericType, D>::processData;
};

} // namespace viennaps
