#pragma once

#include <psMaterials.hpp>
#include <psProcessModel.hpp>

#include <vcVectorType.hpp>

namespace viennaps {

using namespace viennacore;

namespace impl {
template <class NumericType, int D>
class EpitaxyVelocityField : public VelocityField<NumericType, D> {

  std::vector<double> velocities;
  static constexpr double R111 = 0.5;
  static constexpr double R100 = 1.;
  static constexpr double low =
      (D > 2) ? 0.5773502691896257 : 0.7071067811865476;
  static constexpr double high = 1.0;

  const std::vector<std::pair<Material, NumericType>> &materials;

public:
  EpitaxyVelocityField(
      const std::vector<std::pair<Material, NumericType>> &passedmaterials)
      : materials(passedmaterials) {}

  NumericType getScalarVelocity(const Vec3D<NumericType> & /*coordinate*/,
                                int material, const Vec3D<NumericType> &nv,
                                unsigned long /*pointID*/) override {
    for (auto epitaxyMaterial : materials) {
      if (MaterialMap::isMaterial(material, epitaxyMaterial.first)) {
        double vel = std::max(std::abs(nv[0]), std::abs(nv[2]));
        constexpr double factor = (R100 - R111) / (high - low);
        vel = (vel - low) * factor + R111;

        if (std::abs(nv[0]) < std::abs(nv[2])) {
          vel *= 2.;
        }

        return vel * epitaxyMaterial.second;
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

// Model for selective epitaxy process.
template <typename NumericType, int D>
class SelectiveEpitaxy : public ProcessModel<NumericType, D> {
public:
  // The constructor expects the materials where epitaxy is allowed including
  // the corresponding rates.
  SelectiveEpitaxy(
      const std::vector<std::pair<Material, NumericType>> pMaterials)
      : materials(pMaterials) {
    // default surface model
    auto surfModel = SmartPointer<SurfaceModel<NumericType>>::New();

    // velocity field
    auto velField =
        SmartPointer<impl::EpitaxyVelocityField<NumericType, D>>::New(
            materials);

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("SelectiveEpitaxy");
  }

private:
  std::vector<std::pair<Material, NumericType>> materials;
};

} // namespace viennaps
