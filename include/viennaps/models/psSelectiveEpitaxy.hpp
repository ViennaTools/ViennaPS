#pragma once

#include "../materials/psMaterialValueMap.hpp"
#include "../materials/psMaterials.hpp"
#include "../process/psProcessModel.hpp"

namespace viennaps {

using namespace viennacore;

namespace impl {
template <class NumericType, int D>
class EpitaxyVelocityField : public VelocityField<NumericType, D> {

  const double R111 = 0.5;
  const double R100 = 1.;
  static constexpr double low =
      (D > 2) ? 0.5773502691896257 : 0.7071067811865476;
  static constexpr double high = 1.0;
  const double factor;

  const MaterialValueMap<NumericType> &materialRates;

public:
  EpitaxyVelocityField(const MaterialValueMap<NumericType> &materials,
                       NumericType r111, NumericType r100)
      : R111(r111), R100(r100), factor((R100 - R111) / (high - low)),
        materialRates(materials) {}

  NumericType getScalarVelocity(const Vec3D<NumericType> &, int material,
                                const Vec3D<NumericType> &nv,
                                unsigned long) override {

    auto rate = materialRates.get(Material::fromLegacyId(material));
    if (rate > 0) {
      double vel = std::max(std::abs(nv[0]), std::abs(nv[D - 1]));
      vel = (vel - low) * factor + R111;

      if (std::abs(nv[0]) < std::abs(nv[D - 1])) {
        vel *= 2.;
      }

      return -vel * rate;
    }

    // not an epitaxy material
    return 0.;
  }
};
} // namespace impl

// Model for selective epitaxy process.
template <typename NumericType, int D>
class SelectiveEpitaxy : public ProcessModelCPU<NumericType, D> {
public:
  SelectiveEpitaxy(NumericType rate111 = 0.5, NumericType rate100 = 1.)
      : SelectiveEpitaxy(std::vector<std::pair<Material, NumericType>>{},
                         rate111, rate100) {}

  // The constructor expects the materials where epitaxy is allowed including
  // the corresponding rates.
  SelectiveEpitaxy(
      const std::vector<std::pair<Material, NumericType>> pMaterials,
      NumericType rate111 = 0.5, NumericType rate100 = 1.) {
    for (const auto &[material, rate] : pMaterials) {
      materialRates.set(material, rate);
    }

    // default surface model
    auto surfModel = SmartPointer<SurfaceModel<NumericType>>::New();

    // velocity field
    auto velField =
        SmartPointer<impl::EpitaxyVelocityField<NumericType, D>>::New(
            materialRates, rate111, rate100);

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("SelectiveEpitaxy");
  }

  void setMaterialRate(Material material, NumericType rate) {
    materialRates.set(material, rate);
  }

  void initialize(SmartPointer<Domain<NumericType, D>> domain,
                  const NumericType processDuration) final {
    if (firstInit) {
      domainCopy = Domain<NumericType, D>::New(domain);
      const auto numLevelSets = domain->getNumberOfLevelSets();
      if (numLevelSets < 2) {
        VIENNACORE_LOG_ERROR(
            "SelectiveEpitaxy: At least two Level-Sets are required for the "
            "selective epitaxy process.");
      }

      auto topMaterial =
          domain->getMaterialMap()->getMaterialAtIdx(numLevelSets - 1);

      if (!isEpitaxyMaterial(topMaterial)) {
        VIENNACORE_LOG_ERROR(
            "SelectiveEpitaxy: The top material is not an epitaxy material.");
      }

      auto levelSets = domain->getLevelSets();
      auto maskLayer = viennals::Domain<NumericType, D>::New(levelSets.back());
      for (int i = numLevelSets - 2; i >= 0; --i) {
        if (isEpitaxyMaterial(domain->getMaterialMap()->getMaterialAtIdx(i))) {
          auto lsCopy = viennals::Domain<NumericType, D>::New(levelSets[i]);
          if (i > 0) {
            viennals::BooleanOperation(
                lsCopy, levelSets[i - 1],
                viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT)
                .apply();
          }

          viennals::BooleanOperation(
              maskLayer, lsCopy,
              viennals::BooleanOperationEnum::RELATIVE_COMPLEMENT)
              .apply();
        }
      }

      domain->clear();
      domain->insertNextLevelSetAsMaterial(maskLayer, Material::Mask);
      if (Logger::hasDebug())
        domain->saveSurfaceMesh("SelectiveEpitaxyMask");
      domain->insertNextLevelSetAsMaterial(domainCopy->getLevelSets().back(),
                                           topMaterial, false);
      levelSets = domain->getLevelSets();
      viennals::PrepareStencilLocalLaxFriedrichs(levelSets, {false, true});

      firstInit = false;
    }
  }

  void finalize(SmartPointer<Domain<NumericType, D>> domain,
                const NumericType processedDuration) final {
    auto levelSets = domain->getLevelSets();
    viennals::FinalizeStencilLocalLaxFriedrichs(levelSets);
    domain->deepCopy(domainCopy);
    firstInit = true;
  }

private:
  MaterialValueMap<NumericType> materialRates;
  SmartPointer<Domain<NumericType, D>> domainCopy;
  bool firstInit = true;

  bool isEpitaxyMaterial(const Material &material) const {
    return materialRates.get(material) != NumericType(0);
  }
};

} // namespace viennaps
