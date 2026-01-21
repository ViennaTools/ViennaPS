#pragma once

#include "../psMaterials.hpp"
#include "psVelocityField.hpp"

#include <lsVelocityField.hpp>

#include <vcKDTree.hpp>
#include <vcLogger.hpp>
#include <vcSmartPointer.hpp>
#include <vcVectorType.hpp>

namespace viennaps {

using namespace viennacore;

VIENNAPS_TEMPLATE_ND(NumericType, D)
class TranslationField final : public viennals::VelocityField<NumericType> {
  using TranslatorType = std::unordered_map<unsigned long, unsigned long>;

public:
  TranslationField(
      SmartPointer<::viennaps::VelocityField<NumericType, D>> velocityField,
      SmartPointer<MaterialMap> const &materialMap, int translationMethod)
      : modelVelocityField_(velocityField), materialMap_(materialMap),
        translationMethod_(translationMethod) {
    if (!materialMap_) {
      VIENNACORE_LOG_ERROR("TranslationField: material map is required.");
    }
  }

  NumericType getScalarVelocity(const Vec3D<NumericType> &coordinate,
                                int material,
                                const Vec3D<NumericType> &normalVector,
                                unsigned long pointId) override {
    translateLsId(pointId, coordinate);
    material = materialMap_->getMaterialIdAtIdx(material);
    assert(material >= 0);
    return modelVelocityField_->getScalarVelocity(coordinate, material,
                                                  normalVector, pointId);
  }

  Vec3D<NumericType> getVectorVelocity(const Vec3D<NumericType> &coordinate,
                                       int material,
                                       const Vec3D<NumericType> &normalVector,
                                       unsigned long pointId) override {
    translateLsId(pointId, coordinate);
    material = materialMap_->getMaterialIdAtIdx(material);
    assert(material >= 0);
    return modelVelocityField_->getVectorVelocity(coordinate, material,
                                                  normalVector, pointId);
  }

  NumericType
  getDissipationAlpha(int direction, int material,
                      const Vec3D<NumericType> &centralDifferences) override {
    material = materialMap_->getMaterialIdAtIdx(material);
    assert(material >= 0);
    return modelVelocityField_->getDissipationAlpha(direction, material,
                                                    centralDifferences);
  }

  void setTranslator(const SmartPointer<TranslatorType> &translator) {
    translator_ = translator;
  }

  void setKdTree(
      const SmartPointer<KDTree<NumericType, Vec3D<NumericType>>> &kdTree) {
    kdTree_ = kdTree;
  }

  auto &getKdTree() { return kdTree_; }

  void buildKdTree(const std::vector<std::array<NumericType, 3>> &points) {
    if (!kdTree_)
      kdTree_ = SmartPointer<KDTree<NumericType, Vec3D<NumericType>>>::New();
    kdTree_->setPoints(points);
    kdTree_->build();
  }

  void translateLsId(unsigned long &lsId,
                     const Vec3D<NumericType> &coordinate) const {
    switch (translationMethod_) {
    case 1: {
      assert(translator_->size() > 0);
      if (auto it = translator_->find(lsId); it != translator_->end()) {
        lsId = it->second;
      } else {
        Logger::getInstance()
            .addError("Could not extend velocity from surface (" +
                          std::to_string(lsId) + ") to LS point",
                      false)
            .print();
      }
      break;
    }
    case 2: {
      auto nearest = kdTree_->findNearest(coordinate);
      lsId = nearest->first;
      break;
    }
    default:
      break;
    }
  }

  auto const getTranslationMethod() const { return translationMethod_; }

private:
  SmartPointer<TranslatorType> translator_;
  SmartPointer<KDTree<NumericType, Vec3D<NumericType>>> kdTree_;

  const SmartPointer<::viennaps::VelocityField<NumericType, D>>
      modelVelocityField_;
  const SmartPointer<MaterialMap> materialMap_;
  const int translationMethod_ = 1;
};

} // namespace viennaps
