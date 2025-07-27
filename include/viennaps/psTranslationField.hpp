#pragma once

#include "psMaterials.hpp"
#include "psVelocityField.hpp"

#include <lsVelocityField.hpp>

#include <vcKDTree.hpp>
#include <vcLogger.hpp>
#include <vcSmartPointer.hpp>
#include <vcVectorType.hpp>

namespace viennaps {

using namespace viennacore;

template <typename NumericType, int D>
class TranslationField : public viennals::VelocityField<NumericType> {
  using TranslatorType = std::unordered_map<unsigned long, unsigned long>;

public:
  TranslationField(
      SmartPointer<::viennaps::VelocityField<NumericType, D>> velocityField,
      SmartPointer<MaterialMap> const &materialMap)
      : modelVelocityField_(velocityField), materialMap_(materialMap),
        translationMethod_(velocityField->getTranslationFieldOptions()) {}

  NumericType getScalarVelocity(const Vec3D<NumericType> &coordinate,
                                int material,
                                const Vec3D<NumericType> &normalVector,
                                unsigned long pointId) override {
    translateLsId(pointId, coordinate);
    if (materialMap_)
      material = static_cast<int>(materialMap_->getMaterialAtIdx(material));
    return modelVelocityField_->getScalarVelocity(coordinate, material,
                                                  normalVector, pointId);
  }

  Vec3D<NumericType> getVectorVelocity(const Vec3D<NumericType> &coordinate,
                                       int material,
                                       const Vec3D<NumericType> &normalVector,
                                       unsigned long pointId) override {
    translateLsId(pointId, coordinate);
    if (materialMap_)
      material = static_cast<int>(materialMap_->getMaterialAtIdx(material));
    return modelVelocityField_->getVectorVelocity(coordinate, material,
                                                  normalVector, pointId);
  }

  NumericType
  getDissipationAlpha(int direction, int material,
                      const Vec3D<NumericType> &centralDifferences) override {
    if (materialMap_)
      material = static_cast<int>(materialMap_->getMaterialAtIdx(material));
    return modelVelocityField_->getDissipationAlpha(direction, material,
                                                    centralDifferences);
  }

  void setTranslator(const SmartPointer<TranslatorType> &translator) {
    translator_ = translator;
  }

  void buildKdTree(const std::vector<Vec3D<NumericType>> &points) {
    kdTree_.setPoints(points);
    kdTree_.build();
  }

  auto &getKdTree() { return kdTree_; }

  void translateLsId(unsigned long &lsId,
                     const Vec3D<NumericType> &coordinate) const {
    switch (translationMethod_) {
    case 1: {
      if (auto it = translator_->find(lsId); it != translator_->end()) {
        lsId = it->second;
      } else {
        Logger::getInstance()
            .addError("Could not extend velocity from surface to LS point",
                      false)
            .print();
      }
      break;
    }
    case 2: {
      auto nearest = kdTree_.findNearest(coordinate);
      lsId = nearest->first;
      break;
    }
    default:
      break;
    }
  }

private:
  SmartPointer<TranslatorType> translator_;
  KDTree<NumericType, Vec3D<NumericType>> kdTree_;
  const SmartPointer<::viennaps::VelocityField<NumericType, D>>
      modelVelocityField_;
  const SmartPointer<MaterialMap> materialMap_;
  const int translationMethod_ = 1;
};

} // namespace viennaps
