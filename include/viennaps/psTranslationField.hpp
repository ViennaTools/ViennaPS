#pragma once

#include "compact/psKDTree.hpp"
#include "psMaterials.hpp"
#include "psVelocityField.hpp"

#include <lsVelocityField.hpp>

template <typename NumericType>
class psTranslationField : public lsVelocityField<NumericType> {
  using TranslatorType = std::unordered_map<unsigned long, unsigned long>;

public:
  psTranslationField(psSmartPointer<psVelocityField<NumericType>> velocityField,
                     psSmartPointer<psMaterialMap> materialMap)
      : translationMethod_(velocityField->getTranslationFieldOptions()),
        modelVelocityField_(velocityField), materialMap_(materialMap) {}

  NumericType getScalarVelocity(const std::array<NumericType, 3> &coordinate,
                                int material,
                                const std::array<NumericType, 3> &normalVector,
                                unsigned long pointId) {
    translateLsId(pointId, coordinate);
    if (materialMap_)
      material = static_cast<int>(materialMap_->getMaterialAtIdx(material));
    return modelVelocityField_->getScalarVelocity(coordinate, material,
                                                  normalVector, pointId);
  }

  std::array<NumericType, 3>
  getVectorVelocity(const std::array<NumericType, 3> &coordinate, int material,
                    const std::array<NumericType, 3> &normalVector,
                    unsigned long pointId) {
    translateLsId(pointId, coordinate);
    if (materialMap_)
      material = static_cast<int>(materialMap_->getMaterialAtIdx(material));
    return modelVelocityField_->getVectorVelocity(coordinate, material,
                                                  normalVector, pointId);
  }

  NumericType
  getDissipationAlpha(int direction, int material,
                      const std::array<NumericType, 3> &centralDifferences) {
    if (materialMap_)
      material = static_cast<int>(materialMap_->getMaterialAtIdx(material));
    return modelVelocityField_->getDissipationAlpha(direction, material,
                                                    centralDifferences);
  }

  void setTranslator(psSmartPointer<TranslatorType> translator) {
    translator_ = translator;
  }

  void buildKdTree(const std::vector<std::array<NumericType, 3>> &points) {
    kdTree_.setPoints(points);
    kdTree_.build();
  }

  void translateLsId(unsigned long &lsId,
                     const std::array<NumericType, 3> &coordinate) const {
    switch (translationMethod_) {
    case 1: {
      if (auto it = translator_->find(lsId); it != translator_->end()) {
        lsId = it->second;
      } else {
        psLogger::getInstance()
            .addWarning("Could not extend velocity from surface to LS point")
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
  psSmartPointer<TranslatorType> translator_;
  psKDTree<NumericType, std::array<NumericType, 3>> kdTree_;
  const psSmartPointer<psVelocityField<NumericType>> modelVelocityField_;
  const psSmartPointer<psMaterialMap> materialMap_;
  const int translationMethod_ = 1;
};
