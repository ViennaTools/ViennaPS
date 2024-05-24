#pragma once

#include "compact/psKDTree.hpp"
#include "psMaterials.hpp"
#include "psVelocityField.hpp"

#include <lsVelocityField.hpp>

#include <vcLogger.hpp>
#include <vcSmartPointer.hpp>
#include <vcVectorUtil.hpp>

namespace viennaps {

using namespace viennacore;

template <typename NumericType>
class TranslationField : public lsVelocityField<NumericType> {
  using TranslatorType = std::unordered_map<unsigned long, unsigned long>;

public:
  TranslationField(SmartPointer<VelocityField<NumericType>> velocityField,
                   SmartPointer<MaterialMap> materialMap)
      : translationMethod_(velocityField->getTranslationFieldOptions()),
        modelVelocityField_(velocityField), materialMap_(materialMap) {}

  NumericType getScalarVelocity(const Triple<NumericType> &coordinate,
                                int material,
                                const Triple<NumericType> &normalVector,
                                unsigned long pointId) {
    translateLsId(pointId, coordinate);
    if (materialMap_)
      material = static_cast<int>(materialMap_->getMaterialAtIdx(material));
    return modelVelocityField_->getScalarVelocity(coordinate, material,
                                                  normalVector, pointId);
  }

  Triple<NumericType> getVectorVelocity(const Triple<NumericType> &coordinate,
                                        int material,
                                        const Triple<NumericType> &normalVector,
                                        unsigned long pointId) {
    translateLsId(pointId, coordinate);
    if (materialMap_)
      material = static_cast<int>(materialMap_->getMaterialAtIdx(material));
    return modelVelocityField_->getVectorVelocity(coordinate, material,
                                                  normalVector, pointId);
  }

  NumericType
  getDissipationAlpha(int direction, int material,
                      const Triple<NumericType> &centralDifferences) {
    if (materialMap_)
      material = static_cast<int>(materialMap_->getMaterialAtIdx(material));
    return modelVelocityField_->getDissipationAlpha(direction, material,
                                                    centralDifferences);
  }

  void setTranslator(SmartPointer<TranslatorType> translator) {
    translator_ = translator;
  }

  void buildKdTree(const std::vector<Triple<NumericType>> &points) {
    kdTree_.setPoints(points);
    kdTree_.build();
  }

  void translateLsId(unsigned long &lsId,
                     const Triple<NumericType> &coordinate) const {
    switch (translationMethod_) {
    case 1: {
      if (auto it = translator_->find(lsId); it != translator_->end()) {
        lsId = it->second;
      } else {
        Logger::getInstance()
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
  SmartPointer<TranslatorType> translator_;
  KDTree<NumericType, Triple<NumericType>> kdTree_;
  const SmartPointer<VelocityField<NumericType>> modelVelocityField_;
  const SmartPointer<MaterialMap> materialMap_;
  const int translationMethod_ = 1;
};

} // namespace viennaps
