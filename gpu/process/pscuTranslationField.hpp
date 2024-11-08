#pragma once

#include <iostream>
#include <lsVelocityField.hpp>
#include <psKDTree.hpp>
#include <psVelocityField.hpp>

template <typename NumericType, bool translate>
class pscuTranslationField : public lsVelocityField<NumericType> {
public:
  pscuTranslationField(
      psSmartPointer<psVelocityField<NumericType>> passedVeloField,
      psSmartPointer<psKDTree<NumericType, std::array<float, 3>>> passedkdTree,
      psSmartPointer<psMaterialMap> passedMaterialMap)
      : kdTree(passedkdTree), modelVelocityField(passedVeloField),
        materialMap(passedMaterialMap) {}

  NumericType getScalarVelocity(const std::array<NumericType, 3> &coordinate,
                                int material,
                                const std::array<NumericType, 3> &normalVector,
                                unsigned long pointId) {
    if constexpr (translate)
      translateLsId(pointId, coordinate);
    if (materialMap)
      material = static_cast<int>(materialMap->getMaterialAtIdx(material));
    return modelVelocityField->getScalarVelocity(coordinate, material,
                                                 normalVector, pointId);
  }

  std::array<NumericType, 3>
  getVectorVelocity(const std::array<NumericType, 3> &coordinate, int material,
                    const std::array<NumericType, 3> &normalVector,
                    unsigned long pointId) {
    if constexpr (translate)
      translateLsId(pointId, coordinate);
    if (materialMap)
      material = static_cast<int>(materialMap->getMaterialAtIdx(material));
    return modelVelocityField->getVectorVelocity(coordinate, material,
                                                 normalVector, pointId);
  }

  NumericType
  getDissipationAlpha(int direction, int material,
                      const std::array<NumericType, 3> &centralDifferences) {
    if (materialMap)
      material = static_cast<int>(materialMap->getMaterialAtIdx(material));
    return modelVelocityField->getDissipationAlpha(direction, material,
                                                   centralDifferences);
  }

private:
  inline void translateLsId(unsigned long &lsId,
                            const std::array<NumericType, 3> &coordinate) {
    auto nearest = kdTree->findNearest(coordinate);
    lsId = nearest->first;
  }

  const psSmartPointer<psKDTree<NumericType, std::array<float, 3>>> kdTree;
  const psSmartPointer<psVelocityField<NumericType>> modelVelocityField;
  const psSmartPointer<psMaterialMap> materialMap;
};
