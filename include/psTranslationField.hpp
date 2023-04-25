#ifndef PS_TRANSLATIONFIELD_HPP
#define PS_TRANSLATIONFIELD_HPP

#include <iostream>
#include <lsVelocityField.hpp>
#include <psKDTree.hpp>
#include <psVelocityField.hpp>

template <typename NumericType>
class psTranslationField : public lsVelocityField<NumericType> {
  using translatorType = std::unordered_map<unsigned long, unsigned long>;
  const bool useTranslation = true;
  const bool useKdTree = false;

public:
  psTranslationField(
      psSmartPointer<psVelocityField<NumericType>> passedVeloField)
      : useTranslation(passedVeloField->useTranslationField()),
        modelVelocityField(passedVeloField) {}

  NumericType getScalarVelocity(const std::array<NumericType, 3> &coordinate,
                                int material,
                                const std::array<NumericType, 3> &normalVector,
                                unsigned long pointId) {
    if (useTranslation)
      translateLsId(pointId);
    return modelVelocityField->getScalarVelocity(coordinate, material,
                                                 normalVector, pointId);
  }

  std::array<NumericType, 3>
  getVectorVelocity(const std::array<NumericType, 3> &coordinate, int material,
                    const std::array<NumericType, 3> &normalVector,
                    unsigned long pointId) {
    if (useTranslation)
      translateLsId(pointId);
    return modelVelocityField->getVectorVelocity(coordinate, material,
                                                 normalVector, pointId);
  }

  NumericType
  getDissipationAlpha(int direction, int material,
                      const std::array<NumericType, 3> &centralDifferences) {
    return modelVelocityField->getDissipationAlpha(direction, material,
                                                   centralDifferences);
  }

  void setTranslator(psSmartPointer<translatorType> passedTranslator) {
    translator = passedTranslator;
  }

  void buildKdTree(const std::vector<std::array<NumericType, 3>> &points) {
    kdTree.setPoints(points);
    kdTree.build();
  }

private:
  void translateLsId(unsigned long &lsId) {
    // if (useKdTree) {
    // if (auto nearest = kdTree.findNearest(coordinate);
    //     nearest->first < velocities->size()) {
    //   lsId = nearest->first;
    // } else {
    //   psLogger::getInstance()
    //       .addWarning("Could not extent velocity from surface to LS point")
    //       .print();
    // }
    // } else {
    if (auto it = translator->find(lsId); it != translator->end()) {
      lsId = it->second;
    } else {
      psLogger::getInstance()
          .addWarning("Could not extend velocity from surface to LS point")
          .print();
    }
    // }
  }

  psSmartPointer<translatorType> translator;
  psKDTree<NumericType, std::array<NumericType, 3>> kdTree;
  const psSmartPointer<psVelocityField<NumericType>> modelVelocityField;
};

#endif