#ifndef PS_TRANSLATIONFIELD_HPP
#define PS_TRANSLATIONFIELD_HPP

#include <iostream>
#include <lsVelocityField.hpp>
#include <psVelocityField.hpp>

template <typename NumericType>
class psTranslationField : public lsVelocityField<NumericType> {
  using translatorType = std::unordered_map<unsigned long, unsigned long>;

public:
  psTranslationField() = default;

  NumericType getScalarVelocity(const std::array<NumericType, 3> &coordinate,
                                int material,
                                const std::array<NumericType, 3> &normalVector,
                                unsigned long pointId) {
    auto surfacePointId = translateLsId(pointId);
    if (surfacePointId != -1) {
      return modelVelocityField->getScalarVelocity(
          coordinate, material, normalVector, surfacePointId);
    } else {
      return 0;
    }
  }

  std::array<NumericType, 3>
  getVectorVelocity(const std::array<NumericType, 3> &coordinate, int material,
                    const std::array<NumericType, 3> &normalVector,
                    unsigned long pointId) {
    auto surfacePointId = translateLsId(pointId);
    if (surfacePointId != -1) {
      return modelVelocityField->getVectorVelocity(
          coordinate, material, normalVector, surfacePointId);
    } else {
      return {0., 0., 0.};
    }
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

  void setVelocityField(
      psSmartPointer<psVelocityField<NumericType>> passedVeloField) {
    modelVelocityField = passedVeloField;
  }

private:
  long translateLsId(unsigned long lsId) {
    if (auto it = translator->find(lsId); it != translator->end()) {
      return it->second;
    } else {
      return -1;
    }
  }

  psSmartPointer<translatorType> translator;
  psSmartPointer<psVelocityField<NumericType>> modelVelocityField;
};

#endif