#pragma once

#include <lsDomain.hpp>

#include <psSmartPointer.hpp>

template <class NumericType, int D> class psPointValuesToLevelSet {
  using translatorType =
      psSmartPointer<std::unordered_map<unsigned long, unsigned long>>;
  using lsDomainType = psSmartPointer<lsDomain<NumericType, D>>;

  lsDomainType levelSet;
  translatorType translator;
  std::vector<NumericType> *pointValues;
  std::string dataName;

public:
  psPointValuesToLevelSet() {}

  psPointValuesToLevelSet(lsDomainType passedLevelSet,
                          translatorType passedTranslator,
                          std::vector<NumericType> *passedPointValues,
                          std::string passedDataName)
      : levelSet(passedLevelSet), translator(passedTranslator),
        pointValues(passedPointValues), dataName(passedDataName) {}

  void setLevelSet(lsDomainType passedLevelSet) { levelSet = passedLevelSet; }

  void setTranslator(translatorType passedTranslator) {
    translator = passedTranslator;
  }

  void setPointValues(std::vector<NumericType> *passedPointValues) {
    pointValues =
        psSmartPointer<std::vector<NumericType>>::New(passedPointValues);
  }

  void apply() {
    auto data = levelSet->getPointData().getScalarData(dataName);
    if (data != nullptr) {
      data->resize(translator->size());
    } else {
      levelSet->getPointData().insertNextScalarData(
          std::vector<NumericType>(levelSet->getNumberOfPoints()), dataName);
    }
    data = levelSet->getPointData().getScalarData(dataName);
    for (const auto [lsIdx, pointIdx] : *translator) {
      data->at(lsIdx) = pointValues->at(pointIdx);
    }
  }
};