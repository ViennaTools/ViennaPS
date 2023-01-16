#pragma once

#include <iomanip>
#include <iostream>

#include <psAdvectionCallback.hpp>
#include <psCSVWriter.hpp>
#include <psSmartPointer.hpp>

#include "DimensionExtraction.hpp"

template <typename NumericType, int D, int DataDimension>
class AdvectionCallback : public psAdvectionCalback<NumericType, D> {
protected:
  using psAdvectionCalback<NumericType, D>::domain;

public:
  AdvectionCallback() : deltaT(0.5) {}

  AdvectionCallback(NumericType passedDeltaT) : deltaT(passedDeltaT) {}

  void setExtractor(
      psSmartPointer<DimensionExtraction<NumericType, D>> passedExtractor) {
    extractor = passedExtractor;
  }

  void setDataPtr(psSmartPointer<std::vector<NumericType>> passedDataPtr) {
    dataPtr = passedDataPtr;
  }

  void apply() {
    if (!extractor)
      return;

    if (dataPtr) {
      extractor->setDomain(domain);
      extractor->apply();

      auto dimensions = extractor->getDimensions();
      if (dimensions) {
        dataPtr->push_back(processTime / deltaT);
        std::copy(dimensions->begin(), dimensions->end(),
                  std::back_inserter(*dataPtr));
      }
    }
  }

  void applyPreAdvect(const NumericType passedProcessTime) override {
    processTime = passedProcessTime;
  }

  void applyPostAdvect(const NumericType advectionTime) override {
    processTime += advectionTime;
    if (processTime - lastUpdateTime >= deltaT) {
      apply();
      lastUpdateTime = counter * deltaT;
      ++counter;
    }
  }

private:
  NumericType deltaT = 0.5;

  NumericType processTime = 0.0;
  NumericType lastUpdateTime = -deltaT;
  size_t counter = 0;

  psSmartPointer<DimensionExtraction<NumericType, D>> extractor = nullptr;
  psSmartPointer<std::vector<NumericType>> dataPtr = nullptr;
};
