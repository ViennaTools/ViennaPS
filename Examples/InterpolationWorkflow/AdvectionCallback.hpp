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

  void setWriter(
      psSmartPointer<psCSVWriter<NumericType, DataDimension>> passedWriter) {
    writer = passedWriter;
  }

  void
  setPrefixData(psSmartPointer<std::vector<NumericType>> passedPrefixData) {
    prefixData = passedPrefixData;
  }

  void apply() {
    if (!extractor)
      return;

    // std::cout << std::setw(6) << std::setprecision(3) << processTime << ": ";

    extractor->setDomain(domain);
    extractor->apply();
    auto dimensions = extractor->getDimensions();

    // for (auto d : (dimensions ? *dimensions : std::vector<NumericType>{})) {
    //   std::cout << d << ' ';
    // }
    // std::cout << '\n';

    std::vector<NumericType> row;
    row.reserve(DataDimension);

    if (prefixData)
      std::copy(prefixData->begin(), prefixData->end(),
                std::back_inserter(row));

    row.push_back(counter);

    if (dimensions)
      std::copy(dimensions->begin(), dimensions->end(),
                std::back_inserter(row));

    if (writer)
      writer->writeRow(row);

    ++counter;
  }

  void applyPreAdvect(const NumericType passedProcessTime) override {
    processTime = passedProcessTime;
  }

  void applyPostAdvect(const NumericType advectionTime) override {
    processTime += advectionTime;
    if (processTime - lastUpdateTime >= deltaT) {
      apply();
      lastUpdateTime = processTime;
    }
  }

private:
  NumericType deltaT = 0.5;

  NumericType processTime = 0.0;
  NumericType lastUpdateTime = -deltaT;
  size_t counter = 0L;

  psSmartPointer<DimensionExtraction<NumericType, D>> extractor = nullptr;
  psSmartPointer<psCSVWriter<NumericType, DataDimension>> writer = nullptr;
  psSmartPointer<std::vector<NumericType>> prefixData = nullptr;
};
