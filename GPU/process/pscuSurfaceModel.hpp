#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "psPointData.hpp"
#include "psProcessParams.hpp"
#include "psSmartPointer.hpp"
#include "utCudaBuffer.hpp"

template <typename NumericType> class pscuSurfaceModel {
protected:
  utCudaBuffer d_coverages;
  utCudaBuffer d_processParams;
  std::unordered_map<std::string, unsigned int> ratesIndexMap;
  std::unordered_map<std::string, unsigned int> coveragesIndexMap;

public:
  virtual ~pscuSurfaceModel() {}

  virtual void initializeCoverages(unsigned numPoints) {
    // if no coverages get initialized here, they wont be used at all
  }

  virtual void initializeProcessParameters() {
    // if no process parameters get initialized here, they wont be used at all
  }

  virtual psSmartPointer<std::vector<NumericType>>
  calculateVelocities(utCudaBuffer &d_rates,
                      const std::vector<std::array<NumericType, 3>> &points,
                      const std::vector<NumericType> &materialIDs) {
    return psSmartPointer<std::vector<NumericType>>::New();
  }

  virtual void updateCoverages(utCudaBuffer &d_rates, unsigned long numPoints) {
  }

  utCudaBuffer &getCoverages() { return d_coverages; }

  utCudaBuffer &getProcessParameters() { return d_processParams; }

  std::unordered_map<std::string, unsigned int> &getCoverageIndexMap() {
    return coveragesIndexMap;
  }

  void
  setIndexMap(std::unordered_map<std::string, unsigned int> passedIndexMap) {
    ratesIndexMap = passedIndexMap;
  }
};
