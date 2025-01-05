#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <gpu/vcCudaBuffer.hpp>

namespace viennaps {

namespace gpu {

using namespace viennacore;

template <typename NumericType> class SurfaceModel {
protected:
  CudaBuffer d_coverages;
  CudaBuffer d_processParams;
  std::unordered_map<std::string, unsigned int> ratesIndexMap;
  std::unordered_map<std::string, unsigned int> coveragesIndexMap;

public:
  virtual ~SurfaceModel() {}

  virtual void initializeCoverages(unsigned numPoints) {
    // if no coverages get initialized here, they wont be used at all
  }

  virtual void initializeProcessParameters() {
    // if no process parameters get initialized here, they wont be used at all
  }

  virtual SmartPointer<std::vector<NumericType>>
  calculateVelocities(CudaBuffer &d_rates,
                      const std::vector<NumericType> &materialIDs) {
    return SmartPointer<std::vector<NumericType>>::New();
  }

  virtual void updateCoverages(CudaBuffer &d_rates,
                               const std::vector<NumericType> &materialIDs) {}

  CudaBuffer &getCoverages() { return d_coverages; }

  CudaBuffer &getProcessParameters() { return d_processParams; }

  std::unordered_map<std::string, unsigned int> &getCoverageIndexMap() {
    return coveragesIndexMap;
  }

  void
  setIndexMap(std::unordered_map<std::string, unsigned int> passedIndexMap) {
    ratesIndexMap = passedIndexMap;
  }
};

} // namespace gpu
} // namespace viennaps