#pragma once

#include "psProcessParams.hpp"

#include <lsPointData.hpp>
#include <vcSmartPointer.hpp>

#include <vector>

namespace viennaps {

using namespace viennacore;

template <typename NumericType> class SurfaceModel {
protected:
  SmartPointer<lsPointData<NumericType>> coverages = nullptr;
  SmartPointer<ProcessParams<NumericType>> processParams = nullptr;

public:
  virtual ~SurfaceModel() = default;

  virtual void initializeCoverages(unsigned numGeometryPoints) {
    // if no coverages get initialized here, they wont be used at all
  }

  virtual void initializeProcessParameters() {
    // if no process parameters get initialized here, they wont be used at all
  }

  virtual SmartPointer<std::vector<NumericType>> calculateVelocities(
      SmartPointer<lsPointData<NumericType>> rates,
      const std::vector<std::array<NumericType, 3>> &coordinates,
      const std::vector<NumericType> &materialIds) {
    return nullptr;
  }

  virtual void updateCoverages(SmartPointer<lsPointData<NumericType>> rates,
                               const std::vector<NumericType> &materialIds) {}

  // non-virtual functions
  auto getCoverages() const { return coverages; }

  auto getProcessParameters() const { return processParams; }
};

} // namespace viennaps
