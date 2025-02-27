#pragma once

#include "psProcessParams.hpp"

#include <lsPointData.hpp>
#include <vcSmartPointer.hpp>

#include <vector>

namespace viennaps {

using namespace viennacore;

template <typename NumericType> class SurfaceModel {
protected:
  SmartPointer<viennals::PointData<NumericType>> coverages = nullptr;
  SmartPointer<viennals::PointData<NumericType>> surfaceData = nullptr;
  SmartPointer<ProcessParams<NumericType>> processParams = nullptr;

public:
  virtual ~SurfaceModel() = default;

  virtual void initializeCoverages(unsigned numGeometryPoints) {
    // if no coverages get initialized here, they won't be used at all
  }

  virtual void initializeProcessParameters() {
    // if no process parameters get initialized here, they won't be used at all
  }

  virtual void initializeSurfaceData(unsigned numGeometryPoints) {
    // if no surface data get initialized here, they won't be used at all
  }

  virtual SmartPointer<std::vector<NumericType>>
  calculateVelocities(SmartPointer<viennals::PointData<NumericType>> rates,
                      const std::vector<Vec3D<NumericType>> &coordinates,
                      const std::vector<NumericType> &materialIds) {
    return nullptr;
  }

  virtual void
  updateCoverages(SmartPointer<viennals::PointData<NumericType>> rates,
                  const std::vector<NumericType> &materialIds) {}

  // non-virtual functions
  auto getCoverages() const { return coverages; }

  auto getProcessParameters() const { return processParams; }

  auto getSurfaceData() const { return surfaceData; }
};

} // namespace viennaps
