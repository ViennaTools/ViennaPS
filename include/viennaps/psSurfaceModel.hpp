#pragma once

#include "psPointData.hpp"
#include "psProcessParams.hpp"
#include "psSmartPointer.hpp"

#include <vector>

template <typename NumericType> class psSurfaceModel {
protected:
  psSmartPointer<psPointData<NumericType>> coverages = nullptr;
  psSmartPointer<psProcessParams<NumericType>> processParams = nullptr;

public:
  virtual ~psSurfaceModel() = default;

  virtual void initializeCoverages(unsigned numGeometryPoints) {
    // if no coverages get initialized here, they wont be used at all
  }

  virtual void initializeProcessParameters() {
    // if no process parameters get initialized here, they wont be used at all
  }

  virtual psSmartPointer<std::vector<NumericType>> calculateVelocities(
      psSmartPointer<psPointData<NumericType>> rates,
      const std::vector<std::array<NumericType, 3>> &coordinates,
      const std::vector<NumericType> &materialIds) {
    return nullptr;
  }

  virtual void updateCoverages(psSmartPointer<psPointData<NumericType>> rates,
                               const std::vector<NumericType> &materialIds) {}

  // non-virtual functions
  psSmartPointer<psPointData<NumericType>> getCoverages() const {
    return coverages;
  }

  psSmartPointer<psProcessParams<NumericType>> getProcessParameters() const {
    return processParams;
  }
};
