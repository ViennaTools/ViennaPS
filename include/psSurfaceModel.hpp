#ifndef PS_SURFACE_MODEL
#define PS_SURFACE_MODEL

#include <psPointData.hpp>
#include <psSmartPointer.hpp>
#include <vector>

template <typename NumericType> class psSurfaceModel {
protected:
  psSmartPointer<psPointData<NumericType>> Coverages = nullptr;

public:
  virtual void initializeCoverages(unsigned numGeometryPoints) {
    // if no coverages get initialized here, they wont be used at all
  }

  psSmartPointer<psPointData<NumericType>> getCoverages() { return Coverages; }

  virtual psSmartPointer<std::vector<NumericType>>
  calculateVelocities(psSmartPointer<psPointData<NumericType>> Rates,
                      const std::vector<NumericType> &materialIDs,
                      const long numRaysTraced) {
    return psSmartPointer<std::vector<NumericType>>::New();
  }

  virtual void updateCoverages(psSmartPointer<psPointData<NumericType>> Rates,
                               const long numRaysTraced) {}
};

#endif