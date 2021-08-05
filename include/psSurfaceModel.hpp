#ifndef PS_SURFACE_MODEL
#define PS_SURFACE_MODEL

#include <psSmartPointer.hpp>
#include <psPointData.hpp>
#include <vector>

template <typename NumericType>
class psSurfaceModel
{
protected:
  psSmartPointer<psPointData<NumericType>> Coverages = nullptr;

public:
  virtual void initializeCoverages(unsigned numGeometryPoints)
  {
  }

  psSmartPointer<psPointData<NumericType>> getCoverages()
  {
    return Coverages;
  }

  virtual psSmartPointer<std::vector<NumericType>>
  calculateVelocities(psSmartPointer<psPointData<NumericType>> Rates,
                      const std::vector<NumericType> &materialIDs, const long numRaysPerPoint)
  {
    return psSmartPointer<std::vector<NumericType>>::New();
  }

  virtual void updateCoverages(psSmartPointer<psPointData<NumericType>> Rates, const long numRaysPerPoints)
  {
  }
};

#endif