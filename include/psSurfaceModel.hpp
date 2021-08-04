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
    // if (Coverages == nullptr)
    // {
    //   Coverages = psSmartPointer<psPointData<NumericType>>::New();
    // }
    // else
    // {
    //   Coverages->clear();
    // }
    // // Coverages->resize(getNumberOfCoverages());
    // // for (auto &cov : *Coverages)
    // //   cov.resize(numPoints, value);
  }

  psSmartPointer<psPointData<NumericType>> getCoverages()
  {
    return Coverages;
  }

  virtual std::vector<NumericType>
  calculateVelocities(psSmartPointer<psPointData<NumericType>> Rates,
                      std::vector<NumericType> &materialIDs)
  {
    return std::vector<NumericType>{};
  }

  virtual void updateCoverages(psSmartPointer<psPointData<NumericType>> Rates)
  {
  }
};

#endif