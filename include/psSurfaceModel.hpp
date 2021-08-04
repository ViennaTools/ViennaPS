#ifndef PS_SURFACE_MODEL
#define PS_SURFACE_MODEL

#include <psSmartPointer.hpp>
#include <vector>

template <typename NumericType>
class psSurfaceModel
{
private:
  using SurfaceDataType = std::vector<std::vector<NumericType>>;
  psSmartPointer<SurfaceDataType> Coverages = nullptr;

public:
  void initializeCoverages(unsigned numPoints, NumericType value)
  {
    if (Coverages == nullptr)
    {
      Coverages = psSmartPointer<SurfaceDataType>::New();
    }
    else
    {
      Coverages->clear();
    }
    Coverages->resize(getNumberOfCoverages());
    for (auto &cov : *Coverages)
      cov.resize(numPoints, value);
  }

  psSmartPointer<SurfaceDataType> getCoverages()
  {
    return Coverages;
  }

  virtual std::vector<NumericType>
  calculateVelocities(SurfaceDataType &Rates,
                      std::vector<NumericType> &materialIDs)
  {
    return std::vector<NumericType>{};
  }

  virtual void updateCoverages(SurfaceDataType &Rates)
  {
  }

  virtual int getNumberOfCoverages() const 
  {
    return 1;
  }
};

#endif