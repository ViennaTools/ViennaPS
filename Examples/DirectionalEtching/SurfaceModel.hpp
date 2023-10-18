#pragma once

#include <psSurfaceModel.hpp>

template <typename NumericType>
class SurfaceModel : public psSurfaceModel<NumericType>
{
public:
  psSmartPointer<std::vector<NumericType>> calculateVelocities(
      psSmartPointer<psPointData<NumericType>> Rates,
      const std::vector<std::array<NumericType, 3>> &coordinates,
      const std::vector<NumericType> &materialIds) override
  {

    auto flux =
        Rates->getScalarData("particleRate");
    std::vector<NumericType> rates(materialIds.size(), 0.);
    for (std::size_t i = 0; i < rates.size(); i++)
    {
      if (!psMaterialMap::isMaterial(materialIds[i], psMaterial::Mask))
      {
        rates[i] = -0.7 * flux->at(i);
      }
    }

    return psSmartPointer<std::vector<NumericType>>::New(rates);
  }
};

template <class T, int D>
class VelocityField : public psVelocityField<T>
{
public:
  VelocityField() {}

  T getScalarVelocity(const std::array<T, 3> & /*coordinate*/, int material,
                      const std::array<T, 3> &normalVector,
                      unsigned long pointID) override
  {
    return std::exp(-(normalVector[D - 1] - 1) * (normalVector[D - 1] - 1) / fac) * velocities->at(pointID);
  }

  void setVelocities(psSmartPointer<std::vector<T>> passedVelocities) override
  {
    // additional alerations can be made to the velocities here
    velocities = passedVelocities;
  }

private:
  psSmartPointer<std::vector<T>> velocities = nullptr;
  const T fac = .8;
};
