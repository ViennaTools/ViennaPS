#pragma once

#include <iostream>
#include <vector>

#include <psProcessModel.hpp>
#include <psSmartPointer.hpp>
#include <psSurfaceModel.hpp>
#include <psVelocityField.hpp>

template <class T>
class IsoDepositionVelocityField : public psVelocityField<T> {
public:
  IsoDepositionVelocityField() {}

  T getScalarVelocity(const std::array<T, 3> & /*coordinate*/, int material,
                      const std::array<T, 3> &normalVector,
                      unsigned long pointID) override {
    return rate;
  }

  void setVelocities(psSmartPointer<std::vector<T>> passedVelocities) override {
    // additional alerations can be made to the velocities here
    velocities = passedVelocities;
  }

private:
  psSmartPointer<std::vector<T>> velocities = nullptr;
  const T rate = 0.01;
};

template <typename NumericType>
class IsoDepositionSurfaceModel : public psSurfaceModel<NumericType> {

public:
  psSmartPointer<std::vector<NumericType>>
  calculateVelocities(psSmartPointer<psPointData<NumericType>> Rates,
                      const std::vector<NumericType> &materialIds) override {
    return nullptr;
  }
};