#pragma once

#include <iostream>
#include <vector>

#include <psProcessModel.hpp>
#include <psSmartPointer.hpp>
#include <psSurfaceModel.hpp>
#include <psVelocityField.hpp>

template <class T, int D = 3>
class EpitaxyVelocityField : public psVelocityField<T> {
public:
  EpitaxyVelocityField(const int matId, const int ptopMatId)
      : depoMatId(matId), topMatId(ptopMatId) {}

  T getScalarVelocity(const std::array<T, 3> & /*coordinate*/, int material,
                      const std::array<T, 3> &normalVector,
                      unsigned long pointID) override {

    if (material == depoMatId || material == topMatId) {
      double vel = std::max(std::abs(normalVector[0] * 0.5),
                            std::abs(normalVector[1]) * 0.5);
      // std::abs(normalVector[2])); // no vel in y direction
      constexpr double factor = (R100 - R111) / (high - low);
      vel = (vel - low) * factor + R111;
      return vel;
    } else {
      return 0;
    }
  }

  void setVelocities(psSmartPointer<std::vector<T>> passedVelocities) override {
    // additional alerations can be made to the velocities here
    velocities = passedVelocities;
  }

private:
  psSmartPointer<std::vector<T>> velocities = nullptr;
  const T rate = 0.01;
  const int depoMatId = 0;
  const int topMatId = 0;

  static constexpr double R111 = 0.5;
  static constexpr double R100 = 1.;
  static constexpr double low =
      (D > 2) ? 0.5773502691896257 : 0.7071067811865476;
  static constexpr double high = 1.0;
};

template <typename NumericType>
class EpitaxySurfaceModel : public psSurfaceModel<NumericType> {

public:
  psSmartPointer<std::vector<NumericType>>
  calculateVelocities(psSmartPointer<psPointData<NumericType>> Rates,
                      const std::vector<NumericType> &materialIds) override {
    return nullptr;
  }
};