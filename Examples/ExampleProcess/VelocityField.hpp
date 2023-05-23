#pragma once

#include <iostream>
#include <psMaterials.hpp>
#include <psSmartPointer.hpp>
#include <psVelocityField.hpp>
#include <vector>

template <class T> class VelocityField : public psVelocityField<T> {
public:
  VelocityField() {}

  T getScalarVelocity(const std::array<T, 3> & /*coordinate*/, int material,
                      const std::array<T, 3> & /*normalVector*/,
                      unsigned long pointID) override {
    // implement material specific etching/deposition here
    T velocity = 0.;
    if (psMaterialMap::mapToMaterial(material) != psMaterial::Mask) {
      velocity = -velocities->at(pointID);
    }
    return velocity;
  }

  void setVelocities(psSmartPointer<std::vector<T>> passedVelocities) override {
    // additional alerations can be made to the velocities here
    velocities = passedVelocities;
  }

private:
  psSmartPointer<std::vector<T>> velocities = nullptr;
};
