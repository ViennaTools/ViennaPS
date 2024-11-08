#pragma once

#include <psMaterials.hpp>
#include <psVelocityField.hpp>
#include <vector>

template <class T> class VelocityField : public viennaps::VelocityField<T> {
public:
  VelocityField() {}

  T getScalarVelocity(const std::array<T, 3> & /*coordinate*/, int material,
                      const std::array<T, 3> & /*normalVector*/,
                      unsigned long pointID) override {
    // implement material specific etching/deposition here
    T velocity = 0.;
    if (viennaps::MaterialMap::mapToMaterial(material) !=
        viennaps::Material::Mask) {
      velocity = -velocities->at(pointID);
    }
    return velocity;
  }

  void setVelocities(
      viennaps::SmartPointer<std::vector<T>> passedVelocities) override {
    // additional alterations can be made to the velocities here
    velocities = passedVelocities;
  }

private:
  viennaps::SmartPointer<std::vector<T>> velocities = nullptr;
};
