#pragma once

#include <process/psVelocityField.hpp>
#include <psMaterials.hpp>
#include <vector>

template <class T, int D>
class VelocityField : public viennaps::VelocityField<T, D> {
public:
  VelocityField() = default;

  T getScalarVelocity(const viennaps::Vec3D<T> &coordinate, int material,
                      const viennaps::Vec3D<T> &normalVector,
                      unsigned long pointID) override {
    // implement material specific etching/deposition here
    T velocity = 0.;
    if (viennaps::MaterialMap::mapToMaterial(material) !=
        viennaps::Material::Mask) {
      velocity = -velocities->at(pointID);
    }
    return velocity;
  }

  void prepare(viennaps::SmartPointer<viennaps::Domain<T, D>> domain,
               viennaps::SmartPointer<std::vector<T>> passedVelocities,
               const T processTime) override {
    // additional preparation steps can be done here
    velocities = passedVelocities;
  }

private:
  viennaps::SmartPointer<std::vector<T>> velocities = nullptr;
};
