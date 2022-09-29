#ifndef VELOCITY_FIELD_HPP
#define VELOCITY_FIELD_HPP

#include <iostream>
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
    return velocities->at(pointID);
  }

  void setVelocities(psSmartPointer<std::vector<T>> passedVelocities) override {
    // additional alerations can be made to the velocities here
    velocities = passedVelocities;
  }

private:
  psSmartPointer<std::vector<T>> velocities = nullptr;
};

#endif // RT_VELOCITY_FIELD_HPP