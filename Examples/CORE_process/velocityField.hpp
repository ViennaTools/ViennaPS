#ifndef VELOCITY_FIELD_HPP
#define VELOCITY_FIELD_HPP

#include <psSmartPointer.hpp>
#include <psVelocityField.hpp>
#include <vector>

template <class T> class velocityField : public psVelocityField<T> {
public:
  velocityField() {}

  T getScalarVelocity(const std::array<T, 3> & /*coordinate*/, int /*material*/,
                      const std::array<T, 3> & /*normalVector*/,
                      unsigned long pointID) override {
    return velocities->at(pointID);
  }

  void setVelocities(psSmartPointer<std::vector<T>> passedVelocities) override {
    velocities = passedVelocities;
  }

private:
  psSmartPointer<std::vector<T>> velocities = nullptr;
};

#endif // RT_VELOCITY_FIELD_HPP