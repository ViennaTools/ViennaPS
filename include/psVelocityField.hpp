#ifndef PS_VELOCITY_FIELD
#define PS_VELOCITY_FIELD

#include <psSmartPointer.hpp>
#include <vector>

template <typename NumericType> class psVelocityField {
public:
  psVelocityField() {}

  virtual NumericType
  getScalarVelocity(const std::array<NumericType, 3> &coordinate, int material,
                    const std::array<NumericType, 3> &normalVector,
                    unsigned long pointId) {
    return 0;
  }

  virtual std::array<NumericType, 3>
  getVectorVelocity(const std::array<NumericType, 3> &coordinate, int material,
                    const std::array<NumericType, 3> &normalVector,
                    unsigned long pointId) {
    return {0., 0., 0.};
  }

  virtual NumericType
  getDissipationAlpha(int direction, int material,
                      const std::array<NumericType, 3> &centralDifferences) {
    return 0;
  }

  virtual void
  setVelocities(psSmartPointer<std::vector<NumericType>> passedVelocities) {}

  virtual bool useTranslationField() const { return true; }
};

template <typename NumericType>
class psDefaultVelocityField : public psVelocityField<NumericType> {
public:
  virtual NumericType
  getScalarVelocity(const std::array<NumericType, 3> &coordinate, int material,
                    const std::array<NumericType, 3> &normalVector,
                    unsigned long pointId) override {
    return velocities->at(pointId);
  }

  void setVelocities(
      psSmartPointer<std::vector<NumericType>> passedVelocities) override {
    velocities = passedVelocities;
  }

private:
  psSmartPointer<std::vector<NumericType>> velocities;
};

#endif