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

  // translation field options
  // 0: do not translate level set ID to surface ID
  // 1: use unordered map to translate level set ID to surface ID
  // 2: use kd-tree to translate level set ID to surface ID
  virtual int getTranslationFieldOptions() const { return 1; }
};

template <typename NumericType>
class psDefaultVelocityField : public psVelocityField<NumericType> {
public:
  psDefaultVelocityField(const int passedTranslationFieldOptions = 1)
      : translationFieldOptions(passedTranslationFieldOptions) {}

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

  int getTranslationFieldOptions() const override {
    return translationFieldOptions;
  }

private:
  psSmartPointer<std::vector<NumericType>> velocities;
  const int translationFieldOptions = 1; // default: use map translator
};

#endif