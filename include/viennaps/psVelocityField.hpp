#pragma once

#include "psSmartPointer.hpp"

#include <vector>

template <typename NumericType> class psVelocityField {
public:
  virtual ~psVelocityField() = default;

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
  setVelocities(psSmartPointer<std::vector<NumericType>> velocities) {}

  // translation field options
  // 0: do not translate level set ID to surface ID
  // 1: use unordered map to translate level set ID to surface ID
  // 2: use kd-tree to translate level set ID to surface ID
  virtual int getTranslationFieldOptions() const { return 1; }
};

template <typename NumericType>
class psDefaultVelocityField : public psVelocityField<NumericType> {
public:
  psDefaultVelocityField(const int translationFieldOptions = 1)
      : translationFieldOptions_(translationFieldOptions) {}

  virtual NumericType getScalarVelocity(const std::array<NumericType, 3> &, int,
                                        const std::array<NumericType, 3> &,
                                        unsigned long pointId) override {
    return velocities_->at(pointId);
  }

  void
  setVelocities(psSmartPointer<std::vector<NumericType>> velocities) override {
    velocities_ = velocities;
  }

  int getTranslationFieldOptions() const override {
    return translationFieldOptions_;
  }

private:
  psSmartPointer<std::vector<NumericType>> velocities_;
  const int translationFieldOptions_ = 1; // default: use map translator
};
