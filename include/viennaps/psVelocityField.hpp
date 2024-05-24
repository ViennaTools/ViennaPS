#pragma once

#include <vcSmartPointer.hpp>
#include <vcVectorUtil.hpp>

#include <vector>

namespace viennaps {

using namespace viennacore;

template <typename NumericType> class VelocityField {
public:
  virtual ~VelocityField() = default;

  virtual NumericType getScalarVelocity(const Triple<NumericType> &coordinate,
                                        int material,
                                        const Triple<NumericType> &normalVector,
                                        unsigned long pointId) {
    return 0;
  }

  virtual Triple<NumericType>
  getVectorVelocity(const Triple<NumericType> &coordinate, int material,
                    const Triple<NumericType> &normalVector,
                    unsigned long pointId) {
    return {0., 0., 0.};
  }

  virtual NumericType
  getDissipationAlpha(int direction, int material,
                      const Triple<NumericType> &centralDifferences) {
    return 0;
  }

  virtual void
  setVelocities(SmartPointer<std::vector<NumericType>> velocities) {}

  // translation field options
  // 0: do not translate level set ID to surface ID
  // 1: use unordered map to translate level set ID to surface ID
  // 2: use kd-tree to translate level set ID to surface ID
  virtual int getTranslationFieldOptions() const { return 1; }
};

template <typename NumericType>
class DefaultVelocityField : public VelocityField<NumericType> {
public:
  DefaultVelocityField(const int translationFieldOptions = 1)
      : translationFieldOptions_(translationFieldOptions) {}

  virtual NumericType getScalarVelocity(const Triple<NumericType> &, int,
                                        const Triple<NumericType> &,
                                        unsigned long pointId) override {
    return velocities_->at(pointId);
  }

  void
  setVelocities(SmartPointer<std::vector<NumericType>> velocities) override {
    velocities_ = velocities;
  }

  int getTranslationFieldOptions() const override {
    return translationFieldOptions_;
  }

private:
  SmartPointer<std::vector<NumericType>> velocities_;
  const int translationFieldOptions_ = 1; // default: use map translator
};

} // namespace viennaps
