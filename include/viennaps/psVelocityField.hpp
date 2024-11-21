#pragma once

#include <vcSmartPointer.hpp>
#include <vcVectorUtil.hpp>

#include <vector>

namespace viennaps {

using namespace viennacore;

template <typename NumericType> class VelocityField {
public:
  virtual ~VelocityField() = default;

  virtual NumericType getScalarVelocity(const Vec3D<NumericType> &coordinate,
                                        int material,
                                        const Vec3D<NumericType> &normalVector,
                                        unsigned long pointId) {
    return 0;
  }

  virtual Vec3D<NumericType>
  getVectorVelocity(const Vec3D<NumericType> &coordinate, int material,
                    const Vec3D<NumericType> &normalVector,
                    unsigned long pointId) {
    return {0., 0., 0.};
  }

  virtual NumericType
  getDissipationAlpha(int direction, int material,
                      const Vec3D<NumericType> &centralDifferences) {
    return 0;
  }

  virtual void
  setVelocities(SmartPointer<std::vector<NumericType>> velocities) {}

  void setVisibilities(SmartPointer<std::vector<NumericType>> visibilities) {
    visibilities_ = visibilities;
  }

  // translation field options
  // 0: do not translate level set ID to surface ID
  // 1: use unordered map to translate level set ID to surface ID
  // 2: use kd-tree to translate level set ID to surface ID
  virtual int getTranslationFieldOptions() const { return 1; }

  virtual bool useVisibilities() const { return false; }

public:
  SmartPointer<std::vector<NumericType>> visibilities_;
};

template <typename NumericType>
class DefaultVelocityField : public VelocityField<NumericType> {
public:
  DefaultVelocityField(const int translationFieldOptions = 1)
      : translationFieldOptions_(translationFieldOptions) {}

  virtual NumericType getScalarVelocity(const Vec3D<NumericType> &, int,
                                        const Vec3D<NumericType> &,
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
