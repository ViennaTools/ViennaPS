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

  void setVisibilities(SmartPointer<std::vector<NumericType>> visibilities, int rateSetId = 0) {
    visibilities_[rateSetId] = visibilities;
  }

  // translation field options
  // 0: do not translate level set ID to surface ID
  // 1: use unordered map to translate level set ID to surface ID
  // 2: use kd-tree to translate level set ID to surface ID
  virtual int getTranslationFieldOptions() const { return 1; }

  // Return direction vector for a specific rateSet
  virtual Vec3D<NumericType> getDirection(const int rateSetId) const { return Vec3D<NumericType>{0., 0., 0.}; }

  // Check if visibilities are defined for a specific rateSet
  virtual bool useVisibilities(const int rateSetId) const { return false; }

  virtual bool useVisibilities() const { return false; }

  virtual int numRates() const { return 0; }

public:
  std::map<int, SmartPointer<std::vector<NumericType>>> visibilities_;
  // SmartPointer<std::vector<NumericType>> visibilities_;
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
