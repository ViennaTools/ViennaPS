#pragma once

#include "psDomain.hpp"

#include <vector>

namespace viennaps {

using namespace viennacore;

template <typename NumericType, int D> class VelocityField {
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

  // translation field options
  // 0: do not translate level set ID to surface ID
  // 1: use unordered map to translate level set ID to surface ID
  // 2: use kd-tree to translate level set ID to surface ID
  virtual int getTranslationFieldOptions() const { return 1; }

  // Function to override for process-specific preparation
  virtual void
  prepare(SmartPointer<Domain<NumericType, D>> domain, // process domain
          SmartPointer<std::vector<NumericType>>
              velocities, // velocities from SurfaceModel
          const NumericType processTime) {}
};

template <typename NumericType, int D>
class DefaultVelocityField : public VelocityField<NumericType, D> {
public:
  DefaultVelocityField(const int translationFieldOptions = 1)
      : translationFieldOptions_(translationFieldOptions) {}

  NumericType getScalarVelocity(const Vec3D<NumericType> &, int,
                                const Vec3D<NumericType> &,
                                unsigned long pointId) override {
    return velocities_->at(pointId);
  }

  int getTranslationFieldOptions() const override {
    return translationFieldOptions_;
  }

  void prepare(SmartPointer<Domain<NumericType, D>> domain,
               SmartPointer<std::vector<NumericType>> velocities,
               const NumericType) override {
    velocities_ = velocities;
  }

private:
  SmartPointer<std::vector<NumericType>> velocities_;
  const int translationFieldOptions_ = 1; // default: use map translator
};

} // namespace viennaps
