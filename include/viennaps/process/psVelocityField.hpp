#pragma once

#include "../psDomain.hpp"

namespace viennaps {

using namespace viennacore;

VIENNAPS_TEMPLATE_ND(NumericType, D) class VelocityField {
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
    return Vec3D<NumericType>{0., 0., 0.};
  }

  // whether getVectorVelocity can return anything but zero; where it cannot,
  // the translation field looks up no disk for the vector velocity of a
  // point
  virtual bool hasVectorVelocity() const { return true; }

  virtual NumericType
  getDissipationAlpha(int direction, int material,
                      const Vec3D<NumericType> &centralDifferences) {
    return 0;
  }

  // Function to override for process-specific preparation
  virtual void
  prepare(SmartPointer<Domain<NumericType, D>> domain, // process domain
          SmartPointer<std::vector<NumericType>>
              velocities, // velocities from SurfaceModel
          const NumericType processTime) {}
};

VIENNAPS_TEMPLATE_ND(NumericType, D)
class DefaultVelocityField : public VelocityField<NumericType, D> {
public:
  DefaultVelocityField() = default;

  NumericType getScalarVelocity(const Vec3D<NumericType> &, int,
                                const Vec3D<NumericType> &,
                                unsigned long pointId) override {
    return velocities_->at(pointId);
  }

  // scalar velocities only, from the surface model
  bool hasVectorVelocity() const override { return false; }

  void prepare(SmartPointer<Domain<NumericType, D>> domain,
               SmartPointer<std::vector<NumericType>> velocities,
               const NumericType) override {
    velocities_ = velocities;
  }

private:
  SmartPointer<std::vector<NumericType>> velocities_;
};

} // namespace viennaps
