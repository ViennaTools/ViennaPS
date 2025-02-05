#pragma once

#include <psProcessModel.hpp>

#include <rayParticle.hpp>
#include <rayReflection.hpp>

#include <curtParticle.hpp>
#include <pscuProcessModel.hpp>

namespace viennaps {

namespace gpu {

using namespace viennacore;

namespace impl {
template <typename NumericType, int D>
class FaradayCageSurfaceModel : public ::viennaps::SurfaceModel<NumericType> {
public:
  FaradayCageSurfaceModel() {}

  SmartPointer<std::vector<NumericType>> calculateVelocities(
      SmartPointer<viennals::PointData<NumericType>> rates,
      const std::vector<std::array<NumericType, 3>> &coordinates,
      const std::vector<NumericType> &materialIds) override {

    auto velocity =
        SmartPointer<std::vector<NumericType>>::New(materialIds.size(), 0.);
    auto flux = rates->getScalarData("particleFlux");

    for (std::size_t i = 0; i < velocity->size(); i++) {
      if (!MaterialMap::isMaterial(materialIds[i], Material::Mask)) {
        velocity->at(i) = -flux->at(i) / 2.f;
      }
    }

    return velocity;
  }
};
} // namespace impl

// Etching or deposition based on a single particle model with diffuse
// reflections.
template <typename NumericType, int D>
class FaradayCageEtching : public ProcessModel<NumericType, D> {
public:
  // Angles in degrees
  FaradayCageEtching(NumericType stickingProbability,
                     NumericType sourceDistributionPower, NumericType cageAngle,
                     NumericType tiltAngle) {

    float cosTilt = cosf(tiltAngle * M_PIf / 180.);
    float sinTilt = sinf(tiltAngle * M_PIf / 180.);
    float cage_x = cosf(cageAngle * M_PIf / 180.f);
    float cage_y = sinf(cageAngle * M_PIf / 180.f);

    Particle<NumericType> particle1{
        .name = "ion",
        .sticking = stickingProbability,
        .cosineExponent = sourceDistributionPower,
        .direction = Vec3Df{-cage_y * cosTilt, cage_x * cosTilt, -sinTilt}};
    particle1.dataLabels.push_back("particleFlux");
    this->insertNextParticleType(particle1);

    Particle<NumericType> particle2{
        .name = "ion",
        .sticking = stickingProbability,
        .cosineExponent = sourceDistributionPower,
        .direction = Vec3Df{cage_y * cosTilt, -cage_x * cosTilt, -sinTilt}};
    this->insertNextParticleType(particle2);

    // surface model
    auto surfModel =
        SmartPointer<impl::FaradayCageSurfaceModel<NumericType, D>>::New();

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New(2);

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("FaradayCageEtching");
    this->setPipelineFileName("FaradayCagePipeline");
  }
};

} // namespace gpu
} // namespace viennaps