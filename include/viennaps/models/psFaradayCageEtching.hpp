#pragma once

#include "psIonBeamEtching.hpp"

#include "../psMaterials.hpp"
#include "../psProcessModel.hpp"

#include <functional>
#include <random>

namespace viennaps {

using namespace viennacore;

template <typename NumericType> struct FaradayCageParameters {
  IBEParameters<NumericType> ibeParams;

  NumericType cageAngle = 0; // degree
};

namespace impl {

template <typename NumericType, int D>
class PeriodicSource : public viennaray::Source<NumericType> {
public:
  PeriodicSource(const std::array<Vec3D<NumericType>, 2> &boundingBox,
                 const NumericType gridDelta, const NumericType tiltAngle,
                 const NumericType cageAngle, const NumericType cosinePower)
      : sourceExtent_{boundingBox[1][0] - boundingBox[0][0],
                      boundingBox[1][1] - boundingBox[0][1]},
        minPoint_{boundingBox[0][0], boundingBox[0][1]},
        zPos_(boundingBox[1][D - 1] + 2 * gridDelta), gridDelta_(gridDelta),
        ee_{2 / (cosinePower + 1)} {

    NumericType cage_x = std::cos(cageAngle * M_PI / 180.);
    NumericType cage_y = std::sin(cageAngle * M_PI / 180.);
    NumericType cosTilt = std::cos(tiltAngle * M_PI / 180.);
    NumericType sinTilt = std::sin(tiltAngle * M_PI / 180.);

    Vec3D<NumericType> direction;
    direction[0] = -cosTilt * cage_y;
    direction[1] = cosTilt * cage_x;
    direction[2] = -sinTilt;
    if constexpr (D == 2)
      std::swap(direction[1], direction[2]);

    if (Logger::getLogLevel() >= 5) {
      Logger::getInstance()
          .addDebug("FaradayCageEtching: Source direction 1: " +
                    std::to_string(direction[0]) + " " +
                    std::to_string(direction[1]) + " " +
                    std::to_string(direction[2]))
          .print();
    }
    orthoBasis1_ = rayInternal::getOrthonormalBasis(direction);

    direction[0] = cosTilt * cage_y;
    direction[1] = -cosTilt * cage_x;
    direction[2] = -sinTilt;
    if constexpr (D == 2)
      std::swap(direction[1], direction[2]);

    if (Logger::getLogLevel() >= 5) {
      Logger::getInstance()
          .addDebug("FaradayCageEtching: Source direction 2: " +
                    std::to_string(direction[0]) + " " +
                    std::to_string(direction[1]) + " " +
                    std::to_string(direction[2]))
          .print();
    }
    orthoBasis2_ = rayInternal::getOrthonormalBasis(direction);
  }

  std::array<Vec3D<NumericType>, 2>
  getOriginAndDirection(const size_t idx,
                        viennaray::RNG &RngState) const override {
    std::uniform_real_distribution<NumericType> dist(0., 1.);

    Vec3D<NumericType> origin;
    origin[0] = minPoint_[0] + sourceExtent_[0] * dist(RngState);
    if constexpr (D == 3)
      origin[1] = minPoint_[1] + sourceExtent_[1] * dist(RngState);
    origin[D - 1] = zPos_;

    Vec3D<NumericType> direction;
    if (idx % 2 == 0) {
      direction = getCustomDirection(RngState, orthoBasis1_);
    } else {
      direction = getCustomDirection(RngState, orthoBasis2_);
    }
    Normalize(direction);

    return {origin, direction};
  }

  size_t getNumPoints() const override {
    if constexpr (D == 3)
      return sourceExtent_[0] * sourceExtent_[1] / (gridDelta_ * gridDelta_);
    else
      return sourceExtent_[0] / gridDelta_;
  }

  NumericType getSourceArea() const override {
    if constexpr (D == 3)
      return sourceExtent_[0] * sourceExtent_[1];
    else
      return sourceExtent_[0];
  }

  void saveSourcePlane() const {
    auto mesh = viennals::Mesh<NumericType>::New();
    if constexpr (D == 3) {
      Vec3D<NumericType> point{minPoint_[0], minPoint_[1], zPos_};
      mesh->insertNextNode(point);
      point[0] += sourceExtent_[0];
      mesh->insertNextNode(point);
      point[1] += sourceExtent_[1];
      mesh->insertNextNode(point);
      point[0] -= sourceExtent_[0];
      mesh->insertNextNode(point);
      mesh->insertNextTriangle({0, 1, 2});
      mesh->insertNextTriangle({0, 2, 3});
    } else {
      Vec3D<NumericType> point{minPoint_[0], zPos_, NumericType(0)};
      mesh->insertNextNode(point);
      point[0] += sourceExtent_[0];
      mesh->insertNextNode(point);
      mesh->insertNextLine({0, 1});
    }
    viennals::VTKWriter<NumericType>(mesh, "sourcePlane_periodic.vtp").apply();
  }

private:
  Vec3D<NumericType>
  getCustomDirection(viennaray::RNG &rngState,
                     const std::array<Vec3D<NumericType>, 3> &basis) const {
    Vec3D<NumericType> direction;
    std::uniform_real_distribution<NumericType> uniDist;

    Vec3D<NumericType> rndDirection{0., 0., 0.};
    auto r1 = uniDist(rngState);
    auto r2 = uniDist(rngState);

    const NumericType tt = std::pow(r2, ee_);
    rndDirection[0] = std::sqrt(tt);
    rndDirection[1] = std::cos(M_PI * 2. * r1) * std::sqrt(1 - tt);
    rndDirection[2] = std::sin(M_PI * 2. * r1) * std::sqrt(1 - tt);

    direction[0] = basis[0][0] * rndDirection[0] +
                   basis[1][0] * rndDirection[1] +
                   basis[2][0] * rndDirection[2];
    direction[1] = basis[0][1] * rndDirection[0] +
                   basis[1][1] * rndDirection[1] +
                   basis[2][1] * rndDirection[2];
    if constexpr (D == 3) {
      direction[2] = basis[0][2] * rndDirection[0] +
                     basis[1][2] * rndDirection[1] +
                     basis[2][2] * rndDirection[2];
    } else {
      direction[2] = 0.;
      Normalize(direction);
    }

    return direction;
  }

  std::array<NumericType, 2> const sourceExtent_;
  std::array<NumericType, 2> const minPoint_;

  NumericType const zPos_;
  NumericType const gridDelta_;
  const NumericType ee_;

  std::array<Vec3D<NumericType>, 3> orthoBasis1_;
  std::array<Vec3D<NumericType>, 3> orthoBasis2_;
};

} // namespace impl

template <typename NumericType, int D>
class FaradayCageEtching : public ProcessModel<NumericType, D> {
public:
  FaradayCageEtching() = default;

  FaradayCageEtching(const std::vector<Material> &maskMaterials)
      : maskMaterials_(maskMaterials) {}

  FaradayCageEtching(const std::vector<Material> &maskMaterials,
                     const FaradayCageParameters<NumericType> &params)
      : maskMaterials_(maskMaterials), params_(params) {}

  void setMaskMaterials(const std::vector<Material> &maskMaterials) {
    maskMaterials_ = maskMaterials;
  }

  FaradayCageParameters<NumericType> &getParameters() { return params_; }

  void setParameters(const FaradayCageParameters<NumericType> &params) {
    params_ = params;
  }

  void initialize(SmartPointer<Domain<NumericType, D>> domain,
                  const NumericType processDuration) override final {

    auto gridDelta = domain->getGrid().getGridDelta();
    auto boundingBox = domain->getBoundingBox();
    auto source = SmartPointer<impl::PeriodicSource<NumericType, D>>::New(
        boundingBox, gridDelta, params_.ibeParams.tiltAngle, params_.cageAngle,
        params_.ibeParams.exponent);
    this->setSource(source);

    if (firstInit)
      return;

    auto boundaryConditions = domain->getBoundaryConditions();
    if ((D == 3 && (boundaryConditions[0] !=
                        viennals::BoundaryConditionEnum::PERIODIC_BOUNDARY ||
                    boundaryConditions[1] !=
                        viennals::BoundaryConditionEnum::PERIODIC_BOUNDARY)) ||
        (D == 2 && boundaryConditions[0] !=
                       viennals::BoundaryConditionEnum::PERIODIC_BOUNDARY)) {
      Logger::getInstance()
          .addWarning("FaradayCageEtching: Periodic boundary conditions are "
                      "required for the Faraday Cage Etching process.")
          .print();
    }

    // particles
    auto particle =
        std::make_unique<impl::IBEIonWithRedeposition<NumericType, D>>(
            params_.ibeParams);

    // surface model
    auto surfModel = SmartPointer<impl::IBESurfaceModel<NumericType>>::New(
        params_.ibeParams, maskMaterials_);

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New(2);

    if (Logger::getLogLevel() >= 5)
      source->saveSourcePlane();

    this->particles.clear();
    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->insertNextParticleType(particle);
    this->setProcessName("FaradayCageEtching");

    firstInit = true;
  }

  void reset() override final { firstInit = false; }

private:
  bool firstInit = false;
  std::vector<Material> maskMaterials_;
  FaradayCageParameters<NumericType> params_;
};

} // namespace viennaps
