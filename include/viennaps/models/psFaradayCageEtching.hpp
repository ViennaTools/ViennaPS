#pragma once

#include "psIonBeamEtching.hpp"

#include "../process/psProcessModel.hpp"
#include "../psMaterials.hpp"

#ifdef VIENNACORE_COMPILE_GPU
#include <psgPipelineParameters.hpp>
#endif

#include <random>

namespace viennaps {

using namespace viennacore;

template <typename NumericType> struct FaradayCageParameters {
  IBEParameters<NumericType> ibeParams;

  NumericType cageAngle = 0; // degree

  auto toProcessMetaData() const {
    std::unordered_map<std::string, std::vector<double>> processData =
        ibeParams.toProcessMetaData();
    processData["Cage Angle"] = {(cageAngle * M_PI) / 180.};
    return processData;
  }
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

#ifdef VIENNACORE_COMPILE_GPU
namespace gpu {

template <typename NumericType, int D>
class FaradayCageEtching final : public ProcessModelGPU<NumericType, D> {
public:
  // Angles in degrees
  FaradayCageEtching(const FaradayCageParameters<NumericType> &params,
                     const std::vector<Material> &maskMaterials)
      : params_(params), maskMaterials_(maskMaterials) {

    NumericType cosTilt = std::cos(params.ibeParams.tiltAngle * M_PIf / 180.f);
    NumericType sinTilt = std::sin(params.ibeParams.tiltAngle * M_PIf / 180.f);
    NumericType cage_y = std::cos(params.cageAngle * M_PIf / 180.f);
    NumericType cage_x = std::sin(params.cageAngle * M_PIf / 180.f);

    viennaray::gpu::Particle<NumericType> particle1{
        .name = "Ion1",
        .cosineExponent = params.ibeParams.exponent,
        .useCustomDirection = true,
        .direction =
            Vec3D<NumericType>{-cage_x * cosTilt, cage_y * cosTilt, -sinTilt}};
    particle1.dataLabels.push_back(
        ::viennaps::impl::IBESurfaceModel<NumericType>::fluxLabel);
    if (params.ibeParams.redepositionRate > 0.) {
      particle1.dataLabels.push_back(
          ::viennaps::impl::IBESurfaceModel<NumericType>::redepositionLabel);
    }
    this->insertNextParticleType(particle1);

    viennaray::gpu::Particle<NumericType> particle2{
        .name = "Ion2",
        .cosineExponent = params.ibeParams.exponent,
        .useCustomDirection = true,
        .direction =
            Vec3D<NumericType>{cage_x * cosTilt, -cage_y * cosTilt, -sinTilt}};
    // NO ADDITIONAL FLUX ARRAY NEEDED, ALL PARTICLES WRITE TO THE SAME
    this->insertNextParticleType(particle2);

    // Callables
    std::unordered_map<std::string, unsigned> pMap = {{"Ion1", 0}, {"Ion2", 1}};
    std::vector<viennaray::gpu::CallableConfig> cMap = {
        {0, viennaray::gpu::CallableSlot::COLLISION,
         "__direct_callable__faradayCollision"},
        {0, viennaray::gpu::CallableSlot::REFLECTION,
         "__direct_callable__IBEReflection"},
        {0, viennaray::gpu::CallableSlot::INIT, "__direct_callable__IBEInit"},
        {1, viennaray::gpu::CallableSlot::COLLISION,
         "__direct_callable__faradayCollision"},
        {1, viennaray::gpu::CallableSlot::REFLECTION,
         "__direct_callable__IBEReflection"},
        {1, viennaray::gpu::CallableSlot::INIT, "__direct_callable__IBEInit"}};
    this->setParticleCallableMap(pMap, cMap);
    this->setCallableFileName("CallableWrapper");

    // Parameters to upload to device
    impl::IonParams deviceParams;
    deviceParams.thetaRMin =
        static_cast<float>(constants::degToRad(params.ibeParams.thetaRMin));
    deviceParams.thetaRMax =
        static_cast<float>(constants::degToRad(params.ibeParams.thetaRMax));
    deviceParams.meanEnergy = static_cast<float>(params.ibeParams.meanEnergy);
    deviceParams.sigmaEnergy = static_cast<float>(params.ibeParams.sigmaEnergy);
    deviceParams.thresholdEnergy = static_cast<float>(
        std::sqrt(params.ibeParams.thresholdEnergy)); // precompute sqrt
    deviceParams.minAngle =
        static_cast<float>(constants::degToRad(params.ibeParams.minAngle));
    deviceParams.inflectAngle =
        static_cast<float>(constants::degToRad(params.ibeParams.inflectAngle));
    deviceParams.n_l = static_cast<float>(params.ibeParams.n_l);
    deviceParams.B_sp = 0.f; // not used in IBE
    if (params.ibeParams.cos4Yield.isDefined) {
      deviceParams.a1 = static_cast<float>(params.ibeParams.cos4Yield.a1);
      deviceParams.a2 = static_cast<float>(params.ibeParams.cos4Yield.a2);
      deviceParams.a3 = static_cast<float>(params.ibeParams.cos4Yield.a3);
      deviceParams.a4 = static_cast<float>(params.ibeParams.cos4Yield.a4);
      deviceParams.aSum = static_cast<float>(params.ibeParams.cos4Yield.aSum());
    }
    deviceParams.redepositionRate =
        static_cast<float>(params.ibeParams.redepositionRate);
    deviceParams.redepositionThreshold =
        static_cast<float>(params.ibeParams.redepositionThreshold);

    // upload process params
    this->processData.alloc(sizeof(impl::IonParams));
    this->processData.upload(&deviceParams, 1);

    // surface model
    // adjust rate here since we trace two particles per point
    auto surfaceModelParams = params_.ibeParams;
    surfaceModelParams.planeWaferRate *= 0.5;
    for (auto &pair : surfaceModelParams.materialPlaneWaferRate) {
      pair.second *= 0.5;
    }
    surfaceModelParams.redepositionRate *= 0.5;
    auto surfModel =
        SmartPointer<::viennaps::impl::IBESurfaceModel<NumericType>>::New(
            surfaceModelParams, maskMaterials_);

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New();

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("FaradayCageEtching");
    this->processMetaData = params_.toProcessMetaData();
    this->hasGPU = true;
  }

private:
  FaradayCageParameters<NumericType> params_;
  std::vector<Material> maskMaterials_;
};
} // namespace gpu
#endif

template <typename NumericType, int D>
class FaradayCageEtching : public ProcessModelCPU<NumericType, D> {
public:
  FaradayCageEtching(const FaradayCageParameters<NumericType> &params,
                     const std::vector<Material> &maskMaterials)
      : maskMaterials_(maskMaterials), params_(params) {
    // particles
    auto particle =
        std::make_unique<impl::IBEIonWithRedeposition<NumericType, D>>(
            params_.ibeParams);

    // surface model
    auto surfModel = SmartPointer<impl::IBESurfaceModel<NumericType>>::New(
        params_.ibeParams, maskMaterials_);

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New();

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->insertNextParticleType(particle);
    this->setProcessName("FaradayCageEtching");
    this->processMetaData = params_.toProcessMetaData();
    this->hasGPU = true;
  }

  void initialize(SmartPointer<Domain<NumericType, D>> domain,
                  const NumericType processDuration) override final {

    auto gridDelta = domain->getGrid().getGridDelta();
    auto boundingBox = domain->getBoundingBox();
    auto source = SmartPointer<impl::PeriodicSource<NumericType, D>>::New(
        boundingBox, gridDelta, params_.ibeParams.tiltAngle, params_.cageAngle,
        params_.ibeParams.exponent);
    this->setSource(source);

    if (Logger::getLogLevel() >= 5)
      source->saveSourcePlane();

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

    firstInit = true;
  }

  void finalize(SmartPointer<Domain<NumericType, D>> domain,
                const NumericType processedDuration) final {
    firstInit = false;
  }

#ifdef VIENNACORE_COMPILE_GPU
  SmartPointer<ProcessModelBase<NumericType, D>> getGPUModel() final {
    return SmartPointer<gpu::FaradayCageEtching<NumericType, D>>::New(
        params_, maskMaterials_);
  }
#endif

private:
  bool firstInit = false;
  std::vector<Material> maskMaterials_;
  FaradayCageParameters<NumericType> params_;
};

PS_PRECOMPILE_PRECISION_DIMENSION(FaradayCageEtching)

} // namespace viennaps
