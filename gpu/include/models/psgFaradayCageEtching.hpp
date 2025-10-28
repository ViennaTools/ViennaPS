// #pragma once

// #include "psgPipelineParameters.hpp"

// #include <process/psProcessModel.hpp>
// #include <psConstants.hpp>
// #include <rayParticle.hpp>

// #include <vcVectorType.hpp>

// namespace viennaps::gpu {

// using namespace viennacore;

// namespace impl {
// template <typename NumericType, int D>
// class FaradayCageSurfaceModel final
//     : public ::viennaps::SurfaceModel<NumericType> {

//   const NumericType tiltAngle_ = 0;
//   const NumericType rate_ = 0;

// public:
//   FaradayCageSurfaceModel(const NumericType rate, const NumericType
//   tiltAngle)
//       : rate_(rate), tiltAngle_(tiltAngle) {}

//   SmartPointer<std::vector<NumericType>> calculateVelocities(
//       SmartPointer<viennals::PointData<NumericType>> rates,
//       const std::vector<std::array<NumericType, 3>> &coordinates,
//       const std::vector<NumericType> &materialIds) override {

//     auto velocity =
//         SmartPointer<std::vector<NumericType>>::New(materialIds.size(), 0.);
//     auto flux = rates->getScalarData("particleFlux");

//     const NumericType norm = rate_ * .5 / std::cos(tiltAngle_ * M_PI / 180.);

//     for (std::size_t i = 0; i < velocity->size(); i++) {
//       if (!MaterialMap::isHardmask(materialIds[i])) {
//         velocity->at(i) = -flux->at(i) * norm;
//       }
//     }

//     return velocity;
//   }
// };
// } // namespace impl

// // Etching or deposition based on a single particle model with diffuse
// // reflections.
// template <typename NumericType, int D>
// class FaradayCageEtching final : public ProcessModelGPU<NumericType, D> {
// public:
//   // Angles in degrees
//   FaradayCageEtching(NumericType rate, NumericType sourceDistributionPower,
//                      NumericType cageAngle, NumericType tiltAngle) {

//     float cosTilt = cosf(tiltAngle * M_PIf / 180.f);
//     float sinTilt = sinf(tiltAngle * M_PIf / 180.f);
//     float cage_x = cosf(cageAngle * M_PIf / 180.f);
//     float cage_y = sinf(cageAngle * M_PIf / 180.f);

//     viennaray::gpu::Particle<NumericType> particle1{
//         .name = "Ion1",
//         .cosineExponent = sourceDistributionPower,
//         .direction =
//             Vec3D<NumericType>{-cage_y * cosTilt, cage_x * cosTilt,
//             -sinTilt}};
//     particle1.dataLabels.push_back("particleFlux"); // single flux array
//     this->insertNextParticleType(particle1);

//     viennaray::gpu::Particle<NumericType> particle2{
//         .name = "Ion2",
//         .cosineExponent = sourceDistributionPower,
//         .direction =
//             Vec3D<NumericType>{cage_y * cosTilt, -cage_x * cosTilt,
//             -sinTilt}};
//     this->insertNextParticleType(particle2);

//     // Callables
//     this->setCallableFileName("CallableWrapper");
//     std::unordered_map<std::string, unsigned> pMap = {{"Ion1", 0}, {"Ion2",
//     1}}; std::vector<viennaray::gpu::CallableConfig> cMap = {
//         {0, viennaray::gpu::CallableSlot::COLLISION,
//          "__direct_callable__faradayCollision"},
//         {0, viennaray::gpu::CallableSlot::REFLECTION,
//          "__direct_callable__faradayReflection"},
//         {1, viennaray::gpu::CallableSlot::COLLISION,
//          "__direct_callable__faradayCollision"},
//         {1, viennaray::gpu::CallableSlot::REFLECTION,
//          "__direct_callable__faradayReflection"}};
//     this->setParticleCallableMap(pMap, cMap);

//     // surface model
//     auto surfModel =
//         SmartPointer<impl::FaradayCageSurfaceModel<NumericType, D>>::New(
//             rate, tiltAngle);

//     // velocity field
//     auto velField = SmartPointer<DefaultVelocityField<NumericType,
//     D>>::New();

//     this->setSurfaceModel(surfModel);
//     this->setVelocityField(velField);
//     this->setProcessName("FaradayCageEtching");

//     impl::IonParams params;
//     params.thetaRMin = constants::degToRad(70);
//     params.thetaRMax = constants::degToRad(90);
//     params.minAngle = constants::degToRad(80);
//     params.inflectAngle = constants::degToRad(89);
//     params.n = 10;
//     this->processData.allocUploadSingle(params);
//     this->hasGPU = true;

//     // meta data
//     this->processMetaData["PWR"] = {rate};
//     this->processMetaData["stickingProbability"] = {stickingProbability};
//     this->processMetaData["sourceExponent"] = {sourceDistributionPower};
//     this->processMetaData["cageAngle"] = {cageAngle};
//     this->processMetaData["tiltAngle"] = {tiltAngle};
//   }
// };

// } // namespace viennaps::gpu
