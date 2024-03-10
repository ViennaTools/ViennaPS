/*
  This file is used to generate the python module of ViennaPS.
  It uses pybind11 to create the modules.
*/

#define PYBIND11_DETAILED_ERROR_MESSAGES
#define VIENNAPS_PYTHON_BUILD

// correct module name macro
#define TOKENPASTE_INTERNAL(x, y, z) x##y##z
#define TOKENPASTE(x, y, z) TOKENPASTE_INTERNAL(x, y, z)
#define STRINGIZE2(s) #s
#define STRINGIZE(s) STRINGIZE2(s)
#define VIENNAPS_MODULE_VERSION STRINGIZE(VIENNAPS_VERSION)

#include <optional>
#include <vector>

#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

// all header files which define API functions
#include <psDomain.hpp>
#include <psExtrude.hpp>
#include <psGDSGeometry.hpp>
#include <psGDSReader.hpp>
#include <psLogger.hpp>
#include <psMeanFreePath.hpp>
#include <psPlanarize.hpp>
#include <psProcess.hpp>

// geometries
#include <psMakeFin.hpp>
#include <psMakeHole.hpp>
#include <psMakePlane.hpp>
#include <psMakeStack.hpp>
#include <psMakeTrench.hpp>

// model framework
#include <psAdvectionCallback.hpp>
#include <psProcessModel.hpp>
#include <psProcessParams.hpp>
#include <psSurfaceModel.hpp>
#include <psVelocityField.hpp>

// models
#include <psAnisotropicProcess.hpp>
#include <psAtomicLayerProcess.hpp>
#include <psDirectionalEtching.hpp>
#include <psFluorocarbonEtching.hpp>
#include <psGeometricDistributionModels.hpp>
#include <psIsotropicProcess.hpp>
#include <psOxideRegrowth.hpp>
#include <psPlasmaDamage.hpp>
#include <psSF6O2Etching.hpp>
#include <psSingleParticleProcess.hpp>
#include <psTEOSDeposition.hpp>

// visualization
#include <psToDiskMesh.hpp>
#include <psToSurfaceMesh.hpp>
#include <psWriteVisualizationMesh.hpp>

// CellSet
#include <csDenseCellSet.hpp>
#include <csSegmentCells.hpp>

// Compact
#include <psKDTree.hpp>

// other
#include <psUtils.hpp>
#include <rayParticle.hpp>
#include <rayReflection.hpp>
#include <rayTraceDirection.hpp>
#include <rayUtil.hpp>

// always use double for python export
typedef double T;
// get dimension from cmake define
constexpr int D = VIENNAPS_PYTHON_DIMENSION;
typedef psSmartPointer<psDomain<T, D>> DomainType;

PYBIND11_DECLARE_HOLDER_TYPE(Types, psSmartPointer<Types>)

// NOTES:
// PYBIND11_MAKE_OPAQUE(std::vector<T, std::allocator<T>>) can not be used
// constructors with custom enum need lambda to work: seems to be an issue
// with implicit move constructor

// define trampoline classes for interface functions
// ALSO NEED TO ADD TRAMPOLINE CLASSES FOR CLASSES
// WHICH HOLD REFERENCES TO INTERFACE(ABSTRACT) CLASSES
// class PypsSurfaceModel : public psSurfaceModel<T> {
//   using psSurfaceModel<T>::coverages;
//   using psSurfaceModel<T>::processParams;
//   using psSurfaceModel<T>::getCoverages;
//   using psSurfaceModel<T>::getProcessParameters;
//   typedef std::vector<T> vect_type;
// public:
//   void initializeCoverages(unsigned numGeometryPoints) override {
//     PYBIND11_OVERRIDE(void, psSurfaceModel<T>, initializeCoverages,
//                       numGeometryPoints);
//   }
//   void initializeProcessParameters() override {
//     PYBIND11_OVERRIDE(void, psSurfaceModel<T>, initializeProcessParameters,
//     );
//   }
//   psSmartPointer<std::vector<T>>
//   calculateVelocities(psSmartPointer<psPointData<T>> rates,
//                       const std::vector<std::array<T, 3>> &coordinates,
//                       const std::vector<T> &materialIds) override {
//     PYBIND11_OVERRIDE(psSmartPointer<std::vector<T>>, psSurfaceModel<T>,
//                       calculateVelocities, rates, coordinates, materialIds);
//   }
//   void updateCoverages(psSmartPointer<psPointData<T>> rates,
//                        const std::vector<T> &materialIds) override {
//     PYBIND11_OVERRIDE(void, psSurfaceModel<T>, updateCoverages, rates,
//                       materialIds);
//   }
// };

// psAdvectionCallback
class PyAdvectionCallback : public psAdvectionCallback<T, D> {
protected:
  using ClassName = psAdvectionCallback<T, D>;

public:
  using ClassName::domain;

  bool applyPreAdvect(const T processTime) override {
    PYBIND11_OVERRIDE(bool, ClassName, applyPreAdvect, processTime);
  }

  bool applyPostAdvect(const T advectionTime) override {
    PYBIND11_OVERRIDE(bool, ClassName, applyPostAdvect, advectionTime);
  }
};

// Particle Class
template <int D> class psParticle : public rayParticle<psParticle<D>, T> {
  using ClassName = rayParticle<psParticle<D>, T>;

public:
  void surfaceCollision(T rayWeight, const rayTriple<T> &rayDir,
                        const rayTriple<T> &geomNormal,
                        const unsigned int primID, const int materialID,
                        rayTracingData<T> &localData,
                        const rayTracingData<T> *globalData,
                        rayRNG &Rng) override final {
    PYBIND11_OVERRIDE(void, ClassName, surfaceCollision, rayWeight, rayDir,
                      geomNormal, primID, materialID, localData, globalData,
                      Rng);
  }

  std::pair<T, rayTriple<T>>
  surfaceReflection(T rayWeight, const rayTriple<T> &rayDir,
                    const rayTriple<T> &geomNormal, const unsigned int primID,
                    const int materialID, const rayTracingData<T> *globalData,
                    rayRNG &Rng) override final {
    using Pair = std::pair<T, rayTriple<T>>;
    PYBIND11_OVERRIDE(Pair, ClassName, surfaceReflection, rayWeight, rayDir,
                      geomNormal, primID, materialID, globalData, Rng);
  }

  void initNew(rayRNG &RNG) override final {
    PYBIND11_OVERRIDE(void, ClassName, initNew, RNG);
  }

  T getSourceDistributionPower() const override final {
    PYBIND11_OVERRIDE(T, ClassName, getSourceDistributionPower);
  }

  std::vector<std::string> getLocalDataLabels() const override final {
    PYBIND11_OVERRIDE(std::vector<std::string>, ClassName, getLocalDataLabels);
  }
};

// Default particle classes
// template <int D>
// class psDiffuseParticle : public rayParticle<psDiffuseParticle<D>, T> {
//   using ClassName = rayParticle<psDiffuseParticle<D>, T>;
// public:
//   psDiffuseParticle(const T pStickingProbability, const T pCosineExponent,
//                     const std::string &pDataLabel)
//       : stickingProbability(pStickingProbability),
//         cosineExponent(pCosineExponent), dataLabel(pDataLabel) {}
//   void surfaceCollision(T rayWeight, const rayTriple<T> &rayDir,
//                         const rayTriple<T> &geomNormal,
//                         const unsigned int primID, const int materialID,
//                         rayTracingData<T> &localData,
//                         const rayTracingData<T> *globalData,
//                         rayRNG &Rng) override final {
//     localData.getVectorData(0)[primID] += rayWeight;
//   }
//   std::pair<T, rayTriple<T>>
//   surfaceReflection(T rayWeight, const rayTriple<T> &rayDir,
//                     const rayTriple<T> &geomNormal, const unsigned int
//                     primID, const int materialID, const rayTracingData<T>
//                     *globalData, rayRNG &Rng) override final {
//     auto direction = rayReflectionDiffuse<T, D>(geomNormal, Rng);
//     return {stickingProbability, direction};
//   }
//   void initNew(rayRNG &RNG) override final {}
//   T getSourceDistributionPower() const override final { return
//   cosineExponent; }
//   std::vector<std::string> getLocalDataLabels() const override final {
//     return {dataLabel};
//   }
// private:
//   const T stickingProbability = 1.;
//   const T cosineExponent = 1.;
//   const std::string dataLabel = "flux";
// };
// class psSpecularParticle : public rayParticle<psSpecularParticle, T> {
//   using ClassName = rayParticle<psSpecularParticle, T>;
// public:
//   psSpecularParticle(const T pStickingProbability, const T pCosineExponent,
//                      const std::string &pDataLabel)
//       : stickingProbability(pStickingProbability),
//         cosineExponent(pCosineExponent), dataLabel(pDataLabel) {}
//   void surfaceCollision(T rayWeight, const rayTriple<T> &rayDir,
//                         const rayTriple<T> &geomNormal,
//                         const unsigned int primID, const int materialID,
//                         rayTracingData<T> &localData,
//                         const rayTracingData<T> *globalData,
//                         rayRNG &Rng) override final {
//     localData.getVectorData(0)[primID] += rayWeight;
//   }
//   std::pair<T, rayTriple<T>>
//   surfaceReflection(T rayWeight, const rayTriple<T> &rayDir,
//                     const rayTriple<T> &geomNormal, const unsigned int
//                     primID, const int materialID, const rayTracingData<T>
//                     *globalData, rayRNG &Rng) override final {
//     auto direction = rayReflectionSpecular<T>(rayDir, geomNormal);
//     return {stickingProbability, direction};
//   }
//   void initNew(rayRNG &RNG) override final {}
//   T getSourceDistributionPower() const override final { return
//   cosineExponent; }
//   std::vector<std::string> getLocalDataLabels() const override final {
//     return {dataLabel};
//   }
// private:
//   const T stickingProbability = 1.;
//   const T cosineExponent = 1.;
//   const std::string dataLabel = "flux";
// };
// psVelocityField
// class PyVelocityField : public psVelocityField<T> {
//   using psVelocityField<T>::psVelocityField;
// public:
//   T getScalarVelocity(const std::array<T, 3> &coordinate, int material,
//                       const std::array<T, 3> &normalVector,
//                       unsigned long pointId) override {
//     PYBIND11_OVERRIDE(T, psVelocityField<T>, getScalarVelocity, coordinate,
//                       material, normalVector, pointId);
//   }
//   // if we declare a typedef for std::array<T,3>, we will no longer get this
//   // error: the compiler doesn't understand why std::array gets 2 template
//   // arguments
//   // add template argument as the preprocessor becomes confused with the
//   comma
//   // in std::array<T, 3>
//   typedef std::array<T, 3> arrayType;
//   std::array<T, 3> getVectorVelocity(const std::array<T, 3> &coordinate,
//                                      int material,
//                                      const std::array<T, 3> &normalVector,
//                                      unsigned long pointId) override {
//     PYBIND11_OVERRIDE(
//         arrayType, // add template argument here, as the preprocessor becomes
//                    // confused with the comma in std::array<T, 3>
//         psVelocityField<T>, getVectorVelocity, coordinate, material,
//         normalVector, pointId);
//   }
//   T getDissipationAlpha(int direction, int material,
//                         const std::array<T, 3> &centralDifferences) override
//                         {
//     PYBIND11_OVERRIDE(T, psVelocityField<T>, getDissipationAlpha, direction,
//                       material, centralDifferences);
//   }
//   void setVelocities(psSmartPointer<std::vector<T>> passedVelocities)
//   override {
//     PYBIND11_OVERRIDE(void, psVelocityField<T>, setVelocities,
//                       passedVelocities);
//   }
//   int getTranslationFieldOptions() const override {
//     PYBIND11_OVERRIDE(int, psVelocityField<T>, getTranslationFieldOptions, );
//   }
// };
// a function to declare GeometricDistributionModel of type DistType
// template <typename NumericType, int D, typename DistType>
// void declare_GeometricDistributionModel(pybind11::module &m,
//                                         const std::string &typestr) {
//   using Class = psGeometricDistributionModel<NumericType, D, DistType>;
//   pybind11::class_<Class, psSmartPointer<Class>>(m, typestr.c_str())
//       .def(pybind11::init<psSmartPointer<DistType>>(), pybind11::arg("dist"))
//       .def(pybind11::init<psSmartPointer<DistType>,
//                           psSmartPointer<lsDomain<NumericType, D>>>(),
//            pybind11::arg("dist"), pybind11::arg("mask"))
//       .def("apply", &Class::apply);
// }

PYBIND11_MODULE(VIENNAPS_MODULE_NAME, module) {
  module.doc() =
      "ViennaPS is a header-only C++ process simulation library which "
      "includes surface and volume representations, a ray tracer, and physical "
      "models for the simulation of microelectronic fabrication processes. The "
      "main design goals are simplicity and efficiency, tailored towards "
      "scientific simulations.";

  // set version string of python module
  module.attr("__version__") = VIENNAPS_MODULE_VERSION;

  // set dimension
  module.attr("D") = D;

  // wrap omp_set_num_threads to control number of threads
  module.def("setNumThreads", &omp_set_num_threads);

  /****************************************************************************
   *                               MODEL FRAMEWORK                            *
   ****************************************************************************/

  // psProcessModel
  pybind11::class_<psProcessModel<T, D>, psSmartPointer<psProcessModel<T, D>>>
      processModel(module, "ProcessModel");

  // constructors
  processModel
      .def(pybind11::init<>())
      // methods
      .def("setProcessName", &psProcessModel<T, D>::setProcessName)
      .def("getProcessName", &psProcessModel<T, D>::getProcessName)
      .def("getSurfaceModel", &psProcessModel<T, D>::getSurfaceModel)
      .def("getAdvectionCallback", &psProcessModel<T, D>::getAdvectionCallback)
      .def("getGeometricModel", &psProcessModel<T, D>::getGeometricModel)
      .def("getVelocityField", &psProcessModel<T, D>::getVelocityField)
      .def("getParticleLogSize", &psProcessModel<T, D>::getParticleLogSize)
      .def("getParticleTypes",
           [](psProcessModel<T, D> &pm) {
             // Get smart pointer to vector of unique_ptr from the process
             // model
             auto unique_ptrs_sp = pm.getParticleTypes();

             // Dereference the smart pointer to access the vector
             auto &unique_ptrs = *unique_ptrs_sp;

             // Create vector to hold shared_ptr
             std::vector<std::shared_ptr<rayAbstractParticle<T>>> shared_ptrs;

             // Loop over unique_ptrs and create shared_ptrs from them
             for (auto &uptr : unique_ptrs) {
               shared_ptrs.push_back(
                   std::shared_ptr<rayAbstractParticle<T>>(uptr.release()));
             }

             // Return the new vector of shared_ptr
             return shared_ptrs;
           })
      .def("setSurfaceModel",
           [](psProcessModel<T, D> &pm, psSmartPointer<psSurfaceModel<T>> &sm) {
             pm.setSurfaceModel(sm);
           })
      .def("setAdvectionCallback",
           [](psProcessModel<T, D> &pm,
              psSmartPointer<psAdvectionCallback<T, D>> &ac) {
             pm.setAdvectionCallback(ac);
           })
      .def("insertNextParticleType",
           [](psProcessModel<T, D> &pm,
              psSmartPointer<psParticle<D>> &passedParticle) {
             if (passedParticle) {
               auto particle =
                   std::make_unique<psParticle<D>>(*passedParticle.get());
               pm.insertNextParticleType(particle);
             }
           })
      // IMPORTANT: here it may be needed to write this function for any
      // type of passed Particle
      .def("setGeometricModel",
           [](psProcessModel<T, D> &pm,
              psSmartPointer<psGeometricModel<T, D>> &gm) {
             pm.setGeometricModel(gm);
           })
      .def(
          "setVelocityField",
          [](psProcessModel<T, D> &pm, psSmartPointer<psVelocityField<T>> &vf) {
            pm.setVelocityField<psVelocityField<T>>(vf);
          })
      .def("setPrimaryDirection", &psProcessModel<T, D>::setPrimaryDirection)
      .def("getPrimaryDirection", &psProcessModel<T, D>::getPrimaryDirection);

  // psAdvectionCallback
  pybind11::class_<psAdvectionCallback<T, D>,
                   psSmartPointer<psAdvectionCallback<T, D>>,
                   PyAdvectionCallback>(module, "AdvectionCallback")
      // constructors
      .def(pybind11::init<>())
      // methods
      .def("applyPreAdvect", &psAdvectionCallback<T, D>::applyPreAdvect)
      .def("applyPostAdvect", &psAdvectionCallback<T, D>::applyPostAdvect)
      .def_readwrite("domain", &PyAdvectionCallback::domain);

  // psProcessParams
  pybind11::class_<psProcessParams<T>, psSmartPointer<psProcessParams<T>>>(
      module, "ProcessParams")
      .def(pybind11::init<>())
      .def("insertNextScalar", &psProcessParams<T>::insertNextScalar)
      .def("getScalarData", (T & (psProcessParams<T>::*)(int)) &
                                psProcessParams<T>::getScalarData)
      .def("getScalarData", (const T &(psProcessParams<T>::*)(int) const) &
                                psProcessParams<T>::getScalarData)
      .def("getScalarData", (T & (psProcessParams<T>::*)(std::string)) &
                                psProcessParams<T>::getScalarData)
      .def("getScalarDataIndex", &psProcessParams<T>::getScalarDataIndex)
      .def("getScalarData", (std::vector<T> & (psProcessParams<T>::*)()) &
                                psProcessParams<T>::getScalarData)
      .def("getScalarData",
           (const std::vector<T> &(psProcessParams<T>::*)() const) &
               psProcessParams<T>::getScalarData)
      .def("getScalarDataLabel", &psProcessParams<T>::getScalarDataLabel);

  // psSurfaceModel
  //   pybind11::class_<psSurfaceModel<T>, psSmartPointer<psSurfaceModel<T>>,
  //                    PypsSurfaceModel>(module, "SurfaceModel")
  //       .def(pybind11::init<>())
  //       .def("initializeCoverages", &psSurfaceModel<T>::initializeCoverages)
  //       .def("initializeProcessParameters",
  //            &psSurfaceModel<T>::initializeProcessParameters)
  //       .def("getCoverages", &psSurfaceModel<T>::getCoverages)
  //       .def("getProcessParameters",
  //       &psSurfaceModel<T>::getProcessParameters) .def("calculateVelocities",
  //       &psSurfaceModel<T>::calculateVelocities) .def("updateCoverages",
  //       &psSurfaceModel<T>::updateCoverages);
  // psVelocityField
  //   pybind11::class_<psVelocityField<T>, psSmartPointer<psVelocityField<T>>,
  //                    PyVelocityField>
  //       velocityField(module, "VelocityField");
  //   // constructors
  //   velocityField
  //       .def(pybind11::init<>())
  //       // methods
  //       .def("getScalarVelocity", &psVelocityField<T>::getScalarVelocity)
  //       .def("getVectorVelocity", &psVelocityField<T>::getVectorVelocity)
  //       .def("getDissipationAlpha", &psVelocityField<T>::getDissipationAlpha)
  //       .def("getTranslationFieldOptions",
  //            &psVelocityField<T>::getTranslationFieldOptions)
  //       .def("setVelocities", &psVelocityField<T>::setVelocities);
  //   pybind11::class_<psDefaultVelocityField<T>,
  //                    psSmartPointer<psDefaultVelocityField<T>>>(
  //       module, "DefaultVelocityField", velocityField)
  //       // constructors
  //       .def(pybind11::init<>())
  //       // methods
  //       .def("getScalarVelocity",
  //       &psDefaultVelocityField<T>::getScalarVelocity)
  //       .def("getVectorVelocity",
  //       &psDefaultVelocityField<T>::getVectorVelocity)
  //       .def("getDissipationAlpha",
  //            &psDefaultVelocityField<T>::getDissipationAlpha)
  //       .def("getTranslationFieldOptions",
  //            &psDefaultVelocityField<T>::getTranslationFieldOptions)
  //       .def("setVelocities", &psDefaultVelocityField<T>::setVelocities);

  // Shim to instantiate the particle class
  pybind11::class_<psParticle<D>, psSmartPointer<psParticle<D>>> particle(
      module, "Particle");
  particle.def("surfaceCollision", &psParticle<D>::surfaceCollision)
      .def("surfaceReflection", &psParticle<D>::surfaceReflection)
      .def("initNew", &psParticle<D>::initNew)
      .def("getLocalDataLabels", &psParticle<D>::getLocalDataLabels)
      .def("getSourceDistributionPower",
           &psParticle<D>::getSourceDistributionPower);

  // predefined particles
  //   pybind11::class_<psDiffuseParticle<D>,
  //   psSmartPointer<psDiffuseParticle<D>>>(
  //       module, "DiffuseParticle", particle)
  //       .def(pybind11::init(
  //                &psSmartPointer<psDiffuseParticle<D>>::New<const T, const T,
  //                                                           const std::string
  //                                                           &>),
  //            pybind11::arg("stickingProbability") = 1.0,
  //            pybind11::arg("cosineExponent") = 1.,
  //            pybind11::arg("dataLabel") = "flux")
  //       .def("surfaceCollision", &psDiffuseParticle<D>::surfaceCollision)
  //       .def("surfaceReflection", &psDiffuseParticle<D>::surfaceReflection)
  //       .def("initNew", &psDiffuseParticle<D>::initNew)
  //       .def("getLocalDataLabels", &psDiffuseParticle<D>::getLocalDataLabels)
  //       .def("getSourceDistributionPower",
  //            &psDiffuseParticle<D>::getSourceDistributionPower);
  //   pybind11::class_<psSpecularParticle, psSmartPointer<psSpecularParticle>>(
  //       module, "SpecularParticle", particle)
  //       .def(pybind11::init(
  //                &psSmartPointer<psSpecularParticle>::New<const T, const T,
  //                                                         const std::string
  //                                                         &>),
  //            pybind11::arg("stickingProbability") = 1.0,
  //            pybind11::arg("cosineExponent") = 1.,
  //            pybind11::arg("dataLabel") = "flux")
  //       .def("surfaceCollision", &psSpecularParticle::surfaceCollision)
  //       .def("surfaceReflection", &psSpecularParticle::surfaceReflection)
  //       .def("initNew", &psSpecularParticle::initNew)
  //       .def("getLocalDataLabels", &psSpecularParticle::getLocalDataLabels)
  //       .def("getSourceDistributionPower",
  //            &psSpecularParticle::getSourceDistributionPower);

  // ***************************************************************************
  //                                  MODELS
  // ***************************************************************************

  // Enum psMaterial
  pybind11::enum_<psMaterial>(module, "Material")
      .value("Undefined", psMaterial::None) // 1
      .value("Mask", psMaterial::Mask)
      .value("Si", psMaterial::Si)
      .value("SiO2", psMaterial::SiO2)
      .value("Si3N4", psMaterial::Si3N4) // 5
      .value("SiN", psMaterial::SiN)
      .value("SiON", psMaterial::SiON)
      .value("SiC", psMaterial::SiC)
      .value("SiGe", psMaterial::SiGe)
      .value("PolySi", psMaterial::PolySi) // 10
      .value("GaN", psMaterial::GaN)
      .value("W", psMaterial::W)
      .value("Al2O3", psMaterial::Al2O3)
      .value("TiN", psMaterial::TiN)
      .value("Cu", psMaterial::Cu) // 15
      .value("Polymer", psMaterial::Polymer)
      .value("Dielectric", psMaterial::Dielectric)
      .value("Metal", psMaterial::Metal)
      .value("Air", psMaterial::Air)
      .value("GAS", psMaterial::GAS) // 20
      .export_values();

  // Single Particle Process
  pybind11::class_<psSingleParticleProcess<T, D>,
                   psSmartPointer<psSingleParticleProcess<T, D>>>(
      module, "SingleParticleProcess", processModel)
      .def(pybind11::init([](const T rate, const T sticking, const T power,
                             const psMaterial mask) {
             return psSmartPointer<psSingleParticleProcess<T, D>>::New(
                 rate, sticking, power, mask);
           }),
           pybind11::arg("rate") = 1.,
           pybind11::arg("stickingProbability") = 1.,
           pybind11::arg("sourceExponent") = 1.,
           pybind11::arg("maskMaterial") = psMaterial::None);

  // TEOS Deposition
  pybind11::class_<psTEOSDeposition<T, D>,
                   psSmartPointer<psTEOSDeposition<T, D>>>(
      module, "TEOSDeposition", processModel)
      .def(pybind11::init(
               &psSmartPointer<psTEOSDeposition<T, D>>::New<
                   const T /*st1*/, const T /*rate1*/, const T /*order1*/,
                   const T /*st2*/, const T /*rate2*/, const T /*order2*/>),
           pybind11::arg("stickingProbabilityP1"), pybind11::arg("rateP1"),
           pybind11::arg("orderP1"),
           pybind11::arg("stickingProbabilityP2") = 0.,
           pybind11::arg("rateP2") = 0., pybind11::arg("orderP2") = 0.);

  // SF6O2 Etching
  pybind11::class_<psSF6O2Etching<T, D>, psSmartPointer<psSF6O2Etching<T, D>>>(
      module, "SF6O2Etching", processModel)
      .def(pybind11::init(
               &psSmartPointer<psSF6O2Etching<T, D>>::New<
                   const double /*ionFlux*/, const double /*etchantFlux*/,
                   const double /*oxygenFlux*/, const T /*meanIonEnergy*/,
                   const T /*sigmaIonEnergy*/, const T /*ionExponent*/,
                   const T /*oxySputterYield*/, const T /*etchStopDepth*/>),
           pybind11::arg("ionFlux"), pybind11::arg("etchantFlux"),
           pybind11::arg("oxygenFlux"), pybind11::arg("meanIonEnergy") = 100.,
           pybind11::arg("sigmaIonEnergy") = 10.,
           pybind11::arg("ionExponent") = 100.,
           pybind11::arg("oxySputterYield") = 3.,
           pybind11::arg("etchStopDepth") = std::numeric_limits<T>::lowest());

  // Fluorocarbon Etching
  pybind11::class_<psFluorocarbonEtching<T, D>,
                   psSmartPointer<psFluorocarbonEtching<T, D>>>(
      module, "FluorocarbonEtching", processModel)
      .def(pybind11::init<>())
      .def(
          pybind11::init(&psSmartPointer<psFluorocarbonEtching<T, D>>::New<
                         const double /*ionFlux*/, const double /*etchantFlux*/,
                         const double /*polyFlux*/, T /*meanEnergy*/,
                         const T /*sigmaEnergy*/, const T /*ionExponent*/,
                         const T /*deltaP*/, const T /*etchStopDepth*/>),
          pybind11::arg("ionFlux"), pybind11::arg("etchantFlux"),
          pybind11::arg("polyFlux"), pybind11::arg("meanIonEnergy") = 100.,
          pybind11::arg("sigmaIonEnergy") = 10.,
          pybind11::arg("ionExponent") = 100., pybind11::arg("deltaP") = 0.,
          pybind11::arg("etchStopDepth") = std::numeric_limits<T>::lowest())
      .def(pybind11::init(&psSmartPointer<psFluorocarbonEtching<T, D>>::New<
                          const FluorocarbonImplementation::Parameters &>),
           pybind11::arg("parameters"))
      .def("setParameters", &psFluorocarbonEtching<T, D>::setParameters)
      .def("getParameters", &psFluorocarbonEtching<T, D>::getParameters,
           pybind11::return_value_policy::reference);

  // Fluorocarbon Parameters
  pybind11::class_<FluorocarbonImplementation::Parameters::MaskType>(
      module, "FluorocarbonParametersMask")
      .def(pybind11::init<>())
      .def_readwrite("rho",
                     &FluorocarbonImplementation::Parameters::MaskType::rho)
      .def_readwrite("beta_p",
                     &FluorocarbonImplementation::Parameters::MaskType::beta_p)
      .def_readwrite("beta_e",
                     &FluorocarbonImplementation::Parameters::MaskType::beta_e)
      .def_readwrite("A_sp",
                     &FluorocarbonImplementation::Parameters::MaskType::A_sp)
      .def_readwrite("B_sp",
                     &FluorocarbonImplementation::Parameters::MaskType::B_sp)
      .def_readwrite("Eth_sp",
                     &FluorocarbonImplementation::Parameters::MaskType::Eth_sp);

  pybind11::class_<FluorocarbonImplementation::Parameters::SiO2Type>(
      module, "FluorocarbonParametersSiO2")
      .def(pybind11::init<>())
      .def_readwrite("rho",
                     &FluorocarbonImplementation::Parameters::SiO2Type::rho)
      .def_readwrite("E_a",
                     &FluorocarbonImplementation::Parameters::SiO2Type::E_a)
      .def_readwrite("K", &FluorocarbonImplementation::Parameters::SiO2Type::K)
      .def_readwrite("A_sp",
                     &FluorocarbonImplementation::Parameters::SiO2Type::A_sp)
      .def_readwrite("B_sp",
                     &FluorocarbonImplementation::Parameters::SiO2Type::B_sp)
      .def_readwrite("Eth_ie",
                     &FluorocarbonImplementation::Parameters::SiO2Type::Eth_ie)
      .def_readwrite("Eth_sp",
                     &FluorocarbonImplementation::Parameters::SiO2Type::Eth_sp)
      .def_readwrite("A_ie",
                     &FluorocarbonImplementation::Parameters::SiO2Type::A_ie);

  pybind11::class_<FluorocarbonImplementation::Parameters::Si3N4Type>(
      module, "FluorocarbonParametersSi3N4")
      .def(pybind11::init<>())
      .def_readwrite("rho",
                     &FluorocarbonImplementation::Parameters::Si3N4Type::rho)
      .def_readwrite("E_a",
                     &FluorocarbonImplementation::Parameters::Si3N4Type::E_a)
      .def_readwrite("K", &FluorocarbonImplementation::Parameters::Si3N4Type::K)
      .def_readwrite("A_sp",
                     &FluorocarbonImplementation::Parameters::Si3N4Type::A_sp)
      .def_readwrite("B_sp",
                     &FluorocarbonImplementation::Parameters::Si3N4Type::B_sp)
      .def_readwrite("Eth_ie",
                     &FluorocarbonImplementation::Parameters::Si3N4Type::Eth_ie)
      .def_readwrite("Eth_sp",
                     &FluorocarbonImplementation::Parameters::Si3N4Type::Eth_sp)
      .def_readwrite("A_ie",
                     &FluorocarbonImplementation::Parameters::Si3N4Type::A_ie);

  pybind11::class_<FluorocarbonImplementation::Parameters::SiType>(
      module, "FluorocarbonParametersSi")
      .def(pybind11::init<>())
      .def_readwrite("rho",
                     &FluorocarbonImplementation::Parameters::SiType::rho)
      .def_readwrite("E_a",
                     &FluorocarbonImplementation::Parameters::SiType::E_a)
      .def_readwrite("K", &FluorocarbonImplementation::Parameters::SiType::K)
      .def_readwrite("A_sp",
                     &FluorocarbonImplementation::Parameters::SiType::A_sp)
      .def_readwrite("B_sp",
                     &FluorocarbonImplementation::Parameters::SiType::B_sp)
      .def_readwrite("Eth_ie",
                     &FluorocarbonImplementation::Parameters::SiType::Eth_ie)
      .def_readwrite("Eth_sp",
                     &FluorocarbonImplementation::Parameters::SiType::Eth_sp)
      .def_readwrite("A_ie",
                     &FluorocarbonImplementation::Parameters::SiType::A_ie);

  pybind11::class_<FluorocarbonImplementation::Parameters::PolymerType>(
      module, "FluorocarbonParametersPolymer")
      .def(pybind11::init<>())
      .def_readwrite("rho",
                     &FluorocarbonImplementation::Parameters::PolymerType::rho)
      .def_readwrite(
          "Eth_ie",
          &FluorocarbonImplementation::Parameters::PolymerType::Eth_ie)
      .def_readwrite(
          "A_ie", &FluorocarbonImplementation::Parameters::PolymerType::A_ie);

  pybind11::class_<FluorocarbonImplementation::Parameters::IonType>(
      module, "FluorocarbonParametersIons")
      .def(pybind11::init<>())
      .def_readwrite(
          "meanEnergy",
          &FluorocarbonImplementation::Parameters::IonType::meanEnergy)
      .def_readwrite(
          "sigmaEnergy",
          &FluorocarbonImplementation::Parameters::IonType::sigmaEnergy)
      .def_readwrite("exponent",
                     &FluorocarbonImplementation::Parameters::IonType::exponent)
      .def_readwrite(
          "inflectAngle",
          &FluorocarbonImplementation::Parameters::IonType::inflectAngle)
      .def_readwrite("n_l",
                     &FluorocarbonImplementation::Parameters::IonType::n_l)
      .def_readwrite(
          "minAngle",
          &FluorocarbonImplementation::Parameters::IonType::minAngle);

  pybind11::class_<FluorocarbonImplementation::Parameters>(
      module, "FluorocarbonParameters")
      .def(pybind11::init<>())
      .def_readwrite("ionFlux",
                     &FluorocarbonImplementation::Parameters::ionFlux)
      .def_readwrite("etchantFlux",
                     &FluorocarbonImplementation::Parameters::etchantFlux)
      .def_readwrite("polyFlux",
                     &FluorocarbonImplementation::Parameters::polyFlux)
      .def_readwrite("delta_p",
                     &FluorocarbonImplementation::Parameters::delta_p)
      .def_readwrite("etchStopDepth",
                     &FluorocarbonImplementation::Parameters::etchStopDepth)
      .def_readwrite("Mask", &FluorocarbonImplementation::Parameters::Mask)
      .def_readwrite("SiO2", &FluorocarbonImplementation::Parameters::SiO2)
      .def_readwrite("Si3N4", &FluorocarbonImplementation::Parameters::Si3N4)
      .def_readwrite("Si", &FluorocarbonImplementation::Parameters::Si)
      .def_readwrite("Polymer",
                     &FluorocarbonImplementation::Parameters::Polymer)
      .def_readwrite("Ions", &FluorocarbonImplementation::Parameters::Ions);

  // Isotropic Process
  pybind11::class_<psIsotropicProcess<T, D>,
                   psSmartPointer<psIsotropicProcess<T, D>>>(
      module, "IsotropicProcess", processModel)
      .def(pybind11::init([](const T rate, const psMaterial mask) {
             return psSmartPointer<psIsotropicProcess<T, D>>::New(rate, mask);
           }),
           pybind11::arg("rate") = 1.,
           pybind11::arg("maskMaterial") = psMaterial::Mask)
      .def(pybind11::init([](const T rate, const std::vector<psMaterial> mask) {
             return psSmartPointer<psIsotropicProcess<T, D>>::New(rate, mask);
           }),
           pybind11::arg("rate"), pybind11::arg("maskMaterial"));

  // Directional Etching
  pybind11::class_<psDirectionalEtching<T, D>,
                   psSmartPointer<psDirectionalEtching<T, D>>>(
      module, "DirectionalEtching", processModel)
      .def(pybind11::init<const std::array<T, 3> &, const T, const T,
                          const psMaterial>(),
           pybind11::arg("direction"),
           pybind11::arg("directionalVelocity") = 1.,
           pybind11::arg("isotropicVelocity") = 0.,
           pybind11::arg("maskMaterial") = psMaterial::Mask)
      .def(pybind11::init<const std::array<T, 3> &, const T, const T,
                          const std::vector<psMaterial>>(),
           pybind11::arg("direction"), pybind11::arg("directionalVelocity"),
           pybind11::arg("isotropicVelocity"), pybind11::arg("maskMaterial"));

  // Sphere Distribution
  pybind11::class_<psSphereDistribution<T, D>,
                   psSmartPointer<psSphereDistribution<T, D>>>(
      module, "SphereDistribution", processModel)
      .def(pybind11::init([](const T radius, const T gridDelta,
                             psSmartPointer<lsDomain<T, D>> mask) {
             return psSmartPointer<psSphereDistribution<T, D>>::New(
                 radius, gridDelta, mask);
           }),
           pybind11::arg("radius"), pybind11::arg("gridDelta"),
           pybind11::arg("mask"))
      .def(pybind11::init([](const T radius, const T gridDelta) {
             return psSmartPointer<psSphereDistribution<T, D>>::New(
                 radius, gridDelta, nullptr);
           }),
           pybind11::arg("radius"), pybind11::arg("gridDelta"));

  // Box Distribution
  pybind11::class_<psBoxDistribution<T, D>,
                   psSmartPointer<psBoxDistribution<T, D>>>(
      module, "BoxDistribution", processModel)
      .def(
          pybind11::init([](const std::array<T, 3> &halfAxes, const T gridDelta,
                            psSmartPointer<lsDomain<T, D>> mask) {
            return psSmartPointer<psBoxDistribution<T, D>>::New(
                halfAxes, gridDelta, mask);
          }),
          pybind11::arg("halfAxes"), pybind11::arg("gridDelta"),
          pybind11::arg("mask"))
      .def(pybind11::init(
               [](const std::array<T, 3> &halfAxes, const T gridDelta) {
                 return psSmartPointer<psBoxDistribution<T, D>>::New(
                     halfAxes, gridDelta, nullptr);
               }),
           pybind11::arg("halfAxes"), pybind11::arg("gridDelta"));

  // Plasma Damage
  pybind11::class_<psPlasmaDamage<T, D>, psSmartPointer<psPlasmaDamage<T, D>>>(
      module, "PlasmaDamage", processModel)
      .def(pybind11::init([](const T ionEnergy, const T meanFreePath,
                             const psMaterial mask) {
             return psSmartPointer<psPlasmaDamage<T, D>>::New(
                 ionEnergy, meanFreePath, mask);
           }),
           pybind11::arg("ionEnergy") = 100.,
           pybind11::arg("meanFreePath") = 1.,
           pybind11::arg("maskMaterial") = psMaterial::None);

  // Oxide Regrowth
  pybind11::class_<psOxideRegrowth<T, D>,
                   psSmartPointer<psOxideRegrowth<T, D>>>(
      module, "OxideRegrowth", processModel)
      .def(
          pybind11::init(&psSmartPointer<psOxideRegrowth<T, D>>::New<
                         const T, const T, const T, const T, const T, const T,
                         const T, const T, const T, const T, const T, const T>),
          pybind11::arg("nitrideEtchRate"), pybind11::arg("oxideEtchRate"),
          pybind11::arg("redepositionRate"),
          pybind11::arg("redepositionThreshold"),
          pybind11::arg("redepositionTimeInt"),
          pybind11::arg("diffusionCoefficient"), pybind11::arg("sinkStrength"),
          pybind11::arg("scallopVelocity"), pybind11::arg("centerVelocity"),
          pybind11::arg("topHeight"), pybind11::arg("centerWidth"),
          pybind11::arg("stabilityFactor"));

  // Anisotropic Process
  pybind11::class_<psAnisotropicProcess<T, D>,
                   psSmartPointer<psAnisotropicProcess<T, D>>>(
      module, "AnisotropicProcess", processModel)
      .def(pybind11::init(&psSmartPointer<psAnisotropicProcess<T, D>>::New<
                          const std::vector<std::pair<psMaterial, T>>>),
           pybind11::arg("materials"))
      .def(pybind11::init(&psSmartPointer<psAnisotropicProcess<T, D>>::New<
                          const std::array<T, 3> &, const std::array<T, 3> &,
                          const T, const T, const T, const T,
                          const std::vector<std::pair<psMaterial, T>>>),
           pybind11::arg("direction100"), pybind11::arg("direction010"),
           pybind11::arg("rate100"), pybind11::arg("rate110"),
           pybind11::arg("rate111"), pybind11::arg("rate311"),
           pybind11::arg("materials"));

  // Atomic Layer Process
  pybind11::class_<psAtomicLayerProcess<T, D>,
                   psSmartPointer<psAtomicLayerProcess<T, D>>>(
      module, "AtomicLayerProcess")
      .def(pybind11::init<DomainType, const bool>(), pybind11::arg("domain"),
           pybind11::arg("etch") = false)
      .def("setFirstPrecursor",
           pybind11::overload_cast<std::string, T, T, T, T, T>(
               &psAtomicLayerProcess<T, D>::setFirstPrecursor))
      .def("setFirstPrecursor",
           pybind11::overload_cast<
               const psAtomicLayerProcess<T, D>::Precursor &>(
               &psAtomicLayerProcess<T, D>::setFirstPrecursor))
      .def("setSecondPrecursor",
           pybind11::overload_cast<std::string, T, T, T, T, T>(
               &psAtomicLayerProcess<T, D>::setSecondPrecursor))
      .def("setSecondPrecursor",
           pybind11::overload_cast<
               const psAtomicLayerProcess<T, D>::Precursor &>(
               &psAtomicLayerProcess<T, D>::setSecondPrecursor))
      .def("setPurgeParameters",
           &psAtomicLayerProcess<T, D>::setPurgeParameters)
      .def("setReactionOrder", &psAtomicLayerProcess<T, D>::setReactionOrder)
      .def("setMaxLambda", &psAtomicLayerProcess<T, D>::setMaxLambda)
      .def("setStabilityFactor",
           &psAtomicLayerProcess<T, D>::setStabilityFactor)
      .def("setMaxTimeStep", &psAtomicLayerProcess<T, D>::setMaxTimeStep)
      .def("setPrintInterval", &psAtomicLayerProcess<T, D>::setPrintInterval)
      .def("apply", &psAtomicLayerProcess<T, D>::apply);

  pybind11::class_<psAtomicLayerProcess<T, D>::Precursor>(module, "Precursor")
      .def(pybind11::init<>())
      .def_readwrite("name", &psAtomicLayerProcess<T, D>::Precursor::name)
      .def_readwrite(
          "meanThermalVelocity",
          &psAtomicLayerProcess<T, D>::Precursor::meanThermalVelocity)
      .def_readwrite("adsorptionRate",
                     &psAtomicLayerProcess<T, D>::Precursor::adsorptionRate)
      .def_readwrite("desorptionRate",
                     &psAtomicLayerProcess<T, D>::Precursor::desorptionRate)
      .def_readwrite("duration",
                     &psAtomicLayerProcess<T, D>::Precursor::duration)
      .def_readwrite("inFlux", &psAtomicLayerProcess<T, D>::Precursor::inFlux);

  // ***************************************************************************
  //                               GEOMETRIES
  // ***************************************************************************

  // psMakePlane
  pybind11::class_<psMakePlane<T, D>, psSmartPointer<psMakePlane<T, D>>>(
      module, "MakePlane")
      .def(pybind11::init([](DomainType Domain, const T GridDelta,
                             const T XExtent, const T YExtent, const T Height,
                             const bool Periodic, const psMaterial Material) {
             return psSmartPointer<psMakePlane<T, D>>::New(
                 Domain, GridDelta, XExtent, YExtent, Height, Periodic,
                 Material);
           }),
           pybind11::arg("domain"), pybind11::arg("gridDelta"),
           pybind11::arg("xExtent"), pybind11::arg("yExtent"),
           pybind11::arg("height") = 0.,
           pybind11::arg("periodicBoundary") = false,
           pybind11::arg("material") = psMaterial::None)
      .def(pybind11::init(
               [](DomainType Domain, T Height, const psMaterial Material) {
                 return psSmartPointer<psMakePlane<T, D>>::New(Domain, Height,
                                                               Material);
               }),
           pybind11::arg("domain"), pybind11::arg("height") = 0.,
           pybind11::arg("material") = psMaterial::None)
      .def("apply", &psMakePlane<T, D>::apply,
           "Create a plane geometry or add plane to existing geometry.");

  // psMakeTrench
  pybind11::class_<psMakeTrench<T, D>, psSmartPointer<psMakeTrench<T, D>>>(
      module, "MakeTrench")
      .def(pybind11::init([](DomainType Domain, const T GridDelta,
                             const T XExtent, const T YExtent,
                             const T TrenchWidth, const T TrenchDepth,
                             const T TaperingAngle, const T BaseHeight,
                             const bool PeriodicBoundary, const bool MakeMask,
                             const psMaterial Material) {
             return psSmartPointer<psMakeTrench<T, D>>::New(
                 Domain, GridDelta, XExtent, YExtent, TrenchWidth, TrenchDepth,
                 TaperingAngle, BaseHeight, PeriodicBoundary, MakeMask,
                 Material);
           }),
           pybind11::arg("domain"), pybind11::arg("gridDelta"),
           pybind11::arg("xExtent"), pybind11::arg("yExtent"),
           pybind11::arg("trenchWidth"), pybind11::arg("trenchDepth"),
           pybind11::arg("taperingAngle") = 0.,
           pybind11::arg("baseHeight") = 0.,
           pybind11::arg("periodicBoundary") = false,
           pybind11::arg("makeMask") = false,
           pybind11::arg("material") = psMaterial::None)
      .def("apply", &psMakeTrench<T, D>::apply, "Create a trench geometry.");

  // psMakeHole
  pybind11::class_<psMakeHole<T, D>, psSmartPointer<psMakeHole<T, D>>>(
      module, "MakeHole")
      .def(pybind11::init([](DomainType domain, const T GridDelta,
                             const T xExtent, const T yExtent,
                             const T HoleRadius, const T HoleDepth,
                             const T TaperingAngle, const T BaseHeight,
                             const bool PeriodicBoundary, const bool MakeMask,
                             const psMaterial material) {
             return psSmartPointer<psMakeHole<T, D>>::New(
                 domain, GridDelta, xExtent, yExtent, HoleRadius, HoleDepth,
                 TaperingAngle, BaseHeight, PeriodicBoundary, MakeMask,
                 material);
           }),
           pybind11::arg("domain"), pybind11::arg("gridDelta"),
           pybind11::arg("xExtent"), pybind11::arg("yExtent"),
           pybind11::arg("holeRadius"), pybind11::arg("holeDepth"),
           pybind11::arg("taperingAngle") = 0.,
           pybind11::arg("baseHeight") = 0.,
           pybind11::arg("periodicBoundary") = false,
           pybind11::arg("makeMask") = false,
           pybind11::arg("material") = psMaterial::None)
      .def("apply", &psMakeHole<T, D>::apply, "Create a hole geometry.");

  // psMakeFin
  pybind11::class_<psMakeFin<T, D>, psSmartPointer<psMakeFin<T, D>>>(module,
                                                                     "MakeFin")
      .def(pybind11::init([](DomainType domain, const T gridDelta,
                             const T xExtent, const T yExtent, const T finWidth,
                             const T finHeight, const T taperAngle,
                             const T baseHeight, const bool periodicBoundary,
                             const bool makeMask, const psMaterial material) {
             return psSmartPointer<psMakeFin<T, D>>::New(
                 domain, gridDelta, xExtent, yExtent, finWidth, finHeight,
                 taperAngle, baseHeight, periodicBoundary, makeMask, material);
           }),
           pybind11::arg("domain"), pybind11::arg("gridDelta"),
           pybind11::arg("xExtent"), pybind11::arg("yExtent"),
           pybind11::arg("finWidth"), pybind11::arg("finHeight"),
           pybind11::arg("taperAngle") = 0., pybind11::arg("baseHeight") = 0.,
           pybind11::arg("periodicBoundary") = false,
           pybind11::arg("makeMask") = false,
           pybind11::arg("material") = psMaterial::None)
      .def("apply", &psMakeFin<T, D>::apply, "Create a fin geometry.");

  // psMakeStack
  pybind11::class_<psMakeStack<T, D>, psSmartPointer<psMakeStack<T, D>>>(
      module, "MakeStack")
      .def(pybind11::init(&psSmartPointer<psMakeStack<T, D>>::New<
                          DomainType &, const T /*gridDelta*/,
                          const T
                          /*xExtent*/,
                          const T /*yExtent*/,
                          const int
                          /*numLayers*/,
                          const T /*layerHeight*/,
                          const T
                          /*substrateHeight*/,
                          const T /*holeRadius*/,
                          const T
                          /*trenchWidth*/,
                          const T /*maskHeight*/, const bool
                          /*PeriodicBoundary*/>),
           pybind11::arg("domain"), pybind11::arg("gridDelta"),
           pybind11::arg("xExtent"), pybind11::arg("yExtent"),
           pybind11::arg("numLayers"), pybind11::arg("layerHeight"),
           pybind11::arg("substrateHeight"), pybind11::arg("holeRadius"),
           pybind11::arg("trenchWidth"), pybind11::arg("maskHeight"),
           pybind11::arg("periodicBoundary") = false)
      .def("apply", &psMakeStack<T, D>::apply,
           "Create a stack of alternating SiO2 and Si3N4 layers.")
      .def("getTopLayer", &psMakeStack<T, D>::getTopLayer,
           "Returns the number of layers included in the stack")
      .def("getHeight", &psMakeStack<T, D>::getHeight,
           "Returns the total height of the stack.");

  // ***************************************************************************
  //                                 PROCESS
  // ***************************************************************************

  // rayTraceDirection Enum
  pybind11::enum_<rayTraceDirection>(module, "rayTraceDirection")
      .value("POS_X", rayTraceDirection::POS_X)
      .value("POS_Y", rayTraceDirection::POS_Y)
      .value("POS_Z", rayTraceDirection::POS_Z)
      .value("NEG_X", rayTraceDirection::NEG_X)
      .value("NEG_Y", rayTraceDirection::NEG_Y)
      .value("NEG_Z", rayTraceDirection::NEG_Z)
      .export_values();

  // psProcess
  pybind11::class_<psProcess<T, D>>(module, "Process")
      // constructors
      .def(pybind11::init())
      .def(pybind11::init<DomainType>(), pybind11::arg("domain"))
      .def(
          pybind11::init<DomainType, psSmartPointer<psProcessModel<T, D>>, T>(),
          pybind11::arg("domain"), pybind11::arg("model"),
          pybind11::arg("duration"))
      // methods
      .def("apply", &psProcess<T, D>::apply, "Run the process.")
      .def("calculateFlux", &psProcess<T, D>::calculateFlux,
           "Perform a single-pass flux calculation.")
      .def("setDomain", &psProcess<T, D>::setDomain, "Set the process domain.")
      .def("setProcessModel",
           &psProcess<T, D>::setProcessModel<psProcessModel<T, D>>,
           "Set the process model. This has to be a pre-configured process "
           "model.")
      .def("setProcessDuration", &psProcess<T, D>::setProcessDuration,
           "Set the process duration.")
      .def("setSourceDirection", &psProcess<T, D>::setSourceDirection,
           "Set source direction of the process.")
      .def("setNumberOfRaysPerPoint", &psProcess<T, D>::setNumberOfRaysPerPoint,
           "Set the number of rays to traced for each particle in the process. "
           "The number is per point in the process geometry.")
      .def("setMaxCoverageInitIterations",
           &psProcess<T, D>::setMaxCoverageInitIterations,
           "Set the number of iterations to initialize the coverages.")
      .def(" setPrintTimeInterval ", &psProcess<T, D>::setPrintTimeInterval,
           "Sets the minimum time between printing intermediate results during "
           "the process. If this is set to a non-positive value, no "
           "intermediate results are printed.")
      .def("setIntegrationScheme", &psProcess<T, D>::setIntegrationScheme,
           "Set the integration scheme for solving the level-set equation. "
           "Possible integration schemes are specified in "
           "lsIntegrationSchemeEnum.")
      .def("setTimeStepRatio", &psProcess<T, D>::setTimeStepRatio,
           "Set the CFL condition to use during advection. The CFL condition "
           "sets the maximum distance a surface can be moved during one "
           "advection step. It MUST be below 0.5 to guarantee numerical "
           "stability. Defaults to 0.4999.")
      .def("enableFluxSmoothing", &psProcess<T, D>::enableFluxSmoothing,
           "Enable flux smoothing. The flux at each surface point, calculated "
           "by the ray tracer, is averaged over the surface point neighbors.")
      .def("disableFluxSmoothing", &psProcess<T, D>::disableFluxSmoothing,
           "Disable flux smoothing")
      .def("getProcessDuration", &psProcess<T, D>::getProcessDuration,
           "Returns the duration of the recently run process. This duration "
           "can sometimes slightly vary from the set process duration, due to "
           "the maximum time step according to the CFL condition.");

  pybind11::enum_<psLogLevel>(module, "LogLevel")
      .value("ERROR", psLogLevel::ERROR)
      .value("WARNING", psLogLevel::WARNING)
      .value("INFO", psLogLevel::INFO)
      .value("TIMING", psLogLevel::TIMING)
      .value("INTERMEDIATE", psLogLevel::INTERMEDIATE)
      .value("DEBUG", psLogLevel::DEBUG)
      .export_values();

  // some unexpected behaviour can happen as it is working with
  //  multithreading
  pybind11::class_<psLogger, psSmartPointer<psLogger>>(module, "Logger")
      .def_static("setLogLevel", &psLogger::setLogLevel)
      .def_static("getLogLevel", &psLogger::getLogLevel)
      .def_static("getInstance", &psLogger::getInstance,
                  pybind11::return_value_policy::reference)
      .def("addDebug", &psLogger::addDebug)
      .def("addTiming", (psLogger & (psLogger::*)(std::string, double)) &
                            psLogger::addTiming)
      .def("addTiming",
           (psLogger & (psLogger::*)(std::string, double, double)) &
               psLogger::addTiming)
      .def("addInfo", &psLogger::addInfo)
      .def("addWarning", &psLogger::addWarning)
      .def("addError", &psLogger::addError, pybind11::arg("s"),
           pybind11::arg("shouldAbort") = true)
      .def("print", [](psLogger &instance) { instance.print(std::cout); });

  // psDomain
  pybind11::class_<psDomain<T, D>, DomainType>(module, "Domain")
      // constructors
      .def(pybind11::init(&DomainType::New<>))
      // methods
      .def("insertNextLevelSet", &psDomain<T, D>::insertNextLevelSet,
           pybind11::arg("levelset"), pybind11::arg("wrapLowerLevelSet") = true,
           "Insert a level set to domain.")
      .def("insertNextLevelSetAsMaterial",
           &psDomain<T, D>::insertNextLevelSetAsMaterial,
           pybind11::arg("levelSet"), pybind11::arg("material"),
           pybind11::arg("wrapLowerLevelSet") = true,
           "Insert a level set to domain as a material.")
      .def("duplicateTopLevelSet", &psDomain<T, D>::duplicateTopLevelSet)
      .def("removeTopLevelSet", &psDomain<T, D>::removeTopLevelSet)
      .def("applyBooleanOperation", &psDomain<T, D>::applyBooleanOperation)
      .def("setMaterialMap", &psDomain<T, D>::setMaterialMap)
      .def("getMaterialMap", &psDomain<T, D>::getMaterialMap)
      .def("generateCellSet", &psDomain<T, D>::generateCellSet,
           "Generate the cell set.")
      .def("getLevelSets",
           [](psDomain<T, D> &d)
               -> std::optional<std::vector<psSmartPointer<lsDomain<T, D>>>> {
             auto levelsets = d.getLevelSets();
             if (levelsets)
               return *levelsets;
             return std::nullopt;
           })
      .def("getCellSet", &psDomain<T, D>::getCellSet, "Get the cell set.")
      .def("getGrid", &psDomain<T, D>::getGrid, "Get the grid")
      .def("print", &psDomain<T, D>::print)
      .def("saveLevelSetMesh", &psDomain<T, D>::saveLevelSetMesh,
           pybind11::arg("filename"), pybind11::arg("width") = 1,
           "Save the level set grids of layers in the domain.")
      .def("saveSurfaceMesh", &psDomain<T, D>::saveSurfaceMesh,
           pybind11::arg("filename"), pybind11::arg("addMaterialIds") = false,
           "Save the surface of the domain.")
      .def("saveVolumeMesh", &psDomain<T, D>::saveVolumeMesh,
           pybind11::arg("filename"),
           "Save the volume representation of the domain.")
      .def("saveLevelSets", &psDomain<T, D>::saveLevelSets)
      .def("clear", &psDomain<T, D>::clear);

  // psMaterialMap
  pybind11::class_<psMaterialMap, psSmartPointer<psMaterialMap>>(module,
                                                                 "MaterialMap")
      .def(pybind11::init<>())
      .def("insertNextMaterial", &psMaterialMap::insertNextMaterial,
           pybind11::arg("material") = psMaterial::None)
      .def("getMaterialAtIdx", &psMaterialMap::getMaterialAtIdx)
      .def("getMaterialMap", &psMaterialMap::getMaterialMap)
      .def("size", &psMaterialMap::size)
      .def_static("mapToMaterial", &psMaterialMap::mapToMaterial<T>,
                  "Map a float to a material.")
      .def_static("isMaterial", &psMaterialMap::isMaterial<T>);

  //   ***************************************************************************
  //                                  CELL SET
  //  ***************************************************************************

  // csDenseCellSet
  pybind11::class_<csDenseCellSet<T, D>, psSmartPointer<csDenseCellSet<T, D>>>(
      module, "DenseCellSet")
      .def(pybind11::init())
      .def("getBoundingBox", &csDenseCellSet<T, D>::getBoundingBox)
      .def(
          "addScalarData",
          [](csDenseCellSet<T, D> &cellSet, std::string name, T initValue) {
            cellSet.addScalarData(name, initValue);
            // discard return value
          },
          "Add a scalar value to be stored and modified in each cell.")
      .def("getDepth", &csDenseCellSet<T, D>::getDepth,
           "Get the depth of the cell set.")
      .def("getGridDelta", &csDenseCellSet<T, D>::getGridDelta,
           "Get the cell size.")
      .def("getNodes", &csDenseCellSet<T, D>::getNodes,
           "Get the nodes of the cell set which correspond to the corner "
           "points of the cells.")
      .def("getNode", &csDenseCellSet<T, D>::getNode,
           "Get the node at the given index.")
      .def("getElements", &csDenseCellSet<T, D>::getElements,
           "Get elements (cells). The indicies in the elements correspond to "
           "the corner nodes.")
      .def("getElement", &csDenseCellSet<T, D>::getElement,
           "Get the element at the given index.")
      .def("getSurface", &csDenseCellSet<T, D>::getSurface,
           "Get the surface level-set.")
      .def("getCellGrid", &csDenseCellSet<T, D>::getCellGrid,
           "Get the underlying mesh of the cell set.")
      .def("getNumberOfCells", &csDenseCellSet<T, D>::getNumberOfCells,
           "Get the number of cells.")
      .def("getFillingFraction", &csDenseCellSet<T, D>::getFillingFraction,
           "Get the filling fraction of the cell containing the point.")
      .def("getFillingFractions", &csDenseCellSet<T, D>::getFillingFractions,
           "Get the filling fractions of all cells.")
      .def("getAverageFillingFraction",
           &csDenseCellSet<T, D>::getAverageFillingFraction,
           "Get the average filling at a point in some radius.")
      .def("getCellCenter", &csDenseCellSet<T, D>::getCellCenter,
           "Get the center of a cell with given index")
      .def("getScalarData", &csDenseCellSet<T, D>::getScalarData,
           "Get the data stored at each cell. WARNING: This function only "
           "returns a copy of the data")
      .def("getScalarDataLabels", &csDenseCellSet<T, D>::getScalarDataLabels,
           "Get the labels of the scalar data stored in the cell set.")
      .def("getIndex", &csDenseCellSet<T, D>::getIndex,
           "Get the index of the cell containing the given point.")
      .def("getCellSetPosition", &csDenseCellSet<T, D>::getCellSetPosition)
      .def("setCellSetPosition", &csDenseCellSet<T, D>::setCellSetPosition,
           "Set whether the cell set should be created below (false) or above "
           "(true) the surface.")
      .def(
          "setCoverMaterial", &csDenseCellSet<T, D>::setCoverMaterial,
          "Set the material of the cells which are above or below the surface.")
      .def("setPeriodicBoundary", &csDenseCellSet<T, D>::setPeriodicBoundary,
           "Enable periodic boundary conditions in specified dimensions.")
      .def("setFillingFraction",
           pybind11::overload_cast<const int, const T>(
               &csDenseCellSet<T, D>::setFillingFraction),
           "Sets the filling fraction at given cell index.")
      .def("setFillingFraction",
           pybind11::overload_cast<const std::array<T, 3> &, const T>(
               &csDenseCellSet<T, D>::setFillingFraction),
           "Sets the filling fraction for cell which contains given point.")
      .def("addFillingFraction",
           pybind11::overload_cast<const int, const T>(
               &csDenseCellSet<T, D>::addFillingFraction),
           "Add to the filling fraction at given cell index.")
      .def("addFillingFraction",
           pybind11::overload_cast<const std::array<T, 3> &, const T>(
               &csDenseCellSet<T, D>::addFillingFraction),
           "Add to the filling fraction for cell which contains given point.")
      .def("addFillingFractionInMaterial",
           &csDenseCellSet<T, D>::addFillingFractionInMaterial,
           "Add to the filling fraction for cell which contains given point "
           "only if the cell has the specified material ID.")
      .def("writeVTU", &csDenseCellSet<T, D>::writeVTU,
           "Write the cell set as .vtu file")
      .def("writeCellSetData", &csDenseCellSet<T, D>::writeCellSetData,
           "Save cell set data in simple text format.")
      .def("readCellSetData", &csDenseCellSet<T, D>::readCellSetData,
           "Read cell set data from text.")
      .def("clear", &csDenseCellSet<T, D>::clear,
           "Clear the filling fractions.")
      .def("updateMaterials", &csDenseCellSet<T, D>::updateMaterials,
           "Update the material IDs of the cell set. This function should be "
           "called if the level sets, the cell set is made out of, have "
           "changed. This does not work if the surface of the volume has "
           "changed. In this case, call the function 'updateSurface' first.")
      .def("updateSurface", &csDenseCellSet<T, D>::updateSurface,
           "Updates the surface of the cell set. The new surface should be "
           "below the old surface as this function can only remove cells from "
           "the cell set.")
      .def("buildNeighborhood", &csDenseCellSet<T, D>::buildNeighborhood,
           "Generate fast neighbor access for each cell.")
      .def("getNeighbors", &csDenseCellSet<T, D>::getNeighbors,
           "Get the neighbor indices for a cell.");

  // csSegmentCells
  pybind11::class_<csSegmentCells<T, D>, psSmartPointer<csSegmentCells<T, D>>>(
      module, "SegmentCells")
      .def(pybind11::init<psSmartPointer<csDenseCellSet<T, D>>>())
      .def(pybind11::init<psSmartPointer<csDenseCellSet<T, D>>, std::string,
                          psMaterial>(),
           pybind11::arg("cellSet"),
           pybind11::arg("cellTypeString") = "CellType",
           pybind11::arg("bulkMaterial") = psMaterial::GAS)
      .def("setCellSet", &csSegmentCells<T, D>::setCellSet,
           "Set the cell set in the segmenter.")
      .def("setCellTypeString", &csSegmentCells<T, D>::setCellTypeString,
           "Set the cell type string in the segmenter.")
      .def("setBulkMaterial", &csSegmentCells<T, D>::setBulkMaterial,
           "Set the bulk material in the segmenter.")
      .def("apply", &csSegmentCells<T, D>::apply,
           "Segment the cells into surface, material, and gas cells.");

  // ***************************************************************************
  //                                   VISUALIZATION
  //  ***************************************************************************

  // visualization classes are not bound with smart pointer holder types
  // since they should not be passed to other classes
  pybind11::class_<psToDiskMesh<T, D>>(module, "ToDiskMesh")
      .def(pybind11::init<DomainType, psSmartPointer<lsMesh<T>>>(),
           pybind11::arg("domain"), pybind11::arg("mesh"))
      .def(pybind11::init())
      .def("setDomain", &psToDiskMesh<T, D>::setDomain,
           "Set the domain in the mesh converter.")
      .def("setMesh", &psToDiskMesh<T, D>::setMesh,
           "Set the mesh in the mesh converter");
  // static assertion failed: Holder classes are only supported for
  //  custom types
  // .def("setTranslator", &psToDiskMesh<T, D>::setTranslator,
  //      "Set the translator in the mesh converter. It used to convert "
  //      "level-set point IDs to mesh point IDs.")
  // .def("getTranslator", &psToDiskMesh<T, D>::getTranslator,
  //      "Retrieve the translator from the mesh converter.");

  pybind11::class_<psWriteVisualizationMesh<T, D>>(module,
                                                   "WriteVisualizationMesh")
      .def(pybind11::init())
      .def(pybind11::init<DomainType, std::string>(), pybind11::arg("domain"),
           pybind11::arg("fileName"))
      .def("apply", &psWriteVisualizationMesh<T, D>::apply)
      .def("setFileName", &psWriteVisualizationMesh<T, D>::setFileName,
           "Set the output file name. The file name will be appended by"
           "'_volume.vtu'.")
      .def("setDomain", &psWriteVisualizationMesh<T, D>::setDomain,
           "Set the domain in the mesh converter.");

  //   ***************************************************************************
  //                                  OTHER
  //   ***************************************************************************

  // psPlanarize
  pybind11::class_<psPlanarize<T, D>, psSmartPointer<psPlanarize<T, D>>>(
      module, "Planarize")
      .def(pybind11::init(
               &psSmartPointer<psPlanarize<T, D>>::New<DomainType &, const T>),
           pybind11::arg("geometry"), pybind11::arg("cutoffHeight") = 0.)
      .def("apply", &psPlanarize<T, D>::apply, "Apply the planarization.");

  // psMeanFreePath
  pybind11::class_<psMeanFreePath<T, D>, psSmartPointer<psMeanFreePath<T, D>>>(
      module, "MeanFreePath")
      .def(pybind11::init<>())
      .def("setDomain", &psMeanFreePath<T, D>::setDomain)
      .def("setBulkLambda", &psMeanFreePath<T, D>::setBulkLambda)
      .def("setMaterial", &psMeanFreePath<T, D>::setMaterial)
      .def("setNumRaysPerCell", &psMeanFreePath<T, D>::setNumRaysPerCell)
      .def("setReflectionLimit", &psMeanFreePath<T, D>::setReflectionLimit)
      .def("setRngSeed", &psMeanFreePath<T, D>::setRngSeed)
      .def("disableSmoothing", &psMeanFreePath<T, D>::disableSmoothing)
      .def("enableSmoothing", &psMeanFreePath<T, D>::enableSmoothing)
      .def("apply", &psMeanFreePath<T, D>::apply);

#if VIENNAPS_PYTHON_DIMENSION > 2
  // GDS file parsing
  pybind11::class_<psGDSGeometry<T, D>, psSmartPointer<psGDSGeometry<T, D>>>(
      module, "GDSGeometry")
      // constructors
      .def(pybind11::init(&psSmartPointer<psGDSGeometry<T, D>>::New<>))
      .def(pybind11::init(&psSmartPointer<psGDSGeometry<T, D>>::New<const T>),
           pybind11::arg("gridDelta"))
      // methods
      .def("setGridDelta", &psGDSGeometry<T, D>::setGridDelta,
           "Set the grid spacing.")
      .def(
          "setBoundaryConditions",
          [](psGDSGeometry<T, D> &gds,
             std::vector<typename lsDomain<T, D>::BoundaryType> &bcs) {
            if (bcs.size() == D)
              gds.setBoundaryConditions(bcs.data());
          },
          "Set the boundary conditions")
      .def("setBoundaryPadding", &psGDSGeometry<T, D>::setBoundaryPadding,
           "Set padding between the largest point of the geometry and the "
           "boundary of the domain.")
      .def("print", &psGDSGeometry<T, D>::print, "Print the geometry contents.")
      .def("layerToLevelSet", &psGDSGeometry<T, D>::layerToLevelSet,
           "Convert a layer of the GDS geometry to a level set domain.")
      .def(
          "getBounds",
          [](psGDSGeometry<T, D> &gds) -> std::array<double, 6> {
            auto b = gds.getBounds();
            std::array<double, 6> bounds;
            for (unsigned i = 0; i < 6; ++i)
              bounds[i] = b[i];
            return bounds;
          },
          "Get the bounds of the geometry.");

  pybind11::class_<psGDSReader<T, D>, psSmartPointer<psGDSReader<T, D>>>(
      module, "GDSReader")
      // constructors
      .def(pybind11::init(&psSmartPointer<psGDSReader<T, D>>::New<>))
      .def(pybind11::init(&psSmartPointer<psGDSReader<T, D>>::New<
                          psSmartPointer<psGDSGeometry<T, D>> &, std::string>))
      // methods
      .def("setGeometry", &psGDSReader<T, D>::setGeometry,
           "Set the domain to be parsed in.")
      .def("setFileName", &psGDSReader<T, D>::setFileName,
           "Set name of the GDS file.")
      .def("apply", &psGDSReader<T, D>::apply, "Parse the GDS file.");
#else
  // wrap a 3D domain in 2D mode to be used with psExtrude
  // psDomain
  pybind11::class_<psDomain<T, 3>, psSmartPointer<psDomain<T, 3>>>(module,
                                                                   "Domain3D")
      // constructors
      .def(pybind11::init(&psSmartPointer<psDomain<T, 3>>::New<>))
      // methods
      .def("insertNextLevelSet", &psDomain<T, 3>::insertNextLevelSet,
           pybind11::arg("levelSet"), pybind11::arg("wrapLowerLevelSet") = true,
           "Insert a level set to domain.")
      .def("insertNextLevelSetAsMaterial",
           &psDomain<T, 3>::insertNextLevelSetAsMaterial,
           pybind11::arg("levelSet"), pybind11::arg("material"),
           pybind11::arg("wrapLowerLevelSet") = true,
           "Insert a level set to domain as a material.")
      .def("duplicateTopLevelSet", &psDomain<T, 3>::duplicateTopLevelSet)
      .def("applyBooleanOperation", &psDomain<T, 3>::applyBooleanOperation)
      .def("removeTopLevelSet", &psDomain<T, 3>::removeTopLevelSet)
      .def("setMaterialMap", &psDomain<T, 3>::setMaterialMap)
      .def("getMaterialMap", &psDomain<T, 3>::getMaterialMap)
      .def("generateCellSet", &psDomain<T, 3>::generateCellSet,
           pybind11::arg("position"), pybind11::arg("coverMaterial"),
           pybind11::arg("isAboveSurface"), "Generate the cell set.")
      .def("getLevelSets",
           [](psDomain<T, 3> &d)
               -> std::optional<std::vector<psSmartPointer<lsDomain<T, 3>>>> {
             auto levelsets = d.getLevelSets();
             if (levelsets)
               return *levelsets;
             return std::nullopt;
           })
      .def("getCellSet", &psDomain<T, 3>::getCellSet, "Get the cell set.")
      .def("getGrid", &psDomain<T, 3>::getGrid, "Get the grid")
      .def("print", &psDomain<T, 3>::print)
      .def("saveLevelSetMesh", &psDomain<T, 3>::saveLevelSetMesh,
           pybind11::arg("filename"), pybind11::arg("width") = 1,
           "Save the level set grids of layers in the domain.")
      .def("saveSurfaceMesh", &psDomain<T, 3>::saveSurfaceMesh,
           pybind11::arg("filename"), pybind11::arg("addMaterialIds") = true,
           "Save the surface of the domain.")
      .def("saveVolumeMesh", &psDomain<T, 3>::saveVolumeMesh,
           pybind11::arg("filename"),
           "Save the volume representation of the domain.")
      .def("saveLevelSets", &psDomain<T, 3>::saveLevelSets)
      .def("clear", &psDomain<T, 3>::clear);

  pybind11::class_<psExtrude<T>>(module, "Extrude")
      .def(pybind11::init())
      .def(pybind11::init<psSmartPointer<psDomain<T, 2>> &,
                          psSmartPointer<psDomain<T, 3>> &, std::array<T, 2>,
                          const int,
                          std::array<lsBoundaryConditionEnum<3>, 3>>(),
           pybind11::arg("inputDomain"), pybind11::arg("outputDomain"),
           pybind11::arg("extent"), pybind11::arg("extrudeDimension"),
           pybind11::arg("boundaryConditions"))
      .def("setInputDomain", &psExtrude<T>::setInputDomain,
           "Set the input domain to be extruded.")
      .def("setOutputDomain", &psExtrude<T>::setOutputDomain,
           "Set the output domain. The 3D output domain will be overwritten by "
           "the extruded domain.")
      .def("setExtent", &psExtrude<T>::setExtent,
           "Set the min and max extent in the extruded dimension.")
      .def("setExtrudeDimension", &psExtrude<T>::setExtrudeDimension,
           "Set which index of the added dimension (x: 0, y: 1, z: 2).")
      .def("setBoundaryConditions",
           pybind11::overload_cast<std::array<lsBoundaryConditionEnum<3>, 3>>(
               &psExtrude<T>::setBoundaryConditions),
           "Set the boundary conditions in the extruded domain.")
      .def("apply", &psExtrude<T>::apply, "Run the extrusion.");
#endif

  // rayReflection.hpp
  module.def("rayReflectionSpecular", &rayReflectionSpecular<T>,
             "Specular reflection,");
  module.def("rayReflectionDiffuse", &rayReflectionDiffuse<T, D>,
             "Diffuse reflection.");
  module.def("rayReflectionConedCosine", &rayReflectionConedCosine<T, D>,
             "Coned cosine reflection.");

  // psUtils::Timer
  pybind11::class_<psUtils::Timer<std::chrono::high_resolution_clock>>(module,
                                                                       "Timer")
      .def(pybind11::init<>())
      .def("start", &psUtils::Timer<std::chrono::high_resolution_clock>::start,
           "Start the timer.")
      .def("finish",
           &psUtils::Timer<std::chrono::high_resolution_clock>::finish,
           "Stop the timer.")
      .def("reset", &psUtils::Timer<std::chrono::high_resolution_clock>::reset,
           "Reset the timer.")
      .def_readonly(
          "currentDuration",
          &psUtils::Timer<std::chrono::high_resolution_clock>::currentDuration,
          "Get the current duration of the timer in nanoseconds.")
      .def_readonly(
          "totalDuration",
          &psUtils::Timer<std::chrono::high_resolution_clock>::totalDuration,
          "Get the total duration of the timer in nanoseconds.");
}