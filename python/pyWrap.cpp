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
#include <psAtomicLayerProcess.hpp>
#include <psConstants.hpp>
#include <psDomain.hpp>
#include <psExtrude.hpp>
#include <psGDSGeometry.hpp>
#include <psGDSReader.hpp>
#include <psPlanarize.hpp>
#include <psProcess.hpp>

// geometries
#include <geometries/psMakeFin.hpp>
#include <geometries/psMakeHole.hpp>
#include <geometries/psMakePlane.hpp>
#include <geometries/psMakeStack.hpp>
#include <geometries/psMakeTrench.hpp>

// model framework
#include <psAdvectionCallback.hpp>
#include <psProcessModel.hpp>
#include <psProcessParams.hpp>
#include <psSurfaceModel.hpp>
#include <psVelocityField.hpp>

// models
#include <models/psAnisotropicProcess.hpp>
#include <models/psDirectionalEtching.hpp>
#include <models/psFluorocarbonEtching.hpp>
#include <models/psGeometricDistributionModels.hpp>
#include <models/psIsotropicProcess.hpp>
#include <models/psOxideRegrowth.hpp>
#include <models/psSF6O2Etching.hpp>
#include <models/psSingleParticleALD.hpp>
#include <models/psSingleParticleProcess.hpp>
#include <models/psTEOSDeposition.hpp>

// visualization
#include <psToDiskMesh.hpp>

// other
#include <psUtils.hpp>
#include <rayParticle.hpp>
#include <rayReflection.hpp>
#include <rayUtil.hpp>
#include <vcLogger.hpp>

using namespace viennaps;

// always use double for python export
typedef double T;
// get dimension from cmake define
constexpr int D = VIENNAPS_PYTHON_DIMENSION;
typedef SmartPointer<Domain<T, D>> DomainType;

PYBIND11_DECLARE_HOLDER_TYPE(Types, SmartPointer<Types>)

// NOTES:
// PYBIND11_MAKE_OPAQUE(std::vector<T, std::allocator<T>>) can not be used
// constructors with custom enum need lambda to work: seems to be an issue
// with implicit move constructor

// define trampoline classes for interface functions
// ALSO NEED TO ADD TRAMPOLINE CLASSES FOR CLASSES
// WHICH HOLD REFERENCES TO INTERFACE(ABSTRACT) CLASSES
// class PypsSurfaceModel : public SurfaceModel<T> {
//   using SurfaceModel<T>::coverages;
//   using SurfaceModel<T>::processParams;
//   using SurfaceModel<T>::getCoverages;
//   using SurfaceModel<T>::getProcessParameters;
//   typedef std::vector<T> vect_type;
// public:
//   void initializeCoverages(unsigned numGeometryPoints) override {
//     PYBIND11_OVERRIDE(void, SurfaceModel<T>, initializeCoverages,
//                       numGeometryPoints);
//   }
//   void initializeProcessParameters() override {
//     PYBIND11_OVERRIDE(void, SurfaceModel<T>, initializeProcessParameters,
//     );
//   }
//   SmartPointer<std::vector<T>>
//   calculateVelocities(SmartPointer<psPointData<T>> rates,
//                       const std::vector<std::array<T, 3>> &coordinates,
//                       const std::vector<T> &materialIds) override {
//     PYBIND11_OVERRIDE(SmartPointer<std::vector<T>>, SurfaceModel<T>,
//                       calculateVelocities, rates, coordinates, materialIds);
//   }
//   void updateCoverages(SmartPointer<psPointData<T>> rates,
//                        const std::vector<T> &materialIds) override {
//     PYBIND11_OVERRIDE(void, SurfaceModel<T>, updateCoverages, rates,
//                       materialIds);
//   }
// };

// AdvectionCallback
class PyAdvectionCallback : public AdvectionCallback<T, D> {
protected:
  using ClassName = AdvectionCallback<T, D>;

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
template <int D>
class psParticle : public viennaray::Particle<psParticle<D>, T> {
  using ClassName = viennaray::Particle<psParticle<D>, T>;

public:
  void surfaceCollision(T rayWeight, const Vec3D<T> &rayDir,
                        const Vec3D<T> &geomNormal, const unsigned int primID,
                        const int materialID,
                        viennaray::TracingData<T> &localData,
                        const viennaray::TracingData<T> *globalData,
                        RNG &Rng) override final {
    PYBIND11_OVERRIDE(void, ClassName, surfaceCollision, rayWeight, rayDir,
                      geomNormal, primID, materialID, localData, globalData,
                      Rng);
  }

  std::pair<T, Vec3D<T>> surfaceReflection(
      T rayWeight, const Vec3D<T> &rayDir, const Vec3D<T> &geomNormal,
      const unsigned int primID, const int materialID,
      const viennaray::TracingData<T> *globalData, RNG &Rng) override final {
    using Pair = std::pair<T, Vec3D<T>>;
    PYBIND11_OVERRIDE(Pair, ClassName, surfaceReflection, rayWeight, rayDir,
                      geomNormal, primID, materialID, globalData, Rng);
  }

  void initNew(RNG &RNG) override final {
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
//   void surfaceCollision(T rayWeight, const Vec3D<T> &rayDir,
//                         const Vec3D<T> &geomNormal,
//                         const unsigned int primID, const int materialID,
//                         viennaray::TracingData<T> &localData,
//                         const viennaray::TracingData<T> *globalData,
//                         RNG &Rng) override final {
//     localData.getVectorData(0)[primID] += rayWeight;
//   }
//   std::pair<T, Vec3D<T>>
//   surfaceReflection(T rayWeight, const Vec3D<T> &rayDir,
//                     const Vec3D<T> &geomNormal, const unsigned int
//                     primID, const int materialID, const
//                     viennaray::TracingData<T> *globalData, RNG &Rng)
//                     override final {
//     auto direction = rayReflectionDiffuse<T, D>(geomNormal, Rng);
//     return {stickingProbability, direction};
//   }
//   void initNew(RNG &RNG) override final {}
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
//   void surfaceCollision(T rayWeight, const Vec3D<T> &rayDir,
//                         const Vec3D<T> &geomNormal,
//                         const unsigned int primID, const int materialID,
//                         viennaray::TracingData<T> &localData,
//                         const viennaray::TracingData<T> *globalData,
//                         RNG &Rng) override final {
//     localData.getVectorData(0)[primID] += rayWeight;
//   }
//   std::pair<T, Vec3D<T>>
//   surfaceReflection(T rayWeight, const Vec3D<T> &rayDir,
//                     const Vec3D<T> &geomNormal, const unsigned int
//                     primID, const int materialID, const
//                     viennaray::TracingData<T> *globalData, RNG &Rng)
//                     override final {
//     auto direction = rayReflectionSpecular<T>(rayDir, geomNormal);
//     return {stickingProbability, direction};
//   }
//   void initNew(RNG &RNG) override final {}
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
// VelocityField
// class PyVelocityField : public VelocityField<T> {
//   using VelocityField<T>::psVelocityField;
// public:
//   T getScalarVelocity(const std::array<T, 3> &coordinate, int material,
//                       const std::array<T, 3> &normalVector,
//                       unsigned long pointId) override {
//     PYBIND11_OVERRIDE(T, VelocityField<T>, getScalarVelocity, coordinate,
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
//         VelocityField<T>, getVectorVelocity, coordinate, material,
//         normalVector, pointId);
//   }
//   T getDissipationAlpha(int direction, int material,
//                         const std::array<T, 3> &centralDifferences) override
//                         {
//     PYBIND11_OVERRIDE(T, VelocityField<T>, getDissipationAlpha, direction,
//                       material, centralDifferences);
//   }
//   void setVelocities(SmartPointer<std::vector<T>> passedVelocities)
//   override {
//     PYBIND11_OVERRIDE(void, VelocityField<T>, setVelocities,
//                       passedVelocities);
//   }
//   int getTranslationFieldOptions() const override {
//     PYBIND11_OVERRIDE(int, VelocityField<T>, getTranslationFieldOptions, );
//   }
// };
// a function to declare GeometricDistributionModel of type DistType
// template <typename NumericType, int D, typename DistType>
// void declare_GeometricDistributionModel(pybind11::module &m,
//                                         const std::string &typestr) {
//   using Class = psGeometricDistributionModel<NumericType, D, DistType>;
//   pybind11::class_<Class, SmartPointer<Class>>(m, typestr.c_str())
//       .def(pybind11::init<SmartPointer<DistType>>(), pybind11::arg("dist"))
//       .def(pybind11::init<SmartPointer<DistType>,
//                           SmartPointer<viennals::Domain<NumericType, D>>>(),
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

  // Logger
  pybind11::enum_<LogLevel>(module, "LogLevel", pybind11::module_local())
      .value("ERROR", LogLevel::ERROR)
      .value("WARNING", LogLevel::WARNING)
      .value("INFO", LogLevel::INFO)
      .value("TIMING", LogLevel::TIMING)
      .value("INTERMEDIATE", LogLevel::INTERMEDIATE)
      .value("DEBUG", LogLevel::DEBUG)
      .export_values();

  pybind11::class_<Logger, SmartPointer<Logger>>(module, "Logger",
                                                 pybind11::module_local())
      .def_static("setLogLevel", &Logger::setLogLevel)
      .def_static("getLogLevel", &Logger::getLogLevel)
      .def_static("getInstance", &Logger::getInstance,
                  pybind11::return_value_policy::reference)
      .def("addDebug", &Logger::addDebug)
      .def("addTiming",
           (Logger & (Logger::*)(std::string, double)) & Logger::addTiming)
      .def("addTiming", (Logger & (Logger::*)(std::string, double, double)) &
                            Logger::addTiming)
      .def("addInfo", &Logger::addInfo)
      .def("addWarning", &Logger::addWarning)
      .def("addError", &Logger::addError, pybind11::arg("s"),
           pybind11::arg("shouldAbort") = true)
      .def("print", [](Logger &instance) { instance.print(std::cout); });

  /****************************************************************************
   *                               MODEL FRAMEWORK                            *
   ****************************************************************************/

  // ProcessModel
  pybind11::class_<ProcessModel<T, D>, SmartPointer<ProcessModel<T, D>>>(
      module, "ProcessModel")
      .def(pybind11::init<>())
      // methods
      .def("setProcessName", &ProcessModel<T, D>::setProcessName)
      .def("getProcessName", &ProcessModel<T, D>::getProcessName)
      .def("getSurfaceModel", &ProcessModel<T, D>::getSurfaceModel)
      .def("getAdvectionCallback", &ProcessModel<T, D>::getAdvectionCallback)
      .def("getGeometricModel", &ProcessModel<T, D>::getGeometricModel)
      .def("getVelocityField", &ProcessModel<T, D>::getVelocityField)
      .def("getParticleLogSize", &ProcessModel<T, D>::getParticleLogSize)
      .def("getParticleTypes",
           [](ProcessModel<T, D> &pm) {
             // Get smart pointer to vector of unique_ptr from the process
             // model
             auto &unique_ptrs = pm.getParticleTypes();

             // Create vector to hold shared_ptr
             std::vector<std::shared_ptr<viennaray::AbstractParticle<T>>>
                 shared_ptrs;

             // Loop over unique_ptrs and create shared_ptrs from them
             for (auto &uptr : unique_ptrs) {
               shared_ptrs.push_back(
                   std::shared_ptr<viennaray::AbstractParticle<T>>(
                       uptr.release()));
             }

             // Return the new vector of shared_ptr
             return shared_ptrs;
           })
      .def("setSurfaceModel",
           [](ProcessModel<T, D> &pm, SmartPointer<SurfaceModel<T>> &sm) {
             pm.setSurfaceModel(sm);
           })
      .def("setAdvectionCallback",
           [](ProcessModel<T, D> &pm,
              SmartPointer<AdvectionCallback<T, D>> &ac) {
             pm.setAdvectionCallback(ac);
           })
      .def("insertNextParticleType",
           [](ProcessModel<T, D> &pm,
              SmartPointer<psParticle<D>> &passedParticle) {
             if (passedParticle) {
               auto particle =
                   std::make_unique<psParticle<D>>(*passedParticle.get());
               pm.insertNextParticleType(particle);
             }
           })
      // IMPORTANT: here it may be needed to write this function for any
      // type of passed Particle
      .def("setGeometricModel",
           [](ProcessModel<T, D> &pm, SmartPointer<GeometricModel<T, D>> &gm) {
             pm.setGeometricModel(gm);
           })
      .def("setVelocityField",
           [](ProcessModel<T, D> &pm, SmartPointer<VelocityField<T>> &vf) {
             pm.setVelocityField(vf);
           })
      .def("setPrimaryDirection", &ProcessModel<T, D>::setPrimaryDirection)
      .def("getPrimaryDirection", &ProcessModel<T, D>::getPrimaryDirection);

  // AdvectionCallback
  pybind11::class_<AdvectionCallback<T, D>,
                   SmartPointer<AdvectionCallback<T, D>>, PyAdvectionCallback>(
      module, "AdvectionCallback")
      // constructors
      .def(pybind11::init<>())
      // methods
      .def("applyPreAdvect", &AdvectionCallback<T, D>::applyPreAdvect)
      .def("applyPostAdvect", &AdvectionCallback<T, D>::applyPostAdvect)
      .def_readwrite("domain", &PyAdvectionCallback::domain);

  // ProcessParams
  pybind11::class_<ProcessParams<T>, SmartPointer<ProcessParams<T>>>(
      module, "ProcessParams")
      .def(pybind11::init<>())
      .def("insertNextScalar", &ProcessParams<T>::insertNextScalar)
      .def("getScalarData",
           (T & (ProcessParams<T>::*)(int)) & ProcessParams<T>::getScalarData)
      .def("getScalarData", (const T &(ProcessParams<T>::*)(int) const) &
                                ProcessParams<T>::getScalarData)
      .def("getScalarData", (T & (ProcessParams<T>::*)(std::string)) &
                                ProcessParams<T>::getScalarData)
      .def("getScalarDataIndex", &ProcessParams<T>::getScalarDataIndex)
      .def("getScalarData", (std::vector<T> & (ProcessParams<T>::*)()) &
                                ProcessParams<T>::getScalarData)
      .def("getScalarData",
           (const std::vector<T> &(ProcessParams<T>::*)() const) &
               ProcessParams<T>::getScalarData)
      .def("getScalarDataLabel", &ProcessParams<T>::getScalarDataLabel);

  // SurfaceModel
  //   pybind11::class_<SurfaceModel<T>, SmartPointer<SurfaceModel<T>>,
  //                    PypsSurfaceModel>(module, "SurfaceModel")
  //       .def(pybind11::init<>())
  //       .def("initializeCoverages", &SurfaceModel<T>::initializeCoverages)
  //       .def("initializeProcessParameters",
  //            &SurfaceModel<T>::initializeProcessParameters)
  //       .def("getCoverages", &SurfaceModel<T>::getCoverages)
  //       .def("getProcessParameters",
  //       &SurfaceModel<T>::getProcessParameters) .def("calculateVelocities",
  //       &SurfaceModel<T>::calculateVelocities) .def("updateCoverages",
  //       &SurfaceModel<T>::updateCoverages);
  // VelocityField
  //   pybind11::class_<VelocityField<T>, SmartPointer<VelocityField<T>>,
  //                    PyVelocityField>
  //       velocityField(module, "VelocityField");
  //   // constructors
  //   velocityField
  //       .def(pybind11::init<>())
  //       // methods
  //       .def("getScalarVelocity", &VelocityField<T>::getScalarVelocity)
  //       .def("getVectorVelocity", &VelocityField<T>::getVectorVelocity)
  //       .def("getDissipationAlpha", &VelocityField<T>::getDissipationAlpha)
  //       .def("getTranslationFieldOptions",
  //            &VelocityField<T>::getTranslationFieldOptions)
  //       .def("setVelocities", &VelocityField<T>::setVelocities);
  //   pybind11::class_<psDefaultVelocityField<T>,
  //                    SmartPointer<psDefaultVelocityField<T>>>(
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
  pybind11::class_<psParticle<D>, SmartPointer<psParticle<D>>> particle(
      module, "Particle");
  particle.def("surfaceCollision", &psParticle<D>::surfaceCollision)
      .def("surfaceReflection", &psParticle<D>::surfaceReflection)
      .def("initNew", &psParticle<D>::initNew)
      .def("getLocalDataLabels", &psParticle<D>::getLocalDataLabels)
      .def("getSourceDistributionPower",
           &psParticle<D>::getSourceDistributionPower);

  // predefined particles
  //   pybind11::class_<psDiffuseParticle<D>,
  //   SmartPointer<psDiffuseParticle<D>>>(
  //       module, "DiffuseParticle", particle)
  //       .def(pybind11::init(
  //                &SmartPointer<psDiffuseParticle<D>>::New<const T, const T,
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
  //   pybind11::class_<psSpecularParticle, SmartPointer<psSpecularParticle>>(
  //       module, "SpecularParticle", particle)
  //       .def(pybind11::init(
  //                &SmartPointer<psSpecularParticle>::New<const T, const T,
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

  // Enum Material
  pybind11::enum_<Material>(module, "Material")
      .value("Undefined", Material::None) // 1
      .value("Mask", Material::Mask)
      .value("Si", Material::Si)
      .value("SiO2", Material::SiO2)
      .value("Si3N4", Material::Si3N4) // 5
      .value("SiN", Material::SiN)
      .value("SiON", Material::SiON)
      .value("SiC", Material::SiC)
      .value("SiGe", Material::SiGe)
      .value("PolySi", Material::PolySi) // 10
      .value("GaN", Material::GaN)
      .value("W", Material::W)
      .value("Al2O3", Material::Al2O3)
      .value("TiN", Material::TiN)
      .value("Cu", Material::Cu) // 15
      .value("Polymer", Material::Polymer)
      .value("Dielectric", Material::Dielectric)
      .value("Metal", Material::Metal)
      .value("Air", Material::Air)
      .value("GAS", Material::GAS) // 20
      .export_values();

  // Single Particle Process
  pybind11::class_<SingleParticleProcess<T, D>,
                   SmartPointer<SingleParticleProcess<T, D>>>(
      module, "SingleParticleProcess")
      .def(pybind11::init([](const T rate, const T sticking, const T power,
                             const Material mask) {
             return SmartPointer<SingleParticleProcess<T, D>>::New(
                 rate, sticking, power, mask);
           }),
           pybind11::arg("rate") = 1.,
           pybind11::arg("stickingProbability") = 1.,
           pybind11::arg("sourceExponent") = 1.,
           pybind11::arg("maskMaterial") = Material::None)
      .def(pybind11::init([](const T rate, const T sticking, const T power,
                             const std::vector<Material> mask) {
             return SmartPointer<SingleParticleProcess<T, D>>::New(
                 rate, sticking, power, mask);
           }),
           pybind11::arg("rate"), pybind11::arg("stickingProbability"),
           pybind11::arg("sourceExponent"), pybind11::arg("maskMaterials"))
      .def("getProcessModel", &SingleParticleProcess<T, D>::getProcessModel);

  // TEOS Deposition
  pybind11::class_<TEOSDeposition<T, D>, SmartPointer<TEOSDeposition<T, D>>>(
      module, "TEOSDeposition")
      .def(pybind11::init(
               &SmartPointer<TEOSDeposition<T, D>>::New<
                   const T /*st1*/, const T /*rate1*/, const T /*order1*/,
                   const T /*st2*/, const T /*rate2*/, const T /*order2*/>),
           pybind11::arg("stickingProbabilityP1"), pybind11::arg("rateP1"),
           pybind11::arg("orderP1"),
           pybind11::arg("stickingProbabilityP2") = 0.,
           pybind11::arg("rateP2") = 0., pybind11::arg("orderP2") = 0.)
      .def("getProcessModel", &TEOSDeposition<T, D>::getProcessModel);

  // SF6O2 Parameters
  pybind11::class_<SF6O2Parameters<T>::MaskType>(module, "SF6O2ParametersMask")
      .def(pybind11::init<>())
      .def_readwrite("rho", &SF6O2Parameters<T>::MaskType::rho)
      .def_readwrite("beta_F", &SF6O2Parameters<T>::MaskType::beta_F)
      .def_readwrite("beta_O", &SF6O2Parameters<T>::MaskType::beta_O)
      .def_readwrite("A_sp", &SF6O2Parameters<T>::MaskType::A_sp)
      .def_readwrite("B_sp", &SF6O2Parameters<T>::MaskType::B_sp)
      .def_readwrite("Eth_sp", &SF6O2Parameters<T>::MaskType::Eth_sp);

  pybind11::class_<SF6O2Parameters<T>::SiType>(module, "SF6O2ParametersSi")
      .def(pybind11::init<>())
      .def_readwrite("rho", &SF6O2Parameters<T>::SiType::rho)
      .def_readwrite("k_sigma", &SF6O2Parameters<T>::SiType::k_sigma)
      .def_readwrite("beta_sigma", &SF6O2Parameters<T>::SiType::beta_sigma)
      .def_readwrite("A_sp", &SF6O2Parameters<T>::SiType::A_sp)
      .def_readwrite("B_sp", &SF6O2Parameters<T>::SiType::B_sp)
      .def_readwrite("Eth_ie", &SF6O2Parameters<T>::SiType::Eth_ie)
      .def_readwrite("Eth_sp", &SF6O2Parameters<T>::SiType::Eth_sp)
      .def_readwrite("A_ie", &SF6O2Parameters<T>::SiType::A_ie);

  pybind11::class_<SF6O2Parameters<T>::PassivationType>(
      module, "SF6O2ParametersPassivation")
      .def(pybind11::init<>())
      .def_readwrite("Eth_ie", &SF6O2Parameters<T>::PassivationType::Eth_ie)
      .def_readwrite("A_ie", &SF6O2Parameters<T>::PassivationType::A_ie);

  pybind11::class_<SF6O2Parameters<T>::IonType>(module, "SF6O2ParametersIons")
      .def(pybind11::init<>())
      .def_readwrite("meanEnergy", &SF6O2Parameters<T>::IonType::meanEnergy)
      .def_readwrite("sigmaEnergy", &SF6O2Parameters<T>::IonType::sigmaEnergy)
      .def_readwrite("exponent", &SF6O2Parameters<T>::IonType::exponent)
      .def_readwrite("inflectAngle", &SF6O2Parameters<T>::IonType::inflectAngle)
      .def_readwrite("n_l", &SF6O2Parameters<T>::IonType::n_l)
      .def_readwrite("minAngle", &SF6O2Parameters<T>::IonType::minAngle);

  pybind11::class_<SF6O2Parameters<T>>(module, "SF6O2Parameters")
      .def(pybind11::init<>())
      .def_readwrite("ionFlux", &SF6O2Parameters<T>::ionFlux)
      .def_readwrite("etchantFlux", &SF6O2Parameters<T>::etchantFlux)
      .def_readwrite("oxygenFlux", &SF6O2Parameters<T>::oxygenFlux)
      .def_readwrite("etchStopDepth", &SF6O2Parameters<T>::etchStopDepth)
      .def_readwrite("beta_F", &SF6O2Parameters<T>::beta_F)
      .def_readwrite("beta_O", &SF6O2Parameters<T>::beta_O)
      .def_readwrite("Mask", &SF6O2Parameters<T>::Mask)
      .def_readwrite("Si", &SF6O2Parameters<T>::Si)
      .def_readwrite("Polymer", &SF6O2Parameters<T>::Passivation)
      .def_readwrite("Ions", &SF6O2Parameters<T>::Ions);

  // SF6O2 Etching
  pybind11::class_<SF6O2Etching<T, D>, SmartPointer<SF6O2Etching<T, D>>>(
      module, "SF6O2Etching")
      .def(pybind11::init<>())
      .def(pybind11::init(
               &SmartPointer<SF6O2Etching<T, D>>::New<
                   const double /*ionFlux*/, const double /*etchantFlux*/,
                   const double /*oxygenFlux*/, const T /*meanIonEnergy*/,
                   const T /*sigmaIonEnergy*/, const T /*ionExponent*/,
                   const T /*oxySputterYield*/, const T /*etchStopDepth*/>),
           pybind11::arg("ionFlux"), pybind11::arg("etchantFlux"),
           pybind11::arg("oxygenFlux"), pybind11::arg("meanIonEnergy") = 100.,
           pybind11::arg("sigmaIonEnergy") = 10.,
           pybind11::arg("ionExponent") = 100.,
           pybind11::arg("oxySputterYield") = 3.,
           pybind11::arg("etchStopDepth") = std::numeric_limits<T>::lowest())
      .def(pybind11::init(&SmartPointer<SF6O2Etching<T, D>>::New<
                          const SF6O2Parameters<T> &>),
           pybind11::arg("parameters"))
      .def("setParameters", &SF6O2Etching<T, D>::setParameters)
      .def("getParameters", &SF6O2Etching<T, D>::getParameters,
           pybind11::return_value_policy::reference)
      .def("getProcessModel", &SF6O2Etching<T, D>::getProcessModel);

  // Fluorocarbon Parameters
  pybind11::class_<FluorocarbonParameters<T>::MaskType>(
      module, "FluorocarbonParametersMask")
      .def(pybind11::init<>())
      .def_readwrite("rho", &FluorocarbonParameters<T>::MaskType::rho)
      .def_readwrite("beta_p", &FluorocarbonParameters<T>::MaskType::beta_p)
      .def_readwrite("beta_e", &FluorocarbonParameters<T>::MaskType::beta_e)
      .def_readwrite("A_sp", &FluorocarbonParameters<T>::MaskType::A_sp)
      .def_readwrite("B_sp", &FluorocarbonParameters<T>::MaskType::B_sp)
      .def_readwrite("Eth_sp", &FluorocarbonParameters<T>::MaskType::Eth_sp);

  pybind11::class_<FluorocarbonParameters<T>::SiO2Type>(
      module, "FluorocarbonParametersSiO2")
      .def(pybind11::init<>())
      .def_readwrite("rho", &FluorocarbonParameters<T>::SiO2Type::rho)
      .def_readwrite("E_a", &FluorocarbonParameters<T>::SiO2Type::E_a)
      .def_readwrite("K", &FluorocarbonParameters<T>::SiO2Type::K)
      .def_readwrite("A_sp", &FluorocarbonParameters<T>::SiO2Type::A_sp)
      .def_readwrite("B_sp", &FluorocarbonParameters<T>::SiO2Type::B_sp)
      .def_readwrite("Eth_ie", &FluorocarbonParameters<T>::SiO2Type::Eth_ie)
      .def_readwrite("Eth_sp", &FluorocarbonParameters<T>::SiO2Type::Eth_sp)
      .def_readwrite("A_ie", &FluorocarbonParameters<T>::SiO2Type::A_ie);

  pybind11::class_<FluorocarbonParameters<T>::Si3N4Type>(
      module, "FluorocarbonParametersSi3N4")
      .def(pybind11::init<>())
      .def_readwrite("rho", &FluorocarbonParameters<T>::Si3N4Type::rho)
      .def_readwrite("E_a", &FluorocarbonParameters<T>::Si3N4Type::E_a)
      .def_readwrite("K", &FluorocarbonParameters<T>::Si3N4Type::K)
      .def_readwrite("A_sp", &FluorocarbonParameters<T>::Si3N4Type::A_sp)
      .def_readwrite("B_sp", &FluorocarbonParameters<T>::Si3N4Type::B_sp)
      .def_readwrite("Eth_ie", &FluorocarbonParameters<T>::Si3N4Type::Eth_ie)
      .def_readwrite("Eth_sp", &FluorocarbonParameters<T>::Si3N4Type::Eth_sp)
      .def_readwrite("A_ie", &FluorocarbonParameters<T>::Si3N4Type::A_ie);

  pybind11::class_<FluorocarbonParameters<T>::SiType>(
      module, "FluorocarbonParametersSi")
      .def(pybind11::init<>())
      .def_readwrite("rho", &FluorocarbonParameters<T>::SiType::rho)
      .def_readwrite("E_a", &FluorocarbonParameters<T>::SiType::E_a)
      .def_readwrite("K", &FluorocarbonParameters<T>::SiType::K)
      .def_readwrite("A_sp", &FluorocarbonParameters<T>::SiType::A_sp)
      .def_readwrite("B_sp", &FluorocarbonParameters<T>::SiType::B_sp)
      .def_readwrite("Eth_ie", &FluorocarbonParameters<T>::SiType::Eth_ie)
      .def_readwrite("Eth_sp", &FluorocarbonParameters<T>::SiType::Eth_sp)
      .def_readwrite("A_ie", &FluorocarbonParameters<T>::SiType::A_ie);

  pybind11::class_<FluorocarbonParameters<T>::PolymerType>(
      module, "FluorocarbonParametersPolymer")
      .def(pybind11::init<>())
      .def_readwrite("rho", &FluorocarbonParameters<T>::PolymerType::rho)
      .def_readwrite("Eth_ie", &FluorocarbonParameters<T>::PolymerType::Eth_ie)
      .def_readwrite("A_ie", &FluorocarbonParameters<T>::PolymerType::A_ie);

  pybind11::class_<FluorocarbonParameters<T>::IonType>(
      module, "FluorocarbonParametersIons")
      .def(pybind11::init<>())
      .def_readwrite("meanEnergy",
                     &FluorocarbonParameters<T>::IonType::meanEnergy)
      .def_readwrite("sigmaEnergy",
                     &FluorocarbonParameters<T>::IonType::sigmaEnergy)
      .def_readwrite("exponent", &FluorocarbonParameters<T>::IonType::exponent)
      .def_readwrite("inflectAngle",
                     &FluorocarbonParameters<T>::IonType::inflectAngle)
      .def_readwrite("n_l", &FluorocarbonParameters<T>::IonType::n_l)
      .def_readwrite("minAngle", &FluorocarbonParameters<T>::IonType::minAngle);

  pybind11::class_<FluorocarbonParameters<T>>(module, "FluorocarbonParameters")
      .def(pybind11::init<>())
      .def_readwrite("ionFlux", &FluorocarbonParameters<T>::ionFlux)
      .def_readwrite("etchantFlux", &FluorocarbonParameters<T>::etchantFlux)
      .def_readwrite("polyFlux", &FluorocarbonParameters<T>::polyFlux)
      .def_readwrite("delta_p", &FluorocarbonParameters<T>::delta_p)
      .def_readwrite("etchStopDepth", &FluorocarbonParameters<T>::etchStopDepth)
      .def_readwrite("Mask", &FluorocarbonParameters<T>::Mask)
      .def_readwrite("SiO2", &FluorocarbonParameters<T>::SiO2)
      .def_readwrite("Si3N4", &FluorocarbonParameters<T>::Si3N4)
      .def_readwrite("Si", &FluorocarbonParameters<T>::Si)
      .def_readwrite("Polymer", &FluorocarbonParameters<T>::Polymer)
      .def_readwrite("Ions", &FluorocarbonParameters<T>::Ions);

  // Fluorocarbon Etching
  pybind11::class_<FluorocarbonEtching<T, D>,
                   SmartPointer<FluorocarbonEtching<T, D>>>(
      module, "FluorocarbonEtching")
      .def(pybind11::init<>())
      .def(
          pybind11::init(&SmartPointer<FluorocarbonEtching<T, D>>::New<
                         const double /*ionFlux*/, const double /*etchantFlux*/,
                         const double /*polyFlux*/, T /*meanEnergy*/,
                         const T /*sigmaEnergy*/, const T /*ionExponent*/,
                         const T /*deltaP*/, const T /*etchStopDepth*/>),
          pybind11::arg("ionFlux"), pybind11::arg("etchantFlux"),
          pybind11::arg("polyFlux"), pybind11::arg("meanIonEnergy") = 100.,
          pybind11::arg("sigmaIonEnergy") = 10.,
          pybind11::arg("ionExponent") = 100., pybind11::arg("deltaP") = 0.,
          pybind11::arg("etchStopDepth") = std::numeric_limits<T>::lowest())
      .def(pybind11::init(&SmartPointer<FluorocarbonEtching<T, D>>::New<
                          const FluorocarbonParameters<T> &>),
           pybind11::arg("parameters"))
      .def("setParameters", &FluorocarbonEtching<T, D>::setParameters)
      .def("getParameters", &FluorocarbonEtching<T, D>::getParameters,
           pybind11::return_value_policy::reference)
      .def("getProcessModel", &FluorocarbonEtching<T, D>::getProcessModel);

  // Isotropic Process
  pybind11::class_<IsotropicProcess<T, D>,
                   SmartPointer<IsotropicProcess<T, D>>>(module,
                                                         "IsotropicProcess")
      .def(pybind11::init([](const T rate, const Material mask) {
             return SmartPointer<IsotropicProcess<T, D>>::New(rate, mask);
           }),
           pybind11::arg("rate") = 1.,
           pybind11::arg("maskMaterial") = Material::Mask)
      .def(pybind11::init([](const T rate, const std::vector<Material> mask) {
             return SmartPointer<IsotropicProcess<T, D>>::New(rate, mask);
           }),
           pybind11::arg("rate"), pybind11::arg("maskMaterial"))
      .def("getProcessModel", &IsotropicProcess<T, D>::getProcessModel);

  // Directional Etching
  pybind11::class_<DirectionalEtching<T, D>,
                   SmartPointer<DirectionalEtching<T, D>>>(module,
                                                           "DirectionalEtching")
      .def(pybind11::init<const std::array<T, 3> &, const T, const T,
                          const Material>(),
           pybind11::arg("direction"),
           pybind11::arg("directionalVelocity") = 1.,
           pybind11::arg("isotropicVelocity") = 0.,
           pybind11::arg("maskMaterial") = Material::Mask)
      .def(pybind11::init<const std::array<T, 3> &, const T, const T,
                          const std::vector<Material>>(),
           pybind11::arg("direction"), pybind11::arg("directionalVelocity"),
           pybind11::arg("isotropicVelocity"), pybind11::arg("maskMaterial"))
      .def("getProcessModel", &DirectionalEtching<T, D>::getProcessModel);

  // Sphere Distribution
  pybind11::class_<SphereDistribution<T, D>,
                   SmartPointer<SphereDistribution<T, D>>>(module,
                                                           "SphereDistribution")
      .def(pybind11::init([](const T radius, const T gridDelta,
                             SmartPointer<viennals::Domain<T, D>> mask) {
             return SmartPointer<SphereDistribution<T, D>>::New(
                 radius, gridDelta, mask);
           }),
           pybind11::arg("radius"), pybind11::arg("gridDelta"),
           pybind11::arg("mask"))
      .def(pybind11::init([](const T radius, const T gridDelta) {
             return SmartPointer<SphereDistribution<T, D>>::New(
                 radius, gridDelta, nullptr);
           }),
           pybind11::arg("radius"), pybind11::arg("gridDelta"))
      .def("getProcessModel", &SphereDistribution<T, D>::getProcessModel);

  // Box Distribution
  pybind11::class_<BoxDistribution<T, D>, SmartPointer<BoxDistribution<T, D>>>(
      module, "BoxDistribution")
      .def(
          pybind11::init([](const std::array<T, 3> &halfAxes, const T gridDelta,
                            SmartPointer<viennals::Domain<T, D>> mask) {
            return SmartPointer<BoxDistribution<T, D>>::New(halfAxes, gridDelta,
                                                            mask);
          }),
          pybind11::arg("halfAxes"), pybind11::arg("gridDelta"),
          pybind11::arg("mask"))
      .def(pybind11::init(
               [](const std::array<T, 3> &halfAxes, const T gridDelta) {
                 return SmartPointer<BoxDistribution<T, D>>::New(
                     halfAxes, gridDelta, nullptr);
               }),
           pybind11::arg("halfAxes"), pybind11::arg("gridDelta"))
      .def("getProcessModel", &BoxDistribution<T, D>::getProcessModel);

  // Oxide Regrowth
  pybind11::class_<OxideRegrowth<T, D>, SmartPointer<OxideRegrowth<T, D>>>(
      module, "OxideRegrowth")
      .def(
          pybind11::init(&SmartPointer<OxideRegrowth<T, D>>::New<
                         const T, const T, const T, const T, const T, const T,
                         const T, const T, const T, const T, const T, const T>),
          pybind11::arg("nitrideEtchRate"), pybind11::arg("oxideEtchRate"),
          pybind11::arg("redepositionRate"),
          pybind11::arg("redepositionThreshold"),
          pybind11::arg("redepositionTimeInt"),
          pybind11::arg("diffusionCoefficient"), pybind11::arg("sinkStrength"),
          pybind11::arg("scallopVelocity"), pybind11::arg("centerVelocity"),
          pybind11::arg("topHeight"), pybind11::arg("centerWidth"),
          pybind11::arg("stabilityFactor"))
      .def("getProcessModel", &OxideRegrowth<T, D>::getProcessModel);

  // Anisotropic Process
  pybind11::class_<AnisotropicProcess<T, D>,
                   SmartPointer<AnisotropicProcess<T, D>>>(module,
                                                           "AnisotropicProcess")
      .def(pybind11::init(&SmartPointer<AnisotropicProcess<T, D>>::New<
                          const std::vector<std::pair<Material, T>>>),
           pybind11::arg("materials"))
      .def(pybind11::init(&SmartPointer<AnisotropicProcess<T, D>>::New<
                          const std::array<T, 3> &, const std::array<T, 3> &,
                          const T, const T, const T, const T,
                          const std::vector<std::pair<Material, T>>>),
           pybind11::arg("direction100"), pybind11::arg("direction010"),
           pybind11::arg("rate100"), pybind11::arg("rate110"),
           pybind11::arg("rate111"), pybind11::arg("rate311"),
           pybind11::arg("materials"))
      .def("getProcessModel", &AnisotropicProcess<T, D>::getProcessModel);

  // Single Particle ALD
  pybind11::class_<SingleParticleALD<T, D>,
                   SmartPointer<SingleParticleALD<T, D>>>(module,
                                                          "SingleParticleALD")
      .def(pybind11::init(&SmartPointer<SingleParticleALD<T, D>>::New<
                          const T, const T, const T, const T, const T, const T,
                          const T, const T, const T>),
           pybind11::arg("stickingProbability"), pybind11::arg("numCycles"),
           pybind11::arg("growthPerCycle"), pybind11::arg("totalCycles"),
           pybind11::arg("coverageTimeStep"), pybind11::arg("evFlux"),
           pybind11::arg("inFlux"), pybind11::arg("s0"),
           pybind11::arg("gasMFP"))
      .def("getProcessModel", &SingleParticleALD<T, D>::getProcessModel);

  // ***************************************************************************
  //                               GEOMETRIES
  // ***************************************************************************

  // MakePlane
  pybind11::class_<MakePlane<T, D>, SmartPointer<MakePlane<T, D>>>(module,
                                                                   "MakePlane")
      .def(pybind11::init([](DomainType Domain, const T GridDelta,
                             const T XExtent, const T YExtent, const T Height,
                             const bool Periodic, const Material Material) {
             return SmartPointer<MakePlane<T, D>>::New(Domain, GridDelta,
                                                       XExtent, YExtent, Height,
                                                       Periodic, Material);
           }),
           pybind11::arg("domain"), pybind11::arg("gridDelta"),
           pybind11::arg("xExtent"), pybind11::arg("yExtent"),
           pybind11::arg("height") = 0.,
           pybind11::arg("periodicBoundary") = false,
           pybind11::arg("material") = Material::None)
      .def(pybind11::init(
               [](DomainType Domain, T Height, const Material Material) {
                 return SmartPointer<MakePlane<T, D>>::New(Domain, Height,
                                                           Material);
               }),
           pybind11::arg("domain"), pybind11::arg("height") = 0.,
           pybind11::arg("material") = Material::None)
      .def("apply", &MakePlane<T, D>::apply,
           "Create a plane geometry or add plane to existing geometry.");

  // MakeTrench
  pybind11::class_<MakeTrench<T, D>, SmartPointer<MakeTrench<T, D>>>(
      module, "MakeTrench")
      .def(pybind11::init([](DomainType Domain, const T GridDelta,
                             const T XExtent, const T YExtent,
                             const T TrenchWidth, const T TrenchDepth,
                             const T TaperingAngle, const T BaseHeight,
                             const bool PeriodicBoundary, const bool MakeMask,
                             const Material Material) {
             return SmartPointer<MakeTrench<T, D>>::New(
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
           pybind11::arg("material") = Material::None)
      .def("apply", &MakeTrench<T, D>::apply, "Create a trench geometry.");

  // MakeHole
  pybind11::class_<MakeHole<T, D>, SmartPointer<MakeHole<T, D>>>(module,
                                                                 "MakeHole")
      .def(pybind11::init([](DomainType domain, const T GridDelta,
                             const T xExtent, const T yExtent,
                             const T HoleRadius, const T HoleDepth,
                             const T TaperingAngle, const T BaseHeight,
                             const bool PeriodicBoundary, const bool MakeMask,
                             const Material material) {
             return SmartPointer<MakeHole<T, D>>::New(
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
           pybind11::arg("material") = Material::None)
      .def("apply", &MakeHole<T, D>::apply, "Create a hole geometry.");

  // MakeFin
  pybind11::class_<MakeFin<T, D>, SmartPointer<MakeFin<T, D>>>(module,
                                                               "MakeFin")
      .def(pybind11::init([](DomainType domain, const T gridDelta,
                             const T xExtent, const T yExtent, const T finWidth,
                             const T finHeight, const T taperAngle,
                             const T baseHeight, const bool periodicBoundary,
                             const bool makeMask, const Material material) {
             return SmartPointer<MakeFin<T, D>>::New(
                 domain, gridDelta, xExtent, yExtent, finWidth, finHeight,
                 taperAngle, baseHeight, periodicBoundary, makeMask, material);
           }),
           pybind11::arg("domain"), pybind11::arg("gridDelta"),
           pybind11::arg("xExtent"), pybind11::arg("yExtent"),
           pybind11::arg("finWidth"), pybind11::arg("finHeight"),
           pybind11::arg("taperAngle") = 0., pybind11::arg("baseHeight") = 0.,
           pybind11::arg("periodicBoundary") = false,
           pybind11::arg("makeMask") = false,
           pybind11::arg("material") = Material::None)
      .def("apply", &MakeFin<T, D>::apply, "Create a fin geometry.");

  // MakeStack
  pybind11::class_<MakeStack<T, D>, SmartPointer<MakeStack<T, D>>>(module,
                                                                   "MakeStack")
      .def(pybind11::init(&SmartPointer<MakeStack<T, D>>::New<
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
      .def("apply", &MakeStack<T, D>::apply,
           "Create a stack of alternating SiO2 and Si3N4 layers.")
      .def("getTopLayer", &MakeStack<T, D>::getTopLayer,
           "Returns the number of layers included in the stack")
      .def("getHeight", &MakeStack<T, D>::getHeight,
           "Returns the total height of the stack.");

  // ***************************************************************************
  //                                 PROCESS
  // ***************************************************************************

  // rayTraceDirection Enum
  pybind11::enum_<viennaray::TraceDirection>(module, "rayTraceDirection")
      .value("POS_X", viennaray::TraceDirection::POS_X)
      .value("POS_Y", viennaray::TraceDirection::POS_Y)
      .value("POS_Z", viennaray::TraceDirection::POS_Z)
      .value("NEG_X", viennaray::TraceDirection::NEG_X)
      .value("NEG_Y", viennaray::TraceDirection::NEG_Y)
      .value("NEG_Z", viennaray::TraceDirection::NEG_Z)
      .export_values();

  // AtomicLayerProcess
  pybind11::class_<AtomicLayerProcess<T, D>>(module, "AtomicLayerProcess")
      // constructors
      .def(pybind11::init())
      .def(pybind11::init<DomainType>(), pybind11::arg("domain"))
      //  .def(pybind11::init<DomainType, SmartPointer<ProcessModel<T, D>>>(),
      //       pybind11::arg("domain"), pybind11::arg("model"))
      // methods
      .def("apply", &AtomicLayerProcess<T, D>::apply, "Run the process.")
      .def("setDomain", &AtomicLayerProcess<T, D>::setDomain,
           "Set the process domain.")
      .def("setProcessModel", &AtomicLayerProcess<T, D>::setProcessModel,
           "Set the process model.")
      .def("setPulseTime", &AtomicLayerProcess<T, D>::setPulseTime,
           "Set the pulse time.")
      .def("setSourceDirection", &AtomicLayerProcess<T, D>::setSourceDirection,
           "Set source direction of the process.")
      .def("setNumberOfRaysPerPoint",
           &AtomicLayerProcess<T, D>::setNumberOfRaysPerPoint,
           "Set the number of rays to traced for each particle in the process. "
           "The number is per point in the process geometry.")
      .def("setDesorptionRates", &AtomicLayerProcess<T, D>::setDesorptionRates,
           "Set the desorption rate for each surface point.")
      .def("setCoverageTimeStep",
           &AtomicLayerProcess<T, D>::setCoverageTimeStep,
           "Set the time step for the coverage calculation.")
      .def("setIntegrationScheme",
           &AtomicLayerProcess<T, D>::setIntegrationScheme,
           "Set the integration scheme for solving the level-set equation. "
           "Possible integration schemes are specified in "
           "lsIntegrationSchemeEnum.")
      .def("setNumCycles", &AtomicLayerProcess<T, D>::setNumCycles,
           "Set the number of cycles for the process.")
      .def("enableRandomSeeds", &AtomicLayerProcess<T, D>::enableRandomSeeds,
           "Enable random seeds for the ray tracer. This will make the process "
           "results non-deterministic.")
      .def(
          "disableRandomSeeds", &AtomicLayerProcess<T, D>::disableRandomSeeds,
          "Disable random seeds for the ray tracer. This will make the process "
          "results deterministic.");

  // Process
  pybind11::class_<Process<T, D>>(module, "Process")
      // constructors
      .def(pybind11::init())
      .def(pybind11::init<DomainType>(), pybind11::arg("domain"))
      //  .def(pybind11::init<DomainType, SmartPointer<ProcessModel<T, D>>,
      //  T>(),
      //       pybind11::arg("domain"), pybind11::arg("model"),
      //       pybind11::arg("duration"))
      // methods
      .def("apply", &Process<T, D>::apply, "Run the process.")
      .def("calculateFlux", &Process<T, D>::calculateFlux,
           "Perform a single-pass flux calculation.")
      .def("setDomain", &Process<T, D>::setDomain, "Set the process domain.")
      .def("setProcessModel", &Process<T, D>::setProcessModel,
           "Set the process model.")
      .def("setProcessDuration", &Process<T, D>::setProcessDuration,
           "Set the process duration.")
      .def("setSourceDirection", &Process<T, D>::setSourceDirection,
           "Set source direction of the process.")
      .def("setNumberOfRaysPerPoint", &Process<T, D>::setNumberOfRaysPerPoint,
           "Set the number of rays to traced for each particle in the process. "
           "The number is per point in the process geometry.")
      .def("setMaxCoverageInitIterations",
           &Process<T, D>::setMaxCoverageInitIterations,
           "Set the number of iterations to initialize the coverages.")
      .def("setPrintTimeInterval", &Process<T, D>::setPrintTimeInterval,
           "Sets the minimum time between printing intermediate results during "
           "the process. If this is set to a non-positive value, no "
           "intermediate results are printed.")
      .def("setIntegrationScheme", &Process<T, D>::setIntegrationScheme,
           "Set the integration scheme for solving the level-set equation. "
           "Possible integration schemes are specified in "
           "lsIntegrationSchemeEnum.")
      .def("setTimeStepRatio", &Process<T, D>::setTimeStepRatio,
           "Set the CFL condition to use during advection. The CFL condition "
           "sets the maximum distance a surface can be moved during one "
           "advection step. It MUST be below 0.5 to guarantee numerical "
           "stability. Defaults to 0.4999.")
      .def("enableFluxSmoothing", &Process<T, D>::enableFluxSmoothing,
           "Enable flux smoothing. The flux at each surface point, calculated "
           "by the ray tracer, is averaged over the surface point neighbors.")
      .def("disableFluxSmoothing", &Process<T, D>::disableFluxSmoothing,
           "Disable flux smoothing")
      .def("enableRandomSeeds", &Process<T, D>::enableRandomSeeds,
           "Enable random seeds for the ray tracer. This will make the process "
           "results non-deterministic.")
      .def(
          "disableRandomSeeds", &Process<T, D>::disableRandomSeeds,
          "Disable random seeds for the ray tracer. This will make the process "
          "results deterministic.")
      .def("getProcessDuration", &Process<T, D>::getProcessDuration,
           "Returns the duration of the recently run process. This duration "
           "can sometimes slightly vary from the set process duration, due to "
           "the maximum time step according to the CFL condition.");

  // Domain
  pybind11::class_<Domain<T, D>, DomainType>(module, "Domain")
      // constructors
      .def(pybind11::init(&DomainType::New<>))
      // methods
      .def("deepCopy", &Domain<T, D>::deepCopy)
      .def("insertNextLevelSet", &Domain<T, D>::insertNextLevelSet,
           pybind11::arg("levelset"), pybind11::arg("wrapLowerLevelSet") = true,
           "Insert a level set to domain.")
      .def("insertNextLevelSetAsMaterial",
           &Domain<T, D>::insertNextLevelSetAsMaterial,
           pybind11::arg("levelSet"), pybind11::arg("material"),
           pybind11::arg("wrapLowerLevelSet") = true,
           "Insert a level set to domain as a material.")
      .def("duplicateTopLevelSet", &Domain<T, D>::duplicateTopLevelSet)
      .def("removeTopLevelSet", &Domain<T, D>::removeTopLevelSet)
      .def("applyBooleanOperation", &Domain<T, D>::applyBooleanOperation)
      .def("setMaterialMap", &Domain<T, D>::setMaterialMap)
      .def("getMaterialMap", &Domain<T, D>::getMaterialMap)
      .def("generateCellSet", &Domain<T, D>::generateCellSet,
           "Generate the cell set.")
      .def("getLevelSets", &Domain<T, D>::getLevelSets)
      .def("getCellSet", &Domain<T, D>::getCellSet, "Get the cell set.")
      .def("getGrid", &Domain<T, D>::getGrid, "Get the grid")
      .def("print", &Domain<T, D>::print)
      .def("saveLevelSetMesh", &Domain<T, D>::saveLevelSetMesh,
           pybind11::arg("filename"), pybind11::arg("width") = 1,
           "Save the level set grids of layers in the domain.")
      .def("saveSurfaceMesh", &Domain<T, D>::saveSurfaceMesh,
           pybind11::arg("filename"), pybind11::arg("addMaterialIds") = false,
           "Save the surface of the domain.")
      .def("saveVolumeMesh", &Domain<T, D>::saveVolumeMesh,
           pybind11::arg("filename"),
           "Save the volume representation of the domain.")
      .def("saveLevelSets", &Domain<T, D>::saveLevelSets)
      .def("clear", &Domain<T, D>::clear);

  // MaterialMap
  pybind11::class_<MaterialMap, SmartPointer<MaterialMap>>(module,
                                                           "MaterialMap")
      .def(pybind11::init<>())
      .def("insertNextMaterial", &MaterialMap::insertNextMaterial,
           pybind11::arg("material") = Material::None)
      .def("getMaterialAtIdx", &MaterialMap::getMaterialAtIdx)
      .def("getMaterialMap", &MaterialMap::getMaterialMap)
      .def("size", &MaterialMap::size)
      .def_static("mapToMaterial", &MaterialMap::mapToMaterial<T>,
                  "Map a float to a material.")
      .def_static("isMaterial", &MaterialMap::isMaterial<T>);

  // ***************************************************************************
  //                                   VISUALIZATION
  //  ***************************************************************************

  // visualization classes are not bound with smart pointer holder types
  // since they should not be passed to other classes
  pybind11::class_<ToDiskMesh<T, D>>(module, "ToDiskMesh")
      .def(pybind11::init<DomainType, SmartPointer<viennals::Mesh<T>>>(),
           pybind11::arg("domain"), pybind11::arg("mesh"))
      .def(pybind11::init())
      .def("setDomain", &ToDiskMesh<T, D>::setDomain,
           "Set the domain in the mesh converter.")
      .def("setMesh", &ToDiskMesh<T, D>::setMesh,
           "Set the mesh in the mesh converter");
  // static assertion failed: Holder classes are only supported for
  //  custom types
  // .def("setTranslator", &ToDiskMesh<T, D>::setTranslator,
  //      "Set the translator in the mesh converter. It used to convert "
  //      "level-set point IDs to mesh point IDs.")
  // .def("getTranslator", &ToDiskMesh<T, D>::getTranslator,
  //      "Retrieve the translator from the mesh converter.");

  //   ***************************************************************************
  //                                  OTHER
  //   ***************************************************************************

  // Constants
  auto m_constants =
      module.def_submodule("constants", "Physical and material constants.");
  m_constants.attr("kB") = constants::kB;
  m_constants.attr("roomTemperature") = constants::roomTemperature;
  m_constants.attr("N_A") = constants::N_A;
  m_constants.attr("R") = constants::R;
  m_constants.def("torrToPascal", &constants::torrToPascal,
                  "Convert pressure from torr to pascal.");
  m_constants.def("celsiusToKelvin", &constants::celsiusToKelvin,
                  "Convert temperature from Celsius to Kelvin.");
  m_constants.def("gasMeanFreePath", &constants::gasMeanFreePath,
                  "Calculate the mean free path of a gas molecule.");
  m_constants.def("gasMeanThermalVelocity", &constants::gasMeanThermalVelocity,
                  "Calculate the mean thermal velocity of a gas molecule.");

  // Planarize
  pybind11::class_<Planarize<T, D>, SmartPointer<Planarize<T, D>>>(module,
                                                                   "Planarize")
      .def(pybind11::init(&SmartPointer<Planarize<T, D>>::New<>))
      .def(pybind11::init(
               &SmartPointer<Planarize<T, D>>::New<DomainType &, const T>),
           pybind11::arg("geometry"), pybind11::arg("cutoffHeight") = 0.)
      .def("setDomain", &Planarize<T, D>::setDomain,
           "Set the domain in the planarization.")
      .def("setCutoffPosition", &Planarize<T, D>::setCutoffPosition,
           "Set the cutoff height for the planarization.")
      .def("apply", &Planarize<T, D>::apply, "Apply the planarization.");

#if VIENNAPS_PYTHON_DIMENSION > 2
  // GDS file parsing
  pybind11::class_<GDSGeometry<T, D>, SmartPointer<GDSGeometry<T, D>>>(
      module, "GDSGeometry")
      // constructors
      .def(pybind11::init(&SmartPointer<GDSGeometry<T, D>>::New<>))
      .def(pybind11::init(&SmartPointer<GDSGeometry<T, D>>::New<const T>),
           pybind11::arg("gridDelta"))
      // methods
      .def("setGridDelta", &GDSGeometry<T, D>::setGridDelta,
           "Set the grid spacing.")
      .def(
          "setBoundaryConditions",
          [](GDSGeometry<T, D> &gds,
             std::vector<typename viennals::Domain<T, D>::BoundaryType> &bcs) {
            if (bcs.size() == D)
              gds.setBoundaryConditions(bcs.data());
          },
          "Set the boundary conditions")
      .def("setBoundaryPadding", &GDSGeometry<T, D>::setBoundaryPadding,
           "Set padding between the largest point of the geometry and the "
           "boundary of the domain.")
      .def("print", &GDSGeometry<T, D>::print, "Print the geometry contents.")
      .def("layerToLevelSet", &GDSGeometry<T, D>::layerToLevelSet,
           "Convert a layer of the GDS geometry to a level set domain.")
      .def(
          "getBounds",
          [](GDSGeometry<T, D> &gds) -> std::array<double, 6> {
            auto b = gds.getBounds();
            std::array<double, 6> bounds;
            for (unsigned i = 0; i < 6; ++i)
              bounds[i] = b[i];
            return bounds;
          },
          "Get the bounds of the geometry.");

  pybind11::class_<GDSReader<T, D>, SmartPointer<GDSReader<T, D>>>(module,
                                                                   "GDSReader")
      // constructors
      .def(pybind11::init(&SmartPointer<GDSReader<T, D>>::New<>))
      .def(pybind11::init(
          &SmartPointer<GDSReader<T, D>>::New<SmartPointer<GDSGeometry<T, D>> &,
                                              std::string>))
      // methods
      .def("setGeometry", &GDSReader<T, D>::setGeometry,
           "Set the domain to be parsed in.")
      .def("setFileName", &GDSReader<T, D>::setFileName,
           "Set name of the GDS file.")
      .def("apply", &GDSReader<T, D>::apply, "Parse the GDS file.");
#else
  // wrap a 3D domain in 2D mode to be used with psExtrude
  // Domain
  pybind11::class_<Domain<T, 3>, SmartPointer<Domain<T, 3>>>(module, "Domain3D")
      // constructors
      .def(pybind11::init(&SmartPointer<Domain<T, 3>>::New<>))
      // methods
      .def("deepCopy", &Domain<T, 3>::deepCopy)
      .def("insertNextLevelSet", &Domain<T, 3>::insertNextLevelSet,
           pybind11::arg("levelSet"), pybind11::arg("wrapLowerLevelSet") = true,
           "Insert a level set to domain.")
      .def("insertNextLevelSetAsMaterial",
           &Domain<T, 3>::insertNextLevelSetAsMaterial,
           pybind11::arg("levelSet"), pybind11::arg("material"),
           pybind11::arg("wrapLowerLevelSet") = true,
           "Insert a level set to domain as a material.")
      .def("duplicateTopLevelSet", &Domain<T, 3>::duplicateTopLevelSet)
      .def("applyBooleanOperation", &Domain<T, 3>::applyBooleanOperation)
      .def("removeTopLevelSet", &Domain<T, 3>::removeTopLevelSet)
      .def("setMaterialMap", &Domain<T, 3>::setMaterialMap)
      .def("getMaterialMap", &Domain<T, 3>::getMaterialMap)
      .def("generateCellSet", &Domain<T, 3>::generateCellSet,
           pybind11::arg("position"), pybind11::arg("coverMaterial"),
           pybind11::arg("isAboveSurface"), "Generate the cell set.")
      .def("getLevelSets", &Domain<T, 3>::getLevelSets)
      .def("getCellSet", &Domain<T, 3>::getCellSet, "Get the cell set.")
      .def("getGrid", &Domain<T, 3>::getGrid, "Get the grid")
      .def("print", &Domain<T, 3>::print)
      .def("saveLevelSetMesh", &Domain<T, 3>::saveLevelSetMesh,
           pybind11::arg("filename"), pybind11::arg("width") = 1,
           "Save the level set grids of layers in the domain.")
      .def("saveSurfaceMesh", &Domain<T, 3>::saveSurfaceMesh,
           pybind11::arg("filename"), pybind11::arg("addMaterialIds") = true,
           "Save the surface of the domain.")
      .def("saveVolumeMesh", &Domain<T, 3>::saveVolumeMesh,
           pybind11::arg("filename"),
           "Save the volume representation of the domain.")
      .def("saveLevelSets", &Domain<T, 3>::saveLevelSets)
      .def("clear", &Domain<T, 3>::clear);

  pybind11::class_<Extrude<T>>(module, "Extrude")
      .def(pybind11::init())
      .def(pybind11::init<SmartPointer<Domain<T, 2>> &,
                          SmartPointer<Domain<T, 3>> &, std::array<T, 2>,
                          const int,
                          std::array<viennals::BoundaryConditionEnum<3>, 3>>(),
           pybind11::arg("inputDomain"), pybind11::arg("outputDomain"),
           pybind11::arg("extent"), pybind11::arg("extrudeDimension"),
           pybind11::arg("boundaryConditions"))
      .def("setInputDomain", &Extrude<T>::setInputDomain,
           "Set the input domain to be extruded.")
      .def("setOutputDomain", &Extrude<T>::setOutputDomain,
           "Set the output domain. The 3D output domain will be overwritten by "
           "the extruded domain.")
      .def("setExtent", &Extrude<T>::setExtent,
           "Set the min and max extent in the extruded dimension.")
      .def("setExtrudeDimension", &Extrude<T>::setExtrudeDimension,
           "Set which index of the added dimension (x: 0, y: 1, z: 2).")
      .def("setBoundaryConditions",
           pybind11::overload_cast<
               std::array<viennals::BoundaryConditionEnum<3>, 3>>(
               &Extrude<T>::setBoundaryConditions),
           "Set the boundary conditions in the extruded domain.")
      .def("apply", &Extrude<T>::apply, "Run the extrusion.");
#endif

  //   // rayReflection.hpp
  //   module.def("rayReflectionSpecular", &rayReflectionSpecular<T>,
  //              "Specular reflection,");
  //   module.def("rayReflectionDiffuse", &rayReflectionDiffuse<T, D>,
  //              "Diffuse reflection.");
  //   module.def("rayReflectionConedCosine", &rayReflectionConedCosine<T, D>,
  //              "Coned cosine reflection.");
}
