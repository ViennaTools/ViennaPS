
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

// model framework
#include <psAdvectionCallback.hpp>
#include <psProcessModel.hpp>
#include <psProcessParams.hpp>
#include <psSurfaceModel.hpp>
#include <psVelocityField.hpp>

// models
#include <models/psAnisotropicProcess.hpp>
#include <models/psAtomicLayerProcess.hpp>
#include <models/psDirectionalEtching.hpp>
#include <models/psFluorocarbonEtching.hpp>
#include <models/psGeometricDistributionModels.hpp>
#include <models/psIsotropicProcess.hpp>
#include <models/psOxideRegrowth.hpp>
#include <models/psPlasmaDamage.hpp>
#include <models/psSF6O2Etching.hpp>
#include <models/psSingleParticleProcess.hpp>
#include <models/psTEOSDeposition.hpp>

// always use double for python export
typedef double T;
// get dimension from cmake define
inline constexpr int D = VIENNAPS_PYTHON_DIMENSION;
typedef psSmartPointer<psDomain<T, D>> DomainType;

PYBIND11_DECLARE_HOLDER_TYPE(Types, psSmartPointer<Types>)

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

void initModels(pybind11::module_ &module) {
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
           pybind11::arg("maskMaterial") = psMaterial::None)
      .def(pybind11::init([](const T rate, const T sticking, const T power,
                             const std::vector<psMaterial> mask) {
             return psSmartPointer<psSingleParticleProcess<T, D>>::New(
                 rate, sticking, power, mask);
           }),
           pybind11::arg("rate"), pybind11::arg("stickingProbability"),
           pybind11::arg("sourceExponent"), pybind11::arg("maskMaterials"));

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

  // SF6O2 Parameters
  pybind11::class_<SF6O2Implementation::Parameters<T>::MaskType>(
      module, "SF6O2ParametersMask")
      .def(pybind11::init<>())
      .def_readwrite("rho", &SF6O2Implementation::Parameters<T>::MaskType::rho)
      .def_readwrite("beta_F",
                     &SF6O2Implementation::Parameters<T>::MaskType::beta_F)
      .def_readwrite("beta_O",
                     &SF6O2Implementation::Parameters<T>::MaskType::beta_O)
      .def_readwrite("A_sp",
                     &SF6O2Implementation::Parameters<T>::MaskType::A_sp)
      .def_readwrite("B_sp",
                     &SF6O2Implementation::Parameters<T>::MaskType::B_sp)
      .def_readwrite("Eth_sp",
                     &SF6O2Implementation::Parameters<T>::MaskType::Eth_sp);

  pybind11::class_<SF6O2Implementation::Parameters<T>::SiType>(
      module, "SF6O2ParametersSi")
      .def(pybind11::init<>())
      .def_readwrite("rho", &SF6O2Implementation::Parameters<T>::SiType::rho)
      .def_readwrite("k_sigma",
                     &SF6O2Implementation::Parameters<T>::SiType::k_sigma)
      .def_readwrite("beta_sigma",
                     &SF6O2Implementation::Parameters<T>::SiType::beta_sigma)
      .def_readwrite("A_sp", &SF6O2Implementation::Parameters<T>::SiType::A_sp)
      .def_readwrite("B_sp", &SF6O2Implementation::Parameters<T>::SiType::B_sp)
      .def_readwrite("Eth_ie",
                     &SF6O2Implementation::Parameters<T>::SiType::Eth_ie)
      .def_readwrite("Eth_sp",
                     &SF6O2Implementation::Parameters<T>::SiType::Eth_sp)
      .def_readwrite("A_ie", &SF6O2Implementation::Parameters<T>::SiType::A_ie);

  pybind11::class_<SF6O2Implementation::Parameters<T>::PassivationType>(
      module, "SF6O2ParametersPassivation")
      .def(pybind11::init<>())
      .def_readwrite(
          "Eth_ie",
          &SF6O2Implementation::Parameters<T>::PassivationType::Eth_ie)
      .def_readwrite(
          "A_ie", &SF6O2Implementation::Parameters<T>::PassivationType::A_ie);

  pybind11::class_<SF6O2Implementation::Parameters<T>::IonType>(
      module, "SF6O2ParametersIons")
      .def(pybind11::init<>())
      .def_readwrite("meanEnergy",
                     &SF6O2Implementation::Parameters<T>::IonType::meanEnergy)
      .def_readwrite("sigmaEnergy",
                     &SF6O2Implementation::Parameters<T>::IonType::sigmaEnergy)
      .def_readwrite("exponent",
                     &SF6O2Implementation::Parameters<T>::IonType::exponent)
      .def_readwrite("inflectAngle",
                     &SF6O2Implementation::Parameters<T>::IonType::inflectAngle)
      .def_readwrite("n_l", &SF6O2Implementation::Parameters<T>::IonType::n_l)
      .def_readwrite("minAngle",
                     &SF6O2Implementation::Parameters<T>::IonType::minAngle);

  pybind11::class_<SF6O2Implementation::Parameters<T>>(module,
                                                       "SF6O2Parameters")
      .def(pybind11::init<>())
      .def_readwrite("ionFlux", &SF6O2Implementation::Parameters<T>::ionFlux)
      .def_readwrite("etchantFlux",
                     &SF6O2Implementation::Parameters<T>::etchantFlux)
      .def_readwrite("oxygenFlux",
                     &SF6O2Implementation::Parameters<T>::oxygenFlux)
      .def_readwrite("etchStopDepth",
                     &SF6O2Implementation::Parameters<T>::etchStopDepth)
      .def_readwrite("beta_F", &SF6O2Implementation::Parameters<T>::beta_F)
      .def_readwrite("beta_O", &SF6O2Implementation::Parameters<T>::beta_O)
      .def_readwrite("Mask", &SF6O2Implementation::Parameters<T>::Mask)
      .def_readwrite("Si", &SF6O2Implementation::Parameters<T>::Si)
      .def_readwrite("Polymer",
                     &SF6O2Implementation::Parameters<T>::Passivation)
      .def_readwrite("Ions", &SF6O2Implementation::Parameters<T>::Ions);

  // SF6O2 Etching
  pybind11::class_<psSF6O2Etching<T, D>, psSmartPointer<psSF6O2Etching<T, D>>>(
      module, "SF6O2Etching", processModel)
      .def(pybind11::init<>())
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
           pybind11::arg("etchStopDepth") = std::numeric_limits<T>::lowest())
      .def(pybind11::init(&psSmartPointer<psSF6O2Etching<T, D>>::New<
                          const SF6O2Implementation::Parameters<T> &>),
           pybind11::arg("parameters"))
      .def("setParameters", &psSF6O2Etching<T, D>::setParameters)
      .def("getParameters", &psSF6O2Etching<T, D>::getParameters,
           pybind11::return_value_policy::reference);

  // Fluorocarbon Parameters
  pybind11::class_<FluorocarbonImplementation::Parameters<T>::MaskType>(
      module, "FluorocarbonParametersMask")
      .def(pybind11::init<>())
      .def_readwrite("rho",
                     &FluorocarbonImplementation::Parameters<T>::MaskType::rho)
      .def_readwrite(
          "beta_p",
          &FluorocarbonImplementation::Parameters<T>::MaskType::beta_p)
      .def_readwrite(
          "beta_e",
          &FluorocarbonImplementation::Parameters<T>::MaskType::beta_e)
      .def_readwrite("A_sp",
                     &FluorocarbonImplementation::Parameters<T>::MaskType::A_sp)
      .def_readwrite("B_sp",
                     &FluorocarbonImplementation::Parameters<T>::MaskType::B_sp)
      .def_readwrite(
          "Eth_sp",
          &FluorocarbonImplementation::Parameters<T>::MaskType::Eth_sp);

  pybind11::class_<FluorocarbonImplementation::Parameters<T>::SiO2Type>(
      module, "FluorocarbonParametersSiO2")
      .def(pybind11::init<>())
      .def_readwrite("rho",
                     &FluorocarbonImplementation::Parameters<T>::SiO2Type::rho)
      .def_readwrite("E_a",
                     &FluorocarbonImplementation::Parameters<T>::SiO2Type::E_a)
      .def_readwrite("K",
                     &FluorocarbonImplementation::Parameters<T>::SiO2Type::K)
      .def_readwrite("A_sp",
                     &FluorocarbonImplementation::Parameters<T>::SiO2Type::A_sp)
      .def_readwrite("B_sp",
                     &FluorocarbonImplementation::Parameters<T>::SiO2Type::B_sp)
      .def_readwrite(
          "Eth_ie",
          &FluorocarbonImplementation::Parameters<T>::SiO2Type::Eth_ie)
      .def_readwrite(
          "Eth_sp",
          &FluorocarbonImplementation::Parameters<T>::SiO2Type::Eth_sp)
      .def_readwrite(
          "A_ie", &FluorocarbonImplementation::Parameters<T>::SiO2Type::A_ie);

  pybind11::class_<FluorocarbonImplementation::Parameters<T>::Si3N4Type>(
      module, "FluorocarbonParametersSi3N4")
      .def(pybind11::init<>())
      .def_readwrite("rho",
                     &FluorocarbonImplementation::Parameters<T>::Si3N4Type::rho)
      .def_readwrite("E_a",
                     &FluorocarbonImplementation::Parameters<T>::Si3N4Type::E_a)
      .def_readwrite("K",
                     &FluorocarbonImplementation::Parameters<T>::Si3N4Type::K)
      .def_readwrite(
          "A_sp", &FluorocarbonImplementation::Parameters<T>::Si3N4Type::A_sp)
      .def_readwrite(
          "B_sp", &FluorocarbonImplementation::Parameters<T>::Si3N4Type::B_sp)
      .def_readwrite(
          "Eth_ie",
          &FluorocarbonImplementation::Parameters<T>::Si3N4Type::Eth_ie)
      .def_readwrite(
          "Eth_sp",
          &FluorocarbonImplementation::Parameters<T>::Si3N4Type::Eth_sp)
      .def_readwrite(
          "A_ie", &FluorocarbonImplementation::Parameters<T>::Si3N4Type::A_ie);

  pybind11::class_<FluorocarbonImplementation::Parameters<T>::SiType>(
      module, "FluorocarbonParametersSi")
      .def(pybind11::init<>())
      .def_readwrite("rho",
                     &FluorocarbonImplementation::Parameters<T>::SiType::rho)
      .def_readwrite("E_a",
                     &FluorocarbonImplementation::Parameters<T>::SiType::E_a)
      .def_readwrite("K", &FluorocarbonImplementation::Parameters<T>::SiType::K)
      .def_readwrite("A_sp",
                     &FluorocarbonImplementation::Parameters<T>::SiType::A_sp)
      .def_readwrite("B_sp",
                     &FluorocarbonImplementation::Parameters<T>::SiType::B_sp)
      .def_readwrite("Eth_ie",
                     &FluorocarbonImplementation::Parameters<T>::SiType::Eth_ie)
      .def_readwrite("Eth_sp",
                     &FluorocarbonImplementation::Parameters<T>::SiType::Eth_sp)
      .def_readwrite("A_ie",
                     &FluorocarbonImplementation::Parameters<T>::SiType::A_ie);

  pybind11::class_<FluorocarbonImplementation::Parameters<T>::PolymerType>(
      module, "FluorocarbonParametersPolymer")
      .def(pybind11::init<>())
      .def_readwrite(
          "rho", &FluorocarbonImplementation::Parameters<T>::PolymerType::rho)
      .def_readwrite(
          "Eth_ie",
          &FluorocarbonImplementation::Parameters<T>::PolymerType::Eth_ie)
      .def_readwrite(
          "A_ie",
          &FluorocarbonImplementation::Parameters<T>::PolymerType::A_ie);

  pybind11::class_<FluorocarbonImplementation::Parameters<T>::IonType>(
      module, "FluorocarbonParametersIons")
      .def(pybind11::init<>())
      .def_readwrite(
          "meanEnergy",
          &FluorocarbonImplementation::Parameters<T>::IonType::meanEnergy)
      .def_readwrite(
          "sigmaEnergy",
          &FluorocarbonImplementation::Parameters<T>::IonType::sigmaEnergy)
      .def_readwrite(
          "exponent",
          &FluorocarbonImplementation::Parameters<T>::IonType::exponent)
      .def_readwrite(
          "inflectAngle",
          &FluorocarbonImplementation::Parameters<T>::IonType::inflectAngle)
      .def_readwrite("n_l",
                     &FluorocarbonImplementation::Parameters<T>::IonType::n_l)
      .def_readwrite(
          "minAngle",
          &FluorocarbonImplementation::Parameters<T>::IonType::minAngle);

  pybind11::class_<FluorocarbonImplementation::Parameters<T>>(
      module, "FluorocarbonParameters")
      .def(pybind11::init<>())
      .def_readwrite("ionFlux",
                     &FluorocarbonImplementation::Parameters<T>::ionFlux)
      .def_readwrite("etchantFlux",
                     &FluorocarbonImplementation::Parameters<T>::etchantFlux)
      .def_readwrite("polyFlux",
                     &FluorocarbonImplementation::Parameters<T>::polyFlux)
      .def_readwrite("delta_p",
                     &FluorocarbonImplementation::Parameters<T>::delta_p)
      .def_readwrite("etchStopDepth",
                     &FluorocarbonImplementation::Parameters<T>::etchStopDepth)
      .def_readwrite("Mask", &FluorocarbonImplementation::Parameters<T>::Mask)
      .def_readwrite("SiO2", &FluorocarbonImplementation::Parameters<T>::SiO2)
      .def_readwrite("Si3N4", &FluorocarbonImplementation::Parameters<T>::Si3N4)
      .def_readwrite("Si", &FluorocarbonImplementation::Parameters<T>::Si)
      .def_readwrite("Polymer",
                     &FluorocarbonImplementation::Parameters<T>::Polymer)
      .def_readwrite("Ions", &FluorocarbonImplementation::Parameters<T>::Ions);

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
                          const FluorocarbonImplementation::Parameters<T> &>),
           pybind11::arg("parameters"))
      .def("setParameters", &psFluorocarbonEtching<T, D>::setParameters)
      .def("getParameters", &psFluorocarbonEtching<T, D>::getParameters,
           pybind11::return_value_policy::reference);

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
}