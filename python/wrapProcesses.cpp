#include "pyWrapp.hpp"

void wrapProcesses(pybind11::module_ &module) {
  /****************************************************************************
   *                               PROCESS                                    *
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
      .def("setPrintTimeInterval", &psProcess<T, D>::setPrintTimeInterval,
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

  // enums
  pybind11::enum_<rayTraceDirection>(module, "rayTraceDirection")
      .value("POS_X", rayTraceDirection::POS_X)
      .value("POS_Y", rayTraceDirection::POS_Y)
      .value("POS_Z", rayTraceDirection::POS_Z)
      .value("NEG_X", rayTraceDirection::NEG_X)
      .value("NEG_Y", rayTraceDirection::NEG_Y)
      .value("NEG_Z", rayTraceDirection::NEG_Z);
}