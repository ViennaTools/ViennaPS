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

#include <pybind11/functional.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

// all header files which define API functions
#include <pscuProcess.hpp>
#include <pscuSF6O2Etching.hpp>

using namespace viennaps;

// always use double for python export
typedef double T;
constexpr int D = 3;
typedef SmartPointer<Domain<T, D>> DomainType;

PYBIND11_DECLARE_HOLDER_TYPE(Types, SmartPointer<Types>)

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

  // SF6O2 Etching
  pybind11::class_<SF6O2Etching<T, D>, SmartPointer<SF6O2Etching<T, D>>>(
      module, "SF6O2Etching", processModel)
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
           pybind11::return_value_policy::reference);

  //   // Ion Beam Etching
  //   pybind11::class_<IBEParameters<T>>(module, "IBEParameters")
  //       .def(pybind11::init<>())
  //       .def_readwrite("planeWaferRate", &IBEParameters<T>::planeWaferRate)
  //       .def_readwrite("meanEnergy", &IBEParameters<T>::meanEnergy)
  //       .def_readwrite("sigmaEnergy", &IBEParameters<T>::sigmaEnergy)
  //       .def_readwrite("thresholdEnergy", &IBEParameters<T>::thresholdEnergy)
  //       .def_readwrite("exponent", &IBEParameters<T>::exponent)
  //       .def_readwrite("n_l", &IBEParameters<T>::n_l)
  //       .def_readwrite("inflectAngle", &IBEParameters<T>::inflectAngle)
  //       .def_readwrite("minAngle", &IBEParameters<T>::minAngle)
  //       .def_readwrite("tiltAngle", &IBEParameters<T>::tiltAngle)
  //       .def_readwrite("yieldFunction", &IBEParameters<T>::yieldFunction);

  //   pybind11::class_<IonBeamEtching<T, D>, SmartPointer<IonBeamEtching<T,
  //   D>>>(
  //       module, "IonBeamEtching", processModel)
  //       .def(pybind11::init<>())
  //       .def(pybind11::init(&SmartPointer<IonBeamEtching<T, D>>::New<
  //                           const std::vector<Material> &>),
  //            pybind11::arg("maskMaterials"))
  //       .def(pybind11::init(
  //                &SmartPointer<IonBeamEtching<T, D>>::New<
  //                    const std::vector<Material> &, const IBEParameters<T>
  //                    &>),
  //            pybind11::arg("maskMaterials"), pybind11::arg("parameters"))
  //       .def("setParameters", &IonBeamEtching<T, D>::setParameters)
  //       .def("getParameters", &IonBeamEtching<T, D>::getParameters,
  //            pybind11::return_value_policy::reference);

  //   // Faraday Cage Etching
  //   pybind11::class_<FaradayCageParameters<T>>(module,
  //   "FaradayCageParameters")
  //       .def(pybind11::init<>())
  //       .def_readwrite("ibeParams", &FaradayCageParameters<T>::ibeParams)
  //       .def_readwrite("cageAngle", &FaradayCageParameters<T>::cageAngle);

  //   pybind11::class_<FaradayCageEtching<T, D>,
  //                    SmartPointer<FaradayCageEtching<T, D>>>(
  //       module, "FaradayCageEtching", processModel)
  //       .def(pybind11::init<>())
  //       .def(pybind11::init(&SmartPointer<FaradayCageEtching<T, D>>::New<
  //                           const std::vector<Material> &>),
  //            pybind11::arg("maskMaterials"))
  //       .def(pybind11::init(&SmartPointer<FaradayCageEtching<T, D>>::New<
  //                           const std::vector<Material> &,
  //                           const FaradayCageParameters<T> &>),
  //            pybind11::arg("maskMaterials"), pybind11::arg("parameters"))
  //       .def("setParameters", &FaradayCageEtching<T, D>::setParameters)
  //       .def("getParameters", &FaradayCageEtching<T, D>::getParameters,
  //            pybind11::return_value_policy::reference);

  // Process
  pybind11::class_<Process<T, D>>(module, "Process")
      // constructors
      .def(pybind11::init())
      .def(pybind11::init<DomainType>(), pybind11::arg("domain"))
      .def(pybind11::init<DomainType, SmartPointer<ProcessModel<T, D>>, T>(),
           pybind11::arg("domain"), pybind11::arg("model"),
           pybind11::arg("duration"))
      // methods
      .def("apply", &Process<T, D>::apply, "Run the process.")
      .def("calculateFlux", &Process<T, D>::calculateFlux,
           "Perform a single-pass flux calculation.")
      .def("setDomain", &Process<T, D>::setDomain, "Set the process domain.")
      .def("setProcessModel", &Process<T, D>::setProcessModel,
           "Set the process model. This has to be a pre-configured process "
           "model.")
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
      .def("setIntegrationScheme", &Process<T, D>::setIntegrationScheme,
           "Set the integration scheme for solving the level-set equation. "
           "Possible integration schemes are specified in "
           "viennals::IntegrationSchemeEnum.")
      .def("enableAdvectionVelocityOutput",
           &Process<T, D>::enableAdvectionVelocityOutput,
           "Enable the output of the advection velocity field on the ls-mesh.")
      .def("disableAdvectionVelocityOutput",
           &Process<T, D>::disableAdvectionVelocityOutput,
           "Disable the output of the advection velocity field on the ls-mesh.")
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
      .def("setRayTracingDiskRadius", &Process<T, D>::setRayTracingDiskRadius,
           "Set the radius of the disk used for ray tracing. This disk is used "
           "for the intersection calculations at each surface point.")
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
}
