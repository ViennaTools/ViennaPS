/*
  This file is used to generate the python module of viennaps.
  It uses pybind11 to create the modules.
  All necessary headers are included here and the interface
  of the classes which should be exposed defined
*/

// correct module name macro
#define TOKENPASTE_INTERNAL(x, y, z) x##y##z
#define TOKENPASTE(x, y, z) TOKENPASTE_INTERNAL(x, y, z)
#define VIENNAPS_MODULE_NAME TOKENPASTE(viennaps, VIENNAPS_PYTHON_DIMENSION, d)
#define STRINGIZE2(s) #s
#define STRINGIZE(s) STRINGIZE2(s)
#define VIENNAPS_MODULE_VERSION STRINGIZE(VIENNAPS_VERSION)

#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// all header files which define API functions
#include <psDomain.hpp>
#include <psGDSGeometry.hpp>
#include <psGDSReader.hpp>
#include <psProcess.hpp>
#include <psProcessModel.hpp>
#include <psSurfaceModel.hpp>

// geometries
#include <psMakeHole.hpp>
#include <psMakePlane.hpp>
#include <psMakeTrench.hpp>

// models
#include <GeometricUniformDeposition.hpp>
#include <SF6O2Etching.hpp>
#include <SimpleDeposition.hpp>

// other
#include <lsDomain.hpp>
#include <rayTraceDirection.hpp>

// always use double for python export
typedef double T;
// get dimension from cmake define
constexpr int D = VIENNAPS_PYTHON_DIMENSION;

PYBIND11_DECLARE_HOLDER_TYPE(TemplateType, psSmartPointer<TemplateType>);

// define trampoline classes for interface functions
// ALSO NEED TO ADD TRAMPOLINE CLASSES FOR CLASSES
// WHICH HOLD REFERENCES TO INTERFACE(ABSTRACT) CLASSES

// BASE CLASS WRAPPERS
class PypsSurfaceModel : public psSurfaceModel<T> {
  using psSurfaceModel<T>::Coverages;
  using psSurfaceModel<T>::processParams;

public:
  void initializeCoverages(unsigned numGeometryPoints) override {
    PYBIND11_OVERLOAD(T, psSurfaceModel<T>, initializeCoverages,
                      numGeometryPoints);
  }

  void initializeProcessParameters() override {
    PYBIND11_OVERLOAD(T, psSurfaceModel<T>, initializeProcessParameters);
  }

  psSmartPointer<std::vector<T>>
  calculateVelocities(psSmartPointer<psPointData<T>> Rates,
                      const std::vector<T> &materialIDs) override {
    PYBIND11_OVERLOAD(T, psSurfaceModel<T>, calculateVelocities, Rates,
                      materialIDs);
  }

  void updateCoverages(psSmartPointer<psPointData<T>> Rates) override {
    PYBIND11_OVERLOAD(T, psSurfaceModel<T>, updateCoverages, Rates);
  }
};

PYBIND11_MODULE(VIENNAPS_MODULE_NAME, module) {
  module.doc() =
      "ViennaPS is a header-only C++ process simulation library, which "
      "includes surface and volume representations, a ray tracer, and physical "
      "models for the simulation of microelectronic fabrication processes. The "
      "main design goals are simplicity and efficiency, tailored towards "
      "scientific simulations.";

  // set version string of python module
  module.attr("__version__") = VIENNAPS_MODULE_VERSION;

  // wrap omp_set_num_threads to control number of threads
  module.def("setNumThreads", &omp_set_num_threads);

  // psDomain
  pybind11::class_<psDomain<T, D>, psSmartPointer<psDomain<T, D>>>(module,
                                                                   "psDomain")
      // constructors
      .def(pybind11::init(&psSmartPointer<psDomain<T, D>>::New<>))
      .def(pybind11::init(
          &psSmartPointer<psDomain<T, D>>::New<psDomain<T, D>::lsDomainType &>))

      // methods
      .def("insertNextLevelSet", &psDomain<T, D>::insertNextLevelSet,
           "Insert a level set to domain.")
      .def("printSurface", &psDomain<T, D>::printSurface,
           "Print the surface of the domain.");

  pybind11::class_<psMakeTrench<T, D>, psSmartPointer<psMakeTrench<T, D>>>(
      module, "psMakeTrench")
      .def(pybind11::init(
               &psSmartPointer<psMakeTrench<T, D>>::New<
                   psSmartPointer<psDomain<T, D>> &, const T /*GridDelta*/,
                   const T /*XExtent*/, const T /*YExtent*/,
                   const T /*TrenchWidth*/, const T /*TrenchHeight*/,
                   const T /*TaperingAngle*/, const T /*BaseHeight*/,
                   const bool /*PeriodicBoundary*/, const bool /*MakeMask*/>),
           pybind11::arg("psDomain"), pybind11::arg("gridDelta"),
           pybind11::arg("xExtent"), pybind11::arg("yExtent"),
           pybind11::arg("trenchWidth"), pybind11::arg("trenchHeight"),
           pybind11::arg("taperingAngle") = 0.,
           pybind11::arg("baseHeight") = 0.,
           pybind11::arg("periodicBoundary") = false,
           pybind11::arg("makeMask") = false)
      .def("apply", &psMakeTrench<T, D>::apply, "Make trench.");

  pybind11::class_<psMakeHole<T, D>, psSmartPointer<psMakeHole<T, D>>>(
      module, "psMakeHole")
      .def(pybind11::init(
               &psSmartPointer<psMakeHole<T, D>>::New<
                   psSmartPointer<psDomain<T, D>> &, const T /*GridDelta*/,
                   const T /*XExtent*/, const T /*YExtent*/,
                   const T /*HoleRadius*/, const T /*HoleDepth*/,
                   const T /*TaperingAngle*/, const T /*BaseHeight*/,
                   const bool /*PeriodicBoundary*/, const bool /*MakeMask*/>),
           pybind11::arg("psDomain"), pybind11::arg("gridDelta"),
           pybind11::arg("xExtent"), pybind11::arg("yExtent"),
           pybind11::arg("holeRadius"), pybind11::arg("holeDepth"),
           pybind11::arg("taperingAngle") = 0.,
           pybind11::arg("baseHeight") = 0.,
           pybind11::arg("periodicBoundary") = false,
           pybind11::arg("makeMask") = false)
      .def("apply", &psMakeHole<T, D>::apply, "Make hole.");

  pybind11::class_<psMakePlane<T, D>, psSmartPointer<psMakePlane<T, D>>>(
      module, "psMakePlane")
      .def(pybind11::init(
               &psSmartPointer<psMakePlane<T, D>>::New<
                   psSmartPointer<psDomain<T, D>> &, const T /*GridDelta*/,
                   const T /*XExtent*/, const T /*YExtent*/, const T /*Height*/,
                   const bool /*Periodic*/>),
           pybind11::arg("psDomain"), pybind11::arg("gridDelta"),
           pybind11::arg("xExtent"), pybind11::arg("yExtent"),
           pybind11::arg("height") = 0.,
           pybind11::arg("periodicBoundary") = false)
      .def("apply", &psMakePlane<T, D>::apply, "Make plane.");

  // enums
  pybind11::enum_<rayTraceDirection>(module, "rayTraceDirection")
      .value("POS_X", rayTraceDirection::POS_X)
      .value("POS_Y", rayTraceDirection::POS_Y)
      .value("POS_Z", rayTraceDirection::POS_Z)
      .value("NEG_X", rayTraceDirection::NEG_X)
      .value("NEG_Y", rayTraceDirection::NEG_Y)
      .value("NEG_Z", rayTraceDirection::NEG_Z);

  // psProcessModel
  pybind11::class_<psProcessModel<T, D>, psSmartPointer<psProcessModel<T, D>>>(
      module, "psProcessModel")
      // constructors
      .def(pybind11::init(&psSmartPointer<psProcessModel<T, D>>::New<>));

  // psProcess
  pybind11::class_<psProcess<T, D>, psSmartPointer<psProcess<T, D>>>(
      module, "psProcess")
      // constructors
      .def(pybind11::init(&psSmartPointer<psProcess<T, D>>::New<>))

      // methods
      .def("setDomain", &psProcess<T, D>::setDomain, "Set the process domain.")
      .def("setProcessDuration", &psProcess<T, D>::setProcessDuration,
           "Set the process duraction.")
      .def("setSourceDirection", &psProcess<T, D>::setSourceDirection,
           "Set source direction of the process.")
      .def("setNumberOfRaysPerPoint", &psProcess<T, D>::setNumberOfRaysPerPoint,
           "Set the number of rays to traced for each particle in the process. "
           "The number is per point in the process geometry")
      .def("setMaxCoverageInitIerations",
           &psProcess<T, D>::setMaxCoverageInitIterations,
           "Set the number of iterations to initialize the coverages.")
      .def("setPrintIntermediate", &psProcess<T, D>::setPrintIntermediate,
           "Set whether to print disk meshes in the intermediate steps.")
      .def("setProcessModel",
           &psProcess<T, D>::setProcessModel<psProcessModel<T, D>>,
           "Set the process model.")
      .def("apply", &psProcess<T, D>::apply, "Run the process.");

  // models
  pybind11::class_<SimpleDeposition<T, D>,
                   psSmartPointer<SimpleDeposition<T, D>>>(module,
                                                           "SimpleDeposition")
      .def(pybind11::init(
               &psSmartPointer<SimpleDeposition<T, D>>::New<const T, const T>),
           pybind11::arg("stickingProbability") = 0.1,
           pybind11::arg("sourceExponent") = 1.)
      .def("getProcessModel", &SimpleDeposition<T, D>::getProcessModel,
           "Return the deposition process model.");

  pybind11::class_<SF6O2Etching<T, D>, psSmartPointer<SF6O2Etching<T, D>>>(
      module, "SF6O2Etching")
      .def(pybind11::init(&psSmartPointer<SF6O2Etching<T, D>>::New<
                          const double, const double, const double, const T,
                          const T, const int>),
           pybind11::arg("totalIonFlux"), pybind11::arg("totalEtchantFlux"),
           pybind11::arg("totalOxygenFlux"), pybind11::arg("ionEnergy") = 100.,
           pybind11::arg("oxySputterYield") = 3.,
           pybind11::arg("maskMaterial") = 0)
      .def("getProcessModel", &SF6O2Etching<T, D>::getProcessModel,
           "Returns the etching process model");

  pybind11::class_<GeometricUniformDeposition<T, D>,
                   psSmartPointer<GeometricUniformDeposition<T, D>>>(
      module, "GeometricUniformDeposition")
      .def(pybind11::init(
               &psSmartPointer<GeometricUniformDeposition<T, D>>::New<const T>),
           pybind11::arg("layerThickness") = 1.)
      .def("getProcessModel",
           &GeometricUniformDeposition<T, D>::getProcessModel,
           "Return the deposition process model.");

#if VIENNAPS_PYTHON_DIMENSION > 2
  // GDS file parsing
  pybind11::class_<psGDSGeometry<T, D>, psSmartPointer<psGDSGeometry<T, D>>>(
      module, "psGDSGeometry")
      // constructors
      .def(pybind11::init(&psSmartPointer<psGDSGeometry<T, D>>::New<>))
      .def(pybind11::init(&psSmartPointer<psGDSGeometry<T, D>>::New<const T>),
           pybind11::arg("gridDelta"))
      // methods
      .def("setGridDelta", &psGDSGeometry<T, D>::setGridDelta,
           "Set the gird spacing.")
      .def("setBoundaryPadding", &psGDSGeometry<T, D>::setBoundaryPadding,
           "Set padding between the largest point of the geometry and the "
           "boundary of the domain.")
      .def("print", &psGDSGeometry<T, D>::print, "Print the geometry contents.")
      .def("layerToLevelSet", &psGDSGeometry<T, D>::layerToLevelSet,
           "Convert a layer of the GDS geometry to a level set domain.");

  pybind11::class_<psGDSReader<T, D>, psSmartPointer<psGDSReader<T, D>>>(
      module, "psGDSReader")
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
#endif

  // ViennaLS domain setup
  // lsDomain
  pybind11::class_<lsDomain<T, D>, psSmartPointer<lsDomain<T, D>>>(module,
                                                                   "lsDomain")
      // constructors
      .def(pybind11::init(&psSmartPointer<lsDomain<T, D>>::New<>))
      .def(pybind11::init(&psSmartPointer<lsDomain<T, D>>::New<hrleCoordType>))
      .def(pybind11::init(
          &psSmartPointer<lsDomain<T, D>>::New<hrleCoordType *,
                                               lsDomain<T, D>::BoundaryType *>))
      .def(pybind11::init(
          &psSmartPointer<lsDomain<T, D>>::New<
              hrleCoordType *, lsDomain<T, D>::BoundaryType *, hrleCoordType>))
      .def(pybind11::init(
          &psSmartPointer<lsDomain<T, D>>::New<std::vector<hrleCoordType>,
                                               std::vector<unsigned>,
                                               hrleCoordType>))
      .def(pybind11::init(&psSmartPointer<lsDomain<T, D>>::New<
                          lsDomain<T, D>::PointValueVectorType, hrleCoordType *,
                          lsDomain<T, D>::BoundaryType *>))
      .def(pybind11::init(&psSmartPointer<lsDomain<T, D>>::New<
                          lsDomain<T, D>::PointValueVectorType, hrleCoordType *,
                          lsDomain<T, D>::BoundaryType *, hrleCoordType>))
      .def(pybind11::init(&psSmartPointer<lsDomain<T, D>>::New<
                          psSmartPointer<lsDomain<T, D>> &>))
      // methods
      .def("deepCopy", &lsDomain<T, D>::deepCopy,
           "Copy lsDomain in this lsDomain.")
      .def("getNumberOfSegments", &lsDomain<T, D>::getNumberOfSegments,
           "Get the number of segments, the level set structure is divided "
           "into.")
      .def("getNumberOfPoints", &lsDomain<T, D>::getNumberOfPoints,
           "Get the number of defined level set values.")
      .def("getLevelSetWidth", &lsDomain<T, D>::getLevelSetWidth,
           "Get the number of layers of level set points around the explicit "
           "surface.")
      .def("setLevelSetWidth", &lsDomain<T, D>::setLevelSetWidth,
           "Set the number of layers of level set points which should be "
           "stored around the explicit surface.")
      .def("clearMetaData", &lsDomain<T, D>::clearMetaData,
           "Clear all metadata stored in the level set.")
      // allow filehandle to be passed and default to python standard output
      .def("print", [](lsDomain<T, D>& d, pybind11::object fileHandle) {
          if (!(pybind11::hasattr(fileHandle,"write") &&
          pybind11::hasattr(fileHandle,"flush") )){
               throw pybind11::type_error("MyClass::read_from_file_like_object(file): incompatible function argument:  `file` must be a file-like object, but `"
                                        +(std::string)(pybind11::repr(fileHandle))+"` provided"
               );
          }
          pybind11::detail::pythonbuf buf(fileHandle);
          std::ostream stream(&buf);
          d.print(stream);
          }, pybind11::arg("stream") = pybind11::module::import("sys").attr("stdout"));

  // enums
  pybind11::enum_<lsBoundaryConditionEnum<D>>(module, "lsBoundaryConditionEnum")
      .value("REFLECTIVE_BOUNDARY",
             lsBoundaryConditionEnum<D>::REFLECTIVE_BOUNDARY)
      .value("INFINITE_BOUNDARY", lsBoundaryConditionEnum<D>::INFINITE_BOUNDARY)
      .value("PERIODIC_BOUNDARY", lsBoundaryConditionEnum<D>::PERIODIC_BOUNDARY)
      .value("POS_INFINITE_BOUNDARY",
             lsBoundaryConditionEnum<D>::POS_INFINITE_BOUNDARY)
      .value("NEG_INFINITE_BOUNDARY",
             lsBoundaryConditionEnum<D>::NEG_INFINITE_BOUNDARY);
}