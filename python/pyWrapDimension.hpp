/*
  This file is used to generate the python module of ViennaPS.
  It uses pybind11 to create the modules.
*/
#pragma once

#define PYBIND11_DETAILED_ERROR_MESSAGES
#define VIENNATOOLS_PYTHON_BUILD

#include <optional>
#include <vector>

#include <pybind11/functional.h>
#include <pybind11/iostream.h>
#include <pybind11/native_enum.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

// all header files which define API functions
#include <gds/psGDSGeometry.hpp>
#include <gds/psGDSReader.hpp>
#include <psConstants.hpp>
#include <psDomain.hpp>
#include <psDomainSetup.hpp>
#include <psExtrude.hpp>
#include <psModelDb.hpp>
#include <psPlanarize.hpp>
#include <psRateGrid.hpp>
#include <psReader.hpp>
#include <psSlice.hpp>
#include <psUnits.hpp>
#include <psVersion.hpp>
#include <psWriter.hpp>

// geometries
#include <geometries/psGeometryFactory.hpp>
#include <geometries/psMakeFin.hpp>
#include <geometries/psMakeHole.hpp>
#include <geometries/psMakePlane.hpp>
#include <geometries/psMakeStack.hpp>
#include <geometries/psMakeTrench.hpp>

// model framework
#include <process/psAdvectionCallback.hpp>
#include <process/psProcess.hpp>
#include <process/psProcessModel.hpp>
#include <process/psProcessParams.hpp>
#include <process/psSurfaceModel.hpp>
#include <process/psVelocityField.hpp>

// models
#include <models/psCF4O2Etching.hpp>
#include <models/psCSVFileProcess.hpp>
#include <models/psDirectionalProcess.hpp>
#include <models/psFaradayCageEtching.hpp>
#include <models/psFluorocarbonEtching.hpp>
#include <models/psGeometricDistributionModels.hpp>
#include <models/psHBrO2Etching.hpp>
#include <models/psIonBeamEtching.hpp>
#include <models/psIsotropicProcess.hpp>
#include <models/psMultiParticleProcess.hpp>
#include <models/psOxideRegrowth.hpp>
#include <models/psSF6C4F8Etching.hpp>
#include <models/psSF6O2Etching.hpp>
#include <models/psSelectiveEpitaxy.hpp>
#include <models/psSingleParticleALD.hpp>
#include <models/psSingleParticleProcess.hpp>
#include <models/psTEOSDeposition.hpp>
#include <models/psTEOSPECVD.hpp>
#include <models/psWetEtching.hpp>

// Ion implantation and annealing
#include <models/psAnneal.hpp>
#include <models/psImplantProfile.hpp>
#include <models/psIonImplantation.hpp>
#include <csNetDoping.hpp>
#include <csSheetResistance.hpp>

// visualization
#include <psToDiskMesh.hpp>
#include <psVTKRenderWindow.hpp>

// other
#include <csDenseCellSet.hpp>
#include <psUtil.hpp>
#include <rayParticle.hpp>
#include <rayReflection.hpp>
#include <rayUtil.hpp>
#include <vcLogger.hpp>

// GPU
#ifdef VIENNACORE_COMPILE_GPU
#include <vcContext.hpp>
#include <vcCudaBuffer.hpp>
#endif

using namespace viennaps;
namespace py = pybind11;

// always use double for python export
typedef double T;
// get dimension from cmake define
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
template <int D> class PyAdvectionCallback : public AdvectionCallback<T, D> {
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
                        RNG &Rng) final {
    PYBIND11_OVERRIDE(void, ClassName, surfaceCollision, rayWeight, rayDir,
                      geomNormal, primID, materialID, localData, globalData,
                      Rng);
  }

  std::pair<T, Vec3D<T>> surfaceReflection(
      T rayWeight, const Vec3D<T> &rayDir, const Vec3D<T> &geomNormal,
      const unsigned int primID, const int materialID,
      const viennaray::TracingData<T> *globalData, RNG &Rng) final {
    using Pair = std::pair<T, Vec3D<T>>;
    PYBIND11_OVERRIDE(Pair, ClassName, surfaceReflection, rayWeight, rayDir,
                      geomNormal, primID, materialID, globalData, Rng);
  }

  void initNew(RNG &RNG) final {
    PYBIND11_OVERRIDE(void, ClassName, initNew, RNG);
  }

  T getSourceDistributionPower() const final {
    PYBIND11_OVERRIDE(T, ClassName, getSourceDistributionPower);
  }

  std::vector<std::string> getLocalDataLabels() const final {
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
// void declare_GeometricDistributionModel(py::module &m,
//                                         const std::string &typestr) {
//   using Class = psGeometricDistributionModel<NumericType, D, DistType>;
//   py::class_<Class, SmartPointer<Class>>(m, typestr.c_str())
//       .def(py::init<SmartPointer<DistType>>(), py::arg("dist"))
//       .def(py::init<SmartPointer<DistType>,
//                           SmartPointer<viennals::Domain<NumericType, D>>>(),
//            py::arg("dist"), py::arg("mask"))
//       .def("apply", &Class::apply);
// }

template <int D> void bindApi(py::module &module) {

  // ***************************************************************************
  //                      Dense Cell Set from ViennaCS
  // ***************************************************************************
  py::class_<viennacs::DenseCellSet<T, D>,
             SmartPointer<viennacs::DenseCellSet<T, D>>>(module, "DenseCellSet")
      .def(py::init())
      .def("fromLevelSets", &viennacs::DenseCellSet<T, D>::fromLevelSets,
           py::arg("levelSets"), py::arg("materialMap") = nullptr,
           py::arg("depth") = 0.)
      .def("getBoundingBox", &viennacs::DenseCellSet<T, D>::getBoundingBox)
      .def(
          "addScalarData",
          [](viennacs::DenseCellSet<T, D> &cellSet, std::string name,
             T initValue) {
            cellSet.addScalarData(name, initValue);
            // discard return value
          },
          "Add a scalar value to be stored and modified in each cell.")
      .def("getDepth", &viennacs::DenseCellSet<T, D>::getDepth,
           "Get the depth of the cell set.")
      .def("getGridDelta", &viennacs::DenseCellSet<T, D>::getGridDelta,
           "Get the cell size.")
      .def("getNodes", &viennacs::DenseCellSet<T, D>::getNodes,
           "Get the nodes of the cell set which correspond to the corner "
           "points of the cells.")
      .def("getNode", &viennacs::DenseCellSet<T, D>::getNode,
           "Get the node at the given index.")
      .def("getElements", &viennacs::DenseCellSet<T, D>::getElements,
           "Get elements (cells). The indicies in the elements correspond to "
           "the corner nodes.")
      .def("getElement", &viennacs::DenseCellSet<T, D>::getElement,
           "Get the element at the given index.")
      .def("getSurface", &viennacs::DenseCellSet<T, D>::getSurface,
           "Get the surface level-set.")
      .def("getCellGrid", &viennacs::DenseCellSet<T, D>::getCellGrid,
           "Get the underlying mesh of the cell set.")
      .def("getNumberOfCells", &viennacs::DenseCellSet<T, D>::getNumberOfCells,
           "Get the number of cells.")
      .def("getFillingFraction",
           &viennacs::DenseCellSet<T, D>::getFillingFraction,
           "Get the filling fraction of the cell containing the point.")
      .def("getFillingFractions",
           &viennacs::DenseCellSet<T, D>::getFillingFractions,
           "Get the filling fractions of all cells.")
      .def("getAverageFillingFraction",
           &viennacs::DenseCellSet<T, D>::getAverageFillingFraction,
           "Get the average filling at a point in some radius.")
      .def("getCellCenter", &viennacs::DenseCellSet<T, D>::getCellCenter,
           "Get the center of a cell with given index")
      .def("getScalarData", &viennacs::DenseCellSet<T, D>::getScalarData,
           "Get the data stored at each cell. WARNING: This function only "
           "returns a copy of the data")
      .def("setScalarData", &viennacs::DenseCellSet<T, D>::setScalarData,
           py::arg("name"), py::arg("newData"),
           "Overwrite the scalar data associated with 'name' with a new array.")
      .def("getScalarDataLabels",
           &viennacs::DenseCellSet<T, D>::getScalarDataLabels,
           "Get the labels of the scalar data stored in the cell set.")
      .def("getIndex", &viennacs::DenseCellSet<T, D>::getIndex,
           "Get the index of the cell containing the given point.")
      .def("setCellSetPosition",
           &viennacs::DenseCellSet<T, D>::setCellSetPosition,
           "Set whether the cell set should be created below (false) or above "
           "(true) the surface.")
      .def(
          "setCoverMaterial", &viennacs::DenseCellSet<T, D>::setCoverMaterial,
          "Set the material of the cells which are above or below the surface.")
      .def("setPeriodicBoundary",
           &viennacs::DenseCellSet<T, D>::setPeriodicBoundary,
           "Enable periodic boundary conditions in specified dimensions.")
      .def("setFillingFraction",
           py::overload_cast<const int, const T>(
               &viennacs::DenseCellSet<T, D>::setFillingFraction),
           "Sets the filling fraction at given cell index.")
      .def("setFillingFraction",
           py::overload_cast<const std::array<T, 3> &, const T>(
               &viennacs::DenseCellSet<T, D>::setFillingFraction),
           "Sets the filling fraction for cell which contains given point.")
      .def("addFillingFraction",
           py::overload_cast<const int, const T>(
               &viennacs::DenseCellSet<T, D>::addFillingFraction),
           "Add to the filling fraction at given cell index.")
      .def("addFillingFraction",
           py::overload_cast<const std::array<T, 3> &, const T>(
               &viennacs::DenseCellSet<T, D>::addFillingFraction),
           "Add to the filling fraction for cell which contains given point.")
      .def("addFillingFractionInMaterial",
           &viennacs::DenseCellSet<T, D>::addFillingFractionInMaterial,
           "Add to the filling fraction for cell which contains given point "
           "only if the cell has the specified material ID.")
      .def("writeVTU", &viennacs::DenseCellSet<T, D>::writeVTU,
           "Write the cell set as .vtu file")
      .def("writeCellSetData", &viennacs::DenseCellSet<T, D>::writeCellSetData,
           "Save cell set data in simple text format.")
      .def("readCellSetData", &viennacs::DenseCellSet<T, D>::readCellSetData,
           "Read cell set data from text.")
      .def("clear", &viennacs::DenseCellSet<T, D>::clear,
           "Clear the filling fractions.")
      .def("updateMaterials", &viennacs::DenseCellSet<T, D>::updateMaterials,
           "Update the material IDs of the cell set. This function should be "
           "called if the level sets, the cell set is made out of, have "
           "changed. This does not work if the surface of the volume has "
           "changed. In this case, call the function 'updateSurface' first.")
      .def("updateSurface", &viennacs::DenseCellSet<T, D>::updateSurface,
           "Updates the surface of the cell set. The new surface should be "
           "below the old surface as this function can only remove cells from "
           "the cell set.")
      .def("buildNeighborhood",
           &viennacs::DenseCellSet<T, D>::buildNeighborhood,
           "Generate fast neighbor access for each cell.",
           py::arg("forceRebuild") = false)
      .def("getNeighbors", &viennacs::DenseCellSet<T, D>::getNeighbors,
           "Get the neighbor indices for a cell.");

  /****************************************************************************
   *                               DOMAIN                                    *
   ****************************************************************************/
  using DomainType = SmartPointer<Domain<T, D>>;
  using lsDomainType = SmartPointer<::viennals::Domain<T, D>>;

  // Domain Setup
  py::class_<DomainSetup<D>>(module, "DomainSetup")
      .def(py::init<>())
      .def(py::init<double, double, double, BoundaryType>(),
           py::arg("gridDelta"), py::arg("xExtent"), py::arg("yExtent"),
           py::arg("boundary") = BoundaryType::REFLECTIVE_BOUNDARY)
      .def("grid", &DomainSetup<D>::grid)
      .def("gridDelta", &DomainSetup<D>::gridDelta)
      .def("bounds", &DomainSetup<D>::bounds)
      .def("boundaryCons", &DomainSetup<D>::boundaryCons)
      .def("xExtent", &DomainSetup<D>::xExtent)
      .def("yExtent", &DomainSetup<D>::yExtent)
      .def("hasPeriodicBoundary", &DomainSetup<D>::hasPeriodicBoundary)
      .def("isValid", &DomainSetup<D>::isValid)
      .def("print", &DomainSetup<D>::print)
      .def("check", &DomainSetup<D>::check)
      .def("halveXAxis", &DomainSetup<D>::halveXAxis)
      .def("halveYAxis", &DomainSetup<D>::halveYAxis);

  // Domain
  py::class_<Domain<T, D>, DomainType>(module, "Domain")
      // constructors
      .def(py::init(&DomainType::template New<>))
      .def(py::init([](DomainType &domain) { return DomainType::New(domain); }),
           py::arg("domain"), "Deep copy constructor.")
      .def(py::init(
               [](T gridDelta, T xExtent, T yExtent, BoundaryType boundary) {
                 return DomainType::New(gridDelta, xExtent, yExtent, boundary);
               }),
           py::arg("gridDelta"), py::arg("xExtent"), py::arg("yExtent"),
           py::arg("boundary") = BoundaryType::REFLECTIVE_BOUNDARY)
      .def(py::init([](T gridDelta, T xExtent, BoundaryType boundary) {
             return DomainType::New(gridDelta, xExtent, boundary);
           }),
           py::arg("gridDelta"), py::arg("xExtent"),
           py::arg("boundary") = BoundaryType::REFLECTIVE_BOUNDARY)
      .def(py::init([](std::array<double, 2 * D> bounds,
                       std::array<BoundaryType, D> bcs, T gridDelta) {
             return DomainType::New(bounds.data(), bcs.data(), gridDelta);
           }),
           py::arg("bounds"), py::arg("boundaryConditions"),
           py::arg("gridDelta") = 1.0)
      .def(py::init(&DomainType::template New<const DomainSetup<D> &>),
           py::arg("setup"))
      // methods
      .def("setup",
           py::overload_cast<const DomainSetup<D> &>(&Domain<T, D>::setup),
           "Setup the domain.")
      .def("setup",
           py::overload_cast<T, T, T, BoundaryType>(&Domain<T, D>::setup),
           py::arg("gridDelta"), py::arg("xExtent"), py::arg("yExtent") = 0.,
           py::arg("boundary") = BoundaryType::REFLECTIVE_BOUNDARY,
           "Setup the domain.")
      .def("getSetup", &Domain<T, D>::getSetup, "Get the domain setup.")
      .def("deepCopy", &Domain<T, D>::deepCopy)
      .def("insertNextLevelSetAsMaterial",
           py::overload_cast<lsDomainType, std::string, bool>(
               &Domain<T, D>::insertNextLevelSetAsMaterial),
           py::arg("levelSet"), py::arg("material"),
           py::arg("wrapLowerLevelSet") = true,
           "Insert a level set to domain as a material.")
      .def("insertNextLevelSetAsMaterial",
           py::overload_cast<lsDomainType, const Material, bool>(
               &Domain<T, D>::insertNextLevelSetAsMaterial),
           py::arg("levelSet"), py::arg("material"),
           py::arg("wrapLowerLevelSet") = true,
           "Insert a level set to domain as a material.")
      .def("insertMask", &Domain<T, D>::insertMask, py::arg("mask"),
           py::arg("material") = Material::Mask,
           "Insert a mask level set to the domain. The mask is inserted at the "
           "front of the level set vector and can be used to exclude areas "
           "from processes.")
      .def("duplicateTopLevelSet",
           py::overload_cast<const Material>(
               &Domain<T, D>::duplicateTopLevelSet),
           "Duplicate the top level set. Should be used before a deposition "
           "process.")
      .def("duplicateTopLevelSet",
           py::overload_cast<const std::string &>(
               &Domain<T, D>::duplicateTopLevelSet),
           "Duplicate the top level set. Should be used before a deposition "
           "process.")
      .def("removeTopLevelSet", &Domain<T, D>::removeTopLevelSet)
      .def("applyBooleanOperation", &Domain<T, D>::applyBooleanOperation,
           py::arg("levelSet"), py::arg("operation"),
           py::arg("applyToAll") = true,
           "Apply a boolean operation with the passed Level-Set to all (or top "
           "only) Level-Sets in the domain.")
      .def("removeLevelSet", &Domain<T, D>::removeLevelSet)
      .def("removeMaterial", &Domain<T, D>::removeMaterial)
      .def("removeStrayPoints", &Domain<T, D>::removeStrayPoints)
      .def("getNumberOfComponents", &Domain<T, D>::getNumberOfComponents,
           "Get the number of connected components in the domain.")
      .def("getNumberOfLevelSets", &Domain<T, D>::getNumberOfLevelSets,
           "Get the number of level sets in the domain.")
      .def("setMaterialMap", &Domain<T, D>::setMaterialMap)
      .def("getMaterialMap", &Domain<T, D>::getMaterialMap)
      .def("generateCellSet", &Domain<T, D>::generateCellSet,
           "Generate the cell set.")
      .def("getLevelSets", &Domain<T, D>::getLevelSets)
      .def("getMaterialsInDomain", &Domain<T, D>::getMaterialsInDomain,
           "Get the material IDs present in the domain.")
      .def("getMaterialLevelSet", &Domain<T, D>::getMaterialLevelSet,
           py::arg("material"),
           "Returns a Level-Set representing the specified material in the "
           "domain.")
      .def("getSurface", &Domain<T, D>::getSurface,
           "Get the surface level set.")
      .def("getCellSet", &Domain<T, D>::getCellSet, "Get the cell set.")
      .def("getGrid", &Domain<T, D>::getGrid, "Get the grid")
      .def("getGridDelta", &Domain<T, D>::getGridDelta, "Get the grid delta.")
      .def("getBoundingBox", &Domain<T, D>::getBoundingBox,
           "Get the bounding box of the domain.")
      .def("getBoundaryConditions", &Domain<T, D>::getBoundaryConditions,
           "Get the boundary conditions of the domain.")
      .def("getMetaData", &Domain<T, D>::getMetaData,
           "Get meta data (e.g. process data) stored in the domain")
      .def(
          "print",
          [](Domain<T, D> &self, bool hrle) { self.print(std::cout, hrle); },
          "Print the domain information.", py::arg("hrleInfo") = false)
      .def("show", &Domain<T, D>::show, "Render the domain using VTK.")
      // Mesh generation
      .def("getLevelSetMesh", &Domain<T, D>::getLevelSetMesh,
           py::arg("width") = 1,
           "Get the level set grids of layers in the domain.")
      .def("getSurfaceMesh", &Domain<T, D>::getSurfaceMesh,
           py::arg("addInterfaces") = true, py::arg("sharpCorners") = false,
           py::arg("minNodeDistanceFactor") = 0.01,
           "Get the surface mesh of the domain")
      .def("getHullMesh", &Domain<T, D>::getHullMesh,
           py::arg("bottomExtension") = 0.0, py::arg("sharpCorners") = false)
      .def("getDiskMesh", &Domain<T, D>::getDiskMesh)
      // Save to file
      .def("saveLevelSetMesh", &Domain<T, D>::saveLevelSetMesh,
           py::arg("filename"), py::arg("width") = 1,
           "Save the level set grids of layers in the domain.")
      .def("saveSurfaceMesh", &Domain<T, D>::saveSurfaceMesh,
           py::arg("filename"), py::arg("addInterfaces") = true,
           py::arg("sharpCorners") = false,
           py::arg("minNodeDistanceFactor") = 0.01,
           "Save the surface of the domain.")
      .def("saveHullMesh", &Domain<T, D>::saveHullMesh, py::arg("filename"),
           py::arg("bottomExtension") = 0.0, py::arg("sharpCorners") = false,
           "Save the hull of the domain.")
      .def("saveDiskMesh", &Domain<T, D>::saveDiskMesh, py::arg("filename"))
      .def("saveVolumeMesh", &Domain<T, D>::saveVolumeMesh, py::arg("filename"),
           py::arg("wrappingLayerEpsilon") = 1e-2,
           "Save the volume representation of the domain.")
      .def("saveLevelSets", &Domain<T, D>::saveLevelSets, py::arg("filename"))
      .def("clear", &Domain<T, D>::clear)
      .def("clearMetaData", &Domain<T, D>::clearMetaData,
           "Clear meta data from domain.", py::arg("clearDomainData") = false)
      .def(
          "addMetaData",
          py::overload_cast<const std::string &, T>(&Domain<T, D>::addMetaData),
          "Add a single metadata entry to the domain.")
      .def("addMetaData",
           py::overload_cast<const std::string &, const std::vector<T> &>(
               &Domain<T, D>::addMetaData),
           "Add a single metadata entry to the domain.")
      .def("addMetaData",
           py::overload_cast<
               const std::unordered_map<std::string, std::vector<T>> &>(
               &Domain<T, D>::addMetaData),
           "Add metadata to the domain.")
      .def("enableMetaData", &Domain<T, D>::enableMetaData,
           "Enable adding meta data from processes to domain.",
           py::arg("level") = MetaDataLevel::PROCESS)
      .def("disableMetaData", &Domain<T, D>::disableMetaData,
           "Disable adding meta data to domain.")
      .def("getMetaDataLevel", &Domain<T, D>::getMetaDataLevel,
           "Get the current meta data level of the domain.");

  /****************************************************************************
   *                               MODEL FRAMEWORK                            *
   ****************************************************************************/

  // ProcessModelBase
  py::class_<ProcessModelBase<T, D>, SmartPointer<ProcessModelBase<T, D>>>
      processModelBase(module, "ProcessModelBase");

  // ProcessModel
  py::class_<ProcessModelCPU<T, D>, SmartPointer<ProcessModelCPU<T, D>>>
      processModel(module, "ProcessModel", processModelBase);

  // constructors
  processModel
      .def(py::init<>())
      // methods
      .def("setProcessName", &ProcessModelCPU<T, D>::setProcessName)
      .def("getProcessName", &ProcessModelCPU<T, D>::getProcessName)
      .def("getSurfaceModel", &ProcessModelCPU<T, D>::getSurfaceModel)
      .def("getAdvectionCallback", &ProcessModelCPU<T, D>::getAdvectionCallback)
      .def("getGeometricModel", &ProcessModelCPU<T, D>::getGeometricModel)
      .def("getVelocityField", &ProcessModelCPU<T, D>::getVelocityField)
      // .def("getParticleLogSize",
      //  &ProcessModelCPU<T, D>::getParticleLogSize) .def("getParticleTypes",
      //       [](ProcessModelCPU<T, D> &pm) {
      //         // Get smart pointer to vector of unique_ptr from the process
      //         // model
      //         auto &unique_ptrs = pm.getParticleTypes();

      //         // Create vector to hold shared_ptr
      //         std::vector<std::shared_ptr<viennaray::AbstractParticle<T>>>
      //             shared_ptrs;

      //         // Loop over unique_ptrs and create shared_ptrs from them
      //         for (auto &uptr : unique_ptrs) {
      //           shared_ptrs.push_back(
      //               std::shared_ptr<viennaray::AbstractParticle<T>>(
      //                   uptr.release()));
      //         }

      //         // Return the new vector of shared_ptr
      //         return shared_ptrs;
      //       })
      // IMPORTANT: here it may be needed to write this function for any
      // type of passed Particle
      //  .def("insertNextParticleType",
      //       [](ProcessModelCPU<T, D> &pm,
      //          SmartPointer<psParticle<D>> &passedParticle) {
      //         if (passedParticle) {
      //           auto particle =
      //               std::make_unique<psParticle<D>>(*passedParticle.get());
      //           pm.insertNextParticleType(particle);
      //         }
      //       })
      .def("setSurfaceModel",
           [](ProcessModelCPU<T, D> &pm, SmartPointer<SurfaceModel<T>> &sm) {
             pm.setSurfaceModel(sm);
           })
      .def("setAdvectionCallback",
           [](ProcessModelCPU<T, D> &pm,
              SmartPointer<AdvectionCallback<T, D>> &ac) {
             pm.setAdvectionCallback(ac);
           })
      .def("setGeometricModel",
           [](ProcessModelCPU<T, D> &pm,
              SmartPointer<GeometricModel<T, D>> &gm) {
             pm.setGeometricModel(gm);
           })
      .def(
          "setVelocityField",
          [](ProcessModelCPU<T, D> &pm, SmartPointer<VelocityField<T, D>> &vf) {
            pm.setVelocityField(vf);
          })
      .def("setPrimaryDirection", &ProcessModelCPU<T, D>::setPrimaryDirection)
      .def("getPrimaryDirection", &ProcessModelCPU<T, D>::getPrimaryDirection);

  // AdvectionCallback
  py::class_<AdvectionCallback<T, D>, SmartPointer<AdvectionCallback<T, D>>,
             PyAdvectionCallback<D>>(module, "AdvectionCallback")
      // constructors
      .def(py::init<>())
      // methods
      .def("applyPreAdvect", &AdvectionCallback<T, D>::applyPreAdvect)
      .def("applyPostAdvect", &AdvectionCallback<T, D>::applyPostAdvect)
      .def_readwrite("domain", &PyAdvectionCallback<D>::domain);

  // SurfaceModel
  //   py::class_<SurfaceModel<T>, SmartPointer<SurfaceModel<T>>,
  //                    PypsSurfaceModel>(module, "SurfaceModel")
  //       .def(py::init<>())
  //       .def("initializeCoverages", &SurfaceModel<T>::initializeCoverages)
  //       .def("initializeProcessParameters",
  //            &SurfaceModel<T>::initializeProcessParameters)
  //       .def("getCoverages", &SurfaceModel<T>::getCoverages)
  //       .def("getProcessParameters",
  //       &SurfaceModel<T>::getProcessParameters) .def("calculateVelocities",
  //       &SurfaceModel<T>::calculateVelocities) .def("updateCoverages",
  //       &SurfaceModel<T>::updateCoverages);
  // VelocityField
  //   py::class_<VelocityField<T>, SmartPointer<VelocityField<T>>,
  //                    PyVelocityField>
  //       velocityField(module, "VelocityField");
  //   // constructors
  //   velocityField
  //       .def(py::init<>())
  //       // methods
  //       .def("getScalarVelocity", &VelocityField<T>::getScalarVelocity)
  //       .def("getVectorVelocity", &VelocityField<T>::getVectorVelocity)
  //       .def("getDissipationAlpha", &VelocityField<T>::getDissipationAlpha)
  //       .def("getTranslationFieldOptions",
  //            &VelocityField<T>::getTranslationFieldOptions)
  //       .def("setVelocities", &VelocityField<T>::setVelocities);
  //   py::class_<psDefaultVelocityField<T>,
  //                    SmartPointer<psDefaultVelocityField<T>>>(
  //       module, "DefaultVelocityField", velocityField)
  //       // constructors
  //       .def(py::init<>())
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
  //   py::class_<psParticle<D>, SmartPointer<psParticle<D>>> particle(
  //       module, "Particle");
  //   particle.def("surfaceCollision", &psParticle<D>::surfaceCollision)
  //       .def("surfaceReflection", &psParticle<D>::surfaceReflection)
  //       .def("initNew", &psParticle<D>::initNew)
  //       .def("getLocalDataLabels", &psParticle<D>::getLocalDataLabels)
  //       .def("getSourceDistributionPower",
  //            &psParticle<D>::getSourceDistributionPower);

  // ***************************************************************************
  //                                  MODELS
  // ***************************************************************************

  // Single Particle Process
  py::class_<SingleParticleProcess<T, D>,
             SmartPointer<SingleParticleProcess<T, D>>>(
      module, "SingleParticleProcess", processModel)
      .def(py::init<T, T, T, Material>(), py::arg("rate") = 1.,
           py::arg("stickingProbability") = 1., py::arg("sourceExponent") = 1.,
           py::arg("maskMaterial") = Material::Undefined)
      .def(py::init<T, T, T, std::vector<Material>>(), py::arg("rate"),
           py::arg("stickingProbability"), py::arg("sourceExponent"),
           py::arg("maskMaterials"))
      .def(py::init<std::unordered_map<Material, T>, T, T>(),
           py::arg("materialRates"), py::arg("stickingProbability"),
           py::arg("sourceExponent"))
      .def("setDefaultRate", &SingleParticleProcess<T, D>::setDefaultRate)
      .def("setMaterialRate", &SingleParticleProcess<T, D>::setMaterialRate,
           py::arg("material"), py::arg("rate"));

  // Multi Particle Process
  py::class_<MultiParticleProcess<T, D>,
             SmartPointer<MultiParticleProcess<T, D>>>(
      module, "MultiParticleProcess", processModel)
      .def(py::init())
      .def("addNeutralParticle",
           py::overload_cast<T, const std::string &>(
               &MultiParticleProcess<T, D>::addNeutralParticle),
           py::arg("stickingProbability"), py::arg("label") = "neutralFlux")
      .def("addNeutralParticle",
           py::overload_cast<const std::unordered_map<Material, T> &, T,
                             const std::string &>(
               &MultiParticleProcess<T, D>::addNeutralParticle),
           py::arg("materialSticking"),
           py::arg("defaultStickingProbability") = 1.,
           py::arg("label") = "neutralFlux")
      .def("addIonParticle", &MultiParticleProcess<T, D>::addIonParticle,
           py::arg("sourcePower"), py::arg("thetaRMin") = 0.,
           py::arg("thetaRMax") = 90., py::arg("minAngle") = 80.,
           py::arg("B_sp") = -1., py::arg("meanEnergy") = 0.,
           py::arg("sigmaEnergy") = 0., py::arg("thresholdEnergy") = 0.,
           py::arg("inflectAngle") = 0., py::arg("n") = 1,
           py::arg("label") = "ionFlux")
      .def("setRateFunction", &MultiParticleProcess<T, D>::setRateFunction);

  // TEOS Deposition
  py::class_<TEOSDeposition<T, D>, SmartPointer<TEOSDeposition<T, D>>>(
      module, "TEOSDeposition", processModel)
      .def(py::init(&SmartPointer<TEOSDeposition<T, D>>::template New<
                    T /*st1*/, T /*rate1*/, T /*order1*/, T /*st2*/,
                    T /*rate2*/, T /*order2*/>),
           py::arg("stickingProbabilityP1"), py::arg("rateP1"),
           py::arg("orderP1"), py::arg("stickingProbabilityP2") = 0.,
           py::arg("rateP2") = 0., py::arg("orderP2") = 0.);

  // TEOS PE-CVD
  py::class_<TEOSPECVD<T, D>, SmartPointer<TEOSPECVD<T, D>>>(
      module, "TEOSPECVD", processModel)
      .def(
          py::init(&SmartPointer<TEOSPECVD<T, D>>::template New<
                   T /*stR*/, T /*rateR*/, T /*orderR*/, T /*stI*/, T /*rateI*/,
                   T /*orderI*/, T /*exponentI*/, T /*minAngleIon*/>),
          py::arg("stickingProbabilityRadical"),
          py::arg("depositionRateRadical"), py::arg("depositionRateIon"),
          py::arg("exponentIon"), py::arg("stickingProbabilityIon") = 1.,
          py::arg("reactionOrderRadical") = 1.,
          py::arg("reactionOrderIon") = 1., py::arg("minAngleIon") = 0.);

  // SF6O2 Etching
  py::class_<SF6O2Etching<T, D>, SmartPointer<SF6O2Etching<T, D>>>(
      module, "SF6O2Etching", processModel)
      .def(py::init<>())
      .def(py::init(&SmartPointer<SF6O2Etching<T, D>>::template New<
                    double /*ionFlux*/, double /*etchantFlux*/,
                    double /*oxygenFlux*/, T /*meanIonEnergy*/,
                    T /*sigmaIonEnergy*/, T /*ionExponent*/,
                    T /*oxySputterYield*/, T /*etchStopDepth*/>),
           py::arg("ionFlux"), py::arg("etchantFlux"), py::arg("oxygenFlux"),
           py::arg("meanIonEnergy") = 100., py::arg("sigmaIonEnergy") = 10.,
           py::arg("ionExponent") = 100., py::arg("oxySputterYield") = 3.,
           py::arg("etchStopDepth") = std::numeric_limits<T>::lowest())
      .def(py::init(&SmartPointer<SF6O2Etching<T, D>>::template New<
                    const PlasmaEtchingParameters<T> &>),
           py::arg("parameters"))
      .def("setParameters", &SF6O2Etching<T, D>::setParameters)
      .def("getParameters", &SF6O2Etching<T, D>::getParameters,
           py::return_value_policy::reference)
      .def_static("defaultParameters", &SF6O2Etching<T, D>::defaultParameters);

  // HBrO2 Etching
  py::class_<HBrO2Etching<T, D>, SmartPointer<HBrO2Etching<T, D>>>(
      module, "HBrO2Etching", processModel)
      .def(py::init<>())
      .def(py::init(&SmartPointer<HBrO2Etching<T, D>>::template New<
                    double /*ionFlux*/, double /*etchantFlux*/,
                    double /*oxygenFlux*/, T /*meanIonEnergy*/,
                    T /*sigmaIonEnergy*/, T /*ionExponent*/,
                    T /*oxySputterYield*/, T /*etchStopDepth*/>),
           py::arg("ionFlux"), py::arg("etchantFlux"), py::arg("oxygenFlux"),
           py::arg("meanIonEnergy") = 100., py::arg("sigmaIonEnergy") = 10.,
           py::arg("ionExponent") = 100., py::arg("oxySputterYield") = 3.,
           py::arg("etchStopDepth") = std::numeric_limits<T>::lowest())
      .def(py::init(&SmartPointer<HBrO2Etching<T, D>>::template New<
                    const PlasmaEtchingParameters<T> &>),
           py::arg("parameters"))
      .def("setParameters", &HBrO2Etching<T, D>::setParameters)
      .def("getParameters", &HBrO2Etching<T, D>::getParameters,
           py::return_value_policy::reference)
      .def_static("defaultParameters", &HBrO2Etching<T, D>::defaultParameters);

  // SF6C4F8 Etching
  py::class_<SF6C4F8Etching<T, D>, SmartPointer<SF6C4F8Etching<T, D>>>(
      module, "SF6C4F8Etching", processModel)
      .def(py::init<>())
      .def(
          py::init(&SmartPointer<SF6C4F8Etching<T, D>>::template New<
                   double /*ionFlux*/, double /*etchantFlux*/, T /*meanEnergy*/,
                   T /*sigmaEnergy*/, T /*ionExponent*/, T /*etchStopDepth*/>),
          py::arg("ionFlux"), py::arg("etchantFlux"), py::arg("meanEnergy"),
          py::arg("sigmaEnergy"), py::arg("ionExponent") = 300.,
          py::arg("etchStopDepth") = std::numeric_limits<T>::lowest())
      .def(py::init(&SmartPointer<SF6C4F8Etching<T, D>>::template New<
                    const PlasmaEtchingParameters<T> &>),
           py::arg("parameters"))
      .def("setParameters", &SF6C4F8Etching<T, D>::setParameters)
      .def("getParameters", &SF6C4F8Etching<T, D>::getParameters,
           py::return_value_policy::reference)
      .def_static("defaultParameters",
                  &SF6C4F8Etching<T, D>::defaultParameters);

  // CF4O2 Etching
  py::class_<CF4O2Etching<T, D>, SmartPointer<CF4O2Etching<T, D>>>(
      module, "CF4O2Etching", processModel)
      .def(py::init<>())
      .def(py::init(&SmartPointer<CF4O2Etching<T, D>>::template New<
                    double /*ionFlux*/, double /*etchantFlux*/,
                    double /*oxygenFlux*/, double /*polymerFlux*/,
                    T /*meanIonEnergy*/, T /*sigmaIonEnergy*/,
                    T /*ionExponent*/, T /*oxySputterYield*/,
                    T /*polySputterYield*/, T /*etchStopDepth*/>),
           py::arg("ionFlux"), py::arg("etchantFlux"), py::arg("oxygenFlux"),
           py::arg("polymerFlux"), py::arg("meanIonEnergy") = 100.,
           py::arg("sigmaIonEnergy") = 10., py::arg("ionExponent") = 100.,
           py::arg("oxySputterYield") = 3., py::arg("polySputterYield") = 3.,
           py::arg("etchStopDepth") = std::numeric_limits<T>::lowest())
      .def(py::init(&SmartPointer<CF4O2Etching<T, D>>::template New<
                    const CF4O2Parameters<T> &>),
           py::arg("parameters"))
      .def("setParameters", &CF4O2Etching<T, D>::setParameters)
      .def("getParameters", &CF4O2Etching<T, D>::getParameters,
           py::return_value_policy::reference);

  // Fluorocarbon Etching
  py::class_<FluorocarbonEtching<T, D>,
             SmartPointer<FluorocarbonEtching<T, D>>>(
      module, "FluorocarbonEtching", processModel)
      .def(py::init(&SmartPointer<FluorocarbonEtching<T, D>>::template New<
                    const FluorocarbonParameters<T> &>),
           py::arg("parameters"))
      .def("setParameters", &FluorocarbonEtching<T, D>::setParameters);

  // Ion Beam Etching
  py::class_<IonBeamEtching<T, D>, SmartPointer<IonBeamEtching<T, D>>>(
      module, "IonBeamEtching", processModel)
      .def(py::init(&SmartPointer<IonBeamEtching<T, D>>::template New<
                    const IBEParameters<T> &>),
           py::arg("parameters"))
      .def(py::init(&SmartPointer<IonBeamEtching<T, D>>::template New<
                    const IBEParameters<T> &, const std::vector<Material> &>),
           py::arg("parameters"), py::arg("maskMaterials"))
      .def_static("defaultParameters",
                  &IonBeamEtching<T, D>::defaultParameters);

  // Faraday Cage Etching
  py::class_<FaradayCageEtching<T, D>, SmartPointer<FaradayCageEtching<T, D>>>(
      module, "FaradayCageEtching", processModel)
      .def(py::init(&SmartPointer<FaradayCageEtching<T, D>>::template New<
                    const FaradayCageParameters<T> &,
                    const std::vector<Material> &>),
           py::arg("parameters"), py::arg("maskMaterials"));

  // ── Ion Implantation Profile Models ────────────────────────────────────────

  // Abstract profile model base (module_local avoids conflicts with viennacs
  // Python module, which independently binds the same C++ base class).
  auto implantProfileModel =
      py::class_<ImplantProfileModel<T, D>,
                 SmartPointer<ImplantProfileModel<T, D>>>(
          module, "ImplantProfileModel", py::module_local())
          .def("getDepthProfile", &ImplantProfileModel<T, D>::getDepthProfile,
               py::arg("depth"))
          .def("getLateralProfile",
               &ImplantProfileModel<T, D>::getLateralProfile, py::arg("offset"),
               py::arg("depth"))
          .def("getMaxDepth", &ImplantProfileModel<T, D>::getMaxDepth)
          .def("getMaxLateralRange",
               &ImplantProfileModel<T, D>::getMaxLateralRange);

  // Single Pearson IV: depth profile from four moments, Gaussian lateral spread
  py::class_<ImplantPearsonIV<T, D>, SmartPointer<ImplantPearsonIV<T, D>>,
             ImplantProfileModel<T, D>>(module, "ImplantPearsonIV")
      .def(py::init<const PearsonIVParameters<T> &, T, T>(),
           py::arg("params"), py::arg("lateralMu"), py::arg("lateralSigma"),
           "Construct from PearsonIVParameters and Gaussian lateral spread.");

  // Dual Pearson IV: head + tail components (amorphous + channeling)
  py::class_<ImplantDualPearsonIV<T, D>,
             SmartPointer<ImplantDualPearsonIV<T, D>>,
             ImplantProfileModel<T, D>>(module, "ImplantDualPearsonIV")
      .def(py::init<const PearsonIVParameters<T> &,
                    const PearsonIVParameters<T> &, T, T, T, T, T>(),
           py::arg("headParams"), py::arg("tailParams"),
           py::arg("headFraction"),
           py::arg("headLateralMu"), py::arg("headLateralSigma"),
           py::arg("tailLateralMu"), py::arg("tailLateralSigma"),
           "Weighted sum of two Pearson IV components (head fraction in head).");

  // Pearson IV with exponential channeling tail.
  py::class_<ImplantPearsonIVChanneling<T, D>,
             SmartPointer<ImplantPearsonIVChanneling<T, D>>,
             ImplantProfileModel<T, D>>(module, "ImplantPearsonIVChanneling")
      .def(py::init<const PearsonIVParameters<T> &, T, T, T, T, T, T>(),
           py::arg("params"), py::arg("lateralMu"), py::arg("lateralSigma"),
           py::arg("tailFraction"), py::arg("tailStartDepth"),
           py::arg("tailDecayLength"), py::arg("tailBlendWidth") = T(0),
           "Single Pearson IV plus exponential channeling tail.");

  // Hobler damage model: Gaussian/exponential damage depth profile
  py::class_<ImplantDamageHobler<T, D>,
             SmartPointer<ImplantDamageHobler<T, D>>,
             ImplantProfileModel<T, D>>(module, "ImplantDamageHobler")
      .def(py::init<T, T, T, T, T, T>(),
           py::arg("projectedRange"), py::arg("verticalSigma"),
           py::arg("lambda"), py::arg("defectsPerIon"),
           py::arg("lateralSigma"), py::arg("lateralDeltaSigma") = T(0),
           "Hobler damage depth profile with linear-depth-scale lateral spread.");

  py::class_<ImplantTableModel<T, D>, SmartPointer<ImplantTableModel<T, D>>,
             ImplantProfileModel<T, D>>(module, "ImplantTableModel")
      .def(py::init<const std::string &, const std::string &,
                    const std::string &, const std::string &, T, T, T, T, T, T,
                    const std::string &>(),
           py::arg("fileName"), py::arg("species"), py::arg("material"),
           py::arg("substrateType"), py::arg("energyKeV"),
           py::arg("tiltDeg"), py::arg("rotationDeg"),
           py::arg("dosePerCm2") = T(0),
           py::arg("screenThickness") = T(0),
           py::arg("damageLevel") = T(0),
           py::arg("preferredModel") = "auto",
           "Table-backed implant profile model. Pass an explicit modeldb CSV "
           "path; the selected row is interpolated and converted to a profile.");

  py::class_<DamageTableModel<T, D>, SmartPointer<DamageTableModel<T, D>>,
             ImplantProfileModel<T, D>>(module, "DamageTableModel")
      .def(py::init<const std::string &, const std::string &,
                    const std::string &, T, T, T, T, T>(),
           py::arg("fileName"), py::arg("species"), py::arg("material"),
           py::arg("energyKeV"), py::arg("tiltDeg"),
           py::arg("rotationDeg"), py::arg("dosePerCm2") = T(0),
           py::arg("screenThickness") = T(0),
           "Table-backed implant-damage profile model. Pass an explicit "
           "modeldb CSV path.");

  // ── IonImplantation Process Model ──────────────────────────────────────────
  py::class_<IonImplantation<T, D>, SmartPointer<IonImplantation<T, D>>>(
      module, "IonImplantation", processModel)
      .def(py::init(&SmartPointer<IonImplantation<T, D>>::template New<>))
      .def("setImplantModel", &IonImplantation<T, D>::setImplantModel,
           py::arg("model"),
           "Set the dopant concentration profile model.")
      .def("setDamageModel", &IonImplantation<T, D>::setDamageModel,
           py::arg("model"),
           "Set the damage profile model (optional).")
      .def("setDose", &IonImplantation<T, D>::setDose,
           py::arg("dosePerCm2"), "Implant dose in ions/cm².")
      .def("setTiltAngle", &IonImplantation<T, D>::setTiltAngle,
           py::arg("angleDeg"), "Beam tilt angle in degrees (0 = normal).")
      .def("setLengthUnit", &IonImplantation<T, D>::setLengthUnit,
           py::arg("lengthUnitInCm"),
           "Length unit in cm (default 1e-7 = nanometres).")
      .def("setDoseControl", &IonImplantation<T, D>::setDoseControl,
           py::arg("mode"), "Dose control mode: Off, WaferDose, BeamDose.")
      .def("setMaskMaterials", &IonImplantation<T, D>::setMaskMaterials,
           py::arg("materials"), "Materials that completely block the beam.")
      .def("setScreenMaterials", &IonImplantation<T, D>::setScreenMaterials,
           py::arg("materials"),
           "Materials the beam passes through without absorbing dose.")
      .def("setVoidMaterials", &IonImplantation<T, D>::setVoidMaterials,
           py::arg("materials"),
           "Materials ignored by implantation rays; usually set from the "
           "domain cover material automatically.")
      .def("setConcentrationLabel",
           &IonImplantation<T, D>::setConcentrationLabel,
           py::arg("label"), "Cell-set field name for deposited concentration.")
      .def("setDamageLabel", &IonImplantation<T, D>::setDamageLabel,
           py::arg("label"), "Cell-set field name for accumulated damage.")
      .def("setLastDamageLabel", &IonImplantation<T, D>::setLastDamageLabel,
           py::arg("label"), "Field name for damage from the last step only.")
      .def("setBeamHitsLabel", &IonImplantation<T, D>::setBeamHitsLabel,
           py::arg("label"), "Field name for optional beam-hit counts.")
      .def("enableBeamHits", &IonImplantation<T, D>::enableBeamHits,
           py::arg("enable") = true,
           "Write the optional beam-hit count field.")
      .def("setOutputConcentrationInCm3",
           &IonImplantation<T, D>::setOutputConcentrationInCm3,
           py::arg("enable") = true,
           "Store concentration in cm⁻³ instead of length-unit⁻³.")
      .def("setDamageFactor", &IonImplantation<T, D>::setDamageFactor,
           py::arg("factor"),
           "Scale factor for damage accumulation across multiple implants.")
      .def("enableEmbeddedBoundaries",
           &IonImplantation<T, D>::enableEmbeddedBoundaries,
           py::arg("enable") = true,
           "Rebuild the cell set with embedded boundary points before implanting.\n"
           "Enables sub-grid surface offsets via ray-plane intersection for tilt accuracy.");

  // ── Anneal Process Model ────────────────────────────────────────────────────
  py::class_<Anneal<T, D>, SmartPointer<Anneal<T, D>>>(module, "Anneal",
                                                         processModel)
      .def(py::init(&SmartPointer<Anneal<T, D>>::template New<>))
      // Temperature (simple isothermal)
      .def("setTemperature", &Anneal<T, D>::setTemperature,
           py::arg("temperatureK"))
      .def("setDuration", &Anneal<T, D>::setDuration, py::arg("seconds"))
      // Temperature schedule (ramp/soak)
      .def("clearTemperatureSchedule",
           &Anneal<T, D>::clearTemperatureSchedule)
      .def("addIsothermalStep", &Anneal<T, D>::addIsothermalStep,
           py::arg("duration"), py::arg("temperatureK"))
      .def("addRampStep", &Anneal<T, D>::addRampStep,
           py::arg("duration"), py::arg("startT"), py::arg("endT"))
      .def("setTemperatureSchedule", &Anneal<T, D>::setTemperatureSchedule,
           py::arg("durations"), py::arg("temperatures"),
           "N durations + N (isothermal) or N+1 (ramp) temperatures.")
      // Diffusivity
      .def("setDiffusionCoefficient",
           &Anneal<T, D>::setDiffusionCoefficient, py::arg("diffCoeff"))
      .def("setArrheniusParameters", &Anneal<T, D>::setArrheniusParameters,
           py::arg("D0"), py::arg("Ea_eV"))
      .def("setTimeStep", &Anneal<T, D>::setTimeStep, py::arg("dt"))
      .def("setStabilityFactor", &Anneal<T, D>::setStabilityFactor,
           py::arg("factor"))
      // Solver
      .def("setMode", &Anneal<T, D>::setMode, py::arg("mode"))
      .def("setImplicitSolverOptions",
           &Anneal<T, D>::setImplicitSolverOptions,
           py::arg("maxIterations"), py::arg("relativeTolerance"),
           py::arg("relaxation") = T(1))
      // Material roles
      .def("setDiffusionMaterials", &Anneal<T, D>::setDiffusionMaterials,
           py::arg("materials"))
      .def("setBlockingMaterials", &Anneal<T, D>::setBlockingMaterials,
           py::arg("materials"))
      // Field labels
      .def("setSpeciesLabel", &Anneal<T, D>::setSpeciesLabel, py::arg("label"))
      .def("setActiveLabel", &Anneal<T, D>::setActiveLabel, py::arg("label"))
      .def("setMaterialLabel", &Anneal<T, D>::setMaterialLabel,
           py::arg("label"))
      // Solid activation
      .def("enableSolidActivation", &Anneal<T, D>::enableSolidActivation,
           py::arg("enable") = true)
      .def("setSolidSolubilityArrhenius",
           &Anneal<T, D>::setSolidSolubilityArrhenius, py::arg("C0"),
           py::arg("Ea_eV"))
      // Defect coupling
      .def("enableDefectCoupling", &Anneal<T, D>::enableDefectCoupling,
           py::arg("enable") = true)
      .def("resetDefectInitialization",
           &Anneal<T, D>::resetDefectInitialization)
      .def("setDamageLabels", &Anneal<T, D>::setDamageLabels,
           py::arg("damageLabel"), py::arg("lastDamageLabel"))
      .def("setDefectLabels", &Anneal<T, D>::setDefectLabels,
           py::arg("interstitialLabel"), py::arg("vacancyLabel"))
      .def("setDefectSourceWeights", &Anneal<T, D>::setDefectSourceWeights,
           py::arg("historyWeight"), py::arg("lastImpWeight"))
      .def("setDefectPartition", &Anneal<T, D>::setDefectPartition,
           py::arg("interstitialFraction"), py::arg("vacancyFraction"))
      .def("setDefectPartitionFactors",
           &Anneal<T, D>::setDefectPartitionFactors,
           py::arg("interstitialFactor"), py::arg("vacancyFactor"))
      .def("setDefectDiffusivities", &Anneal<T, D>::setDefectDiffusivities,
           py::arg("Di"), py::arg("Dv"))
      .def("setDefectReactionRates", &Anneal<T, D>::setDefectReactionRates,
           py::arg("kRecombination"), py::arg("kInterstitialSink"),
           py::arg("kVacancySink"))
      // Defect equilibrium
      .def("enableDefectEquilibrium",
           &Anneal<T, D>::enableDefectEquilibrium, py::arg("enable") = true)
      .def("setDefectEquilibrium", &Anneal<T, D>::setDefectEquilibrium,
           py::arg("Ieq"), py::arg("Veq"))
      .def("setDefectEquilibriumArrhenius",
           &Anneal<T, D>::setDefectEquilibriumArrhenius,
           py::arg("interstitialC0"), py::arg("interstitialEa_eV"),
           py::arg("vacancyC0"), py::arg("vacancyEa_eV"))
      .def("clearEquilibriumArrhenius",
           &Anneal<T, D>::clearEquilibriumArrhenius)
      // TED
      .def("setDefectEnhancedDiffusion",
           &Anneal<T, D>::setDefectEnhancedDiffusion,
           py::arg("tedCoefficient"), py::arg("normalization"))
      .def("setTEDFromDamageFactor",
           &Anneal<T, D>::setTEDFromDamageFactor,
           py::arg("damageFactor"),
           py::arg("coefficientScale") = T(0.5),
           py::arg("normalization") = T(1e20))
      // Defect clustering
      .def("enableDefectClustering",
           &Anneal<T, D>::enableDefectClustering, py::arg("enable") = true)
      .def("setDefectClusterLabel", &Anneal<T, D>::setDefectClusterLabel,
           py::arg("label"))
      .def("setDefectClusterKinetics",
           &Anneal<T, D>::setDefectClusterKinetics,
           py::arg("kfi"), py::arg("kfc"), py::arg("kr"))
      .def("setDefectClusterInitFraction",
           &Anneal<T, D>::setDefectClusterInitFraction, py::arg("fraction"))
      // Diagnostics
      .def("enableDiagnostics", &Anneal<T, D>::enableDiagnostics,
           py::arg("enable") = true)
      .def("setDiagnosticsMaterialFilter",
           &Anneal<T, D>::setDiagnosticsMaterialFilter,
           py::arg("materialId"))
      .def("clearDefectDiagnostics", &Anneal<T, D>::clearDefectDiagnostics)
      .def("getDefectDiagnostics", &Anneal<T, D>::getDefectDiagnostics,
           py::return_value_policy::reference_internal)
      .def("setClampNonNegative", &Anneal<T, D>::setClampNonNegative,
           py::arg("enable") = true)
      .def("setSourceField", &Anneal<T, D>::setSourceField,
           py::arg("source"),
           "Set an external source term (cell-indexed vector) added each time step.")
      .def("clearSourceField", &Anneal<T, D>::clearSourceField,
           "Remove the previously set source field.")
      .def("applyActivation", &Anneal<T, D>::applyActivation,
           py::arg("domain"),
           "Apply only the solid-activation model without running diffusion.\n\n"
           "Equivalent to Sentaurus 'diffuse time=0': writes the active-\n"
           "concentration field immediately after implantation so that\n"
           "SheetResistance and NetDoping work before the full thermal anneal.\n\n"
           "Prerequisites: enableSolidActivation(True) and\n"
           "setSolidSolubilityArrhenius(C0, Ea) must be configured.");

  // ── SheetResistance ──────────────────────────────────────────────────────
  py::class_<viennacs::SheetResistance<T, D>>(module, "SheetResistance",
      "Compute sheet resistance (Rsh, Ω/□) from an active-concentration\n"
      "field stored in the domain's cell set.\n\n"
      "Default settings target ViennaPS nm-unit domains:\n"
      "  length unit = 1e-7 (nm → cm),  conc unit = 1e21 (nm⁻³ → cm⁻³),\n"
      "  depth axis  = D−1  (y for 2-D, z for 3-D),\n"
      "  surface position = 0  (depth = surface − coordinate).\n\n"
      "Example::\n\n"
      "  sr = SheetResistance()\n"
      "  sr.setCellSet(domain.getCellSet())\n"
      "  sr.setConcentrationLabel(\"P_active\")\n"
      "  rsh = sr.computeElectron()   # Masetti n-type (P in Si)")
      .def(py::init<>())
      .def("setCellSet",
           &viennacs::SheetResistance<T, D>::setCellSet,
           py::arg("cellSet"),
           "Attach the cell set to analyse.")
      .def("setConcentrationLabel",
           &viennacs::SheetResistance<T, D>::setConcentrationLabel,
           py::arg("label"),
           "Name of the scalar field containing the active concentration "
           "(default: 'active_concentration').")
      .def("setDepthAxis",
           &viennacs::SheetResistance<T, D>::setDepthAxis,
           py::arg("axis"),
           "Cell-centre axis index for depth  (default: D−1).")
      .def("setSurfacePosition",
           &viennacs::SheetResistance<T, D>::setSurfacePosition,
           py::arg("surfacePosition"),
           "Wafer-surface coordinate along the depth axis. Depth is computed "
           "as surfacePosition minus the cell-centre coordinate.")
      .def("setLengthUnit",
           &viennacs::SheetResistance<T, D>::setLengthUnit,
           py::arg("lu_cm"),
           "Length-unit → cm conversion factor "
           "(default: 1e-7 for nm domains). "
           "Also updates the concentration unit to stay consistent.")
      .def("setConcentrationUnit",
           &viennacs::SheetResistance<T, D>::setConcentrationUnit,
           py::arg("unit"),
           "Multiplicative factor to convert the cell-set concentration to "
           "cm⁻³ (default: 1e21 for nm⁻³ fields).")
      .def("computeElectron",
           &viennacs::SheetResistance<T, D>::computeElectron,
           "Rsh [Ω/□] using the Masetti-Severi electron mobility model "
           "(n-type, e.g. P-doped Si).")
      .def("computeHole",
           &viennacs::SheetResistance<T, D>::computeHole,
           "Rsh [Ω/□] using the Masetti-Severi hole mobility model "
           "(p-type, e.g. B-doped Si).");

  // ── NetDoping ────────────────────────────────────────────────────────────
  py::class_<viennacs::NetDoping<T, D>>(module, "NetDoping",
      "Compute net doping (Σ donors − Σ acceptors) and extract the\n"
      "metallurgical junction depth from the domain's cell set.\n\n"
      "Typical flow after implanting P and B and calling Anneal.applyActivation:\n\n"
      "  nd = NetDoping()\n"
      "  nd.setCellSet(domain.getCellSet())\n"
      "  nd.addDonorLabel('P_active')\n"
      "  nd.addAcceptorLabel('B_active')\n"
      "  nd.apply()                      # writes 'net_doping' to cell set\n"
      "  xj = nd.junctionDepth()         # nm — metallurgical junction depth\n"
      "  print(nd.junctionCount(), 'junction(s)')")
      .def(py::init<>())
      .def("setCellSet",
           &viennacs::NetDoping<T, D>::setCellSet,
           py::arg("cellSet"),
           "Attach the cell set to analyse.")
      .def("addDonorLabel",
           &viennacs::NetDoping<T, D>::addDonorLabel,
           py::arg("label"),
           "Append one donor (n-type) concentration field name.")
      .def("addAcceptorLabel",
           &viennacs::NetDoping<T, D>::addAcceptorLabel,
           py::arg("label"),
           "Append one acceptor (p-type) concentration field name.")
      .def("setDonorLabels",
           &viennacs::NetDoping<T, D>::setDonorLabels,
           py::arg("labels"),
           "Replace the full donor label list.")
      .def("setAcceptorLabels",
           &viennacs::NetDoping<T, D>::setAcceptorLabels,
           py::arg("labels"),
           "Replace the full acceptor label list.")
      .def("setOutputLabel",
           &viennacs::NetDoping<T, D>::setOutputLabel,
           py::arg("label"),
           "Name of the output field written by apply() (default: 'net_doping').")
      .def("setDepthAxis",
           &viennacs::NetDoping<T, D>::setDepthAxis,
           py::arg("axis"),
           "Cell-centre axis index for depth (default: D−1).")
      .def("setSurfacePosition",
           &viennacs::NetDoping<T, D>::setSurfacePosition,
           py::arg("surfacePosition"),
           "Wafer-surface coordinate along the depth axis. Depth is computed "
           "as surfacePosition minus the cell-centre coordinate.")
      .def("apply", &viennacs::NetDoping<T, D>::apply,
           "Compute net_doping = Σ donors − Σ acceptors and write to the "
           "output field in the cell set.")
      .def("junctionDepth",
           &viennacs::NetDoping<T, D>::junctionDepth,
           "Shallowest depth [nm] where net_doping changes sign. "
           "Returns inf if no junction exists or apply() has not been called.")
      .def("junctionDepths",
           &viennacs::NetDoping<T, D>::junctionDepths,
           "All junction depths [nm], sorted ascending.  "
           "Useful for retrograde profiles with multiple crossings.")
      .def("junctionCount",
           &viennacs::NetDoping<T, D>::junctionCount,
           "Number of metallurgical junctions in the depth profile.")
      .def("lateralJunctionPosition",
           &viennacs::NetDoping<T, D>::lateralJunctionPosition,
           py::arg("atDepth"),
           "Lateral position [nm] where net_doping changes sign at the given "
           "depth.  Use for vertical (lateral) PN junctions where P and B are "
           "implanted side by side.  Returns inf if no crossing exists.")
      .def("lateralJunctionPositions",
           &viennacs::NetDoping<T, D>::lateralJunctionPositions,
           py::arg("atDepth"),
           "All lateral junction positions at the given depth [nm], ascending.");

  // Isotropic Process
  py::class_<IsotropicProcess<T, D>, SmartPointer<IsotropicProcess<T, D>>>(
      module, "IsotropicProcess", processModel)
      .def(py::init<T, Material>(), py::arg("rate") = 1.,
           py::arg("maskMaterial") = Material::Undefined)
      .def(py::init<T, std::vector<Material> const &>(), py::arg("rate") = 1.,
           py::arg("maskMaterials"))
      .def(py::init<std::unordered_map<Material, T>, T>(),
           py::arg("materialRates"), py::arg("defaultRate") = 0.)
      .def("setIsotropicRate", &IsotropicProcess<T, D>::setIsotropicRate)
      .def("setMaterialRate", &IsotropicProcess<T, D>::setMaterialRate,
           py::arg("material"), py::arg("rate"));

  // DirectionalProcess
  py::class_<DirectionalProcess<T, D>, SmartPointer<DirectionalProcess<T, D>>>(
      module, "DirectionalProcess", processModel)
      .def(py::init<const Vec3D<T> &,
                    std::unordered_map<Material, std::pair<T, T>>, T, T>(),
           py::arg("direction"), py::arg("materialRates"),
           py::arg("defaultDirectionalRate") = 0.,
           py::arg("defaultIsotropicRate") = 0.)
      .def(py::init<const Vec3D<T> &, T, T, Material, bool>(),
           py::arg("direction"), py::arg("directionalVelocity"),
           py::arg("isotropicVelocity") = 0.,
           py::arg("maskMaterial") = Material::Mask,
           py::arg("calculateVisibility") = true)
      .def(py::init<const Vec3D<T> &, T, T, const std::vector<Material> &,
                    bool>(),
           py::arg("direction"), py::arg("directionalVelocity"),
           py::arg("isotropicVelocity"), py::arg("maskMaterial"),
           py::arg("calculateVisibility") = true)
      .def(py::init<std::vector<typename DirectionalProcess<T, D>::RateSet>>(),
           py::arg("rateSets"))
      // Constructor accepting a single rate set
      .def(py::init<const typename DirectionalProcess<T, D>::RateSet &>(),
           py::arg("rateSet"));

  // Enum for interpolation
  pybind11::native_enum<typename RateGrid<T, D>::Interpolation>(
      module, "Interpolation", "enum.IntEnum")
      .value("LINEAR", RateGrid<T, D>::Interpolation::LINEAR)
      .value("IDW", RateGrid<T, D>::Interpolation::IDW)
      .value("CUSTOM", RateGrid<T, D>::Interpolation::CUSTOM)
      .finalize();

  // RateGrid
  py::class_<RateGrid<T, D>>(module, "RateGrid")
      .def(py::init<>())
      .def("loadFromCSV", &RateGrid<T, D>::loadFromCSV, py::arg("filename"))
      .def("setOffset", &RateGrid<T, D>::setOffset, py::arg("offset"))
      .def("setInterpolationMode",
           static_cast<void (RateGrid<T, D>::*)(
               typename RateGrid<T, D>::Interpolation)>(
               &RateGrid<T, D>::setInterpolationMode),
           py::arg("mode"))
      .def(
          "setInterpolationMode",
          [](RateGrid<T, D> &self, const std::string &str) {
            self.setInterpolationMode(RateGrid<T, D>::fromString(str));
          },
          py::arg("mode"))
      .def("setIDWNeighbors", &RateGrid<T, D>::setIDWNeighbors, py::arg("k"))
      .def(
          "setCustomInterpolator",
          [](RateGrid<T, D> &self, py::function pyFunc) {
            std::cout << "[ViennaPS] NOTE: Custom Python interpolator requires "
                         "single-threaded execution.\n";
            omp_set_num_threads(1);
            self.setCustomInterpolator([pyFunc](const Vec3D<T> &coord) -> T {
              py::gil_scoped_acquire gil;
              auto result = pyFunc(coord);
              return result.cast<T>();
            });
          },
          py::arg("function"))
      .def("interpolate", &RateGrid<T, D>::interpolate, py::arg("coord"));

  // CSVFileProcess
  py::class_<CSVFileProcess<T, D>, ProcessModelCPU<T, D>,
             SmartPointer<CSVFileProcess<T, D>>>(module, "CSVFileProcess")
      .def(py::init<const std::string &, const Vec3D<T> &, const Vec2D<T> &, T,
                    T, const std::vector<Material> &, bool>(),
           py::arg("ratesFile"), py::arg("direction"), py::arg("offset"),
           py::arg("isotropicComponent") = 0.,
           py::arg("directionalComponent") = 1.,
           py::arg("maskMaterials") = std::vector<Material>{Material::Mask},
           py::arg("calculateVisibility") = true)
      .def(
          "setInterpolationMode",
          [](CSVFileProcess<T, D> &self,
             typename RateGrid<T, D>::Interpolation mode) {
            self.setInterpolationMode(mode);
          },
          py::arg("mode"))
      .def(
          "setInterpolationMode",
          [](CSVFileProcess<T, D> &self, const std::string &str) {
            self.setInterpolationMode(str);
          },
          py::arg("mode"))
      .def("setIDWNeighbors", &CSVFileProcess<T, D>::setIDWNeighbors,
           py::arg("k") = 4)
      .def(
          "setCustomInterpolator",
          [](CSVFileProcess<T, D> &self, py::function pyFunc) {
            std::cout << "[ViennaPS] NOTE: Custom Python interpolator requires "
                         "single-threaded execution.\n";
            omp_set_num_threads(1);
            self.setCustomInterpolator([pyFunc](const Vec3D<T> &coord) -> T {
              py::gil_scoped_acquire gil;
              auto result = pyFunc(coord);
              return result.cast<T>();
            });
          },
          py::arg("function"))
      .def("setOffset", &CSVFileProcess<T, D>::setOffset, py::arg("offset"));

  // Sphere Distribution
  py::class_<SphereDistribution<T, D>, SmartPointer<SphereDistribution<T, D>>>(
      module, "SphereDistribution", processModel)
      .def(py::init([](const T radius,
                       SmartPointer<viennals::Domain<T, D>> mask) {
             return SmartPointer<SphereDistribution<T, D>>::New(radius, mask);
           }),
           py::arg("radius"), py::arg("mask"))
      .def(py::init([](const T radius) {
             return SmartPointer<SphereDistribution<T, D>>::New(radius,
                                                                nullptr);
           }),
           py::arg("radius"))
      .def("addMaskMaterial", &SphereDistribution<T, D>::addMaskMaterial,
           py::arg("material"))
      .def("applyToSingleMaterial",
           &SphereDistribution<T, D>::applyToSingleMaterial,
           py::arg("material"));

  // Box Distribution
  py::class_<BoxDistribution<T, D>, SmartPointer<BoxDistribution<T, D>>>(
      module, "BoxDistribution", processModel)
      .def(py::init([](const std::array<T, 3> &halfAxes,
                       SmartPointer<viennals::Domain<T, D>> mask) {
             return SmartPointer<BoxDistribution<T, D>>::New(halfAxes, mask);
           }),
           py::arg("halfAxes"), py::arg("mask"))
      .def(py::init([](const std::array<T, 3> &halfAxes) {
             return SmartPointer<BoxDistribution<T, D>>::New(halfAxes, nullptr);
           }),
           py::arg("halfAxes"))
      .def("addMaskMaterial", &BoxDistribution<T, D>::addMaskMaterial,
           py::arg("material"))
      .def("applyToSingleMaterial",
           &BoxDistribution<T, D>::applyToSingleMaterial, py::arg("material"));

  // Custom Sphere Distribution
  py::class_<CustomSphereDistribution<T, D>,
             SmartPointer<CustomSphereDistribution<T, D>>>(
      module, "CustomSphereDistribution", processModel)
      .def(py::init([](const std::vector<T> &radii,
                       SmartPointer<viennals::Domain<T, D>> mask) {
             return SmartPointer<CustomSphereDistribution<T, D>>::New(radii,
                                                                      mask);
           }),
           py::arg("radii"), py::arg("mask") = nullptr)
      .def("addMaskMaterial", &CustomSphereDistribution<T, D>::addMaskMaterial,
           py::arg("material"));

  py::class_<GeometricTrenchDeposition<T, D>,
             SmartPointer<GeometricTrenchDeposition<T, D>>>(
      module, "GeometricTrenchDeposition", processModel)
      .def(
          py::init(&SmartPointer<GeometricTrenchDeposition<T, D>>::template New<
                   T, T, T, T, T, T, T>),
          py::arg("trenchWidth"), py::arg("trenchDepth"),
          py::arg("depositionRate"), py::arg("bottomMed"), py::arg("a"),
          py::arg("b"), py::arg("n"));

  // Oxide Regrowth
  py::class_<OxideRegrowth<T, D>, SmartPointer<OxideRegrowth<T, D>>>(
      module, "OxideRegrowth", processModel)
      .def(py::init(&SmartPointer<OxideRegrowth<T, D>>::template New<
                    T, T, T, T, T, T, T, T, T, T, T, T>),
           py::arg("nitrideEtchRate"), py::arg("oxideEtchRate"),
           py::arg("redepositionRate"), py::arg("redepositionThreshold"),
           py::arg("redepositionTimeInt"), py::arg("diffusionCoefficient"),
           py::arg("sinkStrength"), py::arg("scallopVelocity"),
           py::arg("centerVelocity"), py::arg("topHeight"),
           py::arg("centerWidth"), py::arg("stabilityFactor"));

  // Wet Etching Process
  py::class_<WetEtching<T, D>, SmartPointer<WetEtching<T, D>>>(
      module, "WetEtching", processModel)
      .def(py::init(&SmartPointer<WetEtching<T, D>>::template New<
                    const std::vector<std::pair<Material, T>>>),
           py::arg("materialRates"))
      .def(py::init(&SmartPointer<WetEtching<T, D>>::template New<
                    const std::array<T, 3> &, const std::array<T, 3> &, T, T, T,
                    T, const std::vector<std::pair<Material, T>>>),
           py::arg("direction100"), py::arg("direction010"), py::arg("rate100"),
           py::arg("rate110"), py::arg("rate111"), py::arg("rate311"),
           py::arg("materialRates"));

  // Selective Epitaxy Process
  py::class_<SelectiveEpitaxy<T, D>, SmartPointer<SelectiveEpitaxy<T, D>>>(
      module, "SelectiveEpitaxy", processModel)
      .def(py::init(&SmartPointer<SelectiveEpitaxy<T, D>>::template New<T, T>),
           py::arg("rate111") = 0.5, py::arg("rate100") = 1.0)
      .def(py::init(&SmartPointer<SelectiveEpitaxy<T, D>>::template New<
                    const std::vector<std::pair<Material, T>>, T, T>),
           py::arg("materialRates"), py::arg("rate111") = 0.5,
           py::arg("rate100") = 1.0)
      .def(py::init(&SmartPointer<SelectiveEpitaxy<T, D>>::template New<
                    const Vec3D<T> &, T, T>),
           py::arg("nvFactors"), py::arg("rate111") = 0.5,
           py::arg("rate100") = 1.0)
      .def("setMaterialRate", &SelectiveEpitaxy<T, D>::setMaterialRate,
           py::arg("material"), py::arg("rate"));

  // Single Particle ALD
  py::class_<SingleParticleALD<T, D>, SmartPointer<SingleParticleALD<T, D>>>(
      module, "SingleParticleALD", processModel)
      .def(py::init(&SmartPointer<SingleParticleALD<T, D>>::template New<
                    T, int, T, int, T, T, T, T, T>),
           py::arg("stickingProbability"), py::arg("numCycles"),
           py::arg("growthPerCycle"), py::arg("totalCycles"),
           py::arg("coverageTimeStep"), py::arg("evFlux"), py::arg("inFlux"),
           py::arg("s0"), py::arg("gasMFP"));

  // ***************************************************************************
  //                               GEOMETRIES
  // ***************************************************************************

  // Geometry Base
  py::class_<GeometryFactory<T, D>>(module, "GeometryFactory")
      .def(py::init<const DomainSetup<D> &, const std::string &>(),
           py::arg("domainSetup"), py::arg("name") = "GeometryFactory")
      .def("makeMask", &GeometryFactory<T, D>::makeMask, py::arg("base"),
           py::arg("height"))
      .def("makeSubstrate", &GeometryFactory<T, D>::makeSubstrate,
           py::arg("base"))
      .def("makeCylinderStencil", &GeometryFactory<T, D>::makeCylinderStencil,
           py::arg("position"), py::arg("radius"), py::arg("height"),
           py::arg("angle") = 0.)
      .def("makeBoxStencil", &GeometryFactory<T, D>::makeBoxStencil,
           py::arg("position"), py::arg("width"), py::arg("height"),
           py::arg("angle") = 0., py::arg("length") = -1.);

  // Plane
  py::class_<MakePlane<T, D>>(module, "MakePlane")
      .def(py::init<DomainType, T, Material, bool>(), py::arg("domain"),
           py::arg("height") = 0., py::arg("material") = Material::Si,
           py::arg("addToExisting") = false)
      .def(py::init<DomainType, T, T, T, T, bool, Material>(),
           py::arg("domain"), py::arg("gridDelta"), py::arg("xExtent"),
           py::arg("yExtent"), py::arg("height") = 0.,
           py::arg("periodicBoundary") = false,
           py::arg("material") = Material::Si)
      .def("apply", &MakePlane<T, D>::apply,
           "Create a plane geometry or add plane to existing geometry.");

  // Trench
  py::class_<MakeTrench<T, D>> trench(module, "MakeTrench");

  // MaterialLayer struct for multi-layer geometries
  py::class_<typename MakeTrench<T, D>::MaterialLayer>(trench, "MaterialLayer")
      .def(py::init<>())
      .def(py::init<T, T, T, Material, bool>(), py::arg("height"),
           py::arg("width"), py::arg("taperAngle"), py::arg("material"),
           py::arg("isMask"))
      .def_readwrite("height", &MakeTrench<T, D>::MaterialLayer::height,
                     "Layer thickness")
      .def_readwrite("width", &MakeTrench<T, D>::MaterialLayer::width,
                     "Width of cutout for this layer")
      .def_readwrite("taperAngle", &MakeTrench<T, D>::MaterialLayer::taperAngle,
                     "Taper angle for cutout (degrees)")
      .def_readwrite("material", &MakeTrench<T, D>::MaterialLayer::material,
                     "Material type for this layer")
      .def_readwrite("isMask", &MakeTrench<T, D>::MaterialLayer::isMask,
                     "true: apply cutout (mask behavior), false: no cutout");

  trench
      .def(py::init<DomainType, T, T, T, T, T, bool, Material, Material>(),
           py::arg("domain"), py::arg("trenchWidth"), py::arg("trenchDepth"),
           py::arg("trenchTaperAngle") = 0, py::arg("maskHeight") = 0,
           py::arg("maskTaperAngle") = 0, py::arg("halfTrench") = false,
           py::arg("material") = Material::Si,
           py::arg("maskMaterial") = Material::Mask)
      .def(py::init<DomainType, T, T, T, T, T, T, T, bool, bool, Material>(),
           py::arg("domain"), py::arg("gridDelta"), py::arg("xExtent"),
           py::arg("yExtent"), py::arg("trenchWidth"), py::arg("trenchDepth"),
           py::arg("taperingAngle") = 0., py::arg("baseHeight") = 0.,
           py::arg("periodicBoundary") = false, py::arg("makeMask") = false,
           py::arg("material") = Material::Si)
      .def(py::init<
               DomainType,
               const std::vector<typename MakeTrench<T, D>::MaterialLayer> &,
               bool>(),
           py::arg("domain"), py::arg("materialLayers"),
           py::arg("halfTrench") = false)
      .def("apply", &MakeTrench<T, D>::apply, "Create a trench geometry.");

  // Hole
  py::class_<MakeHole<T, D>>(module, "MakeHole")
      .def(py::init<DomainType, T, T, T, T, T, HoleShape, Material, Material>(),
           py::arg("domain"), py::arg("holeRadius"), py::arg("holeDepth"),
           py::arg("holeTaperAngle") = 0., py::arg("maskHeight") = 0.,
           py::arg("maskTaperAngle") = 0.,
           py::arg("holeShape") = HoleShape::FULL,
           py::arg("material") = Material::Si,
           py::arg("maskMaterial") = Material::Mask)
      .def(py::init<DomainType, T, T, T, T, T, T, T, bool, bool, const Material,
                    HoleShape>(),
           py::arg("domain"), py::arg("gridDelta"), py::arg("xExtent"),
           py::arg("yExtent"), py::arg("holeRadius"), py::arg("holeDepth"),
           py::arg("taperingAngle") = 0., py::arg("baseHeight") = 0.,
           py::arg("periodicBoundary") = false, py::arg("makeMask") = false,
           py::arg("material") = Material::Si,
           py::arg("holeShape") = HoleShape::FULL)
      .def("apply", &MakeHole<T, D>::apply, "Create a hole geometry.");

  // Fin
  py::class_<MakeFin<T, D>>(module, "MakeFin")
      .def(py::init<DomainType, T, T, T, T, T, bool, Material, Material>(),
           py::arg("domain"), py::arg("finWidth"), py::arg("finHeight"),
           py::arg("finTaperAngle") = 0., py::arg("maskHeight") = 0,
           py::arg("maskTaperAngle") = 0, py::arg("halfFin") = false,
           py::arg("material") = Material::Si,
           py::arg("maskMaterial") = Material::Mask)
      .def(py::init<DomainType, T, T, T, T, T, T, T, bool, bool, Material>(),
           py::arg("domain"), py::arg("gridDelta"), py::arg("xExtent"),
           py::arg("yExtent"), py::arg("finWidth"), py::arg("finHeight"),
           py::arg("taperAngle") = 0., py::arg("baseHeight") = 0.,
           py::arg("periodicBoundary") = false, py::arg("makeMask") = false,
           py::arg("material") = Material::Si)
      .def("apply", &MakeFin<T, D>::apply, "Create a fin geometry.");

  // Stack
  py::class_<MakeStack<T, D>>(module, "MakeStack")
      .def(py::init<DomainType, int, T, T, T, T, T, T, bool, Material>(),
           py::arg("domain"), py::arg("numLayers"), py::arg("layerHeight"),
           py::arg("substrateHeight") = 0, py::arg("holeRadius") = 0,
           py::arg("trenchWidth") = 0, py::arg("maskHeight") = 0,
           py::arg("taperAngle") = 0, py::arg("halfStack") = false,
           py::arg("maskMaterial") = Material::Mask)
      .def(py::init<DomainType, T, T, T, int, T, T, T, T, T, bool>(),
           py::arg("domain"), py::arg("gridDelta"), py::arg("xExtent"),
           py::arg("yExtent"), py::arg("numLayers"), py::arg("layerHeight"),
           py::arg("substrateHeight"), py::arg("holeRadius"),
           py::arg("trenchWidth"), py::arg("maskHeight"),
           py::arg("periodicBoundary") = false)
      .def("apply", &MakeStack<T, D>::apply,
           "Create a stack of alternating SiO2 and Si3N4 layers.")
      .def("getTopLayer", &MakeStack<T, D>::getTopLayer,
           "Returns the number of layers included in the stack")
      .def("getHeight", &MakeStack<T, D>::getHeight,
           "Returns the total height of the stack.");

  // ***************************************************************************
  //                                 PROCESS
  // ***************************************************************************

  py::class_<lsInternal::StencilLocalLaxFriedrichsScalar<T, D, 1>>(
      module, "StencilLocalLaxFriedrichsScalar", py::module_local())
      .def_static(
          "setMaxDissipation",
          &lsInternal::StencilLocalLaxFriedrichsScalar<T, D,
                                                       1>::setMaxDissipation,
          py::arg("maxDissipation"));

  // Process
  using ProcessTD = Process<T, D>;
  py::class_<ProcessTD>(module, "Process", py::module_local())
      // constructors
      .def(py::init())
      .def(py::init<DomainType>(), py::arg("domain"))
      .def(py::init<DomainType, SmartPointer<ProcessModelBase<T, D>>, T>(),
           py::arg("domain"), py::arg("model"), py::arg("duration") = 0.)
      .def(py::init([](DomainType domain,
                       SmartPointer<ProcessModelBase<T, D>> model, T duration,
                       py::args args) {
             ProcessTD process(domain, model, duration);
             for (auto arg : args) {
               if (py::isinstance<AdvectionParameters>(arg)) {
                 process.setParameters(arg.cast<AdvectionParameters>());
               } else if (py::isinstance<RayTracingParameters>(arg)) {
                 process.setParameters(arg.cast<RayTracingParameters>());
               } else if (py::isinstance<CoverageParameters>(arg)) {
                 process.setParameters(arg.cast<CoverageParameters>());
               } else if (py::isinstance<AtomicLayerProcessParameters>(arg)) {
                 process.setParameters(
                     arg.cast<AtomicLayerProcessParameters>());
               } else {
                 throw py::type_error(
                     "Unsupported parameter type for Process constructor");
               }
             }
             return process;
           }),
           py::arg("domain"), py::arg("model"), py::arg("duration") = 0.)
      // methods
      .def("apply", &ProcessTD::apply,
           //  py::call_guard<py::gil_scoped_release>(),
           "Run the process.")
      .def("calculateFlux", &ProcessTD::calculateFlux,
           //  py::call_guard<py::gil_scoped_release>(),
           "Perform a single-pass flux calculation.")
      .def("setDomain", &ProcessTD::setDomain, "Set the process domain.")
      .def("setProcessModel", &ProcessTD::setProcessModel,
           "Set the process model. This has to be a pre-configured process "
           "model.")
      .def("setProcessDuration", &ProcessTD::setProcessDuration,
           "Set the process duration.")
      .def("setFluxEngineType", &ProcessTD::setFluxEngineType,
           "Set the flux engine type (CPU or GPU).")
      .def("setIntermediateOutputPath", &ProcessTD::setIntermediateOutputPath,
           py::arg("path"),
           "Set the path for intermediate output files during the process.")
      .def("setParameters",
           py::overload_cast<const AdvectionParameters &>(
               &ProcessTD::setParameters),
           py::arg("parameters"),
           "Set the advection parameters for the process.")
      .def("setParameters",
           py::overload_cast<const RayTracingParameters &>(
               &ProcessTD::setParameters),
           py::arg("parameters"),
           "Set the ray tracing parameters for the process.")
      .def("setParameters",
           py::overload_cast<const CoverageParameters &>(
               &ProcessTD::setParameters),
           py::arg("parameters"),
           "Set the coverage parameters for the process.")
      .def("setParameters",
           py::overload_cast<const AtomicLayerProcessParameters &>(
               &ProcessTD::setParameters),
           py::arg("parameters"),
           "Set the atomic layer parameters for the process.");

  // ***************************************************************************
  //                                   VISUALIZATION
  //  ***************************************************************************

  py::class_<ToDiskMesh<T, D>>(module, "ToDiskMesh")
      .def(py::init<DomainType, SmartPointer<viennals::Mesh<T>>>(),
           py::arg("domain"), py::arg("mesh"))
      .def(py::init())
      .def("setDomain", &ToDiskMesh<T, D>::setDomain,
           "Set the domain in the mesh converter.")
      .def("setMesh", &ToDiskMesh<T, D>::setMesh,
           "Set the mesh in the mesh converter")
      .def("apply", &ToDiskMesh<T, D>::apply);
  // static assertion failed: Holder classes are only supported for
  //  custom types
  // .def("setTranslator", &ToDiskMesh<T, D>::setTranslator,
  //      "Set the translator in the mesh converter. It used to convert "
  //      "level-set point IDs to mesh point IDs.")
  // .def("getTranslator", &ToDiskMesh<T, D>::getTranslator,
  //      "Retrieve the translator from the mesh converter.");

#ifdef VIENNALS_VTK_RENDERING
  py::class_<VTKRenderWindow<T, D>>(module, "VTKRenderWindow")
      .def(py::init<>())
      .def(py::init<DomainType>(), py::arg("domain"))
      .def("insertNextDomain", &VTKRenderWindow<T, D>::insertNextDomain,
           "Insert domain to be visualized.", py::arg("domain"),
           py::arg("offset") = std::array<double, 3>{0., 0., 0.})
      .def("setDomainOffset", &VTKRenderWindow<T, D>::setDomainOffset,
           "Set an offset to be applied to the domain during rendering.")
      .def("setBackgroundColor", &VTKRenderWindow<T, D>::setBackgroundColor,
           "Set the background color of the render window.")
      .def("setWindowSize", &VTKRenderWindow<T, D>::setWindowSize,
           "Set the size of the render window.")
      .def("setRenderMode", &VTKRenderWindow<T, D>::setRenderMode,
           "Set the render mode (surface, interfaces, volume).")
      .def("setCameraPosition", &VTKRenderWindow<T, D>::setCameraPosition,
           "Set the camera position in world coordinates.")
      .def("setCameraFocalPoint", &VTKRenderWindow<T, D>::setCameraFocalPoint,
           "Set the camera focal point in world coordinates.")
      .def("setCameraViewUp", &VTKRenderWindow<T, D>::setCameraViewUp,
           "Set the camera view up vector.")
      .def("setCameraView", &VTKRenderWindow<T, D>::setCameraView,
           "Set the camera view along an axix (x,y,z)", py::arg("axis"))
      .def("printCameraInfo", &VTKRenderWindow<T, D>::printCameraInfo,
           "Print the current camera settings to the console.")
      .def("saveScreenshot", &VTKRenderWindow<T, D>::saveScreenshot,
           "Save a screenshot of the current render window.",
           py::arg("fileName"), py::arg("scale") = 1)
      .def("toggleInstructionText",
           &VTKRenderWindow<T, D>::toggleInstructionText,
           "Toggle the instruction text overlay on/off.")
      .def("render", &VTKRenderWindow<T, D>::render,
           "Render the current domain state.");
#endif

  // ***************************************************************************
  //                                 IO OPERATIONS
  // ***************************************************************************

  // Writer
  py::class_<Writer<T, D>>(module, "Writer")
      .def(py::init<>())
      .def(py::init<SmartPointer<Domain<T, D>>>(), py::arg("domain"))
      .def(py::init<SmartPointer<Domain<T, D>>, std::string>(),
           py::arg("domain"), py::arg("fileName"))
      .def("setDomain", &Writer<T, D>::setDomain,
           "Set the domain to be written to a file.")
      .def("setFileName", &Writer<T, D>::setFileName,
           "Set the output file name (should end with .vpsd).")
      .def("apply", &Writer<T, D>::apply,
           "Write the domain to the specified file.");

  // Reader
  py::class_<Reader<T, D>>(module, "Reader")
      .def(py::init<>())
      .def(py::init<std::string>(), py::arg("fileName"))
      .def(py::init<SmartPointer<Domain<T, D>>, std::string>(),
           py::arg("domain"), py::arg("fileName"))
      .def("setDomain", &Reader<T, D>::setDomain,
           "Set the domain to read into.")
      .def("setFileName", &Reader<T, D>::setFileName,
           "Set the input file name to read (should end with .vpsd).")
      .def("apply", &Reader<T, D>::apply,
           "Read the domain from the specified file.");

  //   ***************************************************************************
  //                                  OTHER
  //   ***************************************************************************

  // Planarize
  py::class_<Planarize<T, D>>(module, "Planarize")
      .def(py::init())
      .def(py::init<DomainType &, const T>(), py::arg("geometry"),
           py::arg("cutoffHeight") = 0.)
      .def("setDomain", &Planarize<T, D>::setDomain,
           "Set the domain in the planarization.")
      .def("setCutoffPosition", &Planarize<T, D>::setCutoffPosition,
           "Set the cutoff height for the planarization.")
      .def("apply", &Planarize<T, D>::apply, "Apply the planarization.");

  // GDS file parsing
  py::class_<GDSGeometry<T, D>, SmartPointer<GDSGeometry<T, D>>> gdsGeo(
      module, "GDSGeometry");
  // constructors
  gdsGeo.def(py::init(&SmartPointer<GDSGeometry<T, D>>::template New<>))
      .def(py::init(&SmartPointer<GDSGeometry<T, D>>::template New<const T>),
           py::arg("gridDelta"))
      .def(py::init(
               [](const T gridDelta,
                  std::array<typename ls::Domain<T, D>::BoundaryType, D> bcs) {
                 auto ptr = SmartPointer<GDSGeometry<T, D>>::New(gridDelta);
                 ptr->setBoundaryConditions(bcs.data());
                 return ptr;
               }),
           py::arg("gridDelta"),
           py::arg("boundaryConditions")) // methods
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
      .def(
          "getBounds",
          [](GDSGeometry<T, D> &gds) -> std::array<double, 6> {
            auto b = gds.getBounds();
            std::array<double, 6> bounds;
            for (unsigned i = 0; i < 6; ++i)
              bounds[i] = b[i];
            return bounds;
          },
          "Get the bounds of the geometry.")
      // Blurring
      .def("addBlur", &GDSGeometry<T, D>::addBlur, py::arg("sigmas"),
           py::arg("weights"), py::arg("threshold") = 0.5,
           py::arg("delta") = 0.0, py::arg("gridRefinement") = 4,
           "Set parameters for applying mask blurring.")
      .def("getAllLayers", &GDSGeometry<T, D>::getAllLayers,
           "Return a set of all layers found in the GDS file.")
      .def("getNumberOfStructures", &GDSGeometry<T, D>::getNumberOfStructures,
           "Return number of structure definitions.");
  // Level set conversion
  if constexpr (D == 2) {
    gdsGeo.def(
        "layerToLevelSet",
        static_cast<SmartPointer<viennals::Domain<T, D>> (GDSGeometry<T, D>::*)(
            const int16_t, bool)>(&GDSGeometry<T, D>::layerToLevelSet),
        py::arg("layer"), py::arg("blurLayer") = true);
  } else {
    gdsGeo.def("layerToLevelSet",
               static_cast<SmartPointer<viennals::Domain<T, D>> (
                   GDSGeometry<T, D>::*)(const int16_t, T, T, bool, bool)>(
                   &GDSGeometry<T, D>::layerToLevelSet),
               py::arg("layer"), py::arg("baseHeight") = 0.,
               py::arg("height") = 1., py::arg("mask") = false,
               py::arg("blurLayer") = true);
  }

  py::class_<GDSReader<T, D>>(module, "GDSReader")
      // constructors
      .def(py::init())
      .def(py::init<SmartPointer<GDSGeometry<T, D>> &, std::string>())
      // methods
      .def("setGeometry", &GDSReader<T, D>::setGeometry,
           "Set the domain to be parsed in.")
      .def("setFileName", &GDSReader<T, D>::setFileName,
           "Set name of the GDS file.")
      .def("apply", &GDSReader<T, D>::apply, "Parse the GDS file.");

  // ***************************************************************************
  //                                 GPU MODELS
  // ***************************************************************************

#ifdef VIENNACORE_COMPILE_GPU
  auto m_gpu = module.def_submodule("gpu", "GPU accelerated functions.");

  // GPU ProcessModel
  py::class_<gpu::ProcessModelGPU<T, D>,
             SmartPointer<gpu::ProcessModelGPU<T, D>>>
      processModel_gpu(m_gpu, "ProcessModelGPU", processModelBase);

  py::class_<gpu::SingleParticleProcess<T, D>,
             SmartPointer<gpu::SingleParticleProcess<T, D>>>(
      m_gpu, "SingleParticleProcess", processModel_gpu)
      .def(py::init<MaterialValueMap<T> const &, T, T>(),
           py::arg("materialRates"), py::arg("stickingProbability"),
           py::arg("sourceExponent"));

  // Multi Particle Process
  py::class_<gpu::MultiParticleProcess<T, D>,
             SmartPointer<gpu::MultiParticleProcess<T, D>>>(
      m_gpu, "MultiParticleProcess", processModel_gpu)
      .def(py::init())
      .def("addNeutralParticle",
           py::overload_cast<T, const std::string &>(
               &gpu::MultiParticleProcess<T, D>::addNeutralParticle),
           py::arg("stickingProbability"), py::arg("label") = "neutralFlux")
      .def("addNeutralParticle",
           py::overload_cast<const MaterialValueMap<T> &, T,
                             const std::string &>(
               &gpu::MultiParticleProcess<T, D>::addNeutralParticle),
           py::arg("materialSticking"),
           py::arg("defaultStickingProbability") = 1.,
           py::arg("label") = "neutralFlux")
      .def("addIonParticle", &gpu::MultiParticleProcess<T, D>::addIonParticle,
           py::arg("sourcePower"), py::arg("thetaRMin") = 0.,
           py::arg("thetaRMax") = 90., py::arg("minAngle") = 0.,
           py::arg("B_sp") = -1., py::arg("meanEnergy") = 0.,
           py::arg("sigmaEnergy") = 0., py::arg("thresholdEnergy") = 0.,
           py::arg("inflectAngle") = 0., py::arg("n") = 1,
           py::arg("label") = "ionFlux")
      .def("setRateFunction",
           &gpu::MultiParticleProcess<T, D>::setRateFunction);

  // SF6O2 Etching
  py::class_<gpu::SF6O2Etching<T, D>, SmartPointer<gpu::SF6O2Etching<T, D>>>(
      m_gpu, "SF6O2Etching", processModel_gpu)
      .def(py::init(&SmartPointer<gpu::SF6O2Etching<T, D>>::template New<
                    const PlasmaEtchingParameters<T> &>),
           py::arg("parameters"));

  // HBrO2 Etching
  py::class_<gpu::HBrO2Etching<T, D>, SmartPointer<gpu::HBrO2Etching<T, D>>>(
      m_gpu, "HBrO2Etching", processModel_gpu)
      .def(py::init(&SmartPointer<gpu::HBrO2Etching<T, D>>::template New<
                    const PlasmaEtchingParameters<T> &>),
           py::arg("parameters"));

  // Ion Beam Etching
  py::class_<gpu::IonBeamEtching<T, D>,
             SmartPointer<gpu::IonBeamEtching<T, D>>>(m_gpu, "IonBeamEtching",
                                                      processModel)
      .def(py::init(&SmartPointer<gpu::IonBeamEtching<T, D>>::template New<
                    const IBEParameters<T> &>),
           py::arg("parameters"));

  // Faraday Cage Etching
  py::class_<gpu::FaradayCageEtching<T, D>,
             SmartPointer<gpu::FaradayCageEtching<T, D>>>(
      m_gpu, "FaradayCageEtching", processModel)
      .def(py::init(&SmartPointer<gpu::FaradayCageEtching<T, D>>::template New<
                    const FaradayCageParameters<T> &>),
           py::arg("parameters"));
#endif
}
