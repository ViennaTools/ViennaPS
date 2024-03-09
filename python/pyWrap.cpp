/*
  This file is used to generate the python module of ViennaPS.
  It uses pybind11 to create the modules.

  All necessary headers are included here and the interface
  of the classes which should be exposed defined
*/

#include "pyWrap.hpp"

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

  wrapGeometries(module);

  wrapProcesses(module);

  wrapModels(module);

  pybind11::enum_<psLogLevel>(module, "LogLevel")
      .value("ERROR", psLogLevel::ERROR)
      .value("WARNING", psLogLevel::WARNING)
      .value("INFO", psLogLevel::INFO)
      .value("TIMING", psLogLevel::TIMING)
      .value("INTERMEDIATE", psLogLevel::INTERMEDIATE)
      .value("DEBUG", psLogLevel::DEBUG)
      .export_values();

  // some unexpected behaviour can happen as it is working with multithreading
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
           "returns a "
           "copy of the data")
      .def("getScalarDataLabels", &csDenseCellSet<T, D>::getScalarDataLabels,
           "Get the labels of the scalar data stored in the cell set.")
      .def("getIndex", &csDenseCellSet<T, D>::getIndex,
           "Get the index of the cell containing the given point.")
      .def("getCellSetPosition", &csDenseCellSet<T, D>::getCellSetPosition)
      .def("setCellSetPosition", &csDenseCellSet<T, D>::setCellSetPosition,
           "Set whether the cell set should be created below (false) or above "
           "(true) the surface.")
      .def("setCoverMaterial", &csDenseCellSet<T, D>::setCoverMaterial,
           "Set the material of the cells which are above or below the "
           "surface.")
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

  /****************************************************************************
   *                               VISUALIZATION *
   ****************************************************************************/

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
  // static assertion failed: Holder classes are only supported for custom types
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
           "Set the output file name. The file name will be appended by "
           "'_volume.vtu'.")
      .def("setDomain", &psWriteVisualizationMesh<T, D>::setDomain,
           "Set the domain in the mesh converter.");

  /****************************************************************************
   *                               OTHER                                      *
   ****************************************************************************/

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