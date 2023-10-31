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

  // wrap omp_set_num_threads to control number of threads
  module.def("setNumThreads", &omp_set_num_threads);

  // it was giving an error that it could'nt convert this type to python
  // pybind11::bind_vector<std::vector<double, std::allocator<double>>>(
  //     module, "VectorDouble");

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
  pybind11::class_<psSurfaceModel<T>, psSmartPointer<psSurfaceModel<T>>,
                   PypsSurfaceModel>(module, "SurfaceModel")
      .def(pybind11::init<>())
      .def("initializeCoverages", &psSurfaceModel<T>::initializeCoverages)
      .def("initializeProcessParameters",
           &psSurfaceModel<T>::initializeProcessParameters)
      .def("getCoverages", &psSurfaceModel<T>::getCoverages)
      .def("getProcessParameters", &psSurfaceModel<T>::getProcessParameters)
      .def("calculateVelocities", &psSurfaceModel<T>::calculateVelocities)
      .def("updateCoverages", &psSurfaceModel<T>::updateCoverages);

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

  // psVelocityField
  pybind11::class_<psVelocityField<T>, psSmartPointer<psVelocityField<T>>,
                   PyVelocityField>
      velocityField(module, "VelocityField");
  // constructors
  velocityField
      .def(pybind11::init<>())
      // methods
      .def("getScalarVelocity", &psVelocityField<T>::getScalarVelocity)
      .def("getVectorVelocity", &psVelocityField<T>::getVectorVelocity)
      .def("getDissipationAlpha", &psVelocityField<T>::getDissipationAlpha)
      .def("getTranslationFieldOptions",
           &psVelocityField<T>::getTranslationFieldOptions)
      .def("setVelocities", &psVelocityField<T>::setVelocities);

  pybind11::class_<psDefaultVelocityField<T>,
                   psSmartPointer<psDefaultVelocityField<T>>>(
      module, "DefaultVelocityField", velocityField)
      // constructors
      .def(pybind11::init<>())
      // methods
      .def("getScalarVelocity", &psDefaultVelocityField<T>::getScalarVelocity)
      .def("getVectorVelocity", &psDefaultVelocityField<T>::getVectorVelocity)
      .def("getDissipationAlpha",
           &psDefaultVelocityField<T>::getDissipationAlpha)
      .def("getTranslationFieldOptions",
           &psDefaultVelocityField<T>::getTranslationFieldOptions)
      .def("setVelocities", &psDefaultVelocityField<T>::setVelocities);

  // psDomain
  pybind11::class_<psDomain<T, D>, DomainType>(module, "Domain")
      // constructors
      .def(pybind11::init<bool>())
      .def(pybind11::init(&DomainType::New<>))
      // methods
      .def("insertNextLevelSet", &psDomain<T, D>::insertNextLevelSet,
           pybind11::arg("levelset"), pybind11::arg("wrapLowerLevelSet") = true,
           "Insert a level set to domain.")
      .def("insertNextLevelSetAsMaterial",
           &psDomain<T, D>::insertNextLevelSetAsMaterial)
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
      .def("setUseCellSet", &psDomain<T, D>::setUseCellSet)
      .def("getUseCellSet", &psDomain<T, D>::getUseCellSet)
      .def("print", &psDomain<T, D>::print)
      .def("printSurface", &psDomain<T, D>::printSurface,
           pybind11::arg("filename"), pybind11::arg("addMaterialIds") = false,
           "Print the surface of the domain.")
      .def("writeLevelSets", &psDomain<T, D>::writeLevelSets)
      .def("clear", &psDomain<T, D>::clear);

  // Enum psMaterial
  pybind11::enum_<psMaterial>(module, "Material")
      .value("Undefined", psMaterial::Undefined)
      .value("Mask", psMaterial::Mask)
      .value("Si", psMaterial::Si)
      .value("SiO2", psMaterial::SiO2)
      .value("Si3N4", psMaterial::Si3N4)
      .value("SiN", psMaterial::SiN)
      .value("SiON", psMaterial::SiON)
      .value("SiC", psMaterial::SiC)
      .value("PolySi", psMaterial::PolySi)
      .value("GaN", psMaterial::GaN)
      .value("W", psMaterial::W)
      .value("Al2O3", psMaterial::Al2O3)
      .value("TiN", psMaterial::TiN)
      .value("Cu", psMaterial::Cu)
      .value("Polymer", psMaterial::Polymer)
      .value("Dielectric", psMaterial::Dielectric)
      .value("Metal", psMaterial::Metal)
      .value("Air", psMaterial::Air)
      .value("GAS", psMaterial::GAS)
      .export_values();

  // psMaterialMap
  pybind11::class_<psMaterialMap, psSmartPointer<psMaterialMap>>(module,
                                                                 "MaterialMap")
      .def(pybind11::init<>())
      .def("insertNextMaterial", &psMaterialMap::insertNextMaterial,
           pybind11::arg("material") = psMaterial::Undefined)
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
      .def("getCellGrid", &csDenseCellSet<T, D>::getCellGrid,
           "Get the underlying mesh of the cell set.")
      .def("getDepth", &csDenseCellSet<T, D>::getDepth,
           "Get the depth of the cell set.")
      .def("getGridDelta", &csDenseCellSet<T, D>::getGridDelta,
           "Get the cell size.")
      .def("getNodes", &csDenseCellSet<T, D>::getNodes,
           "Get the nodes of the cell set which correspond to the corner "
           "points of the cells.")
      .def("getElements", &csDenseCellSet<T, D>::getElements,
           "Get elements (cells). The indicies in the elements correspond to "
           "the corner nodes.")
      .def("getSurface", &csDenseCellSet<T, D>::getSurface,
           "Get the surface level-set.")
      .def("getNumberOfCells", &csDenseCellSet<T, D>::getNumberOfCells)
      .def("getFillingFraction", &csDenseCellSet<T, D>::getFillingFraction,
           "Get the filling fraction of the cell containing the point.")
      .def("getScalarData", &csDenseCellSet<T, D>::getScalarData,
           "Get the data stored at each cell.")
      .def("setCellSetPosition", &csDenseCellSet<T, D>::setCellSetPosition,
           "Set whether the cell set should be created below (false) or above "
           "(true) the surface.")
      .def("getCellSetPosition", &csDenseCellSet<T, D>::getCellSetPosition)
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

  // Shim to instantiate the particle class
  pybind11::class_<psParticle<D>, psSmartPointer<psParticle<D>>> particle(
      module, "Particle");
  particle.def("surfaceCollision", &psParticle<D>::surfaceCollision)
      .def("surfaceReflection", &psParticle<D>::surfaceReflection)
      .def("initNew", &psParticle<D>::initNew)
      .def("getLocalDataLabels", &psParticle<D>::getLocalDataLabels)
      .def("getSourceDistributionPower",
           &psParticle<D>::getSourceDistributionPower);

  pybind11::class_<psDiffuseParticle<D>, psSmartPointer<psDiffuseParticle<D>>>(
      module, "DiffuseParticle", particle)
      .def(pybind11::init(
               &psSmartPointer<psDiffuseParticle<D>>::New<const T, const T,
                                                          const std::string &>),
           pybind11::arg("stickingProbability") = 1.0,
           pybind11::arg("cosineExponent") = 1.,
           pybind11::arg("dataLabel") = "flux")
      .def("surfaceCollision", &psDiffuseParticle<D>::surfaceCollision)
      .def("surfaceReflection", &psDiffuseParticle<D>::surfaceReflection)
      .def("initNew", &psDiffuseParticle<D>::initNew)
      .def("getLocalDataLabels", &psDiffuseParticle<D>::getLocalDataLabels)
      .def("getSourceDistributionPower",
           &psDiffuseParticle<D>::getSourceDistributionPower);

  pybind11::class_<psSpecularParticle, psSmartPointer<psSpecularParticle>>(
      module, "SpecularParticle", particle)
      .def(pybind11::init(
               &psSmartPointer<psSpecularParticle>::New<const T, const T,
                                                        const std::string &>),
           pybind11::arg("stickingProbability") = 1.0,
           pybind11::arg("cosineExponent") = 1.,
           pybind11::arg("dataLabel") = "flux")
      .def("surfaceCollision", &psSpecularParticle::surfaceCollision)
      .def("surfaceReflection", &psSpecularParticle::surfaceReflection)
      .def("initNew", &psSpecularParticle::initNew)
      .def("getLocalDataLabels", &psSpecularParticle::getLocalDataLabels)
      .def("getSourceDistributionPower",
           &psSpecularParticle::getSourceDistributionPower);

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
      .def("setVelocityField", [](psProcessModel<T, D> &pm,
                                  psSmartPointer<psVelocityField<T>> &vf) {
        pm.setVelocityField<psVelocityField<T>>(vf);
      });

  // psProcess
  pybind11::class_<psProcess<T, D>, psSmartPointer<psProcess<T, D>>>(module,
                                                                     "Process")
      // constructors
      .def(pybind11::init(&psSmartPointer<psProcess<T, D>>::New<>))

      // methods
      .def("setDomain", &psProcess<T, D>::setDomain, "Set the process domain.")
      .def("setProcessDuration", &psProcess<T, D>::setProcessDuration,
           "Set the process duration.")
      .def("setSourceDirection", &psProcess<T, D>::setSourceDirection,
           "Set source direction of the process.")
      .def("setNumberOfRaysPerPoint", &psProcess<T, D>::setNumberOfRaysPerPoint,
           "Set the number of rays to traced for each particle in the process. "
           "The number is per point in the process geometry")
      .def("setMaxCoverageInitIterations",
           &psProcess<T, D>::setMaxCoverageInitIterations,
           "Set the number of iterations to initialize the coverages.")
      .def("setPrintTimeInterval", &psProcess<T, D>::setPrintTimeInterval,
           "Sets the minimum time between printing intermediate results during "
           "the process. If this is set to a non-positive value, no "
           "intermediate results are printed.")
      .def("setProcessModel",
           &psProcess<T, D>::setProcessModel<psProcessModel<T, D>>,
           "Set the process model.")
      .def("apply", &psProcess<T, D>::apply, "Run the process.")
      .def("setIntegrationScheme", &psProcess<T, D>::setIntegrationScheme,
           "Set the integration scheme for solving the level-set equation. "
           "Should be used out of the ones specified in "
           "lsIntegrationSchemeEnum.")
      .def("setTimeStepRatio", &psProcess<T, D>::setTimeStepRatio,
           "Set the CFL condition to use during advection. The CFL condition "
           "sets the maximum distance a surface can be moved during one "
           "advection step. It MUST be below 0.5 to guarantee numerical "
           "stability. Defaults to 0.4999.");

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

  // enums
  pybind11::enum_<rayTraceDirection>(module, "rayTraceDirection")
      .value("POS_X", rayTraceDirection::POS_X)
      .value("POS_Y", rayTraceDirection::POS_Y)
      .value("POS_Z", rayTraceDirection::POS_Z)
      .value("NEG_X", rayTraceDirection::NEG_X)
      .value("NEG_Y", rayTraceDirection::NEG_Y)
      .value("NEG_Z", rayTraceDirection::NEG_Z);

  /****************************************************************************
   *                               GEOMETRIES                                 *
   ****************************************************************************/

  // constructors with custom enum need lambda to work: seems to be an issue
  // with implicit move constructor

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
           pybind11::arg("material") = psMaterial::Undefined)
      .def(pybind11::init(
               [](DomainType Domain, T Height, const psMaterial Material) {
                 return psSmartPointer<psMakePlane<T, D>>::New(Domain, Height,
                                                               Material);
               }),
           pybind11::arg("domain"), pybind11::arg("height") = 0.,
           pybind11::arg("material") = psMaterial::Undefined)
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
           pybind11::arg("material") = psMaterial::Undefined)
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
           pybind11::arg("material") = psMaterial::Undefined)
      .def("apply", &psMakeHole<T, D>::apply, "Create a hole geometry.");

  // psMakeFin
  pybind11::class_<psMakeFin<T, D>, psSmartPointer<psMakeFin<T, D>>>(module,
                                                                     "MakeFin")
      .def(pybind11::init([](DomainType Domain, const T gridDelta,
                             const T xExtent, const T yExtent, const T FinWidth,
                             const T FinHeight, const T BaseHeight,
                             const bool PeriodicBoundary, const bool MakeMask,
                             const psMaterial material) {
             return psSmartPointer<psMakeFin<T, D>>::New(
                 Domain, gridDelta, xExtent, yExtent, FinWidth, FinHeight,
                 BaseHeight, PeriodicBoundary, MakeMask, material);
           }),
           pybind11::arg("domain"), pybind11::arg("gridDelta"),
           pybind11::arg("xExtent"), pybind11::arg("yExtent"),
           pybind11::arg("finWidth"), pybind11::arg("finHeight"),
           pybind11::arg("baseHeight") = 0.,
           pybind11::arg("periodicBoundary") = false,
           pybind11::arg("makeMask") = false,
           pybind11::arg("material") = psMaterial::Undefined)
      .def("apply", &psMakeFin<T, D>::apply, "Create a fin geometry.");

  // psMakeStack
  pybind11::class_<psMakeStack<T, D>, psSmartPointer<psMakeStack<T, D>>>(
      module, "MakeStack")
      .def(pybind11::init(
               &psSmartPointer<psMakeStack<T, D>>::New<
                   DomainType &, const T /*gridDelta*/, const T /*xExtent*/,
                   const T /*yExtent*/, const int /*numLayers*/,
                   const T /*layerHeight*/, const T /*substrateHeight*/,
                   const T /*holeRadius*/, const T /*trenchWidth*/,
                   const T /*maskHeight*/, const bool /*PeriodicBoundary*/>),
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

  /****************************************************************************
   *                               MODELS                                     *
   ****************************************************************************/
  // Simple Deposition
  pybind11::class_<SimpleDeposition<T, D>,
                   psSmartPointer<SimpleDeposition<T, D>>>(
      module, "SimpleDeposition", processModel)
      .def(pybind11::init(
               &psSmartPointer<SimpleDeposition<T, D>>::New<const T, const T>),
           pybind11::arg("stickingProbability") = 0.1,
           pybind11::arg("sourceExponent") = 1.);

  // TEOS Deposition
  pybind11::class_<TEOSDeposition<T, D>, psSmartPointer<TEOSDeposition<T, D>>>(
      module, "TEOSDeposition", processModel)
      .def(pybind11::init(
               &psSmartPointer<TEOSDeposition<T, D>>::New<
                   const T /*st1*/, const T /*rate1*/, const T /*order1*/,
                   const T /*st2*/, const T /*rate2*/, const T /*order2*/>),
           pybind11::arg("stickingProbabilityP1"), pybind11::arg("rateP1"),
           pybind11::arg("orderP1"),
           pybind11::arg("stickingProbabilityP2") = 0.,
           pybind11::arg("rateP2") = 0., pybind11::arg("orderP2") = 0.);

  // SF6O2 Etching
  pybind11::class_<SF6O2Etching<T, D>, psSmartPointer<SF6O2Etching<T, D>>>(
      module, "SF6O2Etching", processModel)
      .def(pybind11::init(
               &psSmartPointer<SF6O2Etching<T, D>>::New<
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
  pybind11::class_<FluorocarbonEtching<T, D>,
                   psSmartPointer<FluorocarbonEtching<T, D>>>(
      module, "FluorocarbonEtching", processModel)
      .def(
          pybind11::init(&psSmartPointer<FluorocarbonEtching<T, D>>::New<
                         const double /*ionFlux*/, const double /*etchantFlux*/,
                         const double /*polyFlux*/, T /*meanEnergy*/,
                         const T /*sigmaEnergy*/, const T /*ionExponent*/,
                         const T /*deltaP*/, const T /*etchStopDepth*/>),
          pybind11::arg("ionFlux"), pybind11::arg("etchantFlux"),
          pybind11::arg("polyFlux"), pybind11::arg("meanIonEnergy") = 100.,
          pybind11::arg("sigmaIonEnergy") = 10.,
          pybind11::arg("ionExponent") = 100., pybind11::arg("deltaP") = 0.,
          pybind11::arg("etchStopDepth") = std::numeric_limits<T>::lowest());

  // Isotropic Process
  pybind11::class_<IsotropicProcess<T, D>,
                   psSmartPointer<IsotropicProcess<T, D>>>(
      module, "IsotropicProcess", processModel)
      .def(pybind11::init([](const T rate, const psMaterial mask) {
             return psSmartPointer<IsotropicProcess<T, D>>::New(rate, mask);
           }),
           pybind11::arg("isotropic rate"),
           pybind11::arg("mask material") = psMaterial::Mask);

  // Directional Etching
  pybind11::class_<DirectionalEtching<T, D>,
                   psSmartPointer<DirectionalEtching<T, D>>>(
      module, "DirectionalEtching", processModel)
      .def(pybind11::init([](const std::array<T, 3> &direction, const T dirVel,
                             const T isoVel, const psMaterial mask) {
             return psSmartPointer<DirectionalEtching<T, D>>::New(
                 direction, dirVel, isoVel, mask);
           }),
           pybind11::arg("direction"),
           pybind11::arg("directionalVelocity") = 1.,
           pybind11::arg("isotropicVelocity") = 0.,
           pybind11::arg("mask material") = psMaterial::Mask);

  // Sphere Distribution
  pybind11::class_<SphereDistribution<T, D>,
                   psSmartPointer<SphereDistribution<T, D>>>(
      module, "SphereDistribution", processModel)
      .def(pybind11::init([](const T radius, const T gridDelta,
                             psSmartPointer<lsDomain<T, D>> mask) {
             return psSmartPointer<SphereDistribution<T, D>>::New(
                 radius, gridDelta, mask);
           }),
           pybind11::arg("radius"), pybind11::arg("gridDelta"),
           pybind11::arg("mask"))
      .def(pybind11::init([](const T radius, const T gridDelta) {
             return psSmartPointer<SphereDistribution<T, D>>::New(
                 radius, gridDelta, nullptr);
           }),
           pybind11::arg("radius"), pybind11::arg("gridDelta"));

  // Box Distribution
  pybind11::class_<BoxDistribution<T, D>,
                   psSmartPointer<BoxDistribution<T, D>>>(
      module, "BoxDistribution", processModel)
      .def(
          pybind11::init([](const std::array<T, 3> &halfAxes, const T gridDelta,
                            psSmartPointer<lsDomain<T, D>> mask) {
            return psSmartPointer<BoxDistribution<T, D>>::New(halfAxes,
                                                              gridDelta, mask);
          }),
          pybind11::arg("halfAxes"), pybind11::arg("gridDelta"),
          pybind11::arg("mask"))
      .def(pybind11::init(
               [](const std::array<T, 3> &halfAxes, const T gridDelta) {
                 return psSmartPointer<BoxDistribution<T, D>>::New(
                     halfAxes, gridDelta, nullptr);
               }),
           pybind11::arg("halfAxes"), pybind11::arg("gridDelta"));

  // Plasma Damage
  pybind11::class_<PlasmaDamage<T, D>, psSmartPointer<PlasmaDamage<T, D>>>(
      module, "PlasmaDamage", processModel)
      .def(pybind11::init(
               &psSmartPointer<PlasmaDamage<T, D>>::New<const T, const T,
                                                        const int>),
           pybind11::arg("ionEnergy") = 100.,
           pybind11::arg("meanFreePath") = 1.,
           pybind11::arg("maskMaterial") = 0);

  // Oxide Regrowth
  pybind11::class_<OxideRegrowthModel<T, D>,
                   psSmartPointer<OxideRegrowthModel<T, D>>>(
      module, "OxideRegrowthModel", processModel)
      .def(
          pybind11::init(&psSmartPointer<OxideRegrowthModel<T, D>>::New<
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

  pybind11::class_<psPlanarize<T, D>, psSmartPointer<psPlanarize<T, D>>>(
      module, "Planarize")
      .def(pybind11::init(
               &psSmartPointer<psPlanarize<T, D>>::New<DomainType &, const T>),
           pybind11::arg("geometry"), pybind11::arg("cutoffHeight") = 0.)
      .def("apply", &psPlanarize<T, D>::apply, "Apply the planarization.");

#if VIENNAPS_PYTHON_DIMENSION > 2
  pybind11::class_<WetEtching<T, D>, psSmartPointer<WetEtching<T, D>>>(
      module, "WetEtching", processModel)
      .def(pybind11::init(&psSmartPointer<WetEtching<T, D>>::New<const int>),
           pybind11::arg("maskId") = 0);

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
      .def(pybind11::init<bool>())
      .def(pybind11::init(&psSmartPointer<psDomain<T, 3>>::New<>))
      // methods
      .def("insertNextLevelSet", &psDomain<T, 3>::insertNextLevelSet,
           pybind11::arg("levelset"), pybind11::arg("wrapLowerLevelSet") = true,
           "Insert a level set to domain.")
      .def("insertNextLevelSetAsMaterial",
           &psDomain<T, 3>::insertNextLevelSetAsMaterial)
      .def("duplicateTopLevelSet", &psDomain<T, 3>::duplicateTopLevelSet)
      .def("applyBooleanOperation", &psDomain<T, 3>::applyBooleanOperation)
      .def("removeTopLevelSet", &psDomain<T, 3>::removeTopLevelSet)
      .def("setMaterialMap", &psDomain<T, 3>::setMaterialMap)
      .def("getMaterialMap", &psDomain<T, 3>::getMaterialMap)
      .def("generateCellSet", &psDomain<T, 3>::generateCellSet,
           "Generate the cell set.")
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
      .def("setUseCellSet", &psDomain<T, 3>::setUseCellSet)
      .def("getUseCellSet", &psDomain<T, 3>::getUseCellSet)
      .def("print", &psDomain<T, 3>::print)
      .def("printSurface", &psDomain<T, 3>::printSurface,
           pybind11::arg("filename"), pybind11::arg("addMaterialIds") = true,
           "Print the surface of the domain.")
      .def("writeLevelSets", &psDomain<T, 3>::writeLevelSets)
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
}