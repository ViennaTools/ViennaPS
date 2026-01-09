#include "pyWrapDimension.hpp"

struct MaterialInfoPy {
  MaterialInfoPy(Material m)
      : name(info(m).name), category(info(m).category),
        density_gcm3(info(m).density_gcm3), conductive(info(m).conductive),
        colorHex(info(m).colorHex) {}
  MaterialInfoPy(MaterialInfo info)
      : name(info.name), category(info.category),
        density_gcm3(info.density_gcm3), conductive(info.conductive),
        colorHex(info.colorHex) {}

  std::string name;
  MaterialCategory category;
  double density_gcm3;
  bool conductive;
  uint32_t colorHex;
};

PYBIND11_MODULE(VIENNAPS_MODULE_NAME, module) {
  module.doc() =
      "ViennaPS is a header-only C++ process simulation library which "
      "includes surface and volume representations, a ray tracer, and physical "
      "models for the simulation of microelectronic fabrication processes. The "
      "main design goals are simplicity and efficiency, tailored towards "
      "scientific simulations.";

  // set version string of python module
  module.attr("__version__") = versionString();
  module.attr("version") = versionString();

  // wrap omp_set_num_threads to control number of threads
  module.def("setNumThreads", &omp_set_num_threads);

  module.def("gpuAvailable", &gpuAvailable,
             "Check if ViennaPS was compiled with GPU support.");

  // Logger
  py::class_<Logger, SmartPointer<Logger>>(module, "Logger", py::module_local())
      .def_static("setLogLevel", &Logger::setLogLevel)
      .def_static("getLogLevel", &Logger::getLogLevel)
      .def_static("setLogFile", &Logger::setLogFile)
      .def_static("appendToLogFile", &Logger::appendToLogFile)
      .def_static("closeLogFile", &Logger::closeLogFile)
      .def_static("getInstance", &Logger::getInstance,
                  py::return_value_policy::reference)
      .def("addDebug", &Logger::addDebug)
      .def("addTiming", (Logger & (Logger::*)(const std::string &, double)) &
                            Logger::addTiming)
      .def("addTiming",
           (Logger & (Logger::*)(const std::string &, double, double)) &
               Logger::addTiming)
      .def("addInfo", &Logger::addInfo)
      .def("addWarning", &Logger::addWarning)
      .def("addError", &Logger::addError, py::arg("s"),
           py::arg("shouldAbort") = true)
      .def("print", [](Logger &instance) { instance.print(std::cout); });

  // Material enum
  auto matEnum =
      py::native_enum<Material>(module, "Material", "enum.IntEnum",
                                "Material types for domain and level sets");
#define ENUM_BIND(id, sym, cat, dens, cond, color)                             \
  matEnum.value(#sym, Material::sym);
#define ENUM_BIND(id, sym, cat, dens, cond, color)                             \
  matEnum.value(#sym, Material::sym);
  MATERIAL_LIST(ENUM_BIND)
#undef ENUM_BIND
  matEnum.finalize();

  // Material category enum
  py::native_enum<MaterialCategory>(module, "MaterialCategory", "enum.IntEnum")
      .value("Generic", MaterialCategory::Generic)
      .value("Silicon", MaterialCategory::Silicon)
      .value("OxideNitride", MaterialCategory::OxideNitride)
      .value("Hardmask", MaterialCategory::Hardmask)
      .value("Metal", MaterialCategory::Metal)
      .value("Silicide", MaterialCategory::Silicide)
      .value("Compound", MaterialCategory::Compound)
      .value("TwoD", MaterialCategory::TwoD)
      .value("TCO", MaterialCategory::TCO)
      .value("Misc", MaterialCategory::Misc)
      .finalize();

  // MaterialInfo (immutable/read-only)
  py::class_<MaterialInfoPy>(module, "MaterialInfo")
      .def(py::init<Material>())
      .def_readonly("name", &MaterialInfoPy::name)
      .def_readonly("category", &MaterialInfoPy::category)
      .def_readonly("density_gcm3", &MaterialInfoPy::density_gcm3)
      .def_readonly("conductive", &MaterialInfoPy::conductive)
      .def_readonly("color_hex", &MaterialInfoPy::colorHex)
      // convenience: "#RRGGBB"
      .def_property_readonly("color_rgb", [](const MaterialInfoPy &x) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "#%06x",
                      (unsigned)(x.colorHex & 0xFFFFFFu));
        return std::string(buf);
      });

  // MaterialMap
  py::class_<MaterialMap, SmartPointer<MaterialMap>>(module, "MaterialMap")
      .def(py::init<>())
      .def("insertNextMaterial", &MaterialMap::insertNextMaterial,
           py::arg("material") = Material::Undefined)
      .def("getMaterialAtIdx", &MaterialMap::getMaterialAtIdx)
      .def("getMaterialMap", &MaterialMap::getMaterialMap)
      .def("size", &MaterialMap::size)
      .def_static("mapToMaterial", &MaterialMap::mapToMaterial<T>,
                  "Map a float to a material.")
      .def_static("isMaterial", &MaterialMap::isMaterial<T>)
      .def_static("toString",
                  py::overload_cast<const Material>(&MaterialMap::toString),
                  "Get the name of a material.");

  // Meta Data Enum
  py::native_enum<MetaDataLevel>(module, "MetaDataLevel", "enum.IntEnum")
      .value("NONE", MetaDataLevel::NONE)
      .value("GRID", MetaDataLevel::GRID)
      .value("PROCESS", MetaDataLevel::PROCESS)
      .value("FULL", MetaDataLevel::FULL)
      .finalize();

  // Render Mode Enum
  py::native_enum<RenderMode>(module, "RenderMode", "enum.IntEnum")
      .value("SURFACE", RenderMode::SURFACE)
      .value("INTERFACE", RenderMode::INTERFACE)
      .value("VOLUME", RenderMode::VOLUME)
      .finalize();

  // Render Mode Enum
  py::native_enum<RenderMode>(module, "RenderMode", "enum.IntEnum")
      .value("SURFACE", RenderMode::SURFACE)
      .value("INTERFACE", RenderMode::INTERFACE)
      .value("VOLUME", RenderMode::VOLUME)
      .finalize();

  // HoleShape Enum
  py::native_enum<HoleShape>(module, "HoleShape", "enum.IntEnum")
      .value("FULL", HoleShape::FULL)
      .value("HALF", HoleShape::HALF)
      .value("QUARTER", HoleShape::QUARTER)
      .finalize();

  /****************************************************************************
   *                               MODEL FRAMEWORK                            *
   ****************************************************************************/

  // Units
  // Length
  py::native_enum<decltype(units::Length::METER)>(module, "LengthUnit",
                                                  "enum.IntEnum")
      .value("METER", units::Length::METER)
      .value("CENTIMETER", units::Length::CENTIMETER)
      .value("MILLIMETER", units::Length::MILLIMETER)
      .value("MICROMETER", units::Length::MICROMETER)
      .value("NANOMETER", units::Length::NANOMETER)
      .value("ANGSTROM", units::Length::ANGSTROM)
      .value("UNDEFINED", units::Length::UNDEFINED)
      .finalize();

  py::class_<units::Length>(module, "Length")
      .def_static("setUnit", py::overload_cast<const std::string &>(
                                 &units::Length::setUnit))
      .def_static("getInstance", &units::Length::getInstance,
                  py::return_value_policy::reference)
      .def("convertMeter", &units::Length::convertMeter)
      .def("convertCentimeter", &units::Length::convertCentimeter)
      .def("convertMillimeter", &units::Length::convertMillimeter)
      .def("convertMicrometer", &units::Length::convertMicrometer)
      .def("convertNanometer", &units::Length::convertNanometer)
      .def("convertAngstrom", &units::Length::convertAngstrom)
      .def("toString", &units::Length::toString)
      .def("toShortString", &units::Length::toShortString);

  // Time
  py::native_enum<decltype(units::Time::MINUTE)>(module, "TimeUnit",
                                                 "enum.IntEnum")
      .value("MINUTE", units::Time::MINUTE)
      .value("SECOND", units::Time::SECOND)
      .value("MILLISECOND", units::Time::MILLISECOND)
      .value("UNDEFINED", units::Time::UNDEFINED)
      .finalize();

  py::class_<units::Time>(module, "Time")
      .def_static("setUnit",
                  py::overload_cast<const std::string &>(&units::Time::setUnit))
      .def_static("getInstance", &units::Time::getInstance,
                  py::return_value_policy::reference)
      .def("convertMinute", &units::Time::convertMinute)
      .def("convertSecond", &units::Time::convertSecond)
      .def("convertMillisecond", &units::Time::convertMillisecond)
      .def("toString", &units::Time::toString)
      .def("toShortString", &units::Time::toShortString);

  // ProcessParams
  py::class_<ProcessParams<T>, SmartPointer<ProcessParams<T>>>(module,
                                                               "ProcessParams")
      .def(py::init<>())
      .def("insertNextScalar", &ProcessParams<T>::insertNextScalar)
      .def("getScalarData",
           (T & (ProcessParams<T>::*)(int)) & ProcessParams<T>::getScalarData)
      .def("getScalarData", (const T &(ProcessParams<T>::*)(int) const) &
                                ProcessParams<T>::getScalarData)
      .def("getScalarData", (T & (ProcessParams<T>::*)(const std::string &)) &
                                ProcessParams<T>::getScalarData)
      .def("getScalarDataIndex", &ProcessParams<T>::getScalarDataIndex)
      .def("getScalarData", (std::vector<T> & (ProcessParams<T>::*)()) &
                                ProcessParams<T>::getScalarData)
      .def("getScalarData",
           (const std::vector<T> &(ProcessParams<T>::*)() const) &
               ProcessParams<T>::getScalarData)
      .def("getScalarDataLabel", &ProcessParams<T>::getScalarDataLabel);

  // Plasma Etching Parameters
  py::class_<PlasmaEtchingParameters<T>::MaskType>(
      module, "PlasmaEtchingParametersMask")
      .def(py::init<>())
      .def_readwrite("rho", &PlasmaEtchingParameters<T>::MaskType::rho)
      .def_readwrite("A_sp", &PlasmaEtchingParameters<T>::MaskType::A_sp)
      .def_readwrite("B_sp", &PlasmaEtchingParameters<T>::MaskType::B_sp)
      .def_readwrite("Eth_sp", &PlasmaEtchingParameters<T>::MaskType::Eth_sp);

  py::class_<PlasmaEtchingParameters<T>::PolymerType>(
      module, "PlasmaEtchingParametersPolymer")
      .def(py::init<>())
      .def_readwrite("rho", &PlasmaEtchingParameters<T>::PolymerType::rho)
      .def_readwrite("A_sp", &PlasmaEtchingParameters<T>::PolymerType::A_sp)
      .def_readwrite("B_sp", &PlasmaEtchingParameters<T>::PolymerType::B_sp)
      .def_readwrite("Eth_sp",
                     &PlasmaEtchingParameters<T>::PolymerType::Eth_sp);

  py::class_<PlasmaEtchingParameters<T>::MaterialType>(
      module, "PlasmaEtchingParametersSubstrate")
      .def(py::init<>())
      .def_readwrite("rho", &PlasmaEtchingParameters<T>::MaterialType::rho)
      .def_readwrite("k_sigma",
                     &PlasmaEtchingParameters<T>::MaterialType::k_sigma)
      .def_readwrite("beta_sigma",
                     &PlasmaEtchingParameters<T>::MaterialType::beta_sigma)
      .def_readwrite("Eth_sp",
                     &PlasmaEtchingParameters<T>::MaterialType::Eth_sp)
      .def_readwrite("A_sp", &PlasmaEtchingParameters<T>::MaterialType::A_sp)
      .def_readwrite("B_sp", &PlasmaEtchingParameters<T>::MaterialType::B_sp)
      //  .def_readwrite("theta_g_sp",
      //                 &PlasmaEtchingParameters<T>::MaterialType::theta_g_sp)
      .def_readwrite("Eth_ie",
                     &PlasmaEtchingParameters<T>::MaterialType::Eth_ie)
      .def_readwrite("A_ie", &PlasmaEtchingParameters<T>::MaterialType::A_ie)
      .def_readwrite("B_ie", &PlasmaEtchingParameters<T>::MaterialType::B_ie);
  //  .def_readwrite("theta_g_ie",
  //  &PlasmaEtchingParameters<T>::MaterialType::theta_g_ie);

  py::class_<PlasmaEtchingParameters<T>::PassivationType>(
      module, "PlasmaEtchingParametersPassivation")
      .def(py::init<>())
      .def_readwrite("Eth_ie",
                     &PlasmaEtchingParameters<T>::PassivationType::Eth_ie)
      .def_readwrite("A_ie",
                     &PlasmaEtchingParameters<T>::PassivationType::A_ie);

  py::class_<PlasmaEtchingParameters<T>::IonType>(module,
                                                  "PlasmaEtchingParametersIons")
      .def(py::init<>())
      .def_readwrite("meanEnergy",
                     &PlasmaEtchingParameters<T>::IonType::meanEnergy)
      .def_readwrite("sigmaEnergy",
                     &PlasmaEtchingParameters<T>::IonType::sigmaEnergy)
      .def_readwrite("exponent", &PlasmaEtchingParameters<T>::IonType::exponent)
      .def_readwrite("inflectAngle",
                     &PlasmaEtchingParameters<T>::IonType::inflectAngle)
      .def_readwrite("n_l", &PlasmaEtchingParameters<T>::IonType::n_l)
      .def_readwrite("minAngle", &PlasmaEtchingParameters<T>::IonType::minAngle)
      .def_readwrite("thetaRMin",
                     &PlasmaEtchingParameters<T>::IonType::thetaRMin)
      .def_readwrite("thetaRMax",
                     &PlasmaEtchingParameters<T>::IonType::thetaRMax);

  py::class_<PlasmaEtchingParameters<T>>(module, "PlasmaEtchingParameters")
      .def(py::init<>())
      .def_readwrite("ionFlux", &PlasmaEtchingParameters<T>::ionFlux)
      .def_readwrite("etchantFlux", &PlasmaEtchingParameters<T>::etchantFlux)
      .def_readwrite("passivationFlux",
                     &PlasmaEtchingParameters<T>::passivationFlux)
      .def_readwrite("etchStopDepth",
                     &PlasmaEtchingParameters<T>::etchStopDepth)
      .def_readwrite("beta_E", &PlasmaEtchingParameters<T>::beta_E)
      .def_readwrite("beta_P", &PlasmaEtchingParameters<T>::beta_P)
      .def_readwrite("Mask", &PlasmaEtchingParameters<T>::Mask)
      .def_readwrite("Substrate", &PlasmaEtchingParameters<T>::Substrate)
      .def_readwrite("Passivation", &PlasmaEtchingParameters<T>::Passivation)
      .def_readwrite("Ions", &PlasmaEtchingParameters<T>::Ions);

  // CF4O2 Parameters
  py::class_<CF4O2Parameters<T>::MaskType>(module, "CF4O2ParametersMask")
      .def(py::init<>())
      .def_readwrite("rho", &CF4O2Parameters<T>::MaskType::rho)
      .def_readwrite("A_sp", &CF4O2Parameters<T>::MaskType::A_sp)
      .def_readwrite("Eth_sp", &CF4O2Parameters<T>::MaskType::Eth_sp);

  py::class_<CF4O2Parameters<T>::SiType>(module, "CF4O2ParametersSi")
      .def(py::init<>())
      .def_readwrite("rho", &CF4O2Parameters<T>::SiType::rho)
      .def_readwrite("k_sigma", &CF4O2Parameters<T>::SiType::k_sigma)
      .def_readwrite("beta_sigma", &CF4O2Parameters<T>::SiType::beta_sigma)
      .def_readwrite("Eth_sp", &CF4O2Parameters<T>::SiType::Eth_sp)
      .def_readwrite("A_sp", &CF4O2Parameters<T>::SiType::A_sp)
      .def_readwrite("Eth_ie", &CF4O2Parameters<T>::SiType::Eth_ie)
      .def_readwrite("A_ie", &CF4O2Parameters<T>::SiType::A_ie);

  py::class_<CF4O2Parameters<T>::SiGeType>(module, "CF4O2ParametersSiGe")
      .def(py::init<>())
      .def_readwrite("x", &CF4O2Parameters<T>::SiGeType::x)
      .def_readwrite("rho", &CF4O2Parameters<T>::SiGeType::rho)
      .def_readwrite("k_sigma", &CF4O2Parameters<T>::SiGeType::k_sigma)
      .def_readwrite("beta_sigma", &CF4O2Parameters<T>::SiGeType::beta_sigma)
      .def_readwrite("Eth_sp", &CF4O2Parameters<T>::SiGeType::Eth_sp)
      .def_readwrite("A_sp", &CF4O2Parameters<T>::SiGeType::A_sp)
      .def_readwrite("Eth_ie", &CF4O2Parameters<T>::SiGeType::Eth_ie)
      .def_readwrite("A_ie", &CF4O2Parameters<T>::SiGeType::A_ie)
      .def("k_sigma_SiGe", &CF4O2Parameters<T>::SiGeType::k_sigma_SiGe);

  py::class_<CF4O2Parameters<T>::PassivationType>(module,
                                                  "CF4O2ParametersPassivation")
      .def(py::init<>())
      .def_readwrite("Eth_O_ie", &CF4O2Parameters<T>::PassivationType::Eth_O_ie)
      .def_readwrite("Eth_C_ie", &CF4O2Parameters<T>::PassivationType::Eth_C_ie)
      .def_readwrite("A_O_ie", &CF4O2Parameters<T>::PassivationType::A_O_ie)
      .def_readwrite("A_C_ie", &CF4O2Parameters<T>::PassivationType::A_C_ie);

  py::class_<CF4O2Parameters<T>::IonType>(module, "CF4O2ParametersIons")
      .def(py::init<>())
      .def_readwrite("meanEnergy", &CF4O2Parameters<T>::IonType::meanEnergy)
      .def_readwrite("sigmaEnergy", &CF4O2Parameters<T>::IonType::sigmaEnergy)
      .def_readwrite("exponent", &CF4O2Parameters<T>::IonType::exponent)
      .def_readwrite("inflectAngle", &CF4O2Parameters<T>::IonType::inflectAngle)
      .def_readwrite("n_l", &CF4O2Parameters<T>::IonType::n_l)
      .def_readwrite("minAngle", &CF4O2Parameters<T>::IonType::minAngle);

  py::class_<CF4O2Parameters<T>>(module, "CF4O2Parameters")
      .def(py::init<>())
      .def_readwrite("ionFlux", &CF4O2Parameters<T>::ionFlux)
      .def_readwrite("etchantFlux", &CF4O2Parameters<T>::etchantFlux)
      .def_readwrite("oxygenFlux", &CF4O2Parameters<T>::oxygenFlux)
      .def_readwrite("polymerFlux", &CF4O2Parameters<T>::polymerFlux)
      .def_readwrite("etchStopDepth", &CF4O2Parameters<T>::etchStopDepth)
      .def_readwrite("fluxIncludeSticking",
                     &CF4O2Parameters<T>::fluxIncludeSticking)
      .def_readwrite("gamma_F", &CF4O2Parameters<T>::gamma_F)
      .def_readwrite("gamma_F_oxidized", &CF4O2Parameters<T>::gamma_F_oxidized)
      .def_readwrite("gamma_O", &CF4O2Parameters<T>::gamma_O)
      .def_readwrite("gamma_O_passivated",
                     &CF4O2Parameters<T>::gamma_O_passivated)
      .def_readwrite("gamma_C", &CF4O2Parameters<T>::gamma_C)
      .def_readwrite("gamma_C_oxidized", &CF4O2Parameters<T>::gamma_C_oxidized)
      .def_readwrite("Mask", &CF4O2Parameters<T>::Mask)
      .def_readwrite("Si", &CF4O2Parameters<T>::Si)
      .def_readwrite("SiGe", &CF4O2Parameters<T>::SiGe)
      .def_readwrite("Passivation", &CF4O2Parameters<T>::Passivation)
      .def_readwrite("Ions", &CF4O2Parameters<T>::Ions);

  // Fluorocarbon Parameters
  py::class_<FluorocarbonParameters<T>::MaterialParameters>(
      module, "FluorocarbonMaterialParameters")
      .def(py::init<>())
      .def_readwrite("density",
                     &FluorocarbonParameters<T>::MaterialParameters::density)
      .def_readwrite("beta_p",
                     &FluorocarbonParameters<T>::MaterialParameters::beta_p)
      .def_readwrite("beta_e",
                     &FluorocarbonParameters<T>::MaterialParameters::beta_e)
      .def_readwrite("Eth_sp",
                     &FluorocarbonParameters<T>::MaterialParameters::Eth_sp)
      .def_readwrite("Eth_ie",
                     &FluorocarbonParameters<T>::MaterialParameters::Eth_ie)
      .def_readwrite("A_sp",
                     &FluorocarbonParameters<T>::MaterialParameters::A_sp)
      .def_readwrite("B_sp",
                     &FluorocarbonParameters<T>::MaterialParameters::B_sp)
      .def_readwrite("A_ie",
                     &FluorocarbonParameters<T>::MaterialParameters::A_ie)
      .def_readwrite("Eth_sp",
                     &FluorocarbonParameters<T>::MaterialParameters::Eth_sp)
      .def_readwrite("K", &FluorocarbonParameters<T>::MaterialParameters::K)
      .def_readwrite("E_a", &FluorocarbonParameters<T>::MaterialParameters::E_a)
      .def_readwrite("id", &FluorocarbonParameters<T>::MaterialParameters::id);

  py::class_<FluorocarbonParameters<T>::IonType>(module,
                                                 "FluorocarbonParametersIons")
      .def(py::init<>())
      .def_readwrite("meanEnergy",
                     &FluorocarbonParameters<T>::IonType::meanEnergy)
      .def_readwrite("sigmaEnergy",
                     &FluorocarbonParameters<T>::IonType::sigmaEnergy)
      .def_readwrite("exponent", &FluorocarbonParameters<T>::IonType::exponent)
      .def_readwrite("inflectAngle",
                     &FluorocarbonParameters<T>::IonType::inflectAngle)
      .def_readwrite("n_l", &FluorocarbonParameters<T>::IonType::n_l)
      .def_readwrite("minAngle", &FluorocarbonParameters<T>::IonType::minAngle);

  py::class_<FluorocarbonParameters<T>>(module, "FluorocarbonParameters")
      .def(py::init<>())
      .def("addMaterial", &FluorocarbonParameters<T>::addMaterial,
           py::arg("materialParameters"))
      .def("getMaterialParameters",
           &FluorocarbonParameters<T>::getMaterialParameters,
           py::arg("material"))
      .def_readwrite("ionFlux", &FluorocarbonParameters<T>::ionFlux)
      .def_readwrite("etchantFlux", &FluorocarbonParameters<T>::etchantFlux)
      .def_readwrite("polyFlux", &FluorocarbonParameters<T>::polyFlux)
      .def_readwrite("delta_p", &FluorocarbonParameters<T>::delta_p)
      .def_readwrite("etchStopDepth", &FluorocarbonParameters<T>::etchStopDepth)
      .def_readwrite("temperature", &FluorocarbonParameters<T>::temperature)
      .def_readwrite("k_ie", &FluorocarbonParameters<T>::k_ie)
      .def_readwrite("k_ev", &FluorocarbonParameters<T>::k_ev)
      .def_readwrite("Ions", &FluorocarbonParameters<T>::Ions);

  py::class_<IBEParameters<T>::cos4YieldType>(module, "IBEParametersCos4Yield")
      .def(py::init<>())
      .def_readwrite("a1", &IBEParameters<T>::cos4YieldType::a1)
      .def_readwrite("a2", &IBEParameters<T>::cos4YieldType::a2)
      .def_readwrite("a3", &IBEParameters<T>::cos4YieldType::a3)
      .def_readwrite("a4", &IBEParameters<T>::cos4YieldType::a4)
      .def_readwrite("isDefined", &IBEParameters<T>::cos4YieldType::isDefined)
      .def("aSum", &IBEParameters<T>::cos4YieldType::aSum);

  // Ion Beam Etching Parameters
  py::class_<IBEParameters<T>>(module, "IBEParameters")
      .def(py::init<>())
      .def_readwrite("planeWaferRate", &IBEParameters<T>::planeWaferRate)
      .def_readwrite("materialPlaneWaferRate",
                     &IBEParameters<T>::materialPlaneWaferRate)
      .def_readwrite("meanEnergy", &IBEParameters<T>::meanEnergy)
      .def_readwrite("sigmaEnergy", &IBEParameters<T>::sigmaEnergy)
      .def_readwrite("thresholdEnergy", &IBEParameters<T>::thresholdEnergy)
      .def_readwrite("n_l", &IBEParameters<T>::n_l)
      .def_readwrite("inflectAngle", &IBEParameters<T>::inflectAngle)
      .def_readwrite("minAngle", &IBEParameters<T>::minAngle)
      .def_readwrite("tiltAngle", &IBEParameters<T>::tiltAngle)
      .def_readwrite("exponent", &IBEParameters<T>::exponent)
      //   .def_readwrite("yieldFunction", &IBEParameters<T>::yieldFunction) //
      //   problem with GIL
      .def_readwrite("cos4Yield", &IBEParameters<T>::cos4Yield)
      .def_readwrite("thetaRMin", &IBEParameters<T>::thetaRMin)
      .def_readwrite("thetaRMax", &IBEParameters<T>::thetaRMax)
      .def_readwrite("redepositionThreshold",
                     &IBEParameters<T>::redepositionThreshold)
      .def_readwrite("redepositionRate", &IBEParameters<T>::redepositionRate)
      .def("toProcessMetaData", &IBEParameters<T>::toProcessMetaData,
           "Convert the IBE parameters to a metadata dict.");

  // Faraday Cage Etching
  py::class_<FaradayCageParameters<T>>(module, "FaradayCageParameters")
      .def(py::init<>())
      .def_readwrite("ibeParams", &FaradayCageParameters<T>::ibeParams)
      .def_readwrite("cageAngle", &FaradayCageParameters<T>::cageAngle);

  // Expose RateSet struct to Python
  py::class_<impl::RateSet<T>>(module, "RateSet")
      .def(py::init<const Vec3D<T> &, T, T, const std::vector<Material> &,
                    bool>(),
           py::arg("direction") = std::array<T, 3>{0., 0., 0.},
           py::arg("directionalVelocity") = 0.,
           py::arg("isotropicVelocity") = 0.,
           py::arg("maskMaterials") = std::vector<Material>{Material::Mask},
           py::arg("calculateVisibility") = true)
      .def_readwrite("direction", &impl::RateSet<T>::direction)
      .def_readwrite("directionalVelocity",
                     &impl::RateSet<T>::directionalVelocity)
      .def_readwrite("isotropicVelocity", &impl::RateSet<T>::isotropicVelocity)
      .def_readwrite("maskMaterials", &impl::RateSet<T>::maskMaterials)
      .def_readwrite("calculateVisibility",
                     &impl::RateSet<T>::calculateVisibility)
      .def("print", &impl::RateSet<T>::print);

  // ***************************************************************************
  //                                 PROCESS
  // ***************************************************************************

  // Normalization Enum
  py::native_enum<viennaray::NormalizationType>(module, "NormalizationType",
                                                "enum.IntEnum")
      .value("SOURCE", viennaray::NormalizationType::SOURCE)
      .value("MAX", viennaray::NormalizationType::MAX)
      .finalize();

  // Flux Engine Type Enum
  py::native_enum<FluxEngineType>(module, "FluxEngineType", "enum.IntEnum")
      .value("AUTO", FluxEngineType::AUTO)
      .value("CPU_DISK", FluxEngineType::CPU_DISK)
      .value("CPU_TRIANGLE", FluxEngineType::CPU_TRIANGLE)
      .value("GPU_DISK", FluxEngineType::GPU_DISK)
      .value("GPU_TRIANGLE", FluxEngineType::GPU_TRIANGLE)
      .value("GPU_LINE", FluxEngineType::GPU_LINE)
      .finalize();

  // RayTracingParameters
  py::class_<RayTracingParameters>(module, "RayTracingParameters")
      .def(py::init<>())
      .def_readwrite("normalizationType",
                     &RayTracingParameters::normalizationType)
      .def_readwrite("raysPerPoint", &RayTracingParameters::raysPerPoint)
      .def_readwrite("diskRadius", &RayTracingParameters::diskRadius)
      .def_readwrite("useRandomSeeds", &RayTracingParameters::useRandomSeeds)
      .def_readwrite("rngSeed", &RayTracingParameters::rngSeed)
      .def_readwrite("ignoreFluxBoundaries",
                     &RayTracingParameters::ignoreFluxBoundaries)
      .def_readwrite("smoothingNeighbors",
                     &RayTracingParameters::smoothingNeighbors)
      .def_readwrite("minNodeDistanceFactor",
                     &RayTracingParameters::minNodeDistanceFactor)
      .def_readwrite("maxReflections", &RayTracingParameters::maxReflections)
      .def_readwrite("maxBoundaryHits", &RayTracingParameters::maxBoundaryHits)
      .def("toMetaData", &RayTracingParameters::toMetaData,
           "Convert the ray tracing parameters to a metadata dict.")
      .def("toMetaDataString", &RayTracingParameters::toMetaDataString,
           "Convert the ray tracing parameters to a metadata string.");

  // AdvectionParameters
  py::class_<AdvectionParameters>(module, "AdvectionParameters")
      .def(py::init<>())
      .def_readwrite("spatialScheme", &AdvectionParameters::spatialScheme)
      // integrationScheme is depreciated
      .def_property(
          "integrationScheme",
          [](AdvectionParameters &self) {
            VIENNACORE_LOG_WARNING("The parameter 'integrationScheme' is "
                                   "deprecated and will be removed in a future "
                                   "release. Please use 'spatialScheme' "
                                   "instead.");
            return self.spatialScheme;
          },
          [](AdvectionParameters &self, viennals::SpatialSchemeEnum scheme) {
            VIENNACORE_LOG_WARNING("The parameter 'integrationScheme' is "
                                   "deprecated and will be removed in a future "
                                   "release. Please use 'spatialScheme' "
                                   "instead.");
            self.spatialScheme = scheme;
          })
      .def_readwrite("temporalScheme", &AdvectionParameters::temporalScheme)
      .def_readwrite("timeStepRatio", &AdvectionParameters::timeStepRatio)
      .def_readwrite("dissipationAlpha", &AdvectionParameters::dissipationAlpha)
      .def_readwrite("checkDissipation", &AdvectionParameters::checkDissipation)
      .def_readwrite("velocityOutput", &AdvectionParameters::velocityOutput)
      .def_readwrite("ignoreVoids", &AdvectionParameters::ignoreVoids)
      .def_readwrite("adaptiveTimeStepping",
                     &AdvectionParameters::adaptiveTimeStepping)
      .def_readwrite("adaptiveTimeStepSubdivisions",
                     &AdvectionParameters::adaptiveTimeStepSubdivisions)
      .def_readwrite("calculateIntermediateVelocities",
                     &AdvectionParameters::calculateIntermediateVelocities)
      .def("toMetaData", &AdvectionParameters::toMetaData,
           "Convert the advection parameters to a metadata dict.")
      .def("toMetaDataString", &AdvectionParameters::toMetaDataString,
           "Convert the advection parameters to a metadata string.");

  // CoverageParameters
  py::class_<CoverageParameters>(module, "CoverageParameters")
      .def(py::init<>())
      .def_readwrite("tolerance", &CoverageParameters::tolerance)
      .def_readwrite("maxIterations", &CoverageParameters::maxIterations)
      .def("toMetaData", &CoverageParameters::toMetaData,
           "Convert the coverage parameters to a metadata dict.")
      .def("toMetaDataString", &CoverageParameters::toMetaDataString,
           "Convert the coverage parameters to a metadata string.");

  // AtomicLayerProcessParameters
  py::class_<AtomicLayerProcessParameters>(module,
                                           "AtomicLayerProcessParameters")
      .def(py::init<>())
      .def_readwrite("numCycles", &AtomicLayerProcessParameters::numCycles)
      .def_readwrite("pulseTime", &AtomicLayerProcessParameters::pulseTime)
      .def_readwrite("coverageTimeStep",
                     &AtomicLayerProcessParameters::coverageTimeStep)
      .def_readwrite("purgePulseTime",
                     &AtomicLayerProcessParameters::purgePulseTime)
      .def("toMetaData", &AtomicLayerProcessParameters::toMetaData,
           "Convert the ALD process parameters to a metadata dict.")
      .def("toMetaDataString", &AtomicLayerProcessParameters::toMetaDataString,
           "Convert the ALD process parameters to a metadata string.");

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

  // Utility functions
  auto m_util = module.def_submodule("util", "Utility functions.");
  m_util.def("convertSpatialScheme", &util::convertSpatialScheme,
             "Convert a string to an discretization scheme.");
  // convertIntegrationScheme is deprecated
  m_util.attr("convertIntegrationScheme") = m_util.attr("convertSpatialScheme");
  m_util.def("convertFluxEngineType", &util::convertFluxEngineType,
             "Convert a string to a flux engine type.");
  m_util.def("convertTemporalScheme", &util::convertTemporalScheme,
             "Convert a string to a time integration scheme.");

  //   ***************************************************************************
  //                                  MAIN API
  //   ***************************************************************************
  // Submodule for 2D
  auto m2 = module.def_submodule("d2", "2D bindings");
  m2.attr("__name__") = "viennaps.d2";
  m2.attr("__package__") = "viennaps";
  bindApi<2>(m2);

  // Submodule for 3D
  auto m3 = module.def_submodule("d3", "3D bindings");
  m3.attr("__name__") = "viennaps.d3";
  m3.attr("__package__") = "viennaps";
  bindApi<3>(m3);

  // Extrude domain (2D to 3D)
  py::class_<Extrude<T>>(module, "Extrude")
      .def(py::init())
      .def(py::init<SmartPointer<Domain<T, 2>> &, SmartPointer<Domain<T, 3>> &,
                    std::array<T, 2>, int, std::array<BoundaryType, 3>>(),
           py::arg("inputDomain"), py::arg("outputDomain"), py::arg("extent"),
           py::arg("extrusionAxis"), py::arg("boundaryConditions"))
      .def("setInputDomain", &Extrude<T>::setInputDomain,
           "Set the input domain to be extruded.")
      .def("setOutputDomain", &Extrude<T>::setOutputDomain,
           "Set the output domain. The 3D output domain will be overwritten by "
           "the extruded domain.")
      .def("setExtent", &Extrude<T>::setExtent,
           "Set the min and max extent in the extruded dimension.")
      .def("setExtrusionAxis", &Extrude<T>::setExtrusionAxis,
           "Set the axis along which to extrude (0, 1, or 2).")
      .def("setBoundaryConditions",
           py::overload_cast<std::array<BoundaryType, 3>>(
               &Extrude<T>::setBoundaryConditions),
           "Set the boundary conditions in the extruded domain.")
      .def("apply", &Extrude<T>::apply, "Run the extrusion.");

  // Slice domain (3D to 2D)
  py::class_<Slice<T>>(module, "Slice")
      .def(py::init())
      .def(py::init<SmartPointer<Domain<T, 3>> &, SmartPointer<Domain<T, 2>> &,
                    int, T>(),
           py::arg("inputDomain"), py::arg("outputDomain"),
           py::arg("sliceDimension"), py::arg("slicePosition"))
      .def("setInputDomain", &Slice<T>::setInputDomain,
           "Set the input domain to be sliced.")
      .def("setOutputDomain", &Slice<T>::setOutputDomain,
           "Set the output domain. The 2D output domain will be overwritten by "
           "the sliced domain.")
      .def("setSliceDimension", &Slice<T>::setSliceDimension,
           "Set the dimension along which to slice (0, 1).")
      .def("setSlicePosition", &Slice<T>::setSlicePosition,
           "Set the position along the slice dimension at which to slice.")
      .def("setReflectX", &Slice<T>::setReflectX,
           "Set whether to reflect the slice along the X axis.")
      .def("apply", &Slice<T>::apply, "Run the slicing.");

  // ***************************************************************************
  //                                 GPU SUPPORT
  // ***************************************************************************

#ifdef VIENNACORE_COMPILE_GPU
  auto m_gpu = module.def_submodule("gpu", "GPU support functions.");

  py::class_<std::filesystem::path>(m_gpu, "Path").def(py::init<std::string>());
  py::implicitly_convertible<std::string, std::filesystem::path>();

  py::class_<DeviceContext, std::shared_ptr<DeviceContext>>(m_gpu, "Context")
      .def(py::init())
      //  .def_readwrite("modulePath", &Context::modulePath)
      //  .def_readwrite("moduleNames", &Context::moduleNames)
      //  .def_readwrite("cuda", &Context::cuda, "Cuda context.")
      //  .def_readwrite("optix", &Context::optix, "Optix context.")
      //  .def_readwrite("deviceProps", &Context::deviceProps,
      //                 "Device properties.")
      //  .def("getModule", &Context::getModule)
      .def_static("createContext", &DeviceContext::createContext,
                  "Create a new context.",
                  py::arg("modulePath") = VIENNACORE_KERNELS_PATH,
                  py::arg("deviceID") = 0, py::arg("registerInGlobal") = true)
      .def_static("getContextFromRegistry",
                  &DeviceContext::getContextFromRegistry,
                  "Get a context from the global registry by device ID.",
                  py::arg("deviceID") = 0)
      .def_static(
          "hasContextInRegistry", &DeviceContext::hasContextInRegistry,
          "Check if a context exists in the global registry by device ID.",
          py::arg("deviceID") = 0)
      .def_static("getRegisteredDeviceIDs",
                  &DeviceContext::getRegisteredDeviceIDs,
                  "Get a list of all device IDs with registered contexts.")
      .def("create", &DeviceContext::create, "Create a new context.",
           py::arg("modulePath") = VIENNACORE_KERNELS_PATH,
           py::arg("deviceID") = 0)
      .def("destroy", &DeviceContext::destroy, "Destroy the context.")
      .def("addModule", &DeviceContext::addModule,
           "Add a module to the context.")
      .def("getModulePath", &DeviceContext::getModulePath,
           "Get the module path.")
      .def_readwrite("deviceID", &DeviceContext::deviceID, "Device ID.");
#endif
}
