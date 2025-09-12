#include "pyWrapDimension.hpp"

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

  // Logger
  pybind11::class_<Logger, SmartPointer<Logger>>(module, "Logger",
                                                 pybind11::module_local())
      .def_static("setLogLevel", &Logger::setLogLevel)
      .def_static("getLogLevel", &Logger::getLogLevel)
      .def_static("setLogFile", &Logger::setLogFile)
      .def_static("appendToLogFile", &Logger::appendToLogFile)
      .def_static("closeLogFile", &Logger::closeLogFile)
      .def_static("getInstance", &Logger::getInstance,
                  pybind11::return_value_policy::reference)
      .def("addDebug", &Logger::addDebug)
      .def("addTiming", (Logger & (Logger::*)(const std::string &, double)) &
                            Logger::addTiming)
      .def("addTiming",
           (Logger & (Logger::*)(const std::string &, double, double)) &
               Logger::addTiming)
      .def("addInfo", &Logger::addInfo)
      .def("addWarning", &Logger::addWarning)
      .def("addError", &Logger::addError, pybind11::arg("s"),
           pybind11::arg("shouldAbort") = true)
      .def("print", [](Logger &instance) { instance.print(std::cout); });

  // Material enum
  pybind11::native_enum<Material>(module, "Material", "enum.IntEnum",
                                  "Material types for domain and level sets.")
      .value("Undefined", Material::Undefined) // -1
      .value("Mask", Material::Mask)           // 0
      .value("Si", Material::Si)
      .value("SiO2", Material::SiO2)
      .value("Si3N4", Material::Si3N4) // 3
      .value("SiN", Material::SiN)
      .value("SiON", Material::SiON)
      .value("SiC", Material::SiC)
      .value("SiGe", Material::SiGe)
      .value("PolySi", Material::PolySi) // 8
      .value("GaN", Material::GaN)
      .value("W", Material::W)
      .value("Al2O3", Material::Al2O3)
      .value("HfO2", Material::HfO2)
      .value("TiN", Material::TiN) // 13
      .value("Cu", Material::Cu)
      .value("Polymer", Material::Polymer)
      .value("Dielectric", Material::Dielectric)
      .value("Metal", Material::Metal)
      .value("Air", Material::Air) // 18
      .value("GAS", Material::GAS)
      .finalize();

  // MaterialMap
  pybind11::class_<MaterialMap, SmartPointer<MaterialMap>>(module,
                                                           "MaterialMap")
      .def(pybind11::init<>())
      .def("insertNextMaterial", &MaterialMap::insertNextMaterial,
           pybind11::arg("material") = Material::Undefined)
      .def("getMaterialAtIdx", &MaterialMap::getMaterialAtIdx)
      .def("getMaterialMap", &MaterialMap::getMaterialMap)
      .def("size", &MaterialMap::size)
      .def_static("mapToMaterial", &MaterialMap::mapToMaterial<T>,
                  "Map a float to a material.")
      .def_static("isMaterial", &MaterialMap::isMaterial<T>)
      .def_static("getMaterialName", &MaterialMap::getMaterialName<Material>,
                  "Get the name of a material.");

  // Meta Data Enum
  pybind11::native_enum<MetaDataLevel>(module, "MetaDataLevel", "enum.IntEnum")
      .value("NONE", MetaDataLevel::NONE)
      .value("GRID", MetaDataLevel::GRID)
      .value("PROCESS", MetaDataLevel::PROCESS)
      .value("FULL", MetaDataLevel::FULL)
      .finalize();

  // Hole
  pybind11::native_enum<HoleShape>(module, "HoleShape", "enum.IntEnum")
      .value("FULL", HoleShape::FULL)
      .value("HALF", HoleShape::HALF)
      .value("QUARTER", HoleShape::QUARTER)
      .finalize();

  /****************************************************************************
   *                               MODEL FRAMEWORK                            *
   ****************************************************************************/

  // Units
  // Length
  pybind11::native_enum<decltype(units::Length::METER)>(module, "LengthUnit",
                                                        "enum.IntEnum")
      .value("METER", units::Length::METER)
      .value("CENTIMETER", units::Length::CENTIMETER)
      .value("MILLIMETER", units::Length::MILLIMETER)
      .value("MICROMETER", units::Length::MICROMETER)
      .value("NANOMETER", units::Length::NANOMETER)
      .value("ANGSTROM", units::Length::ANGSTROM)
      .value("UNDEFINED", units::Length::UNDEFINED)
      .finalize();

  pybind11::class_<units::Length>(module, "Length")
      .def_static("setUnit", pybind11::overload_cast<const std::string &>(
                                 &units::Length::setUnit))
      .def_static("getInstance", &units::Length::getInstance,
                  pybind11::return_value_policy::reference)
      .def("convertMeter", &units::Length::convertMeter)
      .def("convertCentimeter", &units::Length::convertCentimeter)
      .def("convertMillimeter", &units::Length::convertMillimeter)
      .def("convertMicrometer", &units::Length::convertMicrometer)
      .def("convertNanometer", &units::Length::convertNanometer)
      .def("convertAngstrom", &units::Length::convertAngstrom)
      .def("toString", &units::Length::toString)
      .def("toShortString", &units::Length::toShortString);

  // Time
  pybind11::native_enum<decltype(units::Time::MINUTE)>(module, "TimeUnit",
                                                       "enum.IntEnum")
      .value("MINUTE", units::Time::MINUTE)
      .value("SECOND", units::Time::SECOND)
      .value("MILLISECOND", units::Time::MILLISECOND)
      .value("UNDEFINED", units::Time::UNDEFINED)
      .finalize();

  pybind11::class_<units::Time>(module, "Time")
      .def_static("setUnit", pybind11::overload_cast<const std::string &>(
                                 &units::Time::setUnit))
      .def_static("getInstance", &units::Time::getInstance,
                  pybind11::return_value_policy::reference)
      .def("convertMinute", &units::Time::convertMinute)
      .def("convertSecond", &units::Time::convertSecond)
      .def("convertMillisecond", &units::Time::convertMillisecond)
      .def("toString", &units::Time::toString)
      .def("toShortString", &units::Time::toShortString);

  // ProcessParams
  pybind11::class_<ProcessParams<T>, SmartPointer<ProcessParams<T>>>(
      module, "ProcessParams")
      .def(pybind11::init<>())
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
  pybind11::class_<PlasmaEtchingParameters<T>::MaskType>(
      module, "PlasmaEtchingParametersMask")
      .def(pybind11::init<>())
      .def_readwrite("rho", &PlasmaEtchingParameters<T>::MaskType::rho)
      .def_readwrite("A_sp", &PlasmaEtchingParameters<T>::MaskType::A_sp)
      .def_readwrite("B_sp", &PlasmaEtchingParameters<T>::MaskType::B_sp)
      .def_readwrite("Eth_sp", &PlasmaEtchingParameters<T>::MaskType::Eth_sp);

  pybind11::class_<PlasmaEtchingParameters<T>::PolymerType>(
      module, "PlasmaEtchingParametersPolymer")
      .def(pybind11::init<>())
      .def_readwrite("rho", &PlasmaEtchingParameters<T>::PolymerType::rho)
      .def_readwrite("A_sp", &PlasmaEtchingParameters<T>::PolymerType::A_sp)
      .def_readwrite("B_sp", &PlasmaEtchingParameters<T>::PolymerType::B_sp)
      .def_readwrite("Eth_sp",
                     &PlasmaEtchingParameters<T>::PolymerType::Eth_sp);

  pybind11::class_<PlasmaEtchingParameters<T>::MaterialType>(
      module, "PlasmaEtchingParametersSubstrate")
      .def(pybind11::init<>())
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

  pybind11::class_<PlasmaEtchingParameters<T>::PassivationType>(
      module, "PlasmaEtchingParametersPassivation")
      .def(pybind11::init<>())
      .def_readwrite("Eth_ie",
                     &PlasmaEtchingParameters<T>::PassivationType::Eth_ie)
      .def_readwrite("A_ie",
                     &PlasmaEtchingParameters<T>::PassivationType::A_ie);

  pybind11::class_<PlasmaEtchingParameters<T>::IonType>(
      module, "PlasmaEtchingParametersIons")
      .def(pybind11::init<>())
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

  pybind11::class_<PlasmaEtchingParameters<T>>(module,
                                               "PlasmaEtchingParameters")
      .def(pybind11::init<>())
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
  pybind11::class_<CF4O2Parameters<T>::MaskType>(module, "CF4O2ParametersMask")
      .def(pybind11::init<>())
      .def_readwrite("rho", &CF4O2Parameters<T>::MaskType::rho)
      .def_readwrite("A_sp", &CF4O2Parameters<T>::MaskType::A_sp)
      .def_readwrite("Eth_sp", &CF4O2Parameters<T>::MaskType::Eth_sp);

  pybind11::class_<CF4O2Parameters<T>::SiType>(module, "CF4O2ParametersSi")
      .def(pybind11::init<>())
      .def_readwrite("rho", &CF4O2Parameters<T>::SiType::rho)
      .def_readwrite("k_sigma", &CF4O2Parameters<T>::SiType::k_sigma)
      .def_readwrite("beta_sigma", &CF4O2Parameters<T>::SiType::beta_sigma)
      .def_readwrite("Eth_sp", &CF4O2Parameters<T>::SiType::Eth_sp)
      .def_readwrite("A_sp", &CF4O2Parameters<T>::SiType::A_sp)
      .def_readwrite("Eth_ie", &CF4O2Parameters<T>::SiType::Eth_ie)
      .def_readwrite("A_ie", &CF4O2Parameters<T>::SiType::A_ie);

  pybind11::class_<CF4O2Parameters<T>::SiGeType>(module, "CF4O2ParametersSiGe")
      .def(pybind11::init<>())
      .def_readwrite("x", &CF4O2Parameters<T>::SiGeType::x)
      .def_readwrite("rho", &CF4O2Parameters<T>::SiGeType::rho)
      .def_readwrite("k_sigma", &CF4O2Parameters<T>::SiGeType::k_sigma)
      .def_readwrite("beta_sigma", &CF4O2Parameters<T>::SiGeType::beta_sigma)
      .def_readwrite("Eth_sp", &CF4O2Parameters<T>::SiGeType::Eth_sp)
      .def_readwrite("A_sp", &CF4O2Parameters<T>::SiGeType::A_sp)
      .def_readwrite("Eth_ie", &CF4O2Parameters<T>::SiGeType::Eth_ie)
      .def_readwrite("A_ie", &CF4O2Parameters<T>::SiGeType::A_ie)
      .def("k_sigma_SiGe", &CF4O2Parameters<T>::SiGeType::k_sigma_SiGe);

  pybind11::class_<CF4O2Parameters<T>::PassivationType>(
      module, "CF4O2ParametersPassivation")
      .def(pybind11::init<>())
      .def_readwrite("Eth_O_ie", &CF4O2Parameters<T>::PassivationType::Eth_O_ie)
      .def_readwrite("Eth_C_ie", &CF4O2Parameters<T>::PassivationType::Eth_C_ie)
      .def_readwrite("A_O_ie", &CF4O2Parameters<T>::PassivationType::A_O_ie)
      .def_readwrite("A_C_ie", &CF4O2Parameters<T>::PassivationType::A_C_ie);

  pybind11::class_<CF4O2Parameters<T>::IonType>(module, "CF4O2ParametersIons")
      .def(pybind11::init<>())
      .def_readwrite("meanEnergy", &CF4O2Parameters<T>::IonType::meanEnergy)
      .def_readwrite("sigmaEnergy", &CF4O2Parameters<T>::IonType::sigmaEnergy)
      .def_readwrite("exponent", &CF4O2Parameters<T>::IonType::exponent)
      .def_readwrite("inflectAngle", &CF4O2Parameters<T>::IonType::inflectAngle)
      .def_readwrite("n_l", &CF4O2Parameters<T>::IonType::n_l)
      .def_readwrite("minAngle", &CF4O2Parameters<T>::IonType::minAngle);

  pybind11::class_<CF4O2Parameters<T>>(module, "CF4O2Parameters")
      .def(pybind11::init<>())
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

  // Ion Beam Etching Parameters
  pybind11::class_<IBEParameters<T>>(module, "IBEParameters")
      .def(pybind11::init<>())
      .def_readwrite("planeWaferRate", &IBEParameters<T>::planeWaferRate)
      .def_readwrite("materialPlaneWaferRate",
                     &IBEParameters<T>::materialPlaneWaferRate)
      .def_readwrite("meanEnergy", &IBEParameters<T>::meanEnergy)
      .def_readwrite("sigmaEnergy", &IBEParameters<T>::sigmaEnergy)
      .def_readwrite("thresholdEnergy", &IBEParameters<T>::thresholdEnergy)
      .def_readwrite("exponent", &IBEParameters<T>::exponent)
      .def_readwrite("n_l", &IBEParameters<T>::n_l)
      .def_readwrite("inflectAngle", &IBEParameters<T>::inflectAngle)
      .def_readwrite("minAngle", &IBEParameters<T>::minAngle)
      .def_readwrite("tiltAngle", &IBEParameters<T>::tiltAngle)
      .def_readwrite("yieldFunction", &IBEParameters<T>::yieldFunction)
      .def_readwrite("redepositionThreshold",
                     &IBEParameters<T>::redepositionThreshold)
      .def_readwrite("redepositionRate", &IBEParameters<T>::redepositionRate);

  // Faraday Cage Etching
  pybind11::class_<FaradayCageParameters<T>>(module, "FaradayCageParameters")
      .def(pybind11::init<>())
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
  pybind11::native_enum<viennaray::NormalizationType>(
      module, "NormalizationType", "enum.IntEnum")
      .value("SOURCE", viennaray::NormalizationType::SOURCE)
      .value("MAX", viennaray::NormalizationType::MAX)
      .finalize();

  // RayTracingParameters
  pybind11::class_<RayTracingParameters>(module, "RayTracingParameters")
      .def(pybind11::init<>())
      .def_readwrite("normalizationType",
                     &RayTracingParameters::normalizationType)
      .def_readwrite("raysPerPoint", &RayTracingParameters::raysPerPoint)
      .def_readwrite("diskRadius", &RayTracingParameters::diskRadius)
      .def_readwrite("useRandomSeeds", &RayTracingParameters::useRandomSeeds)
      .def_readwrite("ignoreFluxBoundaries",
                     &RayTracingParameters::ignoreFluxBoundaries)
      .def_readwrite("smoothingNeighbors",
                     &RayTracingParameters::smoothingNeighbors)
      .def("toMetaData", &RayTracingParameters::toMetaData,
           "Convert the ray tracing parameters to a metadata dict.")
      .def("toMetaDataString", &RayTracingParameters::toMetaDataString,
           "Convert the ray tracing parameters to a metadata string.");

  // AdvectionParameters
  pybind11::class_<AdvectionParameters>(module, "AdvectionParameters")
      .def(pybind11::init<>())
      .def_readwrite("integrationScheme",
                     &AdvectionParameters::integrationScheme)
      .def_readwrite("timeStepRatio", &AdvectionParameters::timeStepRatio)
      .def_readwrite("dissipationAlpha", &AdvectionParameters::dissipationAlpha)
      .def_readwrite("checkDissipation", &AdvectionParameters::checkDissipation)
      .def_readwrite("velocityOutput", &AdvectionParameters::velocityOutput)
      .def_readwrite("ignoreVoids", &AdvectionParameters::ignoreVoids)
      .def("toMetaData", &AdvectionParameters::toMetaData,
           "Convert the advection parameters to a metadata dict.")
      .def("toMetaDataString", &AdvectionParameters::toMetaDataString,
           "Convert the advection parameters to a metadata string.");

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
  m_util.def("convertIntegrationScheme", &util::convertIntegrationScheme,
             "Convert a string to an integration scheme.");

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
  pybind11::class_<Extrude<T>>(module, "Extrude")
      .def(pybind11::init())
      .def(pybind11::init<SmartPointer<Domain<T, 2>> &,
                          SmartPointer<Domain<T, 3>> &, std::array<T, 2>,
                          const int,
                          std::array<viennals::BoundaryConditionEnum, 3>>(),
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
               std::array<viennals::BoundaryConditionEnum, 3>>(
               &Extrude<T>::setBoundaryConditions),
           "Set the boundary conditions in the extruded domain.")
      .def("apply", &Extrude<T>::apply, "Run the extrusion.");

  // ***************************************************************************
  //                                 GPU SUPPORT
  // ***************************************************************************

#ifdef VIENNACORE_COMPILE_GPU
  auto m_gpu = module.def_submodule("gpu", "GPU support functions.");

  pybind11::class_<DeviceContext, std::shared_ptr<DeviceContext>>(m_gpu,
                                                                  "Context")
      .def(pybind11::init())
      //  .def_readwrite("modulePath", &Context::modulePath)
      //  .def_readwrite("moduleNames", &Context::moduleNames)
      //  .def_readwrite("cuda", &Context::cuda, "Cuda context.")
      //  .def_readwrite("optix", &Context::optix, "Optix context.")
      //  .def_readwrite("deviceProps", &Context::deviceProps,
      //                 "Device properties.")
      //  .def("getModule", &Context::getModule)
      .def_static("createContext", &DeviceContext::createContext,
                  "Create a new context.",
                  pybind11::arg("modulePath") = VIENNACORE_KERNELS_PATH,
                  pybind11::arg("deviceID") = 0,
                  pybind11::arg("registerInGlobal") = true)
      .def_static("getContextFromRegistry",
                  &DeviceContext::getContextFromRegistry,
                  "Get a context from the global registry by device ID.",
                  pybind11::arg("deviceID") = 0)
      .def_static(
          "hasContextInRegistry", &DeviceContext::hasContextInRegistry,
          "Check if a context exists in the global registry by device ID.",
          pybind11::arg("deviceID") = 0)
      .def_static("getRegisteredDeviceIDs",
                  &DeviceContext::getRegisteredDeviceIDs,
                  "Get a list of all device IDs with registered contexts.")
      .def("create", &DeviceContext::create, "Create a new context.",
           pybind11::arg("modulePath") = VIENNACORE_KERNELS_PATH,
           pybind11::arg("deviceID") = 0)
      .def("destroy", &DeviceContext::destroy, "Destroy the context.")
      .def("addModule", &DeviceContext::addModule,
           "Add a module to the context.")
      .def("getModulePath", &DeviceContext::getModulePath,
           "Get the module path.")
      .def_readwrite("deviceID", &DeviceContext::deviceID, "Device ID.");
#endif
}
