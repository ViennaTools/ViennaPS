#include "pyWrap.hpp"

void wrapModels(pybind11::module_ &module) {
  /****************************************************************************
   *                               MODELS *
   ****************************************************************************/
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
           pybind11::arg("maskMaterial") = psMaterial::None);

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

  // SF6O2 Etching
  pybind11::class_<psSF6O2Etching<T, D>, psSmartPointer<psSF6O2Etching<T, D>>>(
      module, "SF6O2Etching", processModel)
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
           pybind11::arg("etchStopDepth") = std::numeric_limits<T>::lowest());

  // Fluorocarbon Etching
  pybind11::class_<psFluorocarbonEtching<T, D>,
                   psSmartPointer<psFluorocarbonEtching<T, D>>>(
      module, "FluorocarbonEtching", processModel)
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
          pybind11::arg("etchStopDepth") = std::numeric_limits<T>::lowest());

  // Fluorocarbon Parameters
  pybind11::class_<FluorocarbonImplementation::Parameters>(
      module, "FluorocarbonParameters")
      .def(pybind11::init<>())
      .def_readwrite("ionFlux",
                     &FluorocarbonImplementation::Parameters::ionFlux)
      .def_readwrite("etchantFlux",
                     &FluorocarbonImplementation::Parameters::etchantFlux)
      .def_readwrite("polyFlux",
                     &FluorocarbonImplementation::Parameters::polyFlux)
      .def_readwrite("delta_p",
                     &FluorocarbonImplementation::Parameters::delta_p)
      .def_readwrite("etchStopDepth",
                     &FluorocarbonImplementation::Parameters::etchStopDepth)
      .def_readwrite("Mask", &FluorocarbonImplementation::Parameters::Mask)
      .def_readwrite("SiO2", &FluorocarbonImplementation::Parameters::SiO2)
      .def_readwrite("Si3N4", &FluorocarbonImplementation::Parameters::Si3N4)
      .def_readwrite("Si", &FluorocarbonImplementation::Parameters::Si)
      .def_readwrite("Polymer",
                     &FluorocarbonImplementation::Parameters::Polymer)
      .def_readwrite("Ions", &FluorocarbonImplementation::Parameters::Ions);

  pybind11::class_<FluorocarbonImplementation::Parameters::MaskType>(
      module, "FluorocarbonParametersMask")
      .def(pybind11::init<>())
      .def_readwrite("rho",
                     &FluorocarbonImplementation::Parameters::MaskType::rho)
      .def_readwrite("beta_p",
                     &FluorocarbonImplementation::Parameters::MaskType::beta_p)
      .def_readwrite("beta_e",
                     &FluorocarbonImplementation::Parameters::MaskType::beta_e)
      .def_readwrite("A_sp",
                     &FluorocarbonImplementation::Parameters::MaskType::A_sp)
      .def_readwrite("B_sp",
                     &FluorocarbonImplementation::Parameters::MaskType::B_sp)
      .def_readwrite("Eth_sp",
                     &FluorocarbonImplementation::Parameters::MaskType::Eth_sp);

  pybind11::class_<FluorocarbonImplementation::Parameters::SiO2Type>(
      module, "FluorocarbonParametersSiO2")
      .def(pybind11::init<>())
      .def_readwrite("rho",
                     &FluorocarbonImplementation::Parameters::SiO2Type::rho)
      .def_readwrite("E_a",
                     &FluorocarbonImplementation::Parameters::SiO2Type::E_a)
      .def_readwrite("K", &FluorocarbonImplementation::Parameters::SiO2Type::K)
      .def_readwrite("A_sp",
                     &FluorocarbonImplementation::Parameters::SiO2Type::A_sp)
      .def_readwrite("B_sp",
                     &FluorocarbonImplementation::Parameters::SiO2Type::B_sp)
      .def_readwrite("Eth_ie",
                     &FluorocarbonImplementation::Parameters::SiO2Type::Eth_ie)
      .def_readwrite("Eth_sp",
                     &FluorocarbonImplementation::Parameters::SiO2Type::Eth_sp)
      .def_readwrite("A_ie",
                     &FluorocarbonImplementation::Parameters::SiO2Type::A_ie);

  pybind11::class_<FluorocarbonImplementation::Parameters::Si3N4Type>(
      module, "FluorocarbonParametersSi3N4")
      .def(pybind11::init<>())
      .def_readwrite("rho",
                     &FluorocarbonImplementation::Parameters::Si3N4Type::rho)
      .def_readwrite("E_a",
                     &FluorocarbonImplementation::Parameters::Si3N4Type::E_a)
      .def_readwrite("K", &FluorocarbonImplementation::Parameters::Si3N4Type::K)
      .def_readwrite("A_sp",
                     &FluorocarbonImplementation::Parameters::Si3N4Type::A_sp)
      .def_readwrite("B_sp",
                     &FluorocarbonImplementation::Parameters::Si3N4Type::B_sp)
      .def_readwrite("Eth_ie",
                     &FluorocarbonImplementation::Parameters::Si3N4Type::Eth_ie)
      .def_readwrite("Eth_sp",
                     &FluorocarbonImplementation::Parameters::Si3N4Type::Eth_sp)
      .def_readwrite("A_ie",
                     &FluorocarbonImplementation::Parameters::Si3N4Type::A_ie);

  pybind11::class_<FluorocarbonImplementation::Parameters::SiType>(
      module, "FluorocarbonParametersSi")
      .def(pybind11::init<>())
      .def_readwrite("rho",
                     &FluorocarbonImplementation::Parameters::SiType::rho)
      .def_readwrite("E_a",
                     &FluorocarbonImplementation::Parameters::SiType::E_a)
      .def_readwrite("K", &FluorocarbonImplementation::Parameters::SiType::K)
      .def_readwrite("A_sp",
                     &FluorocarbonImplementation::Parameters::SiType::A_sp)
      .def_readwrite("B_sp",
                     &FluorocarbonImplementation::Parameters::SiType::B_sp)
      .def_readwrite("Eth_ie",
                     &FluorocarbonImplementation::Parameters::SiType::Eth_ie)
      .def_readwrite("Eth_sp",
                     &FluorocarbonImplementation::Parameters::SiType::Eth_sp)
      .def_readwrite("A_ie",
                     &FluorocarbonImplementation::Parameters::SiType::A_ie);

  pybind11::class_<FluorocarbonImplementation::Parameters::PolymerType>(
      module, "FluorocarbonParametersPolymer")
      .def(pybind11::init<>())
      .def_readwrite("rho",
                     &FluorocarbonImplementation::Parameters::PolymerType::rho)
      .def_readwrite(
          "Eth_ie",
          &FluorocarbonImplementation::Parameters::PolymerType::Eth_ie)
      .def_readwrite(
          "A_ie", &FluorocarbonImplementation::Parameters::PolymerType::A_ie);

  pybind11::class_<FluorocarbonImplementation::Parameters::IonType>(
      module, "FluorocarbonParametersIons")
      .def(pybind11::init<>())
      .def_readwrite(
          "meanEnergy",
          &FluorocarbonImplementation::Parameters::IonType::meanEnergy)
      .def_readwrite(
          "sigmaEnergy",
          &FluorocarbonImplementation::Parameters::IonType::sigmaEnergy)
      .def_readwrite("exponent",
                     &FluorocarbonImplementation::Parameters::IonType::exponent)
      .def_readwrite(
          "inflectAngle",
          &FluorocarbonImplementation::Parameters::IonType::inflectAngle)
      .def_readwrite("n_l",
                     &FluorocarbonImplementation::Parameters::IonType::n_l)
      .def_readwrite(
          "minAngle",
          &FluorocarbonImplementation::Parameters::IonType::minAngle);

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

  // psAtomicLayerProcess
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