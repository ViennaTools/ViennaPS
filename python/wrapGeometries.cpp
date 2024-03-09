#include "pyWrap.hpp"

void wrapGeometries(pybind11::module_ &module) {
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
           pybind11::arg("material") = psMaterial::None)
      .def(pybind11::init(
               [](DomainType Domain, T Height, const psMaterial Material) {
                 return psSmartPointer<psMakePlane<T, D>>::New(Domain, Height,
                                                               Material);
               }),
           pybind11::arg("domain"), pybind11::arg("height") = 0.,
           pybind11::arg("material") = psMaterial::None)
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
           pybind11::arg("material") = psMaterial::None)
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
           pybind11::arg("material") = psMaterial::None)
      .def("apply", &psMakeHole<T, D>::apply, "Create a hole geometry.");

  // psMakeFin
  pybind11::class_<psMakeFin<T, D>, psSmartPointer<psMakeFin<T, D>>>(module,
                                                                     "MakeFin")
      .def(pybind11::init([](DomainType domain, const T gridDelta,
                             const T xExtent, const T yExtent, const T finWidth,
                             const T finHeight, const T taperAngle,
                             const T baseHeight, const bool periodicBoundary,
                             const bool makeMask, const psMaterial material) {
             return psSmartPointer<psMakeFin<T, D>>::New(
                 domain, gridDelta, xExtent, yExtent, finWidth, finHeight,
                 taperAngle, baseHeight, periodicBoundary, makeMask, material);
           }),
           pybind11::arg("domain"), pybind11::arg("gridDelta"),
           pybind11::arg("xExtent"), pybind11::arg("yExtent"),
           pybind11::arg("finWidth"), pybind11::arg("finHeight"),
           pybind11::arg("taperAngle") = 0., pybind11::arg("baseHeight") = 0.,
           pybind11::arg("periodicBoundary") = false,
           pybind11::arg("makeMask") = false,
           pybind11::arg("material") = psMaterial::None)
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
}