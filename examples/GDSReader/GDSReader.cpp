#include <psDomain.hpp>
#include <psGDSReader.hpp>
#include <psMaterials.hpp>

namespace ps = viennaps;
namespace ls = viennals;

int main(int argc, char **argv) {
  using NumericType = double;
  constexpr int D = 3;

  // read GDS mask file
  ps::Logger::setLogLevel(ps::LogLevel::DEBUG);

  constexpr NumericType gridDelta = 0.01;
  constexpr NumericType exposureDelta = 0.005;
  double forwardSigma = 5.;
  double backsSigma = 50.;

  ls::BoundaryConditionEnum boundaryConds[D] = {
      ls::BoundaryConditionEnum::REFLECTIVE_BOUNDARY,
      ls::BoundaryConditionEnum::REFLECTIVE_BOUNDARY,
      ls::BoundaryConditionEnum::INFINITE_BOUNDARY};

  auto mask = ps::SmartPointer<ps::GDSGeometry<NumericType, D>>::New(
      gridDelta, boundaryConds);
  mask->addBlur({forwardSigma, backsSigma}, // Gaussian sigmas
                {0.8, 0.2},                 // Weights
                0.5,                        // Threshold
                exposureDelta);             // Exposure grid delta
  ps::GDSReader<NumericType, D>(mask, "myTest.gds").apply();

  // geometry setup
  auto geometry = ps::SmartPointer<ps::Domain<NumericType, D>>::New();

  // substrate plane
  auto bounds = mask->getBounds();
  NumericType origin[D] = {0., 0., 0.};
  NumericType normal[D] = {0., 0., 1.};
  auto plane = ps::SmartPointer<ls::Domain<NumericType, D>>::New(
      bounds, boundaryConds, gridDelta);
  ls::MakeGeometry<NumericType, D>(
      plane, ps::SmartPointer<ls::Plane<NumericType, D>>::New(origin, normal))
      .apply();
  geometry->insertNextLevelSetAsMaterial(plane, ps::Material::Si);
  geometry->saveSurfaceMesh("Substrate.vtp");

  auto layer0 = mask->layerToLevelSet(0, 0.0, 0.1);
  geometry->insertNextLevelSetAsMaterial(layer0, ps::Material::Mask);

  auto layer1 = mask->layerToLevelSet(1, -0.1, 0.3);
  geometry->insertNextLevelSetAsMaterial(layer1, ps::Material::SiO2);

  auto layer2 = mask->layerToLevelSet(2, 0., 0.15, false, false); // no blur
  geometry->insertNextLevelSetAsMaterial(layer2, ps::Material::Si3N4);

  auto layer3 = mask->layerToLevelSet(3, 0, 0.25);
  geometry->insertNextLevelSetAsMaterial(layer3, ps::Material::Cu);

  auto layer4 = mask->layerToLevelSet(4, 0, 0.4, false, false); // no blur
  geometry->insertNextLevelSetAsMaterial(layer4, ps::Material::W);

  auto layer5 = mask->layerToLevelSet(5, 0, 0.2);
  geometry->insertNextLevelSetAsMaterial(layer5, ps::Material::PolySi);

  geometry->saveSurfaceMesh("Geometry.vtp");

  return 0;
}