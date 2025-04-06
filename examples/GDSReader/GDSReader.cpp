#include <hrleGrid.hpp>
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
  auto bounds = mask->getBounds();
  auto geometry = ps::SmartPointer<ps::Domain<NumericType, D>>::New();

  // substrate plane
  NumericType origin[D] = {0., 0., 0.};
  NumericType normal[D] = {0., 0., 1.};
  auto plane = ps::SmartPointer<ls::Domain<NumericType, D>>::New(
      bounds, boundaryConds, gridDelta);
  ls::MakeGeometry<NumericType, D>(
      plane, ps::SmartPointer<ls::Plane<NumericType, D>>::New(origin, normal))
      .apply();
  geometry->insertNextLevelSetAsMaterial(plane, viennaps::Material::Si);

  auto layer0 = mask->layerToLevelSet(0, 0.0, 0.1, true);
  geometry->insertNextLevelSetAsMaterial(layer0, viennaps::Material::Mask);

  auto layer1 = mask->layerToLevelSet(1, -0.1, 0.3, true);
  geometry->insertNextLevelSetAsMaterial(layer1, viennaps::Material::SiO2);

  auto layer2 = mask->layerToLevelSet(2, 0., 0.15, true, false);
  geometry->insertNextLevelSetAsMaterial(layer2, viennaps::Material::Si3N4);

  auto layer3 = mask->layerToLevelSet(3, 0, 0.25, true);
  geometry->insertNextLevelSetAsMaterial(layer3, viennaps::Material::Cu);

  auto layer4 = mask->layerToLevelSet(4, 0, 0.4, true, false);
  geometry->insertNextLevelSetAsMaterial(layer4, viennaps::Material::W);

  auto layer5 = mask->layerToLevelSet(5, 0, 0.2, true);
  geometry->insertNextLevelSetAsMaterial(layer5, viennaps::Material::PolySi);

  geometry->saveSurfaceMesh("Geometry.vtp", false /* add material IDs */);
  geometry->saveVolumeMesh("Geometry");

  return 0;
}