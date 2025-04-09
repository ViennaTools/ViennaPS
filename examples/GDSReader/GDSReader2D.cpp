#include <psGDSReader.hpp>

#include <lsToSurfaceMesh.hpp>
#include <lsVTKWriter.hpp>

namespace ps = viennaps;
namespace ls = viennals;

int main(int argc, char **argv) {
  using NumericType = double;
  constexpr int D = 2;

  // read GDS mask file
  ps::Logger::setLogLevel(ps::LogLevel::DEBUG);

  constexpr NumericType gridDelta = 0.01;
  constexpr NumericType exposureDelta = 0.005;
  double forwardSigma = 5.;
  double backsSigma = 50.;

  ls::BoundaryConditionEnum boundaryConds[D] = {
      ls::BoundaryConditionEnum::REFLECTIVE_BOUNDARY,
      ls::BoundaryConditionEnum::REFLECTIVE_BOUNDARY};

  auto mask = ps::SmartPointer<ps::GDSGeometry<NumericType, D>>::New(
      gridDelta, boundaryConds);
  mask->addBlur({forwardSigma, backsSigma}, // Gaussian sigmas
                {0.8, 0.2},                 // Weights
                0.5,                        // Threshold
                exposureDelta);             // Exposure grid delta
  ps::GDSReader<NumericType, D>(mask, "myTest.gds").apply();

  auto maskLayer = mask->layerToLevelSet(0, false);
  auto mesh = ls::SmartPointer<ls::Mesh<NumericType>>::New();
  ls::ToSurfaceMesh<NumericType, D>(maskLayer, mesh).apply();
  ls::VTKWriter<NumericType>(mesh, "maskLayer.vtp").apply();

  auto blurredLayer = mask->layerToLevelSet(0, true);
  ls::ToSurfaceMesh<NumericType, D>(blurredLayer, mesh).apply();
  ls::VTKWriter<NumericType>(mesh, "blurredLayer.vtp").apply();

  return 0;
}