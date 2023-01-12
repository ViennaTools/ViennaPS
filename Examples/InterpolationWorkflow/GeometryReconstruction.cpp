#include <iostream>

#include <psSmartPointer.hpp>

#include "AdvectionCallback.hpp"
#include "DimensionExtraction.hpp"
#include "GeometryReconstruction.hpp"
#include "Parameters.hpp"
#include "TrenchDeposition.hpp"

int main() {
  using NumericType = double;
  static constexpr int D = 2;

  static constexpr int numberOfSamples = 20;

  Parameters<NumericType> params;

  auto geometry = psSmartPointer<psDomain<NumericType, D>>::New();
  psMakeTrench<NumericType, D>(
      geometry, params.gridDelta /* grid delta */, params.xExtent /*x extent*/,
      params.yExtent /*y extent*/, params.trenchWidth /*trench width*/,
      params.trenchHeight /*trench height*/,
      params.taperAngle /* tapering angle */)
      .apply();

  executeProcess<NumericType, D>(geometry, params);
  geometry->printSurface("simulation.vtp");

  DimensionExtraction<NumericType, D> extractor;
  extractor.setDomain(geometry);
  extractor.setNumberOfSamples(numberOfSamples);
  extractor.setEdgeAffinity(4.);

  extractor.apply();

  auto sampleLocations = extractor.getSampleLocations();

  auto dimensions = extractor.getDimensions();

  assert(sampleLocations.size() == dimensions->size() - 1);

  /**
   * Now reconstruct the geometry based on the extracted diameters
   */
  NumericType origin[D] = {0.};
  origin[D - 1] = params.processTime + params.trenchHeight;

  auto ls = psSmartPointer<lsDomain<NumericType, D>>::New(
      geometry->getLevelSets()->back()->getGrid());

  GeometryReconstruction<NumericType, D>(ls, origin, sampleLocations,
                                         *dimensions)
      .apply();

  auto reconstruction = psSmartPointer<psDomain<NumericType, D>>::New();

  reconstruction->insertNextLevelSet(ls);

  reconstruction->printSurface("reconstruction.vtp");
}