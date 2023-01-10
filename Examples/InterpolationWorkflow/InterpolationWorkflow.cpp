#include <filesystem>
#include <iostream>
#include <string>

#include <lsDomain.hpp>

#include <psCSVDataSource.hpp>
#include <psDataScaler.hpp>
#include <psMakeTrench.hpp>
#include <psNearestNeighborsInterpolation.hpp>

#include "GeometryReconstruction.hpp"
#include "Parameters.hpp"
#include "TrenchDeposition.hpp"

namespace fs = std::filesystem;

template <typename NumericType, int D>
psSmartPointer<lsDomain<NumericType, D>>
createEmptyLevelset(const Parameters<NumericType> &params) {

  double bounds[2 * D];
  bounds[0] = -params.xExtent / 2.;
  bounds[1] = params.xExtent / 2.;

  if constexpr (D == 3) {
    bounds[2] = -params.yExtent / 2.;
    bounds[3] = params.yExtent / 2.;
    bounds[4] = -params.gridDelta;
    bounds[5] = params.gridDelta;
  } else {
    bounds[2] = -params.gridDelta;
    bounds[3] = params.gridDelta;
  }

  typename lsDomain<NumericType, D>::BoundaryType boundaryCons[D];

  for (int i = 0; i < D - 1; i++)
    boundaryCons[i] =
        lsDomain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY;

  boundaryCons[D - 1] =
      lsDomain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY;

  return psSmartPointer<lsDomain<NumericType, D>>::New(bounds, boundaryCons,
                                                       params.gridDelta);
}

int main(int argc, char *argv[]) {
  using NumericType = double;
  static constexpr int D = 2;

  // Input dimensions: taperAngle, stickingProbability and time
  static constexpr int InputDim = 3;

  // Target Dimensions: depth, diameters
  static constexpr int TargetDim = 20;

  static constexpr int DataDim = InputDim + TargetDim;

  fs::path dataFile = "./data.csv";
  if (argc > 1)
    dataFile = fs::path{argv[1]};

  Parameters<NumericType> params;
  params.taperAngle = 0.;
  params.stickingProbability = 0.5;
  params.processTime = 5.;

  // Interpolation based on previous simulation results
  {
    psCSVDataSource<NumericType, DataDim> dataSource;
    dataSource.setFilename(dataFile.string());

    auto sampleLocations = dataSource.getPositionalParameters();

    int numberOfNeighbors = 5;
    NumericType distanceExponent = 2.;

    psNearestNeighborsInterpolation<
        NumericType, InputDim, TargetDim,
        psStandardScaler<NumericType, InputDim, DataDim>>
        estimator(numberOfNeighbors, distanceExponent);

    auto data = dataSource.get();
    estimator.setData(psSmartPointer<decltype(data)>::New(data));

    if (!estimator.initialize())
      return EXIT_FAILURE;

    std::array<NumericType, InputDim> x = {
        params.taperAngle, params.stickingProbability, params.processTime};
    auto [result, distance] = estimator.estimate(x);

    std::cout << distance << std::endl;

    auto dimensions = psSmartPointer<std::vector<NumericType>>::New();
    dimensions->reserve(TargetDim);
    std::copy(result.begin(), result.end(), std::back_inserter(*dimensions));

    NumericType origin[D] = {0.};
    origin[D - 1] = params.processTime + params.trenchHeight;

    auto interpolated = psSmartPointer<psDomain<NumericType, D>>::New();
    auto substrate = createEmptyLevelset<NumericType, D>(params);
    interpolated->insertNextLevelSet(substrate);
    auto geometry = psSmartPointer<lsDomain<NumericType, D>>::New(substrate);

    GeometryReconstruction<NumericType, D>(geometry, origin, sampleLocations,
                                           *dimensions)
        .apply();

    interpolated->insertNextLevelSet(geometry);
    interpolated->printSurface("interpolated.vtp");
  }

  // Actual simulation for reference
  {
    auto geometry = psSmartPointer<psDomain<NumericType, D>>::New();
    psMakeTrench<NumericType, D>(geometry, params.gridDelta /* grid delta */,
                                 params.xExtent /*x extent*/,
                                 params.yExtent /*y extent*/,
                                 params.trenchWidth /*trench width*/,
                                 params.trenchHeight /*trench height*/,
                                 params.taperAngle /* tapering angle */)
        .apply();

    executeProcess<NumericType, D>(geometry, params);
    geometry->printSurface("reference.vtp");
  }
  return EXIT_SUCCESS;
}
