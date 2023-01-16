#include <chrono>
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

  using Clock = std::chrono::high_resolution_clock;
  using Duration = std::chrono::duration<double>;

  // Input dimensions: taperAngle, stickingProbability
  static constexpr int InputDim = 2;

  // How long to run the process and at which intervals to do the extraction
  static constexpr NumericType processDuration = 5.0;
  static constexpr NumericType extractionInterval = 1.0;

  // The number of heights at which we are going to measure the diameter of the
  // trench
  static constexpr int numberOfSamples = 30;

  // Total number of timesteps during the advection process at which the
  // geometry parameters are extracted.
  static constexpr int numberOfTimesteps =
      processDuration / extractionInterval + 1;

  // Target Dimensions: (time, depth, diameters) x numberOfTimesteps
  static constexpr int TargetDim = (numberOfSamples + 1) * numberOfTimesteps;

  static constexpr int DataDim = InputDim + TargetDim;

  fs::path dataFile = "./data.csv";
  if (argc > 1)
    dataFile = fs::path{argv[1]};

  Parameters<NumericType> params;
  if (argc > 2) {
    auto config = psUtils::readConfigFile(argv[2]);
    if (config.empty()) {
      std::cerr << "Empty config provided" << std::endl;
      return -1;
    }
    params.fromMap(config);
  }

  // Interpolation based on previous simulation results
  auto start = Clock::now();
  {
    psCSVDataSource<NumericType, DataDim> dataSource;
    dataSource.setFilename(dataFile.string());

    auto sampleLocations = dataSource.getPositionalParameters();

    int numberOfNeighbors = 3;
    NumericType distanceExponent = 2.;

    psNearestNeighborsInterpolation<
        NumericType, InputDim, TargetDim,
        psMedianDistanceScaler<NumericType, InputDim>>
        estimator(numberOfNeighbors, distanceExponent);

    // Copy the data from the data source
    auto data = dataSource.get();

    estimator.setData(psSmartPointer<decltype(data)>::New(data));

    if (!estimator.initialize())
      return EXIT_FAILURE;

    std::array<NumericType, InputDim> x = {params.taperAngle,
                                           params.stickingProbability};
    auto [result, distance] = estimator.estimate(x);

    std::cout << std::setw(40) << "Distance to nearest data point: ";
    std::cout << distance << std::endl;

    // Now determine which two timesteps we should consider for interpolating
    // along the time axis
    NumericType extractionStep = params.processTime / extractionInterval;

    int stepSize = (numberOfSamples + 1);
    int lowerIdx = std::clamp(static_cast<int>(std::floor(extractionStep)) *
                                  (numberOfSamples + 1),
                              0, TargetDim - stepSize - 1);

    int upperIdx = std::clamp(static_cast<int>(std::ceil(extractionStep)) *
                                  (numberOfSamples + 1),
                              0, TargetDim - stepSize - 1);
    NumericType distanceToLower = 1.0 * lowerIdx - extractionStep;
    NumericType distanceToUpper = 1.0 * extractionStep - upperIdx;
    NumericType totalDistance = distanceToUpper + distanceToLower;

    auto dimensions = psSmartPointer<std::vector<NumericType>>::New();

    // Copy the data corresponding to the dimensions of the lower timestep
    auto lowerDimensions =
        psSmartPointer<std::vector<NumericType>>::New(numberOfSamples);
    for (unsigned i = 0; i < numberOfSamples; ++i)
      lowerDimensions->at(i) = result.at(lowerIdx + 1 + i);

    if (totalDistance > 0) {
      // Copy the data corresponding to the dimensions of the upper timestep
      auto upperDimensions =
          psSmartPointer<std::vector<NumericType>>::New(numberOfSamples);
      for (unsigned i = 0; i < numberOfSamples; ++i)
        upperDimensions->at(i) = result.at(upperIdx + 1 + i);

      // Now for each individual dimension do linear interpolation between upper
      // and lower value based on the relative distance
      for (unsigned i = 0; i < lowerDimensions->size(); ++i)
        dimensions->emplace_back((distanceToUpper * lowerDimensions->at(i) +
                                  distanceToLower * upperDimensions->at(i)) /
                                 totalDistance);
    } else {
      std::copy(lowerDimensions->begin(), lowerDimensions->end(),
                std::back_inserter(*dimensions));
    }

    NumericType origin[D] = {0.};
    origin[D - 1] = params.processTime + params.trenchHeight;

    auto interpolated = psSmartPointer<psDomain<NumericType, D>>::New();
    auto substrate = createEmptyLevelset<NumericType, D>(params);
    interpolated->insertNextLevelSet(substrate);
    auto geometry = psSmartPointer<lsDomain<NumericType, D>>::New(substrate);

    // Now retrieve the dimension data of the two neighboring timesteps

    GeometryReconstruction<NumericType, D>(geometry, origin, sampleLocations,
                                           *dimensions)
        .apply();

    interpolated->insertNextLevelSet(geometry);
    interpolated->printSurface("interpolated.vtp");
  }
  auto stop = Clock::now();
  auto interpDuration = Duration(stop - start).count();
  std::cout << std::setw(40) << "Interpolation and reconstruction took: ";
  std::cout << std::scientific << interpDuration << "s\n";

  // Actual simulation for reference
  start = Clock::now();
  {
    auto geometry = psSmartPointer<psDomain<NumericType, D>>::New();
    psMakeTrench<NumericType, D>(geometry, params.gridDelta /* grid delta */,
                                 params.xExtent /*x extent*/,
                                 params.yExtent /*y extent*/,
                                 params.trenchWidth /*trench width*/,
                                 params.trenchHeight /*trench height*/,
                                 params.taperAngle /* tapering angle */)
        .apply();

    geometry->printSurface("initial.vtp");

    params.processTime = params.processTime / params.stickingProbability;

    executeProcess<NumericType, D>(geometry, params);
    geometry->printSurface("reference.vtp");
  }
  stop = Clock::now();
  auto simDuration = Duration(stop - start).count();
  std::cout << std::setw(40) << "Physical simulation took: ";
  std::cout << std::scientific << simDuration << "s\n";

  std::cout << std::setw(40) << "Speedup: ";
  std::cout << std::fixed << simDuration / interpDuration << '\n';
  return EXIT_SUCCESS;
}