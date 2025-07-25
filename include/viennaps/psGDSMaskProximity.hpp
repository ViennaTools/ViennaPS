#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "psDomain.hpp"

#include <lsDomain.hpp>

#include <vcSmartPointer.hpp>

#include <hrleDenseIterator.hpp>
#include <hrleGrid.hpp>

namespace ls = viennals;

namespace viennaps {

template <typename NumericType> class GDSMaskProximity {
  using DomainType2D = SmartPointer<ls::Domain<NumericType, 2>>;
  using Grid2D = std::vector<std::vector<NumericType>>;

public:
  GDSMaskProximity(DomainType2D inputLS, int delta,
                   const std::vector<NumericType> &sigmas,
                   const std::vector<NumericType> &weights)
      : inputLevelSet(inputLS), deltaRatio(delta), sigmas(sigmas),
        weights(weights) {
    assert(sigmas.size() == weights.size());
    assert(inputLS != nullptr);
    initializeGrid();
  }

  void apply() {
    for (auto sigma : sigmas)
      blurredGrids.push_back(applyGaussianBlur(sigma));

    finalGrid = combineExposures();
  }

  const Grid2D &getExposedGrid() const { return finalGrid; }

  const Grid2D &getExposureMap() const { return exposureMap; }

  // Save the final grid
  void saveGridToCSV(const std::string &filename) {
    saveGridToCSV(filename, finalGrid);
  }

  void saveGridToCSV(const std::string &filename, Grid2D grid) {
    std::ofstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Error: Cannot open file '" << filename << "' for writing!"
                << std::endl;
      return;
    }

    for (const auto &row : grid) {
      for (size_t i = 0; i < row.size(); ++i) {
        file << row[i];
        if (i < row.size() - 1)
          file << ",";
      }
      file << "\n";
    }

    file.close();
  }

  double exposureAt(double xReal, double yReal) {
    const double delta = inputLevelSet->getGrid().getGridDelta() / deltaRatio;
    auto boundaryConds_ = inputLevelSet->getGrid().getBoundaryConditions();

    double xExpId = (xReal) / delta;
    double yExpId = (yReal) / delta;

    int x0 = static_cast<int>(std::floor(xExpId));
    int x1 = x0 + 1;
    int y0 = static_cast<int>(std::floor(yExpId));
    int y1 = y0 + 1;

    int maxY = static_cast<int>(finalGrid.size()) - 1;
    int maxX = static_cast<int>(finalGrid[0].size()) - 1;

    double dx = xExpId - x0;
    double dy = yExpId - y0;

    // Check if all four points are in-bounds
    if (x0 > 0 && x0 < maxX && y0 > 0 && y0 < maxY) {
      double v00 = finalGrid[y0][x0];
      double v10 = finalGrid[y0][x1];
      double v01 = finalGrid[y1][x0];
      double v11 = finalGrid[y1][x1];

      return (1 - dx) * (1 - dy) * v00 + dx * (1 - dy) * v10 +
             (1 - dx) * dy * v01 + dx * dy * v11;
    }

    // Handle boundary condition fallback
    auto getSafeValue = [&](int x, int y) -> double {
      int tx = x, ty = y;
      if (!applyBoundaryCondition(tx, ty, maxX, maxY, boundaryConds_))
        return 0.0;
      return finalGrid[ty][tx];
    };

    double v00 = getSafeValue(x0, y0);
    double v10 = getSafeValue(x1, y0);
    double v01 = getSafeValue(x0, y1);
    double v11 = getSafeValue(x1, y1);

    return (1 - dx) * (1 - dy) * v00 + dx * (1 - dy) * v10 +
           (1 - dx) * dy * v01 + dx * dy * v11;
  }

private:
  std::vector<NumericType> sigmas, weights;
  int deltaRatio;

  DomainType2D inputLevelSet;
  Grid2D finalGrid;
  Grid2D exposureMap;

  std::vector<Grid2D> blurredGrids;
  int gridSizeX = 0, gridSizeY = 0;

  void initializeGrid() {
    auto minIdx = inputLevelSet->getGrid().getMinIndex();
    auto maxIdx = inputLevelSet->getGrid().getMaxIndex();
    gridSizeX = maxIdx[0] - minIdx[0] + 1;
    gridSizeY = maxIdx[1] - minIdx[1] + 1;

    exposureMap.resize(gridSizeY, std::vector<NumericType>(gridSizeX, 0.0));

    viennahrle::DenseIterator<typename ls::Domain<NumericType, 2>::DomainType>
        it(inputLevelSet->getDomain());
    for (; !it.isFinished(); ++it) {
      auto idx = it.getIndices();
      NumericType val = it.getValue();
      int x = idx[0] - minIdx[0];
      int y = idx[1] - minIdx[1];
      exposureMap[y][x] = (val < 0.) ? 1.0 : 0.0; // Binary mask
    }
  }

  Grid2D applyGaussianBlur(NumericType sigma) {
    if (exposureMap.empty() || exposureMap[0].empty()) {
      std::cerr << "Error: input grid is empty!" << std::endl;
      return {};
    }

    // Coarse grid (used for Gaussian convolution)
    const int coarseSizeY = exposureMap.size();
    const int coarseSizeX = exposureMap[0].size();
    const double coarseDelta = inputLevelSet->getGrid().getGridDelta();

    // Fine grid (used for kernel and output)
    const int fineSizeY = coarseSizeY * deltaRatio;
    const int fineSizeX = coarseSizeX * deltaRatio;
    const double fineDelta = coarseDelta / deltaRatio;

    // Create output exposure grid on the fine grid
    Grid2D output(fineSizeY, std::vector<NumericType>(fineSizeX, 0.0));

    // Kernel size based on fine grid resolution
    int kernelSize =
        std::min(static_cast<int>(21 * deltaRatio),
                 std::max(3, static_cast<int>(std::ceil(6 * sigma))));
    if (kernelSize % 2 == 0)
      kernelSize += 1;
    int halfSize = kernelSize / 2;

    // Create Gaussian kernel (on fine grid spacing)
    std::vector<std::vector<double>> kernel(kernelSize,
                                            std::vector<double>(kernelSize));
    double sum = 0.0;
    for (int i = 0; i < kernelSize; ++i) {
      for (int j = 0; j < kernelSize; ++j) {
        double x = (i - halfSize) * fineDelta;
        double y = (j - halfSize) * fineDelta;
        kernel[i][j] = std::exp(-0.5 * (x * x + y * y) / (sigma * sigma));
        sum += kernel[i][j];
      }
    }
    for (auto &row : kernel)
      for (auto &val : row)
        val /= sum;

    // Apply Gaussian convolution centered at every beam location (coarse grid
    // spacing)
    for (int y = 0; y < coarseSizeY; y++) {
      for (int x = 0; x < coarseSizeX; x++) {
        if (exposureMap[y][x] <= 0.0)
          continue;

        // Apply Gaussian centered at (x,y)
        for (int ky = -halfSize; ky <= halfSize; ++ky) {
          for (int kx = -halfSize; kx <= halfSize; ++kx) {
            int nx = x * deltaRatio + kx;
            int ny = y * deltaRatio + ky;

            if (nx >= 0 && nx < fineSizeX && ny >= 0 && ny < fineSizeY) {
              output[ny][nx] +=
                  exposureMap[y][x] * kernel[ky + halfSize][kx + halfSize];
            }
          }
        }
      }
    }
    return output;
  }

  Grid2D combineExposures() {
    auto fineGridSizeY = gridSizeY * deltaRatio;
    auto fineGridSizeX = gridSizeX * deltaRatio;
    Grid2D output(fineGridSizeY, std::vector<NumericType>(fineGridSizeX, 0.0));
    NumericType maxValue = 0.0;

    // Step 1: Compute the weighted sum and find max value in one pass
    for (int y = 0; y < fineGridSizeY; ++y) {
      for (int x = 0; x < fineGridSizeX; ++x) {
        NumericType combinedValue = 0.0;
        for (size_t i = 0; i < blurredGrids.size(); ++i) {
          combinedValue += weights[i] * blurredGrids[i][y][x];
        }
        output[y][x] = combinedValue;
        maxValue = std::max(maxValue, combinedValue);
      }
    }

    // Step 2: Normalize to max of 1 (if maxValue > 0)
    if (maxValue > 0.0) {
      double invMax = 1.0 / maxValue;
      for (int y = 0; y < fineGridSizeY; ++y) {
        for (int x = 0; x < fineGridSizeX; ++x) {
          output[y][x] *= invMax;
        }
      }
    }

    return output;
  }

  bool applyBoundaryCondition(
      int &x, int &y, int maxX, int maxY,
      const std::array<BoundaryType, 2> &boundaryConditions) {
    // X
    if (x < 0) {
      if (boundaryConditions[0] == BoundaryType::INFINITE_BOUNDARY)
        return false;
      else if (boundaryConditions[0] == BoundaryType::REFLECTIVE_BOUNDARY)
        x = -x;
      else if (boundaryConditions[0] == BoundaryType::PERIODIC_BOUNDARY)
        x = maxX - 1;
      else
        return false;
    } else if (x > maxX) {
      if (boundaryConditions[0] == BoundaryType::INFINITE_BOUNDARY)
        return false;
      else if (boundaryConditions[0] == BoundaryType::REFLECTIVE_BOUNDARY)
        x = 2 * maxX - x - 1;
      else if (boundaryConditions[0] == BoundaryType::PERIODIC_BOUNDARY)
        x = 0;
      else
        return false;
    }

    // Y
    if (y < 0) {
      if (boundaryConditions[1] == BoundaryType::INFINITE_BOUNDARY)
        return false;
      else if (boundaryConditions[1] == BoundaryType::REFLECTIVE_BOUNDARY)
        y = -y;
      else if (boundaryConditions[1] == BoundaryType::PERIODIC_BOUNDARY)
        y = maxY - 1;
      else
        return false;
    } else if (y > maxY) {
      if (boundaryConditions[1] == BoundaryType::INFINITE_BOUNDARY)
        return false;
      else if (boundaryConditions[1] == BoundaryType::REFLECTIVE_BOUNDARY)
        y = 2 * maxY - y - 1;
      else if (boundaryConditions[1] == BoundaryType::PERIODIC_BOUNDARY)
        y = 0;
      else
        return false;
    }

    return true;
  }
};

} // namespace viennaps
