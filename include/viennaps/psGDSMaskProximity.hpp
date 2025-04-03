#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <string>

#include "psDomain.hpp"

#include <lsDomain.hpp>

#include <vcSmartPointer.hpp>

#include <hrleDenseIterator.hpp>

namespace ls = viennals;

namespace viennaps {

template <typename NumericType>
class GDSMaskProximity {
  using DomainType2D = SmartPointer<ls::Domain<NumericType, 2>>;
  using Grid2D = std::vector<std::vector<NumericType>>;

public:
  GDSMaskProximity(DomainType2D inputLS, double delta, const std::vector<double> &sigmas,
                          const std::vector<double> &weights)
      : inputLevelSet(inputLS), exposureDelta(delta), sigmas(sigmas), weights(weights) {
    assert(sigmas.size() == weights.size());
    assert(inputLevelSet != nullptr);
    initializeGrid();
  }

  void apply() {
    for (double sigma : sigmas) {
      blurredGrids.push_back(applyGaussianBlur(sigma));
    }

    finalGrid = combineExposures();
  }

  const Grid2D& getExposedGrid() const { return finalGrid; }

  const Grid2D& getExposureMap() const { return exposureMap; }

  void saveGridToCSV(const Grid2D &grid, const std::string &filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Error: Cannot open file '" << filename << "' for writing!" << std::endl;
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
  
private:
  std::vector<double> sigmas, weights;
  double exposureDelta;

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

    viennahrle::DenseIterator<typename ls::Domain<double, 2>::DomainType> it(inputLevelSet->getDomain());
    for (; !it.isFinished(); ++it) {
        auto idx = it.getIndices();
        double val = it.getValue();
        int x = idx[0] - minIdx[0];
        int y = idx[1] - minIdx[1];
        exposureMap[y][x] = (val < 0.) ? 1.0 : 0.0; // Binary mask
    }
  }

    Grid2D applyGaussianBlur(double sigma) {
        if (exposureMap.empty() || exposureMap[0].empty()) {
            std::cerr << "Error: input grid is empty!" << std::endl;
            return {};
        }

        int kernelSize = std::min(21, std::max(3, static_cast<int>(6 * sigma)));  
        if (kernelSize % 2 == 0) kernelSize += 1;

        int halfSize = kernelSize / 2;
        Grid2D output(gridSizeY, std::vector<double>(gridSizeX, 0.0));

        // Create 2D Gaussian kernel
        std::vector<std::vector<double>> kernel(kernelSize, std::vector<double>(kernelSize));
        double sum = 0.0;
        for (int i = 0; i < kernelSize; ++i) {
            for (int j = 0; j < kernelSize; ++j) {
                double x = i - halfSize;
                double y = j - halfSize;
                kernel[i][j] = std::exp(-0.5 * (x * x + y * y) / (sigma * sigma));
                sum += kernel[i][j];
            }
        }
        for (auto& row : kernel)
            for (auto& val : row)
                val /= sum;  

        // Perform 2D convolution
        for (int y = 0; y < gridSizeY; ++y) {
            for (int x = 0; x < gridSizeX; ++x) {
                double value = 0.0;

                for (int ky = -halfSize; ky <= halfSize; ++ky) {
                    for (int kx = -halfSize; kx <= halfSize; ++kx) {
                        int srcY = std::clamp(y + ky, 0, gridSizeY - 1);
                        int srcX = std::clamp(x + kx, 0, gridSizeX - 1);
    
                        // Apply kernel while ensuring proper boundary handling
                        value += exposureMap[srcY][srcX] * kernel[ky + halfSize][kx + halfSize];
                    }
                }

                output[y][x] = value;
            }
        }

        return output;
    }

    Grid2D combineExposures() {
        Grid2D output(gridSizeY, std::vector<double>(gridSizeX, 0.0));
        double maxValue = 0.0;
    
        // Step 1: Compute the weighted sum and find max value in one pass
        for (int y = 0; y < gridSizeY; ++y) {
            for (int x = 0; x < gridSizeX; ++x) {
                double combinedValue = 0.0;
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
            for (int y = 0; y < gridSizeY; ++y) {
                for (int x = 0; x < gridSizeX; ++x) {
                    output[y][x] *= invMax;
                }
            }
        }
    
        return output;
    }
};

} // namespace viennaps
