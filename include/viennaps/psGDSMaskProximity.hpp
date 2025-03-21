#pragma once

#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <queue>
#include <fstream>

namespace viennaps {

template <class NumericType> class psGDSMaskProximity {
    public:
        using Grid = std::vector<std::vector<NumericType>>;
        using Polygon = std::vector<std::pair<double, double>>;
    
        // Constructor: Initialize rasterization grid using GDSGeometry bounds_
        psGDSMaskProximity(double gridRes, const double bounds[6])
            : gridResolution(gridRes) {
            // Compute grid size based on total bounds
            gridMinX = bounds[0];
            gridMaxX = bounds[1];
            gridMinY = bounds[2];
            gridMaxY = bounds[3];
        
            gridSizeX = static_cast<int>(std::ceil((gridMaxX - gridMinX) / gridResolution));
            gridSizeY = static_cast<int>(std::ceil((gridMaxY - gridMinY) / gridResolution));

            // Initialize the exposure grid with full domain size
            exposureGrid = Grid(gridSizeY, std::vector<double>(gridSizeX, 0.0));
        }
    
        void addPolygons(const std::vector<Polygon>& polygons) {
            for (const auto& polygon : polygons) {
                rasterizePolygon(polygon);
            }
        }
    
        void applyProximityEffects(double sigmaForward, double sigmaBack, int layer = 0) {
            forwardScattering = applyGaussianBlur2D(exposureGrid, sigmaForward);
            backScattering = applyGaussianBlur2D(exposureGrid, sigmaBack);
            finalExposure = combineExposures();
            // if (lsInternal::Logger::getLogLevel() >= 4)
                saveGridsToCSV();
        }
    
        std::vector<Polygon> extractContoursAtThreshold(double threshold = 0.5) {
        
            if (finalExposure.empty() || finalExposure[0].empty()) {
                std::cerr << "Error: finalExposure grid is empty!" << std::endl;
                return {};
            }
        
            size_t rows = finalExposure.size();
            size_t cols = finalExposure[0].size();
            std::vector<Polygon> modifiedPolygons;
            Polygon allContours;
        
            std::vector<std::pair<int, int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
        
            for (size_t y = 0; y < rows; ++y) {
                for (size_t x = 0; x < cols; ++x) {
                    if (std::abs(finalExposure[y][x] - threshold) < 1e-6) {
                        allContours.push_back({x, y});
                        continue;
                    }

                    if (finalExposure[y][x] < threshold) continue;

                    if (y == 0 || y == rows - 1 || x == 0 || x == cols - 1) {
                        allContours.push_back({x, y});
                        continue;
                    }
        
                    // Only process if this point is above threshold and touches a lower value
                    for (auto [dy, dx] : directions) {
                        size_t ny = y + dy, nx = x + dx;
                        if (ny >= 0 && ny < rows && nx >= 0 && nx < cols && finalExposure[ny][nx] < threshold) {
                            double T_high = finalExposure[y][x];  // Current pixel (above threshold)
                            double T_low = finalExposure[ny][nx]; // Neighbor pixel (below threshold)
                            double interpFactor = (T_high - threshold) / (T_high - T_low);
                            double newX = x + (dx * interpFactor);
                            double newY = y + (dy * interpFactor);
                            allContours.push_back({newX, newY});
                        }
                    }
                }
            }

            static int count = 0;
            viennaps::GDS::saveContoursToCSV(allContours, "layer" + std::to_string(count++) + "_allContours.csv");
        
            // Group allContours into separate polygons
            std::vector<bool> visited(allContours.size(), false);
            double distanceThreshold = 1.5;

            for (size_t i = 0; i < allContours.size(); ++i) {
                if (visited[i]) continue;  

                Polygon currentPolygon;
                size_t currentIndex = i;

                while (true) {
                    visited[currentIndex] = true;
                    currentPolygon.push_back(allContours[currentIndex]);

                    // Search for the closest unvisited neighbor in a looped manner
                    double minDist = std::numeric_limits<double>::max();
                    size_t closestIdx = -1;

                    for (size_t j = 0; j < allContours.size(); ++j) {
                        if (!visited[j]) {
                            double dist = std::hypot(allContours[currentIndex].first - allContours[j].first, 
                                                    allContours[currentIndex].second - allContours[j].second);
                            if (dist < minDist) {
                                minDist = dist;
                                closestIdx = j;
                            }
                        }
                    }

                    if (minDist > distanceThreshold) break;
                    currentIndex = closestIdx;
                }
                // currentPolygon.push_back(currentPolygon.front());  // Close the polygon
                modifiedPolygons.push_back(currentPolygon);
            }

            return modifiedPolygons;
        }
                                
        const Grid& getFinalExposure() const { return finalExposure; }
        double getGridResolution() const { return gridResolution; }
    
    private:
        double gridResolution;
        int gridSizeX, gridSizeY;
        double gridMinX, gridMaxX, gridMinY, gridMaxY;
        Grid exposureGrid, forwardScattering, backScattering, finalExposure;
   
        void saveGridsToCSV() {
            static int count = 0;
            std::string baseFilename = "layer" + std::to_string(count++);
        
            struct GridInfo {
                std::string name;
                const Grid* grid;
            };
        
            std::vector<GridInfo> grids = {
                {"finalExposure", &finalExposure},
                {"exposure", &exposureGrid}
            };
        
            for (const auto& gridInfo : grids) {
                if (gridInfo.grid->empty() || (*gridInfo.grid)[0].empty()) {
                    std::cerr << "Error: " << gridInfo.name << " grid is empty, cannot save CSV!" << std::endl;
                    continue;
                }
        
                std::string filename = baseFilename + "_" + gridInfo.name + ".csv";
                std::ofstream file(filename);
                if (!file) {
                    std::cerr << "Error: Cannot open file " << filename << " for writing!" << std::endl;
                    continue;
                }
        
                for (const auto& row : *gridInfo.grid) {
                    for (size_t x = 0; x < row.size(); ++x) {
                        file << row[x];
                        if (x < row.size() - 1) file << ",";  // CSV format
                    }
                    file << "\n";  // New row for next Y line
                }
        
                file.close();
            }
        }        

        void rasterizePolygon(const Polygon& polygon) {
            std::vector<int> nodeX;
        
            // Convert bounds to grid index space
            int xMin = static_cast<int>(gridMinX / gridResolution - 0.5);
            int xMax = static_cast<int>(gridMaxX / gridResolution + 0.5);
            int yMin = static_cast<int>(gridMinY / gridResolution - 0.5);
            int yMax = static_cast<int>(gridMaxY / gridResolution + 0.5);
            
            for (int y = yMin; y <= yMax; ++y) {
                nodeX.clear();
        
                // Find intersections with edges
                for (size_t i = 0; i < polygon.size(); ++i) {
                    size_t j = (i + 1) % polygon.size();
                    double x1 = polygon[i].first / gridResolution - xMin;
                    double y1 = polygon[i].second / gridResolution;
                    double x2 = polygon[j].first / gridResolution - xMin;
                    double y2 = polygon[j].second / gridResolution;
        
                    if ((y1 <= y && y2 > y) || (y2 <= y && y1 > y)) {
                        double xInt = x1 + (y - y1) * (x2 - x1) / (y2 - y1);
                        nodeX.push_back(static_cast<int>(xInt));
                    }
                }
        
                // Sort X intersections
                std::sort(nodeX.begin(), nodeX.end());
        
                // Fill using Even-Odd rule
                bool inside = false;
                for (size_t i = 0; i < nodeX.size(); ++i) {
                    inside = !inside;  // Toggle inside status
                    if (inside && i + 1 < nodeX.size()) {
                        for (int x = nodeX[i]; x <= nodeX[i + 1]; ++x) {
                            if (x >= 0 && x < gridSizeX && y >= 0 && y < gridSizeY) {
                                exposureGrid[y][x] = 1.0;  // Fill inside pixels
                            }
                        }
                    }
                }
            }
        }

        Grid applyGaussianBlur2D(const Grid& input, double sigma) {
            if (input.empty() || input[0].empty()) {
                std::cerr << "Error: input grid is empty!" << std::endl;
                return {};
            }

            int kernelSize = std::min(21, std::max(3, static_cast<int>(6 * sigma)));  
            if (kernelSize % 2 == 0) kernelSize += 1;
    
            int halfSize = kernelSize / 2;
            Grid output(gridSizeY, std::vector<double>(gridSizeX, 0.0));
    
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
                            value += input[srcY][srcX] * kernel[ky + halfSize][kx + halfSize];
                        }
                    }

                    output[y][x] = value;
                }
            }
    
            return output;
        }
    
        Grid combineExposures(double backScatterFactor = 0.2) {
            Grid output(gridSizeY, std::vector<double>(gridSizeX, 0.0));
            double maxValue = 0.0;
        
            // Step 1: Compute the combined exposure and find max value
            for (int y = 0; y < gridSizeY; ++y) {
                for (int x = 0; x < gridSizeX; ++x) {
                    output[y][x] = forwardScattering[y][x] + backScatterFactor * backScattering[y][x];
                    if (output[y][x] > maxValue) {
                        maxValue = output[y][x];
                    }
                }
            }
        
            // Step 2: Normalize to max of 1 (if maxValue > 0)
            if (maxValue > 0.0) {
                for (int y = 0; y < gridSizeY; ++y) {
                    for (int x = 0; x < gridSizeX; ++x) {
                        output[y][x] /= maxValue;
                    }
                }
            }
        
            return output;
        }
    };

} // namespace viennaps