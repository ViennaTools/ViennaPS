#pragma once

#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <queue>
#include <fstream>

class psGDSMaskProximity {
    public:
        using Grid = std::vector<std::vector<double>>;
        using Polygon = std::vector<std::pair<double, double>>;
            
        psGDSMaskProximity(double gridRes, int gridSizeX, int gridSizeY)
            : gridResolution(gridRes), gridSizeX(gridSizeX), gridSizeY(gridSizeY) {
            exposureGrid = Grid(gridSizeY, std::vector<double>(gridSizeX, 0.0));
        }
    
        void addPolygons(const std::vector<Polygon>& polygons) {
            for (const auto& polygon : polygons) {
                rasterizePolygon(polygon);
            }
        }
    
        void applyProximityEffects(double sigmaForward, double sigmaBack) {
            forwardScattering = applyGaussianBlur2D(exposureGrid, sigmaForward);
            backScattering = applyGaussianBlur2D(exposureGrid, sigmaBack);
            finalExposure = combineExposures();
            // if (lsInternal::Logger::getLogLevel() >= 4)
                saveFinalExposureToCSV("final_exposure.csv");
        }
    
        std::vector<Polygon> extractContoursAtThreshold(double threshold = 0.5) {
            // std::cout << "Extracting contours at threshold " << threshold << std::endl;
        
            if (finalExposure.empty() || finalExposure[0].empty()) {
                std::cerr << "Error: finalExposure grid is empty!" << std::endl;
                return {};
            }
        
            size_t rows = finalExposure.size();
            size_t cols = finalExposure[0].size();
            std::vector<Polygon> modifiedPolygons;
            Polygon allContours;
        
            // Directions for Manhattan neighbors (up, down, left, right)
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
                            double newX = x + (dx * interpFactor); // * gridResolution;
                            double newY = y + (dy * interpFactor); // * gridResolution;
                            allContours.push_back({newX, newY});
                        }
                    }
                }
            }

            saveContoursToCSV(allContours, "contours.csv");
        
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

            int i = 0;
            for (auto currentPoly : modifiedPolygons) {
                saveContoursToCSV(currentPoly, "polygon_" + std::to_string(i) + ".csv");
                i++;
            }

            return modifiedPolygons;
        }
                                
        void saveFinalExposureToCSV(const std::string& filename) {
            if (finalExposure.empty() || finalExposure[0].empty()) {
                std::cerr << "Error: finalExposure grid is empty, cannot save CSV!" << std::endl;
                return;
            }
        
            std::ofstream file(filename);
            if (!file) {
                std::cerr << "Error: Cannot open file " << filename << " for writing!" << std::endl;
                return;
            }
        
            for (size_t y = 0; y < finalExposure.size(); ++y) {
                for (size_t x = 0; x < finalExposure[y].size(); ++x) {
                    file << finalExposure[y][x];
                    if (x < finalExposure[y].size() - 1)
                        file << ",";
                }
                file << "\n";
            }
        
            file.close();
            // std::cout << "Saved finalExposure to " << filename << std::endl;
        }

        void saveContoursToCSV(const Polygon &allContours, const std::string &filename) {
            std::ofstream file(filename);
            if (!file.is_open()) {
                std::cerr << "Error opening file: " << filename << std::endl;
                return;
            }
        
            for (const auto &point : allContours) {
                file << point.first << "," << point.second << "\n";
            }
        
            file.close();
            // std::cout << "Contours saved to " << filename << std::endl;
        }

        const Grid& getFinalExposure() const { return finalExposure; }
        double getGridResolution() const { return gridResolution; }
    
    private:
        double gridResolution;
        int gridSizeX, gridSizeY;
        Grid exposureGrid, forwardScattering, backScattering, finalExposure;
    
        void rasterizePolygon(const Polygon& polygon) {
            std::vector<int> nodeX;  
        
            // Find min/max Y values
            double minY = polygon[0].second, maxY = polygon[0].second;
            for (const auto& point : polygon) {
                minY = std::min(minY, point.second);
                maxY = std::max(maxY, point.second);
            }
        
            int yMin = static_cast<int>(minY / gridResolution);
            int yMax = static_cast<int>(maxY / gridResolution);
        
            for (int y = yMin; y <= yMax; ++y) {
                nodeX.clear();
        
                // Find intersections with edges
                for (size_t i = 0; i < polygon.size(); ++i) {
                    size_t j = (i + 1) % polygon.size();
                    double x1 = polygon[i].first / gridResolution;
                    double y1 = polygon[i].second / gridResolution;
                    double x2 = polygon[j].first / gridResolution;
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
                            if ((y + ky) >= 0 && (y + ky) < gridSizeY && (x + kx) >= 0 && (x + kx) < gridSizeX) {
                                value += input[y + ky][x + kx] * kernel[ky + halfSize][kx + halfSize];
                            }                            
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
        
            // std::cout << "Exposure grid normalized to max value of 1." << std::endl;
            return output;
        }
    };