#pragma once

#include <fstream>
#include <string>
#include <vector>

template <class NumericType> struct BivariateDistribution {
  std::vector<std::vector<NumericType>> pdf; // 2D grid of values
  std::vector<NumericType> support_x;        // (x-values)
  std::vector<NumericType> support_y;        // (y-values)
};

template <typename NumericType>
auto loadDistributionFromFile(const std::string &fileName) {
  BivariateDistribution<NumericType> distribution;

  // Load the distribution from the file
  std::ifstream file(fileName);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + fileName);
  }

  std::string line;
  // Header
  std::getline(file, line);

  // Read y-support
  std::getline(file, line);
  std::istringstream yStream(line);
  NumericType yValue;
  while (yStream >> yValue) {
    distribution.support_y.push_back(yValue);
  }
  distribution.support_y.shrink_to_fit();

  // Read x-support
  std::getline(file, line);
  std::istringstream xStream(line);
  NumericType xValue;
  while (xStream >> xValue) {
    distribution.support_x.push_back(xValue);
  }
  distribution.support_x.shrink_to_fit();

  // Read PDF values
  size_t rowSize = 0;
  while (std::getline(file, line)) {
    if (line.empty())
      continue; // Skip empty lines

    std::istringstream iss(line);
    std::vector<NumericType> pdfRow;
    if (rowSize > 0)
      pdfRow.reserve(rowSize); // Reserve space if row size is known

    NumericType pdfValue;
    while (iss >> pdfValue) {
      pdfRow.push_back(pdfValue);
    }

    rowSize = pdfRow.size();
    distribution.pdf.push_back(pdfRow);
  }

  return distribution;
}