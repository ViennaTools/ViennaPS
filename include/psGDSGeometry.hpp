#pragma once

#include <psGDSUtils.hpp>

template <class NumericType, int D = 3> class psGDSGeometry {
  std::vector<psGDSStructure<NumericType>> structures;
  std::string libName = "";
  double units;
  double userUnits;

public:
  psGDSGeometry() {}

  void setLibName(const char *str) { libName = str; }

  void insertNextStructure(psGDSStructure<NumericType> &structure) {
    structures.push_back(structure);
  }

  void print() {
    std::cout << "======= STRUCTURES ========" << std::endl;
    for (auto &s : structures) {
      s.print();
    }
    std::cout << "============================" << std::endl;
  }
};