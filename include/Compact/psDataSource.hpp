#ifndef PS_DATA_SOURCE_HPP
#define PS_DATA_SOURCE_HPP

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

#include <psSmartPointer.hpp>

template <typename NumericType, int D> class psDataSource {

public:
  using DataPtr = psSmartPointer<std::vector<std::array<NumericType, D>>>;
  virtual DataPtr getAll() = 0;

  virtual std::vector<NumericType> getPositionalParameters() { return {}; }

  virtual std::unordered_map<std::string, NumericType> getNamedParameters() {
    return {};
  }
};

#endif