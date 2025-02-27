#pragma once

#include <cassert>
#include <string>
#include <vector>

#include <curtParticle.hpp>

namespace viennaps::gpu {

class IndexMap {
  std::vector<std::string> dataLabels;

public:
  IndexMap() = default;

  template <class T>
  explicit IndexMap(const std::vector<Particle<T>> &particles) {
    for (size_t pIdx = 0; pIdx < particles.size(); pIdx++) {
      for (size_t dIdx = 0; dIdx < particles[pIdx].dataLabels.size(); dIdx++) {
        dataLabels.push_back(particles[pIdx].dataLabels[dIdx]);
      }
    }
  }

  void insertNextDataLabel(std::string dataLabel) {
    dataLabels.push_back(std::move(dataLabel));
  }

  std::size_t getIndex(const std::string &label) const {
    for (std::size_t idx = 0; idx < dataLabels.size(); idx++) {
      if (dataLabels[idx] == label) {
        return idx;
      }
    }
    assert(false && "Data label not found");
    return 0;
  }

  [[nodiscard]] const std::string &getLabel(std::size_t idx) const {
    assert(idx < dataLabels.size());
    return dataLabels[idx];
  }

  std::size_t getNumberOfData() const { return dataLabels.size(); }

  std::vector<std::string>::const_iterator begin() const {
    return dataLabels.cbegin();
  }

  std::vector<std::string>::const_iterator end() const {
    return dataLabels.cend();
  }
};

} // namespace viennaps::gpu
