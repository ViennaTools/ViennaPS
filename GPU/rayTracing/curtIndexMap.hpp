#pragma once

#include <cassert>
#include <string>
#include <unordered_map>
#include <vector>

#include <curtParticle.hpp>

template <typename T> class curtIndexMap {
public:
  using indexMap = std::unordered_map<std::string, unsigned int>;

  static indexMap
  getPointDataIndexMap(std::vector<curtParticle<T>> &particles) {
    indexMap imap;
    unsigned int offset = 0;
    for (size_t pIdx = 0; pIdx < particles.size(); pIdx++) {
      assert(particles[pIdx].numberOfData == particles[pIdx].dataLabels.size());
      for (size_t dIdx = 0; dIdx < particles[pIdx].numberOfData; dIdx++) {
        imap.insert(
            std::make_pair(particles[pIdx].dataLabels[dIdx], offset + dIdx));
      }
      offset += particles[pIdx].numberOfData;
    }
    return imap;
  }
};