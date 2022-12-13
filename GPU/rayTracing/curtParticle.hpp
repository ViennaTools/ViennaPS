#pragma once

#include <string>
#include <vector>

template <typename T> struct curtParticle {
  std::string name;
  int numberOfData = 1;
  std::vector<std::string> dataLabels;
  float sticking = 1.f;
  float cosineExponent = 1.f;
};
