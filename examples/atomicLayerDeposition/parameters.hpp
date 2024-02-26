#pragma once

#include <string>
#include <unordered_map>

#include <psUtils.hpp>

struct Parameters {
  std::unordered_map<std::string, std::string> m;

  void readConfigFile(const std::string &fileName) {
    m = psUtils::readConfigFile(fileName);
  }

  template <typename T = double> T get(const std::string &key) {
    if (m.find(key) == m.end()) {
      std::cout << "Key not found in parameters: " << key << std::endl;
      exit(1);
      return T();
    }
    return psUtils::convert<T>(m[key]);
  }
};
