#pragma once

#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace viennaps::gpu {

struct SourceCache {
  using CacheType = std::map<std::string, std::string *>;
  CacheType map;

  ~SourceCache() {
    for (auto &it : map)
      delete it.second;
  }
};
static SourceCache g_sourceCache;

inline bool readSourceFile(std::string &str, const std::string &filename) {
  // Try to open file
  std::ifstream file(filename.c_str(), std::ios::binary);
  if (file.good()) {
    // Found usable source file
    std::vector<unsigned char> buffer =
        std::vector<unsigned char>(std::istreambuf_iterator<char>(file), {});
    str.assign(buffer.begin(), buffer.end());
    return true;
  }
  return false;
}

inline void getInputDataFromFile(std::string &inputData, const char *filename) {

  const std::string sourceFilePath = filename;

  // Try to open source file
  if (!readSourceFile(inputData, sourceFilePath)) {
    std::string err = "Couldn't open source file " + sourceFilePath;
    throw std::runtime_error(err.c_str());
  }
}

inline const char *getInputData(const char *filename, size_t &dataSize) {

  std::string *inputData;
  const auto key = std::string(filename);

  if (auto elem = g_sourceCache.map.find(key);
      elem == g_sourceCache.map.end()) {
    inputData = new std::string();

    getInputDataFromFile(*inputData, filename);
    g_sourceCache.map[key] = inputData;
  } else {
    inputData = elem->second;
  }

  dataSize = inputData->size();
  return inputData->c_str();
}

} // namespace viennaps::gpu
