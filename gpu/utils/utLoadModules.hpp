#include <fstream>
#include <map>
#include <string>
#include <vector>

struct SourceCache {
  std::map<std::string, std::string *> map;
  ~SourceCache() {
    for (std::map<std::string, std::string *>::const_iterator it = map.begin();
         it != map.end(); ++it)
      delete it->second;
  }
};
static SourceCache g_sourceCache;

static bool readSourceFile(std::string &str, const std::string &filename) {
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

static void getInputDataFromFile(std::string &inputData, const char *filename) {

  const std::string sourceFilePath = filename;

  // Try to open source file
  if (!readSourceFile(inputData, sourceFilePath)) {
    std::string err = "Couldn't open source file " + sourceFilePath;
    throw std::runtime_error(err.c_str());
  }
}

const char *getInputData(const char *filename, size_t &dataSize,
                         const char **log,
                         const std::vector<const char *> &compilerOptions) {
  if (log)
    *log = NULL;

  std::string *inputData, cu;
  std::string key = std::string(filename);
  std::map<std::string, std::string *>::iterator elem =
      g_sourceCache.map.find(key);

  if (elem == g_sourceCache.map.end()) {
    inputData = new std::string();

    getInputDataFromFile(*inputData, filename);
    g_sourceCache.map[key] = inputData;
  } else {
    inputData = elem->second;
  }
  dataSize = inputData->size();
  return inputData->c_str();
}