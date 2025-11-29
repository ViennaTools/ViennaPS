#pragma once

#include <vcLogger.hpp>
#include <vcUtil.hpp>

#include <fstream>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace viennaps {

using namespace viennacore;

// Simple class for reading CSV files
template <class NumericType> class CSVReader {
  // Regex to find trailing and leading whitespaces
  const std::regex wsRegex = std::regex("^ +| +$|( ) +");

  std::string filename;
  char delimiter = ',';
  int numCols = 0;

public:
  CSVReader() {}
  CSVReader(std::string passedFilename, char passedDelimiter = ',')
      : filename(passedFilename), delimiter(passedDelimiter) {}

  void setFilename(std::string passedFilename) { filename = passedFilename; }

  void setDelimiter(char passedDelimiter) { delimiter = passedDelimiter; }

  std::optional<std::string> readHeader() {
    std::ifstream file(filename);
    std::string header;
    if (file.is_open()) {
      std::string line;
      // Iterate over each line
      while (std::getline(file, line)) {
        // Remove trailing and leading whitespaces
        line = std::regex_replace(line, wsRegex, "$1");

        // Skip empty lines at the top of the file
        if (line.empty())
          continue;

        // If the line is marked as comment and it is located before any data
        // at the top of the file, add it to the header string. Otherwise return
        // the header string, since we are now reading data.
        if (line.rfind('#') == 0) {
          header += '\n' + line;
        } else {
          file.close();
          break;
        }
      }
    } else {
      VIENNACORE_LOG_WARNING("Couldn't open file '" + filename + "'");
      return {};
    }
    return {header};
  }

  std::optional<std::vector<std::vector<NumericType>>> readContent() {
    std::ifstream file(filename);
    if (file.is_open()) {
      auto data = std::vector<std::vector<NumericType>>();

      std::string line;
      int lineCount = 0;

      // Iterate over each line
      while (std::getline(file, line)) {
        ++lineCount;

        // Remove trailing and leading whitespaces
        line = std::regex_replace(line, wsRegex, "$1");
        // Skip this line if it is marked as a comment
        if (line.rfind('#') == 0)
          continue;

        std::istringstream iss(line);
        std::string tmp;
        std::vector<NumericType> a;
        int i = 0;
        while (std::getline(iss, tmp, delimiter)) {
          auto valueOpt = util::safeConvert<NumericType>(tmp);
          if (valueOpt)
            a.push_back(valueOpt.value());
          else {
            VIENNACORE_LOG_WARNING("Error while reading line " +
                                   std::to_string(lineCount - 1) + " in '" +
                                   filename + "'");
            return {};
          }
          ++i;
        }

        // The first row of actual data determins the data dimension
        if (numCols == 0)
          numCols = i;

        if (i != numCols) {
          VIENNACORE_LOG_WARNING("Invalid number of columns in line " +
                                 std::to_string(lineCount - 1) + " in '" +
                                 filename + "'");
          return {};
        }
        data.push_back(a);
      }
      file.close();
      return data;
    } else {
      VIENNACORE_LOG_WARNING("Couldn't open file '" + filename + "'");
      return {};
    }
  }
};

} // namespace viennaps
