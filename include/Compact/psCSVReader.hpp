#ifndef PS_CSV_READER_HPP
#define PS_CSV_READER_HPP

#include <array>
#include <fstream>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <psSmartPointer.hpp>
#include <psUtils.hpp>

// Simple class for reading CSV files
template <class NumericType, int NumCols, bool continueOnError = false>
class psCSVReader {
  // Regex to find trailing and leading whitespaces
  const std::regex wsRegex = std::regex("^ +| +$|( ) +");

  std::string filename;
  int offset = 0;
  char delimiter = ',';

public:
  psCSVReader() {}
  psCSVReader(std::string passedFilename, int passedOffset = 0,
              char passedDelimiter = ',')
      : filename(passedFilename), offset(passedOffset),
        delimiter(passedDelimiter) {}

  void setFilename(std::string passedFilename) { filename = passedFilename; }

  void setOffset(int passedOffset) { offset = passedOffset; }

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
      std::cout << "Couldn't open file '" << filename << "'\n";
      return std::nullopt;
    }
    return {header};
  }

  psSmartPointer<std::vector<std::array<NumericType, NumCols>>> apply() {
    std::ifstream file(filename);
    if (file.is_open()) {
      auto data =
          psSmartPointer<std::vector<std::array<NumericType, NumCols>>>::New();

      std::string line;
      int lineCount = 0;

      // Iterate over each line
      while (std::getline(file, line)) {
        ++lineCount;
        if (lineCount <= offset)
          continue;

        // Remove trailing and leading whitespaces
        line = std::regex_replace(line, wsRegex, "$1");
        // Skip this line if it is marked as a comment
        if (line.rfind('#') == 0)
          continue;

        std::istringstream iss(line);
        std::string tmp;
        std::array<NumericType, NumCols> a{0};
        int i = 0;
        while (std::getline(iss, tmp, delimiter) && i < NumCols) {
          auto v = psUtils::safeConvert<NumericType>(tmp);
          if (v.has_value())
            a[i] = v.value();
          else {
            std::cout << "Error while reading line " << lineCount - 1 << " in '"
                      << filename << "'\n";
            if constexpr (!continueOnError) {
              return nullptr;
            }
          }
          ++i;
        }
        if (i != NumCols) {
          std::cout << "Invalid number of columns in line " << lineCount - 1
                    << " in '" << filename << "'\n";
          if constexpr (!continueOnError)
            return nullptr;
        }
        data->push_back(a);
      }
      file.close();
      return data;
    } else {
      std::cout << "Couldn't open file '" << filename << "'\n";
      return nullptr;
    }
  }
};

#endif