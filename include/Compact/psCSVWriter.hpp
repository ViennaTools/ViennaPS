#ifndef PS_CSV_WRITER_HPP
#define PS_CSV_WRITER_HPP

#include <array>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <string>
#include <vector>

#include <psSmartPointer.hpp>

// A simple CSV writer class
template <class NumericType, int NumCols> class psCSVWriter {

  std::string filename;
  std::ofstream file;
  std::string header;
  bool initialized = false;

  template <class Iterator>
  static std::string join(Iterator begin, Iterator end,
                          const std::string &separator = ",") {
    std::ostringstream ostr;
    if (begin != end)
      ostr << *begin++;
    while (begin != end)
      ostr << separator << *begin++;
    return ostr.str();
  }

public:
  psCSVWriter() {}
  psCSVWriter(std::string passedFilename, std::string passedHeader = "")
      : filename(passedFilename), header(passedHeader) {}

  bool initialize() {
    if (filename.empty()) {
      std::cout << "No filename provided!" << std::endl;
      return false;
    }
    if (file.is_open())
      return false;
    file.open(filename);
    if (file.is_open()) {
      if (!header.empty()) {
        std::string line;
        std::istringstream iss(header);
        while (std::getline(iss, line)) {
          file << "# " << line << '\n';
        }
      }
    } else {
      return false;
    }
    initialized = true;
    return true;
  }

  void setFilename(std::string passedFilename) { filename = passedFilename; }

  void setHeader(std::string passedHeader) { header = passedHeader; }

  void writeRow(const std::array<NumericType, NumCols> &data) {
    if (!initialized)
      if (!initialize())
        return;

    if (file.is_open())
      file << join(data.cbegin(), data.cend()) << '\n';
  }

  void writeRow(const std::vector<NumericType> &data) {
    if (!initialized)
      if (!initialize())
        return;

    if (data.size() != NumCols) {
      std::cout << "Unexpected number of items in the provided row!\n";
      return;
    }
    if (!file.is_open()) {
      std::cout << "Couldn't open file `" << filename << "`\n";
      return;
    }
    file << join(data.cbegin(), data.cend()) << '\n';
  }

  void writeRow(std::initializer_list<NumericType> data) {
    if (!initialized)
      if (!initialize())
        return;

    if (data.size() != NumCols) {
      std::cout << "Unexpected number of items in the provided row!\n";
      return;
    }
    if (!file.is_open()) {
      std::cout << "Couldn't open file `" << filename << "`\n";
      return;
    }
    file << join(data.begin(), data.end()) << '\n';
  }

  void flush() {
    if (file.is_open())
      file.flush();
  }

  ~psCSVWriter() {
    if (file.is_open())
      file.close();
  }
};
#endif
