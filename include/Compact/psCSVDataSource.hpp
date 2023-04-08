#ifndef PS_CSV_DATA_SOURCE_HPP
#define PS_CSV_DATA_SOURCE_HPP

#include <array>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <psCSVReader.hpp>
#include <psCSVWriter.hpp>
#include <psDataSource.hpp>
#include <psSmartPointer.hpp>
#include <psUtils.hpp>

template <typename NumericType>
class psCSVDataSource : public psDataSource<NumericType> {
  using Parent = psDataSource<NumericType>;

  using Parent::namedParameters;
  using Parent::positionalParameters;

  psCSVReader<NumericType> reader;
  psCSVWriter<NumericType> writer;

  std::string header;

  bool parametersInitialized = false;

  static void
  processPositionalParam(const std::string &input,
                         std::vector<NumericType> &positionalParameters) {
    // Positional parameter
    auto v = psUtils::safeConvert<NumericType>(input);
    if (v.has_value())
      positionalParameters.push_back(v.value());
    else {
      psLogger::getInstance()
          .addWarning("Error while converting parameter '" + input +
                      "' to numeric type.")
          .print();
    }
  }

  static void processNamedParam(
      const std::string &input,
      std::unordered_map<std::string, NumericType> &namedParameters) {
    const std::string keyValueRegex =
        R"rgx(^[ \t]*([0-9a-zA-Z_-]+)[ \t]*=[ \t]*([0-9a-zA-Z_\-\.+]+)[ \t]*$)rgx";
    const std::regex rgx(keyValueRegex);

    std::smatch smatch;
    if (std::regex_search(input, smatch, rgx) && smatch.size() == 3) {
      auto v = psUtils::safeConvert<NumericType>(smatch[2]);
      if (v.has_value())
        namedParameters.insert({smatch[1], v.value()});
      else {
        psLogger::getInstance()
            .addWarning("Error while converting value of parameter '" +
                        std::string(smatch[1]) + "'")
            .print();
      }
    } else {
      psLogger::getInstance()
          .addWarning("Error while parsing parameter line '" + input + "'")
          .print();
    }
  }

  static void processParamLine(
      const std::string &line, std::vector<NumericType> &positionalParameters,
      std::unordered_map<std::string, NumericType> &namedParameters) {
    std::istringstream iss(line);
    std::string tmp;

    // Skip the '#!' characters
    char c;
    iss >> c >> c;

    // Split the string at commas
    while (std::getline(iss, tmp, ',')) {
      if (tmp.find('=') == std::string::npos) {
        processPositionalParam(tmp, positionalParameters);
      } else {
        processNamedParam(tmp, namedParameters);
      }
    }
  }

  void processHeader() {
    auto opt = reader.readHeader();
    if (opt.has_value()) {
      header = opt.value();
      std::istringstream hdr(header);
      std::string line;

      // Go over each comment line
      while (std::getline(hdr, line)) {
        // Check if the line is marked as a parameter line
        if (line.rfind("#!") == 0) {
          processParamLine(line, positionalParameters, namedParameters);
        }
      }
      parametersInitialized = true;
    }
  }

public:
  using typename Parent::ItemType;
  using typename Parent::VectorType;

  psCSVDataSource() {}

  psCSVDataSource(std::string passedFilename) {
    reader.setFilename(passedFilename);
    writer.setFilename(passedFilename);
  }

  void setFilename(std::string passedFilename) {
    reader.setFilename(passedFilename);
    writer.setFilename(passedFilename);
  }

  void setHeader(const std::string &passedHeader) { header = passedHeader; }

  VectorType read() override {
    auto opt = reader.readHeader();
    header = opt.value_or("");
    auto contentOpt = reader.readContent();
    if (contentOpt)
      return contentOpt.value();
    return {};
  }

  bool write(const VectorType &data) override {
    std::string extendedHeader = header;
    if (!positionalParameters.empty())
      extendedHeader +=
          "!" + join(positionalParameters.begin(), positionalParameters.end()) +
          "\n";

    if (!namedParameters.empty()) {
      std::vector<std::string> np;
      for (const auto &[key, value] : namedParameters)
        np.push_back(key + "=" + std::to_string(value));

      extendedHeader += "!" + join(np.begin(), np.end()) + "\n";
    }

    writer.setHeader(extendedHeader);
    writer.initialize();
    for (auto &row : data)
      if (!writer.writeRow(row))
        return false;

    return true;
  }

  std::vector<NumericType> getPositionalParameters() override {
    if (!parametersInitialized)
      processHeader();
    return positionalParameters;
  }

  std::unordered_map<std::string, NumericType> getNamedParameters() override {
    if (!parametersInitialized)
      processHeader();
    return namedParameters;
  }
};

#endif