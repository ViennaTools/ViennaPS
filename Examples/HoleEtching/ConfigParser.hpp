#pragma once

#include <fstream>
#include <iostream>
#include <istream>
#include <regex>
#include <string>
#include <unordered_map>

template <class NumericType>
std::unordered_map<std::string, NumericType> parseConfig(std::string filename) {
  std::ifstream f(filename);
  if (!f.is_open()) {
    std::cout << "Failed to open config file" << std::endl;
    return {};
  }
  const std::string regexPattern =
      "^([a-zA-Z_]+)[\\ \\t]*=[\\ \\t]*([0-9e\\.\\-\\+]+)";
  const std::regex rgx(regexPattern);

  std::unordered_map<std::string, NumericType> paramMap;
  std::string line;
  while (std::getline(f, line)) {
    std::smatch smatch;
    if (std::regex_search(line, smatch, rgx)) {
      if (smatch.size() < 3) {
        std::cerr << "Malformed line:\n " << line;
        continue;
      }

      NumericType value;
      try {
        value = std::stof(smatch[2]);
      } catch (std::exception e) {
        std::cerr << "Error parsing value in line:\n " << line;
        continue;
      }
      paramMap.insert({smatch[1], value});
    }
  }
  return paramMap;
}
