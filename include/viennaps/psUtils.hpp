#pragma once

#include <lsAdvect.hpp>
#include <lsDomain.hpp>
#include <rayBoundary.hpp>

#include <chrono>
#include <fstream>
#include <iostream>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>

namespace viennaps {

namespace utils {

// Checks if a string starts with a - or not
[[nodiscard]] inline bool isSigned(const std::string &s) {
  auto pos = s.find_first_not_of(' ');
  if (pos == s.npos)
    return false;
  if (s[pos] == '-')
    return true;
  return false;
}

[[nodiscard]] viennals::IntegrationSchemeEnum
convertIntegrationScheme(const std::string &s) {
  if (s == "ENGQUIST_OSHER_1ST_ORDER" || s == "EO_1")
    return viennals::IntegrationSchemeEnum::ENGQUIST_OSHER_1ST_ORDER;
  if (s == "ENGQUIST_OSHER_2ND_ORDER" || s == "EO_2")
    return viennals::IntegrationSchemeEnum::ENGQUIST_OSHER_2ND_ORDER;
  if (s == "LAX_FRIEDRICHS_1ST_ORDER" || s == "LF_1")
    return viennals::IntegrationSchemeEnum::LAX_FRIEDRICHS_1ST_ORDER;
  if (s == "LAX_FRIEDRICHS_2ND_ORDER" || s == "LF_2")
    return viennals::IntegrationSchemeEnum::LAX_FRIEDRICHS_2ND_ORDER;
  if (s == "LOCAL_LAX_FRIEDRICHS_ANALYTICAL_1ST_ORDER" || s == "LLFA_1")
    return viennals::IntegrationSchemeEnum::
        LOCAL_LAX_FRIEDRICHS_ANALYTICAL_1ST_ORDER;
  if (s == "LOCAL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER" || s == "LLLF_1")
    return viennals::IntegrationSchemeEnum::
        LOCAL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER;
  if (s == "LOCAL_LOCAL_LAX_FRIEDRICHS_2ND_ORDER" || s == "LLLF_2")
    return viennals::IntegrationSchemeEnum::
        LOCAL_LOCAL_LAX_FRIEDRICHS_2ND_ORDER;
  if (s == "LOCAL_LAX_FRIEDRICHS_1ST_ORDER" || s == "LLF_1")
    return viennals::IntegrationSchemeEnum::LOCAL_LAX_FRIEDRICHS_1ST_ORDER;
  if (s == "LOCAL_LAX_FRIEDRICHS_2ND_ORDER" || s == "LLF_2")
    return viennals::IntegrationSchemeEnum::LOCAL_LAX_FRIEDRICHS_2ND_ORDER;
  if (s == "STENCIL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER" || s == "SLLF_1")
    return viennals::IntegrationSchemeEnum::
        STENCIL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER;
  throw std::invalid_argument(
      "The value must be one of the following: "
      "ENGQUIST_OSHER_1ST_ORDER, ENGQUIST_OSHER_2ND_ORDER, "
      "LAX_FRIEDRICHS_1ST_ORDER, LAX_FRIEDRICHS_2ND_ORDER, "
      "LOCAL_LAX_FRIEDRICHS_ANALYTICAL_1ST_ORDER, "
      "LOCAL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER, "
      "LOCAL_LOCAL_LAX_FRIEDRICHS_2ND_ORDER, "
      "LOCAL_LAX_FRIEDRICHS_1ST_ORDER, "
      "LOCAL_LAX_FRIEDRICHS_2ND_ORDER, "
      "STENCIL_LOCAL_LAX_FRIEDRICHS_1ST_ORDER");
}

// Converts string to the given numeric datatype
template <typename T> [[nodiscard]] T convert(const std::string &s) {
  if constexpr (std::is_same_v<T, int>) {
    return std::stoi(s);
  } else if constexpr (std::is_same_v<T, unsigned int>) {
    if (isSigned(s))
      throw std::invalid_argument("The value must be unsigned");
    unsigned long int val = std::stoul(s);
    unsigned int num = static_cast<unsigned int>(val);
    if (val != num)
      throw std::out_of_range("The value is larger than the largest value "
                              "representable by `unsigned int`.");
    return num;
  } else if constexpr (std::is_same_v<T, long int>) {
    return std::stol(s);
  } else if constexpr (std::is_same_v<T, unsigned long int>) {
    if (isSigned(s))
      throw std::invalid_argument("The value must be unsigned");
    return std::stoul(s);
  } else if constexpr (std::is_same_v<T, long long int>) {
    return std::stoll(s);
  } else if constexpr (std::is_same_v<T, unsigned long long int>) {
    if (isSigned(s))
      throw std::invalid_argument("The value must be unsigned");
    return std::stoull(s);
  } else if constexpr (std::is_same_v<T, float>) {
    return std::stof(s);
  } else if constexpr (std::is_same_v<T, double>) {
    return std::stod(s);
  } else if constexpr (std::is_same_v<T, long double>) {
    return std::stold(s);
  } else if constexpr (std::is_same_v<T, std::string>) {
    return s;
  } else if constexpr (std::is_same_v<T, bool>) {
    if (s == "true")
      return true;
    if (s == "false")
      return false;
    throw std::invalid_argument("The value must be either 'true' or 'false'");
  } else if constexpr (std::is_same_v<T, viennals::IntegrationSchemeEnum>) {
    return convertIntegrationScheme(s);
  } else {
    // Throws a compile time error for all types but void
    return;
  }
}

// safeConvert wraps the convert function to catch exceptions. If an error
// occurs the default initialized value is returned.
template <typename T> std::optional<T> safeConvert(const std::string &s) {
  T value;
  try {
    value = convert<T>(s);
  } catch (std::exception &e) {
    std::cout << '\'' << s << "' couldn't be converted to type  '"
              << typeid(value).name() << "'\n";
    return std::nullopt;
  }
  return {value};
}

inline std::unordered_map<std::string, std::string>
parseConfigStream(std::istream &input) {
  // Regex to find trailing and leading whitespaces
  const auto wsRegex = std::regex("^ +| +$|( ) +");

  // Regular expression for extracting key and value separated by '=' as two
  // separate capture groups
  const auto keyValueRegex = std::regex(
      R"rgx([ \t]*([0-9a-zA-Z_\-\.+]+)[ \t]*=[ \t]*([0-9a-zA-Z_\-\.+]+).*$)rgx");

  // Reads a simple config file containing a single <key>=<value> pair per line
  // and returns the content as an unordered map
  std::unordered_map<std::string, std::string> paramMap;
  std::string line;
  while (std::getline(input, line)) {
    // Remove trailing and leading whitespaces
    line = std::regex_replace(line, wsRegex, "$1");
    // Skip this line if it is marked as a comment
    if (line.rfind('#') == 0 || line.empty())
      continue;

    // Extract key and value
    std::smatch smatch;
    if (std::regex_search(line, smatch, keyValueRegex)) {
      if (smatch.size() < 3) {
        std::cerr << "Malformed line:\n " << line;
        continue;
      }

      paramMap.insert({smatch[1], smatch[2]});
    }
  }
  return paramMap;
}

// Opens a file and forwards its stream to the config stream parser.
inline std::unordered_map<std::string, std::string>
readFile(const std::string &filename) {
  std::ifstream f(filename);
  if (!f.is_open()) {
    std::cout << "Failed to open config file '" << filename << "'\n";
    return {};
  }
  return parseConfigStream(f);
}

// Class that can be used during the assigning process of a param map to the
// param struct
template <typename K, typename V, typename C = decltype(&convert<V>)>
class Item {
private:
  C conv;

public:
  K key;
  V &value;

  Item(K key_, V &value_) : key(key_), value(value_), conv(&convert<V>) {}

  Item(K key_, V &value_, C conv_) : key(key_), value(value_), conv(conv_) {}

  void operator()(const std::string &k) {
    try {
      value = conv(k);
    } catch (std::exception &e) {
      std::cout << '\'' << k << "' couldn't be converted to type of parameter '"
                << key << "'\n";
    }
  }
};

// If the key is found in the unordered_map, then the
template <typename K, typename V, typename C>
void AssignItems(std::unordered_map<std::string, std::string> &map,
                 Item<K, V, C> &&item) {
  if (auto it = map.find(item.key); it != map.end()) {
    item(it->second);
    // Remove the item from the map, since it is now 'consumed'.
    map.erase(it);
  } else {
    std::cout << "Couldn't find '" << item.key
              << "' in parameter file. Using default value instead.\n";
  }
}

// Peels off items from parameter pack
template <typename K, typename V, typename C, typename... ARGS>
void AssignItems(std::unordered_map<std::string, std::string> &map,
                 Item<K, V, C> &&item, ARGS &&...args) {
  AssignItems(map, std::forward<Item<K, V, C>>(item));
  AssignItems(map, std::forward<ARGS>(args)...);
}

template <class NumericType, std::size_t D>
std::string arrayToString(const std::array<NumericType, D> arr) {
  std::stringstream arrayStr;
  arrayStr << "[";
  for (std::size_t i = 0; i < D - 1; i++) {
    arrayStr << arr[i] << ", ";
  }
  arrayStr << arr[D - 1] << "]";
  return arrayStr.str();
}

struct Parameters {
  std::unordered_map<std::string, std::string> m;

  void readConfigFile(const std::string &fileName) { m = readFile(fileName); }

  bool exists(const std::string &key) const { return m.find(key) != m.end(); }

  template <typename T = double>
  [[nodiscard]] T get(const std::string &key) const {
    if (!exists(key)) {
      std::cout << "Key not found in parameters: " << key << std::endl;
      exit(1);
      return T();
    }
    return convert<T>(m.at(key));
  }
};

[[nodiscard]] inline viennaray::BoundaryCondition convertBoundaryCondition(
    viennals::BoundaryConditionEnum originalBoundaryCondition) {
  switch (originalBoundaryCondition) {
  case viennals::BoundaryConditionEnum::REFLECTIVE_BOUNDARY:
    return viennaray::BoundaryCondition::REFLECTIVE;

  case viennals::BoundaryConditionEnum::INFINITE_BOUNDARY:
    return viennaray::BoundaryCondition::IGNORE;

  case viennals::BoundaryConditionEnum::PERIODIC_BOUNDARY:
    return viennaray::BoundaryCondition::PERIODIC;

  case viennals::BoundaryConditionEnum::POS_INFINITE_BOUNDARY:
    return viennaray::BoundaryCondition::IGNORE;

  case viennals::BoundaryConditionEnum::NEG_INFINITE_BOUNDARY:
    return viennaray::BoundaryCondition::IGNORE;
  }
  return viennaray::BoundaryCondition::IGNORE;
}

inline std::string boolString(const int in) {
  return in == 0 ? "false" : "true";
}

}; // namespace utils
} // namespace viennaps
