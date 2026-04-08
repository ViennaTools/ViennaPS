#pragma once

#include "psMaterialDBEntry.hpp"
#include "psMaterialMap.hpp"
#include "psMaterialValueMap.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace viennaps {

class MaterialDataBase {
public:
  using Entry = materials::DBEntry;
  using Json = nlohmann::json;

  static constexpr int kSchemaVersion = 1;

  [[nodiscard]] bool hasEntry(Material material) const {
    return materialData_.has(material);
  }

  [[nodiscard]] Entry getEntry(Material material) const {
    return materialData_.get(material);
  }

  void setEntry(Material material, const Entry &entry) {
    materialData_.set(material, entry);
  }

  void clear() { materialData_.clearAll(); }

  void writeToFile(const std::filesystem::path &filePath) const {
    Json root;
    root["schemaVersion"] = kSchemaVersion;
    root["generatedBy"] = "ViennaPS";

    Json materialsJson = Json::object();
    // Serialize all entries, including built-in materials without user-provided
    // data
    for (const auto &[material, entry] : materialData_) {
      materialsJson[MaterialMap::toString(material)] = entry.serialize();
    }
    root["materials"] = std::move(materialsJson);

    if (const auto parent = filePath.parent_path(); !parent.empty()) {
      std::filesystem::create_directories(parent);
    }

    const auto tmpPath = filePath.string() + ".tmp";
    {
      std::ofstream out(tmpPath, std::ios::trunc);
      if (!out.good()) {
        throw std::runtime_error("Failed to open material database file for "
                                 "writing: " +
                                 tmpPath);
      }
      out << root.dump(2) << '\n';
    }

    std::error_code error;
    std::filesystem::rename(tmpPath, filePath, error);
    if (error) {
      std::filesystem::remove(filePath, error);
      error.clear();
      std::filesystem::rename(tmpPath, filePath, error);
      if (error) {
        std::filesystem::remove(tmpPath, error);
        throw std::runtime_error("Failed to finalize material database file: " +
                                 filePath.string());
      }
    }
  }

  void readFromFile(const std::filesystem::path &filePath) {
    std::ifstream in(filePath);
    if (!in.good()) {
      throw std::runtime_error("Failed to open material database file for "
                               "reading: " +
                               filePath.string());
    }

    Json root;
    try {
      in >> root;
    } catch (const std::exception &error) {
      throw std::runtime_error("Failed to parse material database JSON: " +
                               std::string(error.what()));
    }

    const auto schemaVersion = requireInt(root, "schemaVersion", "root");
    if (schemaVersion != kSchemaVersion) {
      throw std::runtime_error("Unsupported material database schemaVersion: " +
                               std::to_string(schemaVersion));
    }

    if (!root.contains("materials") || !root["materials"].is_object()) {
      throw std::runtime_error(
          "Material database JSON must contain object field 'materials'.");
    }

    materialData_.clearAll();
    const auto &materials = root["materials"];
    for (auto it = materials.begin(); it != materials.end(); ++it) {
      const auto &materialName = it.key();
      const auto material = MaterialMap::fromString(materialName);
      const auto entry = Entry::deserializeEntry(it.value(), materialName);
      materialData_.set(material, entry);
    }
  }

private:
  static int requireInt(const Json &node, const char *field,
                        const std::string &path) {
    if (!node.contains(field) || !node[field].is_number_integer()) {
      throw std::runtime_error("Expected integer field '" + std::string(field) +
                               "' at " + path + ".");
    }
    return node[field].get<int>();
  }

  MaterialValueMap<Entry> materialData_;
};

} // namespace viennaps