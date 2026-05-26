#pragma once

#include "models/psModelNames.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace viennaps {

inline std::string &modelDbRootStorage() {
  static std::string s;
  return s;
}

inline void setModelDbRoot(const std::string &path) {
  modelDbRootStorage() = path;
}

inline const std::string &getModelDbRoot() { return modelDbRootStorage(); }

inline void initModelDbRoot() {
#ifdef VIENNAPS_MODELDB_DIR
  if (modelDbRootStorage().empty())
    setModelDbRoot(VIENNAPS_MODELDB_DIR);
#endif
  if (const char *dbRoot = std::getenv("VIENNAPS_MODELDB_ROOT")) {
    if (dbRoot[0] != '\0')
      setModelDbRoot(dbRoot);
  }
}

namespace modeldb {

class ModelDbError final : public std::runtime_error {
public:
  explicit ModelDbError(const std::string &message)
      : std::runtime_error(message) {}
};

enum class ModelDataKind { Implant, Damage, Anneal };

inline std::string modelDataKindName(const ModelDataKind kind) {
  switch (kind) {
  case ModelDataKind::Implant:
    return "implant";
  case ModelDataKind::Damage:
    return "damage";
  case ModelDataKind::Anneal:
    return "anneal";
  }
  return "model";
}

inline std::string missingModelDataMessage(const ModelDataKind kind,
                                           const std::string &path,
                                           const std::string &details = "") {
  std::ostringstream msg;
  msg << "ViennaPS model data is missing";
  if (!path.empty())
    msg << ":\n  " << path;
  msg << "\n\n";
  if (!details.empty())
    msg << details << "\n\n";
  msg << "This " << modelDataKindName(kind)
      << " model is table/model-DB driven, but the private model DB is not "
         "available at the configured path.\n\n"
      << "You can fix this in one of these ways:\n"
      << "  1. Install the private ViennaPS model DB at `ViennaPS/modeldb`, "
         "or set `VIENNAPS_MODELDB_ROOT` to its location.\n"
      << "  2. Provide your own CSV table and pass that path through the "
         "corresponding recipe/table-file parameter.\n"
      << "  3. Provide calibrated parameters manually instead of using table "
         "lookup. In the ion implantation examples this means using explicit "
         "implant moments and manual anneal parameters in the config file; "
         "damage also needs manual damage moments if defect-coupled anneal is "
         "used without the DB.\n\n"
      << "To request access to the private model DB, contact "
         "filipovic@iue.tuwien.ac.at with subject `ViennaPS Model Data "
         "Request` and include your affiliation, usage context, and the "
         "dopant/material/process range you need.";
  return msg.str();
}

inline int reportModelDbError(const std::exception &error,
                              std::ostream &out = std::cerr) {
  std::cout.flush();
  out << "\nViennaPS model data error\n"
      << "-------------------------\n"
      << error.what() << "\n";
  return 2;
}

template <typename Fn> int runWithModelDbErrors(Fn &&fn) {
  try {
    return fn();
  } catch (const ModelDbError &error) {
    return reportModelDbError(error);
  }
}

using viennaps::model::canonicalMaterialName;
using viennaps::model::canonicalMaterialToken;
using viennaps::model::canonicalSpeciesName;
using viennaps::model::canonicalSpeciesToken;
using viennaps::model::lower;
using viennaps::model::trim;

} // namespace modeldb

} // namespace viennaps
