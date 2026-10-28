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
                                           const std::string &details = "",
                                           const std::string &species = "",
                                           const std::string &material = "") {
  const auto kindName = modelDataKindName(kind);
  const bool named = !species.empty() || !material.empty();
  std::ostringstream msg;

  if (named) {
    msg << "ViennaPS: the public model database has no " << kindName
        << " data for ";
    if (!species.empty())
      msg << species;
    if (!species.empty() && !material.empty())
      msg << " in ";
    if (!material.empty())
      msg << material;
    msg << ".";
  } else {
    msg << "ViennaPS: the " << kindName << " model database could not be read.";
  }
  if (!path.empty())
    msg << "\n  looked for: " << path;
  msg << "\n\n";
  if (!details.empty())
    msg << details << "\n\n";

  msg << "This is the public release of ViennaPS. Its bundled model database "
         "is generic and literature-based, and ships tables for a limited set "
         "of dopants and materials (the initial public release covers boron "
         "and phosphorus in silicon). Other dopants, materials, energies, or "
         "process conditions are not part of the public data set.\n\n";

  msg << "You can proceed by:\n";
  if (!named)
    msg << "  - making sure the model DB is installed and reachable (it ships "
           "under `ViennaPS/modeldb`; set `VIENNAPS_MODELDB_ROOT` to its "
           "location if it is not found automatically),\n";
  msg << "  - providing your own CSV table for this " << kindName
      << " and passing its path through the recipe/table-file parameter,\n"
      << "  - or using explicit, manually specified parameters instead of a "
         "table lookup (in the ion implantation example: explicit implant "
         "moments and manual anneal parameters in the config file, plus manual "
         "damage moments for a defect-coupled anneal run without the DB).\n\n";

  msg << "The extended, measurement-calibrated model database covers "
         "additional dopants, materials, and process ranges. To request it, or "
         "coverage for a specific dopant / material / process range, contact "
         "filipovic@iue.tuwien.ac.at with subject `ViennaPS Model Data "
         "Request` and include your affiliation and usage context.";
  return msg.str();
}

// Message for a request that is inside the covered species/material but outside
// the tabulated *range* (an energy beyond the grid, or a tilt away from the
// single tabulated channeling geometry). `details` is the specific range
// message from the table layer.
inline std::string outOfRangeModelDataMessage(const std::string &details) {
  std::ostringstream msg;
  msg << "ViennaPS: " << details << "\n\n"
      << "The public model database interpolates within the tabulated range "
         "but does not extrapolate, and it provides a single representative "
         "tilt/twist geometry. Choose a condition inside the tabulated range, "
         "provide your own CSV table, or use explicit implant moments in the "
         "config file instead of a table lookup.\n\n"
      << "The extended, measurement-calibrated model database covers a wider "
         "range of energies, tilts, and process conditions. To request it, "
         "contact filipovic@iue.tuwien.ac.at with subject `ViennaPS Model "
         "Data Request` and include your affiliation and usage context.";
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
