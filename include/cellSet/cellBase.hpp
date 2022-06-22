#ifndef CELL_BASE_HPP
#define CELL_BASE_HPP

#include <istream>
#include <ostream>
#include <unordered_map>

/// Base class for a cell describing a volume voxel
/// of the simulation domain. It implements the
/// filling fraction and overloads the stream operators
/// >> and <<. If the concrete cell implementation
/// has data which should be output in text format,
/// these operators should be overloaded and call the
/// respective operators of this base class. Data which
/// will be exported to binary formats should
/// be returned by a call to "std::ostream serialize(std::ostream)"
/// which should also call serialize of this base class.
class cellBase {
public:
  using MaterialFractionType = std::unordered_map<unsigned, float>;

private:
  MaterialFractionType materialFractions;

public:
  cellBase() {}

  cellBase(float fillingFraction, unsigned baseMaterial = 0) {
    materialFractions.insert(std::make_pair(baseMaterial, fillingFraction));
  }

  cellBase(const cellBase &passedCell) {
    materialFractions = passedCell.materialFractions;
  }

  /// Only used internally to initialize a cell
  void setInitialFillingFraction(float fillingFraction,
                                 unsigned baseMaterial = 0) {
    materialFractions.insert(std::make_pair(baseMaterial, fillingFraction));
  }

  /// Return the filling fractions of each material, which
  /// is stored as an std::map<unsigned, float>
  const MaterialFractionType &getMaterialFractions() const {
    return materialFractions;
  }

  /// Return the filling fractions of each material, which
  /// is stored as an std::unordered_map<unsigned, float>
  MaterialFractionType &getMaterialFractions() { return materialFractions; }

  /// Set a MaterialFractionType to describe the filling fractions of this cell
  void setMaterialFractions(MaterialFractionType passedMaterialFractions) {
    materialFractions = passedMaterialFractions;
  }

  /// Compare all fililng fractions of two cells. Only returns true if all of
  /// them are the same.
  virtual bool operator==(cellBase passedCell) {
    // for (unsigned i = 0; i < materialFractions.size(); ++i) {
    //   if (materialFractions[i].first !=
    //   passedCell.materialFractions[i].first) {
    //     return false;
    //   }
    //   if (std::abs(materialFractions[i].second -
    //       passedCell.materialFractions[i].second) > 1e-6) {
    //     return false;
    //   }
    // }
    // return true;
    return materialFractions == passedCell.materialFractions;
  }

  /// Serialize this cell into a binary stream.
  virtual std::ostream &serialize(std::ostream &s) {
    s.write(reinterpret_cast<const char *>(&materialFractions),
            sizeof(materialFractions));
    return s;
  }

  /// Deserialize this cell from a binary stream.
  virtual std::istream &deserialize(std::istream &s) {
    s.read(reinterpret_cast<char *>(&materialFractions),
           sizeof(materialFractions));
    return s;
  }

  virtual ~cellBase(){};
};

/// Write this cell to a character stream (e.g. stdout)
std::ostream &operator<<(std::ostream &s, const cellBase &cell) {
  s << "materialFractions: ";
  const auto &fractions = cell.getMaterialFractions();
  for (auto &f : fractions) {
    s << "[" << f.first << ": " << f.second << "], ";
  }
  return s;
}

#endif // CELL_BASE_HPP
