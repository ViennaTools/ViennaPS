#ifndef CELL_BASE_HPP
#define CELL_BASE_HPP

#include <istream>
#include <ostream>
#include <vector>

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
  using MaterialFractionType = std::vector<std::pair<unsigned, float>>;

private:
  MaterialFractionType materialFractions;

public:
  // cellBase() {
  //   materialFractions.push_back(std::make_pair(0, 1.0));
  // }

  cellBase(float fillingFraction = 0.0, unsigned baseMaterial = 0) {
    materialFractions.push_back(std::make_pair(baseMaterial, fillingFraction));
  }

  cellBase(const cellBase &passedCell) {
    materialFractions = passedCell.materialFractions;
  }

  const MaterialFractionType &getMaterialFractions() const { return materialFractions; }

  MaterialFractionType &getMaterialFractions() { return materialFractions; }

  void setMaterialFractions(MaterialFractionType passedMaterialFractions) {
    materialFractions = passedMaterialFractions;
  }

  virtual bool operator==(cellBase passedCell) {
    for(unsigned i = 0; i < materialFractions.size(); ++i) {
      if(materialFractions[i].first != passedCell.materialFractions[i].first) {
        return false;
      }
      if(materialFractions[i].second != passedCell.materialFractions[i].second) {
        return false;
      }
    }
    return true;
  }

  virtual std::ostream &serialize(std::ostream &s) {
    s.write(reinterpret_cast<const char *>(&materialFractions),
            sizeof(materialFractions));
    return s;
  }

  virtual std::istream &deserialize(std::istream &s) {
    s.read(reinterpret_cast<char *>(&materialFractions), sizeof(materialFractions));
    return s;
  }

  virtual ~cellBase(){};
};

template <class S> S &operator<<(S &s, const cellBase &cell) {
  s << "materialFractions: ";
  const auto &fractions = cell.getMaterialFractions();
  for(auto& f : fractions) {
    s << "[" << f.first << ": " << f.second << "], ";
  }
  return s;
}

#endif // CELL_BASE_HPP
