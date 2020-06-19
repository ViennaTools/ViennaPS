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
  double fillingFraction = 0.;

public:
  cellBase() {}

  cellBase(const cellBase &passedCell) {
    fillingFraction = passedCell.fillingFraction;
  }

  double getFillingFraction() const { return fillingFraction; }

  void setFillingFraction(double passedFillingFraction) {
    fillingFraction = passedFillingFraction;
  }

  virtual bool operator==(cellBase passedCell) {
    return fillingFraction == passedCell.fillingFraction;
  }

  virtual std::ostream &serialize(std::ostream &s) {
    s.write(reinterpret_cast<const char *>(&fillingFraction),
            sizeof(fillingFraction));
    return s;
  }

  virtual std::istream &deserialize(std::istream &s) {
    s.read(reinterpret_cast<char *>(&fillingFraction), sizeof(fillingFraction));
    return s;
  }

  virtual ~cellBase(){};
};

template <class S> S &operator<<(S &s, const cellBase &cell) {
  s << "fillingFraction: " << cell.getFillingFraction();
  return s;
}

#endif // CELL_BASE_HPP
