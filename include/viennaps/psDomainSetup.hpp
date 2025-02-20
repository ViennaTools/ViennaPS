#pragma once

#include "psUtils.hpp"
#include <hrleGrid.hpp>
#include <vcLogger.hpp>

namespace viennaps {

using namespace viennacore;
using BoundaryType = hrleBoundaryType;

template <class NumericType, int D> class DomainSetup {
  NumericType gridDelta_;
  double bounds_[2 * D];
  BoundaryType boundaryCons_[D];
  hrleGrid<D> grid_;

public:
  DomainSetup() : gridDelta_(0.0) {
    for (int i = 0; i < D; i++) {
      bounds_[2 * i] = 0.0;
      bounds_[2 * i + 1] = 0.0;
      boundaryCons_[i] = BoundaryType::INFINITE_BOUNDARY;
    }
  }

  DomainSetup(NumericType gridDelta, NumericType xExtent, NumericType yExtent,
              BoundaryType boundary = BoundaryType::REFLECTIVE_BOUNDARY)
      : gridDelta_(gridDelta) {
    if (xExtent <= 0.0) {
      Logger::getInstance()
          .addWarning("Invalid 'x' extent for domain setup.")
          .print();
    }

    bounds_[0] = -xExtent / 2.;
    bounds_[1] = xExtent / 2.;

    if constexpr (D == 3) {
      if (yExtent <= 0.0) {
        Logger::getInstance()
            .addWarning("Invalid 'y' extent for domain setup.")
            .print();
      }
      bounds_[2] = -yExtent / 2.;
      bounds_[3] = yExtent / 2.;
      bounds_[4] = -gridDelta;
      bounds_[5] = gridDelta;
    } else {
      bounds_[2] = -gridDelta;
      bounds_[3] = gridDelta;
    }

    for (int i = 0; i < D - 1; i++) {
      boundaryCons_[i] = boundary;
    }
    boundaryCons_[D - 1] = BoundaryType::INFINITE_BOUNDARY;
    init();
  }

  auto &grid() const { return grid_; }

  NumericType gridDelta() const { return gridDelta_; }

  std::array<double, 2 * D> bounds() const {
    std::array<double, 2 * D> boundsArray;
    for (int i = 0; i < 2 * D; i++) {
      boundsArray[i] = bounds_[i];
    }
    return boundsArray;
  }

  std::array<BoundaryType, D> boundaryCons() const {
    std::array<BoundaryType, D> boundaryConsArray;
    for (int i = 0; i < D; i++) {
      boundaryConsArray[i] = boundaryCons_[i];
    }
    return boundaryConsArray;
  }

  NumericType xExtent() const { return bounds_[1] - bounds_[0]; }

  NumericType yExtent() const { return bounds_[3] - bounds_[2]; }

  bool hasPeriodicBoundary() const {
    return boundaryCons_[0] == BoundaryType::PERIODIC_BOUNDARY ||
           boundaryCons_[1] == BoundaryType::PERIODIC_BOUNDARY;
  }

  bool isValid() const {
    return gridDelta_ > 0.0 && xExtent() > 0.0 &&
           (D == 2 || (D == 3 && yExtent() > 0.0));
  }

  void print() const {
    std::cout << "Domain setup:" << std::endl;
    std::cout << "\tGrid delta: " << gridDelta_ << std::endl;
    std::cout << "\tX extent: " << xExtent() << std::endl;
    if constexpr (D == 3)
      std::cout << "\tY extent: " << yExtent() << std::endl;
    std::cout << "\tPeriodic boundary: "
              << utils::boolString(hasPeriodicBoundary()) << std::endl;
  }

  void halveXAxis() {
    assert(isValid());
    if (hasPeriodicBoundary()) {
      Logger::getInstance()
          .addWarning("Half geometry cannot be created with "
                      "periodic boundaries!")
          .print();
    } else {
      bounds_[0] = 0.0;
      init();
    }
  }

  void halveYAxis() {
    assert(isValid());
    if (hasPeriodicBoundary()) {
      Logger::getInstance()
          .addWarning("Half geometry cannot be created with "
                      "periodic boundaries!")
          .print();
    } else {
      bounds_[2] = 0.0;
      init();
    }
  }

  void init() {
    assert(isValid());
    hrleIndexType gridMin[D], gridMax[D];
    for (unsigned i = 0; i < D; ++i) {
      gridMin[i] = std::floor(bounds_[2 * i] / gridDelta_);
      gridMax[i] = std::ceil(bounds_[2 * i + 1] / gridDelta_);
    }
    grid_ = hrleGrid<D>(gridMin, gridMax, gridDelta_, boundaryCons_);
  }

  void init(hrleGrid<D> grid) {
    gridDelta_ = grid.getGridDelta();
    for (int i = 0; i < D; i++) {
      bounds_[2 * i] = grid.getMinBounds(i) * gridDelta_;
      bounds_[2 * i + 1] = grid.getMaxBounds(i) * gridDelta_;
      boundaryCons_[i] = grid.getBoundaryConditions(i);
    }
    grid_ = grid;
    assert(isValid());
  }
};

} // namespace viennaps