#pragma once

#include "psPreCompileMacros.hpp"
#include "psUtil.hpp"

#include <hrleGrid.hpp>
#include <vcLogger.hpp>

namespace viennaps {

using namespace viennacore;
using BoundaryType = viennahrle::BoundaryType;

template <int D> class DomainSetup {
  double gridDelta_;
  double bounds_[2 * D] = {0.};
  BoundaryType boundaryCons_[D] = {};
  viennahrle::Grid<D> grid_;

public:
  DomainSetup() : gridDelta_(0.0) {
    for (int i = 0; i < D; i++) {
      bounds_[2 * i] = 0.0;
      bounds_[2 * i + 1] = 0.0;
      boundaryCons_[i] = BoundaryType::INFINITE_BOUNDARY;
    }
  }

  DomainSetup(double bounds[2 * D], BoundaryType boundaryCons[D],
              double gridDelta)
      : gridDelta_(gridDelta) {
    for (int i = 0; i < D; i++) {
      bounds_[2 * i] = bounds[2 * i];
      bounds_[2 * i + 1] = bounds[2 * i + 1];
      boundaryCons_[i] = boundaryCons[i];
    }
    init();
  }

  DomainSetup(double gridDelta, double xExtent, double yExtent,
              BoundaryType boundary = BoundaryType::REFLECTIVE_BOUNDARY)
      : gridDelta_(gridDelta) {
    if (xExtent <= 0.0) {
      VIENNACORE_LOG_ERROR("Invalid 'x' extent for domain setup.");
    }

    bounds_[0] = -xExtent / 2.;
    bounds_[1] = xExtent / 2.;

    if constexpr (D == 3) {
      if (yExtent <= 0.0) {
        VIENNACORE_LOG_ERROR("Invalid 'y' extent for domain setup.");
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

  auto &grid() const {
    check();
    return grid_;
  }

  double gridDelta() const {
    check();
    return gridDelta_;
  }

  std::array<double, 2 * D> bounds() const {
    check();
    std::array<double, 2 * D> boundsArray;
    for (int i = 0; i < 2 * D; i++) {
      boundsArray[i] = bounds_[i];
    }
    return boundsArray;
  }

  std::array<BoundaryType, D> boundaryCons() const {
    check();
    std::array<BoundaryType, D> boundaryConsArray;
    for (int i = 0; i < D; i++) {
      boundaryConsArray[i] = boundaryCons_[i];
    }
    return boundaryConsArray;
  }

  double xExtent() const { return bounds_[1] - bounds_[0]; }

  double yExtent() const { return bounds_[3] - bounds_[2]; }

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
              << util::boolString(hasPeriodicBoundary()) << std::endl;
  }

  void halveXAxis() {
    check();
    if (hasPeriodicBoundary()) {
      VIENNACORE_LOG_WARNING(
          "Half geometry cannot be created with periodic boundaries.");
    } else {
      bounds_[0] = 0.0;
      init();
    }
  }

  void halveYAxis() {
    check();
    if (hasPeriodicBoundary()) {
      VIENNACORE_LOG_WARNING(
          "Half geometry cannot be created with periodic boundaries.");
    } else {
      bounds_[2] = 0.0;
      init();
    }
  }

  void init() {
    check();
    viennahrle::IndexType gridMin[D], gridMax[D];
    for (unsigned i = 0; i < D; ++i) {
      gridMin[i] = std::floor(bounds_[2 * i] / gridDelta_);
      gridMax[i] = std::ceil(bounds_[2 * i + 1] / gridDelta_);
    }
    grid_ = viennahrle::Grid<D>(gridMin, gridMax, gridDelta_, boundaryCons_);
  }

  void init(viennahrle::Grid<D> grid) {
    gridDelta_ = grid.getGridDelta();
    for (int i = 0; i < D; i++) {
      bounds_[2 * i] = grid.getMinBounds(i) * gridDelta_;
      bounds_[2 * i + 1] = grid.getMaxBounds(i) * gridDelta_;
      boundaryCons_[i] = grid.getBoundaryConditions(i);
    }
    grid_ = grid;
  }

  void check() const {
    if (!isValid()) {
      print();
      VIENNACORE_LOG_ERROR("Domain setup is not correctly initialized.");
    }
  }
};

} // namespace viennaps