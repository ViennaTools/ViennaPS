#pragma once

#include <type_traits>

#include <lsGeometricAdvect.hpp>
#include <lsGeometricAdvectDistributions.hpp>

#include "../process/psGeometricModel.hpp"
#include "../process/psProcessModel.hpp"

namespace viennaps {

using namespace viennacore;

template <typename NumericType, int D>
class SphereDistribution : public ProcessModelCPU<NumericType, D> {
  using LSPtr = SmartPointer<viennals::Domain<NumericType, D>>;

public:
  SphereDistribution(NumericType radius, LSPtr mask = nullptr) {
    auto dist = SmartPointer<
        viennals::SphereDistribution<viennahrle::CoordType, D>>::New(radius);

    auto geomModel =
        SmartPointer<GeometricModel<NumericType, D>>::New(dist, mask);

    this->setGeometricModel(geomModel);
    this->setProcessName("SphereDistribution");
    this->processMetaData["Radius"] = std::vector<double>{radius};
  }
};

template <typename NumericType, int D>
class BoxDistribution : public ProcessModelCPU<NumericType, D> {
  using LSPtr = SmartPointer<viennals::Domain<NumericType, D>>;

public:
  BoxDistribution(const std::array<viennahrle::CoordType, 3> &halfAxes,
                  LSPtr mask = nullptr) {
    auto dist =
        SmartPointer<viennals::BoxDistribution<viennahrle::CoordType, D>>::New(
            halfAxes);

    auto geomModel =
        SmartPointer<GeometricModel<NumericType, D>>::New(dist, mask);

    this->setGeometricModel(geomModel);
    this->setProcessName("BoxDistribution");
    this->processMetaData["HalfAxes"] = std::vector<double>{
        static_cast<double>(halfAxes[0]), static_cast<double>(halfAxes[1]),
        static_cast<double>(halfAxes[2])};
  }
};

template <typename NumericType, int D>
class CustomSphereDistribution : public ProcessModelCPU<NumericType, D> {
  using LSPtr = SmartPointer<viennals::Domain<NumericType, D>>;

public:
  CustomSphereDistribution(const std::vector<NumericType> &radii,
                           LSPtr mask = nullptr) {
    auto dist =
        SmartPointer<viennals::CustomSphereDistribution<viennahrle::CoordType,
                                                        D>>::New(radii);

    auto geomModel =
        SmartPointer<GeometricModel<NumericType, D>>::New(dist, mask);

    this->setGeometricModel(geomModel);
    this->setProcessName("CustomSphereDistribution");
  }
};

namespace impl {

template <class T, int D>
class TrenchDistribution : public viennals::GeometricAdvectDistribution<T, D> {

  const T trenchWidth_;
  const T trenchDepth_;

  const T rate_;
  const T bottomMed_;
  const T a_, b_, n_;

  T gridDelta_;

public:
  TrenchDistribution(const T trenchWidth, const T trenchDepth, const T rate,
                     const T bottomMed = 1.0, const T a = 1.0, const T b = 1.0,
                     const T n = 1.0)
      : trenchWidth_(trenchWidth), trenchDepth_(trenchDepth), rate_(rate),
        bottomMed_(bottomMed), a_(a), b_(b), n_(n) {}

  T getSignedDistance(const Vec3D<viennahrle::CoordType> &initial,
                      const Vec3D<viennahrle::CoordType> &candidate,
                      unsigned long pointId) const override {
    T distance = std::numeric_limits<T>::max();
    Vec3D<viennahrle::CoordType> v{};
    for (unsigned i = 0; i < D; ++i) {
      v[i] = candidate[i] - initial[i];
    }

    T radius = 0;
    // if (std::abs(initial[D - 1]) < gridDelta_) {
    // radius = rate_;
    // } else
    if (std::abs(initial[D - 1] + trenchDepth_) < gridDelta_) {
      radius = bottomMed_;
    } else {
      radius =
          a_ * std::pow(1. - std::abs(initial[D - 1]) / trenchDepth_, n_) + b_;
      // radius = a_ * std::exp(-n_ * std::abs(initial[D - 1])) + b_;
    }

    if (std::abs(radius) <= gridDelta_) {
      distance =
          std::max(std::max(std::abs(v[0]), std::abs(v[1])), std::abs(v[2])) -
          std::abs(radius);
    } else {
      for (unsigned i = 0; i < D; ++i) {
        T y = (v[(i + 1) % D]);
        T z = 0;
        if constexpr (D == 3)
          z = (v[(i + 2) % D]);
        T x = radius * radius - y * y - z * z;
        if (x < 0.)
          continue;
        T dirRadius = std::abs(v[i]) - std::sqrt(x);
        if (std::abs(dirRadius) < std::abs(distance))
          distance = dirRadius;
      }
    }
    // return distance;
    if (radius < 0) {
      return -distance;
    } else {
      return distance;
    }
  }

  std::array<viennahrle::CoordType, 6> getBounds() const override {
    std::array<viennahrle::CoordType, 6> bounds = {};
    for (unsigned i = 0; i < D; ++i) {
      bounds[2 * i] = -rate_;
      bounds[2 * i + 1] = rate_;
    }
    return bounds;
  }

  bool useSurfacePointId() const override { return true; }

  void prepare(SmartPointer<viennals::Domain<T, D>> domain) override {
    gridDelta_ = domain->getGrid().getGridDelta();
  }
};
} // namespace impl

template <typename NumericType, int D>
class GeometricTrenchDeposition : public ProcessModelCPU<NumericType, D> {
  using LSPtr = SmartPointer<viennals::Domain<NumericType, D>>;

public:
  GeometricTrenchDeposition(NumericType trenchWidth, NumericType trenchDepth,
                            NumericType rate, NumericType bottomMed = 1.0,
                            NumericType a = 1.0, NumericType b = 1.0,
                            NumericType n = 1.0) {
    auto dist =
        SmartPointer<impl::TrenchDistribution<viennahrle::CoordType, D>>::New(
            trenchWidth, trenchDepth, rate, bottomMed, a, b, n);

    auto geomModel = SmartPointer<GeometricModel<NumericType, D>>::New(dist);

    this->setGeometricModel(geomModel);
    this->setProcessName("GeometricTrenchDeposition");
  }
};

PS_PRECOMPILE_PRECISION_DIMENSION(SphereDistribution)
PS_PRECOMPILE_PRECISION_DIMENSION(BoxDistribution)
PS_PRECOMPILE_PRECISION_DIMENSION(CustomSphereDistribution)
PS_PRECOMPILE_PRECISION_DIMENSION(GeometricTrenchDeposition)

} // namespace viennaps
