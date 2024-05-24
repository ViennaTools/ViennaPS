#pragma once

#include "psDomain.hpp"

#include <vcSmartPointer.hpp>

namespace viennaps {

using namespace viennacore;

template <typename NumericType, int D> class GeometricModel {
protected:
  SmartPointer<Domain<NumericType, D>> domain = nullptr;

public:
  virtual ~GeometricModel() = default;

  void setDomain(SmartPointer<Domain<NumericType, D>> passedDomain) {
    domain = passedDomain;
  }

  virtual void apply() {}
};

} // namespace viennaps
