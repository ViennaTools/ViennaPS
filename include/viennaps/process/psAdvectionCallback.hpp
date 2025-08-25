#pragma once

#include "../psDomain.hpp"

namespace viennaps {

using namespace viennacore;

template <typename NumericType, int D> class AdvectionCallback {
protected:
  SmartPointer<Domain<NumericType, D>> domain = nullptr;

public:
  virtual ~AdvectionCallback() = default;

  void setDomain(SmartPointer<Domain<NumericType, D>> passedDomain) {
    domain = passedDomain;
  }

  virtual bool applyPreAdvect(const NumericType processTime) { return true; }

  virtual bool applyPostAdvect(const NumericType advectionTime) { return true; }
};

} // namespace viennaps
