#ifndef PS_GEOMETRIC_MODEL_HPP
#define PS_GEOMETRIC_MODEL_HPP

#include <psDomain.hpp>
#include <psSmartPointer.hpp>

template <typename NumericType, int D> class psGeometricModel {
protected:
  psSmartPointer<psDomain<NumericType, D>> domain = nullptr;

public:
  void setDomain(psSmartPointer<psDomain<NumericType, D>> passedDomain) {
    domain = passedDomain;
  }

  virtual void apply(){};
};

#endif