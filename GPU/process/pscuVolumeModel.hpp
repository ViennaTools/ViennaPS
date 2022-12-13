#pragma once

#include "csTracing.hpp"

#include "psDomain.hpp"

#include "psSmartPointer.hpp"

template <typename NumericType> class pscuVolumeModel {
protected:
  psSmartPointer<psDomain<NumericType, 3>> domain;
  csTracing<NumericType, 3> tracer;

public:
  void setDomain(psSmartPointer<psDomain<NumericType, 3>> passedDomain) {
    domain = passedDomain;
  }

  virtual void applyPreAdvect(const NumericType processTime) {}
  virtual void applyPostAdvect(const NumericType advectionTime) {}
};
