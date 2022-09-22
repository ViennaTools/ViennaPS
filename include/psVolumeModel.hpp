#ifndef PS_VOLUME_MODEL
#define PS_VOLUME_MODEL

#include <csTracing.hpp>
#include <psDomain.hpp>
#include <psSmartPointer.hpp>

template <typename NumericType, int D> class psVolumeModel {
protected:
  psSmartPointer<psDomain<NumericType, D>> domain;
  csTracing<NumericType, D> tracer;

public:
  void setDomain(psSmartPointer<psDomain<NumericType, 3>> passedDomain) {
    domain = passedDomain;
  }

  virtual void applyPreAdvect(const NumericType processTime) {}
  virtual void applyPostAdvect(const NumericType advectionTime) {}
};

#endif