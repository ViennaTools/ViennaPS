#ifndef PS_ADVECTION_CALLBACK
#define PS_ADVECTION_CALLBACK

#include <psDomain.hpp>
#include <psSmartPointer.hpp>

template <typename NumericType, int D> class psAdvectionCalback {
protected:
  psSmartPointer<psDomain<NumericType, D>> domain = nullptr;

public:
  void setDomain(psSmartPointer<psDomain<NumericType, D>> passedDomain) {
    domain = passedDomain;
  }

  virtual void applyPreAdvect(const NumericType processTime) {}

  virtual void applyPostAdvect(const NumericType advectionTime) {}
};

#endif