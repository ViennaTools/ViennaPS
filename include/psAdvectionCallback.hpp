#ifndef PS_ADVECTION_CALLBACK
#define PS_ADVECTION_CALLBACK

#include <psDomain.hpp>
#include <psSmartPointer.hpp>

template <typename NumericType, int D> class psAdvectionCallback {
protected:
  psSmartPointer<psDomain<NumericType, D>> domain = nullptr;

public:
  void setDomain(psSmartPointer<psDomain<NumericType, D>> passedDomain) {
    domain = passedDomain;
  }

  virtual bool applyPreAdvect(const NumericType processTime) { return true; }

  virtual bool applyPostAdvect(const NumericType advectionTime) { return true; }
};

#endif