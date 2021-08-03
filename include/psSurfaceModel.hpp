#ifndef PS_SURFACE_MODEL
#define PS_SURFACE_MODEL

#include <vector>

template <typename NumericType> class psSurfaceModel {
private:
  std::vector<std::vector<NumericType>> Coverages;

public:
  std::vector<NumericType>
  calculateVelocities(std::vector<std::vector<NumericType>> &Rates,
                      std::vector<NumericType> &materialIDs) {
    return std::vector<NumericType>{};
  }
  std::shared_ptr<std::vector<NumericType>> getCoverages() {
    return std::make_shared<std::vector<std::vector<NumericType>>>();
  }

  void setTotalFluxes() {} // Is it needed for setting the coverages

  // TO DO: figure out how to initialize coverages
  void setCoverages(unsigned numPoints, NumericType value) {
    for (auto &i : Coverages) {
    }
  }
};

#endif