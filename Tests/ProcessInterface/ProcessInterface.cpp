#include <psProcess.hpp>

// needs to be derived for interacting particles, if only non-interacting can
// use base implementation with predefined sticky/non-sticky particles if
// interacting particles should be used, particle types may be defined as
// standalone classes or as subclasses of this class In any case, this class
// must hold references to particleType objects which are returned on a call to
// getParticleTypes()
// template <typename NumericType>
// class myModel : public psProcessModel<NumericType> {
//   // implementation of particle types and surface model here or somewhere
//   // outside the class std::vector<std::unique_ptr<rayParticle>>
//   // getParticleTypes() override
//   // {
//   // }
//   psSmartPointer<psSurfaceModel<NumericType>> getSurfaceModel() override {}
// };

int main() { return 0; }
