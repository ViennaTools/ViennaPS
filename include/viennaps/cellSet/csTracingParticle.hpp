#pragma once

#include "csUtil.hpp"

#include <rayRNG.hpp>
#include <vcVectorUtil.hpp>

namespace viennacs {

using namespace viennacore;

template <typename T> class AbstractParticle {
public:
  virtual ~AbstractParticle() = default;
  virtual std::unique_ptr<AbstractParticle> clone() const = 0;

  virtual void initNew(viennaray::RNG &Rng) = 0;

  virtual std::pair<T, Triple<T>> surfaceHit(const Triple<T> &rayDir,
                                             const Triple<T> &geomNormal,
                                             bool &reflect,
                                             viennaray::RNG &Rng) = 0;
  virtual T getSourceDistributionPower() const = 0;
  virtual csPair<T> getMeanFreePath() const = 0;
  virtual T collision(csVolumeParticle<T> &particle, viennaray::RNG &RNG,
                      std::vector<csVolumeParticle<T>> &particleStack) = 0;
};

template <typename Derived, typename T>
class Particle : public AbstractParticle<T> {
public:
  std::unique_ptr<AbstractParticle<T>> clone() const override final {
    return std::make_unique<Derived>(static_cast<Derived const &>(*this));
  }
  virtual void initNew(viennaray::RNG &Rng) override {}
  virtual std::pair<T, Triple<T>> surfaceHit(const Triple<T> &rayDir,
                                             const Triple<T> &geomNormal,
                                             bool &reflect,
                                             viennaray::RNG &Rng) override {
    reflect = false;
    return std::pair<T, Triple<T>>{1., Triple<T>{0., 0., 0.}};
  }
  virtual T getSourceDistributionPower() const override { return 1.; }
  virtual csPair<T> getMeanFreePath() const override { return {1., 1.}; }
  virtual T
  collision(csVolumeParticle<T> &particle, viennaray::RNG &RNG,
            std::vector<csVolumeParticle<T>> &particleStack) override {
    return 0.;
  }

protected:
  // We make clear Particle class needs to be inherited
  Particle() = default;
  Particle(const Particle &) = default;
  Particle(Particle &&) = default;
};

} // namespace viennacs
