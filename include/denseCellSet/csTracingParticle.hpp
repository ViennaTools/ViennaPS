#pragma once

#include <rayRNG.hpp>
#include <rayReflection.hpp>
#include <rayUtil.hpp>

template <typename T> class csAbstractParticle {
public:
  virtual ~csAbstractParticle() = default;
  virtual std::unique_ptr<csAbstractParticle> clone() const = 0;

  virtual void initNew(rayRNG &Rng) = 0;

  virtual std::pair<T, rayTriple<T>> surfaceHit(const rayTriple<T> &rayDir,
                                                const rayTriple<T> &geomNormal,
                                                bool &reflect, rayRNG &Rng) = 0;
  virtual T getSourceDistributionPower() const = 0;
};

template <typename Derived, typename T>
class csParticle : public csAbstractParticle<T> {
public:
  std::unique_ptr<csAbstractParticle<T>> clone() const override final {
    return std::make_unique<Derived>(static_cast<Derived const &>(*this));
  }
  virtual void initNew(rayRNG &Rng) override {}
  virtual std::pair<T, rayTriple<T>> surfaceHit(const rayTriple<T> &rayDir,
                                                const rayTriple<T> &geomNormal,
                                                bool &reflect,
                                                rayRNG &Rng) override {
    reflect = false;
    return std::pair<T, rayTriple<T>>{1., rayTriple<T>{0., 0., 0.}};
  }
  virtual T getSourceDistributionPower() const override { return 1.; }

protected:
  // We make clear csParticle class needs to be inherited
  csParticle() = default;
  csParticle(const csParticle &) = default;
  csParticle(csParticle &&) = default;
};