#ifndef RT_PARTICLE_HPP
#define RT_PARTICLE_HPP

#include <rti/particle/i_particle.hpp>

using NumericType = float;

class rtParticle1 : public rti::particle::i_particle<NumericType> {
public:
  NumericType
  get_sticking_probability(RTCRay &rayin, RTCHit &hitin,
                           rti::geo::meta_geometry<NumericType> &geometry,
                           rti::rng::i_rng &rng,
                           rti::rng::i_rng::i_state &rngstate) override final {
    // return the sticking probability for this hit
    return 0.01;
  }

  void init_new() override final {}
};

class rtParticle2 : public rti::particle::i_particle<NumericType> {
public:
  NumericType
  get_sticking_probability(RTCRay &rayin, RTCHit &hitin,
                           rti::geo::meta_geometry<NumericType> &geometry,
                           rti::rng::i_rng &rng,
                           rti::rng::i_rng::i_state &rngstate) override final {
    // return the sticking probability for this hit
    ++hitCounter;
    // do something with energy
    return 0.1;
  }

  void init_new() override final { totalEnergy = 1.; }

private:
  unsigned hitCounter = 0;
  NumericType totalEnergy;
};

#endif // RT_PARTICLE_HPP