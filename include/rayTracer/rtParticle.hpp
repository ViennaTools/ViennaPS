#ifndef RT_PARTICLE_HPP
#define RT_PARTICLE_HPP

#include <rti/particle/i_particle.hpp>

using NumericTypePart = float;

class rtParticle1 : public rti::particle::i_particle<NumericTypePart> {
public:
  NumericTypePart
  get_sticking_probability(RTCRay &rayin, RTCHit &hitin,
                           rti::geo::meta_geometry<NumericTypePart> &geometry,
                           rti::rng::i_rng &rng,
                           rti::rng::i_rng::i_state &rngstate) override final {
    // return the sticking probability for this hit
    return 0.01;
  }

  void init_new() override final {}
};

class rtParticle2 : public rti::particle::i_particle<NumericTypePart> {
public:
  NumericTypePart
  get_sticking_probability(RTCRay &rayin, RTCHit &hitin,
                           rti::geo::meta_geometry<NumericTypePart> &geometry,
                           rti::rng::i_rng &rng,
                           rti::rng::i_rng::i_state &rngstate) override final {
    // return the sticking probability for this hit

    // do something with energy
    totalEnergy += 0.01;
    return totalEnergy;
  }

  void init_new() override final { totalEnergy = 0.01; }

private:
  NumericTypePart totalEnergy;
};

#endif // RT_PARTICLE_HPP