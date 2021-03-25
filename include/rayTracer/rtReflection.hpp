#ifndef RT_REFLECTION_HPP
#define RT_REFLECTION_HPP

#include <iostream>
#include <psSmartPointer.hpp>
#include <rti/device.hpp>

using NumericTypeReflect = float;

typedef rti::reflection::diffuse<NumericTypeReflect> rtDiffuseReflection;
typedef rti::reflection::specular<NumericTypeReflect> rtSpecularReflection;

class rtCustomReflection
    : public rti::reflection::i_reflection<NumericTypeReflect> {
public:
  rti::util::pair<rti::util::triple<NumericTypeReflect>>
  use(RTCRay &rayin, RTCHit &hitin,
      rti::geo::meta_geometry<NumericTypeReflect> &geometry,
      rti::rng::i_rng &rng, rti::rng::i_rng::i_state &rngstate) {
    // Use `rng.get(rngstate)` to aquire a random number.
    // This way of drawing random numbers is compatible with RTI's parallel
    // execution and Monte Carlo simulation.
    auto rndm = ((double)rng.get(rngstate)) / rng.max(); // random in [0,1]

    // Incoming ray direction
    auto indir = std::array<NumericTypeReflect, 3>{rayin.dir_x, rayin.dir_y,
                                                   rayin.dir_z};
    rti::util::inv(indir);
    // Surface normal
    auto normal = geometry.get_normal(hitin.primID);
    auto cosphi = rti::util::dot_product(indir, normal) /
                  rti::util::length_of_vec(indir) /
                  rti::util::length_of_vec(normal);

    if (cosphi < thrshld || cosphi < rndm) {
      return diffuse.use(rayin, hitin, geometry, rng, rngstate);
    }
    return specular.use(rayin, hitin, geometry, rng, rngstate);
  }

private:
  NumericTypeReflect thrshld = 0.5;
  rtDiffuseReflection diffuse;
  rtSpecularReflection specular;
};

#endif // RT_REFLECTION_HPP