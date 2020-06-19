#pragma once

#include <hrleVectorType.hpp>
#include <lsVelocityField.hpp>

#include <psParticleTypeBase.hpp>

/**
  This base class sets the interface for all models, which
  generate a velocity either geometric parameters or
  ray tracing results.
*/

class psModelBase : lsVelocityField<T> {
public:
  // is used to find whether a ray tracer is needed
  virtual constexpr bool isRayTracing() { return true; }

  // the ray tracer calls this to set how a particle should move
  virtual void generateParticle(ParticleTypeBase *particle){};

  // the ray tracer calls this when the particle collides with the surface
  virtual void collideParticle(ParticleTypeBase *particle){};

  // the ray tracer calls this when a particle should be reflected
  virtual void reflectParticle(ParticleTypeBase *particle){};
};
