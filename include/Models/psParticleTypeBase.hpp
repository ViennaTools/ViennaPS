#pragma once

class psParticleTypeBase : public alex_rti_particle_type {
public:
  hrleVectorType<double, 3> position;
  hrleVectorType<double, 3> direction;
};
