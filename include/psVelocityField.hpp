#ifndef PS_VELOCITY_FIELD
#define PS_VELOCITY_FIELD

#include <psSmartPointer.hpp>
#include <vector>

template <typename NumericType>
class psVelocityField
{
private:
  psSmartPointer<std::vector<NumericType>> velocities = nullptr;

public:
  psVelocityField() {}

  NumericType getScalarVelocity(const std::array<NumericType, 3> &coordinate,
                                int material,
                                const std::array<NumericType, 3> &normalVector,
                                unsigned long pointId)
  {
    return 0;
  }

  std::array<NumericType, 3>
  getVectorVelocity(const std::array<NumericType, 3> &coordinate, int material,
                    const std::array<NumericType, 3> &normalVector,
                    unsigned long pointId)
  {
    return {0., 0., 0.};
  }

  NumericType getDissipationAlpha(int direction, int material,
                                  const std::array<NumericType, 3> &centralDifferences)
  {
    return 0;
  }
};

#endif