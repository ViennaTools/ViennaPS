#ifndef VELOCITY_FIELD_HPP
#define VELOCITY_FIELD_HPP

#include <lsSmartPointer.hpp>
#include <psVelocityField.hpp>
#include <unordered_map>
#include <vector>
#include <cmath>

template <class T>
class velocityField : public psVelocityField<T>
{
public:
    velocityField() {}

    T getScalarVelocity(const std::array<T, 3> & /*coordinate*/, int material,
                        const std::array<T, 3> & /*normalVector*/,
                        unsigned long pointID) override
    {
        if (material != 0)
        {
            if (auto velId = this->getVelocityId(pointID); velId != -1)
            {
                return this->getVelocity(velId);
            }
            else
            {
                return 0.;
            }
        }
        else
        {
            return 0.;
        }
    }
};

#endif // RT_VELOCITY_FIELD_HPP