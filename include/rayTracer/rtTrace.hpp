#ifndef RT_TRACE_HPP
#define RT_TRACE_HPP

#include <iostream>
#include <lsDomain.hpp>
#include <lsToDiskMesh.hpp>
#include <psSmartPointer.hpp>
#include <rti/device.hpp>

using NumericType = float;

template <class particle, class reflection, int D>
class rtTrace
{
public:
    typedef psSmartPointer<rti::device<NumericType, particle, reflection>>
        rtDeviceType;

    /// Enumeration for the different types of
    /// reflections supported by rtTrace
    // enum struct lsReflectionEnum : unsigned
    // {
    //     DIFFUSE = 0,
    //     SPECULAR = 1,
    // };

private:
    rtDeviceType rtiDevice = nullptr;
    lsSmartPointer<lsDomain<NumericType, D>> domain = nullptr;
    NumericType discRadius;
    size_t numberOfRaysMult = 100;

public:
    rtTrace() { rtiDevice = rtDeviceType::New(); };

    rtTrace(lsSmartPointer<lsDomain<NumericType, D>> passedlsDomain,
            const NumericType _discRadius) : domain(passedlsDomain), discRadius(_discRadius)
    {
        rtiDevice = rtDeviceType::New();
    }

    void apply()
    {
        {
            auto mesh = lsSmartPointer<lsMesh<NumericType>>::New();
            lsToDiskMesh<NumericType, D>(domain, mesh).apply();
            auto points = mesh.get()->getNodes();
            auto normals = *mesh.get()->getVectorData("Normals");

            rtiDevice.get()->set_points(points);
            rtiDevice.get()->set_normals(normals);
            rtiDevice.get()->set_grid_spacing(discRadius);
            rtiDevice.get()->set_number_of_rays(numberOfRaysMult*points.size());
        }

        rtiDevice.get()->run();
    }

    std::vector<NumericType> getMcEstimates()
    {
        return rtiDevice.get()->get_mc_estimates();
    }

    std::vector<NumericType> getHitCounts()
    {
        return rtiDevice.get()->get_hit_counts();
    }

    void setDiscRadius(const NumericType _discRadius)
    {
        discRadius = _discRadius;
    }

    void setPowerCosineDirection(const NumericType exp)
    {
        auto direction = rti::ray::power_cosine_direction_z<NumericType>{exp};
        rtiDevice.get()->set(direction);
    }

    void setNumberOfRays(size_t num) { numberOfRaysMult = num; }

    void setDomain(lsSmartPointer<lsDomain<NumericType, D>> passedlsDomain,
                   const NumericType _discRadius)
    {
        domain = passedlsDomain;
        discRadius = _discRadius;
    }
};

#endif // RT_TRACE_HPP