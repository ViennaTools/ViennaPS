#ifndef RT_TRACE_HPP
#define RT_TRACE_HPP

#include <iostream>
#include <rti/device.hpp>
#include <lsDomain.hpp>
#include <lsToDiskMesh.hpp>
#include <psSmartPointer.hpp>

using NumericType = float;

template <class particle, class reflection, int D>
class rtTrace
{
public:
    typedef psSmartPointer<rti::device<NumericType, particle, reflection>> rtDeviceType;

private:
    rtDeviceType rtiDevice = nullptr;

public:
    rtTrace()
    {
        rtiDevice = rtDeviceType::New();
    };

    rtTrace(lsSmartPointer<lsDomain<NumericType, D>> passedlsDomain,
            const NumericType discRadius)
    {
        rtiDevice = rtDeviceType::New();
        auto mesh = lsSmartPointer<lsMesh<NumericType>>::New();
        lsToDiskMesh<NumericType, D>(passedlsDomain, mesh).apply();
        auto points = mesh.get()->getNodes();
        auto normals = *mesh.get()->getVectorData("Normals");

        rtiDevice.get()->set_points(points);
        rtiDevice.get()->set_normals(normals);
        rtiDevice.get()->set_grid_spacing(discRadius);
        rtiDevice.get()->set_number_of_rays(points.size() * 100);
    }

    rtTrace(std::vector<NumericType> &points, std::vector<NumericType> &normals,
            const NumericType discRadius)
    {
        assert(points.size() == normals.size() && "Assumption");

        rtiDevice = rtDeviceType::New();
        rtiDevice.get()->set_points(points);
        rtiDevice.get()->set_normals(normals);
        rtiDevice.get()->set_grid_spacing(discRadius);
        rtiDevice.get()->set_number_of_rays(points.size() * 100);
    }

    void apply()
    {
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

    void setPoints(std::vector<NumericType> &points)
    {
        rtiDevice.get()->set_points(points);
    }

    void setNormals(std::vector<NumericType> &normals)
    {
        rtiDevice.get()->set_points(normals);
    }

    void setDiscRadius(const NumericType discRadius)
    {
        rtiDevice.get()->set_grid_spacing(discRadius);
    }

    void setPowerCosineDirection(const NumericType exp)
    {
        auto direction = rti::ray::power_cosine_direction_z<NumericType>{exp};
        rtiDevice.get()->set(direction);
    }

    void setNumOfRays(size_t num)
    {
        rtiDevice.get()->set_number_of_rays(num);
    }

    void setDomain(lsSmartPointer<lsDomain<NumericType, D>> passedlsDomain,
                   const NumericType discRadius)
    {
        auto mesh = lsSmartPointer<lsMesh<NumericType>>::New();
        lsToDiskMesh<NumericType, D>(passedlsDomain, mesh).apply();
        auto points = mesh.get()->getNodes();
        auto normals = *mesh.get()->getVectorData("Normals");

        rtiDevice.get()->set_points(points);
        rtiDevice.get()->set_normals(normals);
        rtiDevice.get()->set_grid_spacing(discRadius);
        rtiDevice.get()->set_number_of_rays(points.size() * 100);
    }
};

#endif // RT_TRACE_HPP