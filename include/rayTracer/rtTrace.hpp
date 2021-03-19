#ifndef RT_TRACE_HPP
#define RT_TRACE_HPP

#include <iostream>
#include <lsDomain.hpp>
#include <lsToDiskMesh.hpp>
#include <psSmartPointer.hpp>
#include <rti/device.hpp>

using NumericType = float;

/**
    This class represent a ray tracer, simulating particle interactions
    with the domain surface.
*/
template <class particle, class reflection, int D> class rtTrace {
public:
  typedef psSmartPointer<rti::device<NumericType, particle, reflection>>
      rtDeviceType;

  enum struct rtBoundary : unsigned {
    REFLECTIVE = 0,
    PERIODIC = 1,
  };

private:
  rtDeviceType rtiDevice = nullptr;
  lsSmartPointer<lsDomain<NumericType, D>> domain = nullptr;
  size_t numberOfRaysMult = 1000;

public:
  rtTrace() { rtiDevice = rtDeviceType::New(); };

  rtTrace(lsSmartPointer<lsDomain<NumericType, D>> passedlsDomain,
          const NumericType discRadius)
      : domain(passedlsDomain) {
    rtiDevice = rtDeviceType::New();
    rtiDevice.get()->set_grid_spacing(discRadius);
  }

  /// Extract the mesh points and normals from the LS domain
  /// and then run the ray tracer
  void apply() {
    {
      auto mesh = lsSmartPointer<lsMesh<NumericType>>::New();
      lsToDiskMesh<NumericType, D>(domain, mesh).apply();
      auto points = mesh.get()->getNodes();
      auto normals = *mesh.get()->getVectorData("Normals");

      rtiDevice.get()->set_points(points);
      rtiDevice.get()->set_normals(normals);
      rtiDevice.get()->set_number_of_rays(numberOfRaysMult * points.size());
    }

    rtiDevice.get()->run();
  }

  // Returns the particle hit counts for each grid point
  std::vector<size_t> getHitCounts() { return rtiDevice.get()->get_hit_cnts(); }

  /// Returns the particle hit counts for each grid point
  /// normalized to the overall maximal hit count
  std::vector<NumericType> getMcEstimates() {
    return rtiDevice.get()->get_mc_estimates();
  }

  void setDiscRadius(const NumericType discRadius) {
    rtiDevice.get()->set_grid_spacing(discRadius);
  }

  void setPowerCosineDirection(const NumericType exp) {
    auto direction = rti::ray::power_cosine_direction_z<NumericType>{exp};
    rtiDevice.get()->set(direction);
  }

  void setNumberOfRays(size_t num) { numberOfRaysMult = num; }

  void setDomain(lsSmartPointer<lsDomain<NumericType, D>> passedlsDomain,
                 const NumericType discRadius) {
    domain = passedlsDomain;
    rtiDevice.get()->set_grid_spacing(discRadius);
  }

  void setBoundaryX(rtBoundary bound) {
    switch (bound) {
    case rtBoundary::REFLECTIVE:
      rtiDevice.get()->set_x(rti::geo::bound_condition::REFLECTIVE);
      break;
    case rtBoundary::PERIODIC:
      rtiDevice.get()->set_x(rti::geo::bound_condition::PERIODIC);
      break;
    default:
      std::cout << "Invalid boundary condition" << std::endl;
      break;
    }
  }

  void setBoundaryY(rtBoundary bound) {
    switch (bound) {
    case rtBoundary::REFLECTIVE:
      rtiDevice.get()->set_y(rti::geo::bound_condition::REFLECTIVE);
      break;
    case rtBoundary::PERIODIC:
      rtiDevice.get()->set_y(rti::geo::bound_condition::PERIODIC);
      break;
    default:
      std::cout << "Invalid boundary condition" << std::endl;
      break;
    }
  }
};

#endif // RT_TRACE_HPP