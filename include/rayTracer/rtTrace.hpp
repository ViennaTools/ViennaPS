#ifndef RT_TRACE_HPP
#define RT_TRACE_HPP

#include <iostream>
#include <lsDomain.hpp>
#include <lsToDiskMesh.hpp>
#include <psSmartPointer.hpp>
#include <rtVelocityField.hpp>
#include <rti/device.hpp>

enum struct rtTraceBoundary : unsigned {
  REFLECTIVE = 0,
  PERIODIC = 1,
};

/**
    This class represent a ray tracer, simulating particle interactions
    with the domain surface.
*/
template <class particle, class reflection, int D> class rtTrace {
public:
  using numeric_type = float;
  typedef psSmartPointer<rti::device<numeric_type, particle, reflection>>
      rtDeviceType;

private:
  rtDeviceType rtiDevice = nullptr;
  lsSmartPointer<lsDomain<numeric_type, D>> domain = nullptr;
  lsSmartPointer<rtVelocityField<numeric_type>> velocityField = nullptr;
  size_t numberOfRaysMult = 1000;

public:
  rtTrace() {
    rtiDevice = rtDeviceType::New();
    velocityField = lsSmartPointer<rtVelocityField<numeric_type>>::New();
  };

  rtTrace(lsSmartPointer<lsDomain<numeric_type, D>> passedlsDomain,
          const numeric_type discRadius)
      : domain(passedlsDomain) {
    rtiDevice = rtDeviceType::New();
    rtiDevice->set_grid_spacing(discRadius);
    velocityField = lsSmartPointer<rtVelocityField<numeric_type>>::New();
  }

  /// Extract the mesh points, normals and pointId translator from the LS domain
  /// and then run the ray tracer
  void apply() {
    {
      auto translator = lsSmartPointer<
          std::unordered_map<unsigned long, unsigned long>>::New();
      auto mesh = lsSmartPointer<lsMesh<numeric_type>>::New();
      lsToDiskMesh<numeric_type, D>(domain, mesh, translator).apply();
      auto points = mesh->getNodes();
      auto normals = *mesh->getVectorData("Normals");
      velocityField->setTranslator(translator);

      rtiDevice->set_points(points);
      rtiDevice->set_normals(normals);
      rtiDevice->set_number_of_rays(numberOfRaysMult * points.size());
    }

    rtiDevice->run();
    auto mcestimates = lsSmartPointer<std::vector<numeric_type>>::New(
        rtiDevice->get_mc_estimates());
    velocityField->setMcEstimates(mcestimates);
  }

  // Returns the particle hit counts for each grid point
  std::vector<size_t> getHitCounts() { return rtiDevice.get()->get_hit_cnts(); }

  // Returns the particle hit counts for each grid point
  // normalized to the overall maximal hit count
  std::vector<numeric_type> getMcEstimates() {
    return rtiDevice->get_mc_estimates();
  }

  // Return the velocity field needed for advection
  lsSmartPointer<rtVelocityField<numeric_type>> getVelocityField() {
    return velocityField;
  }

  void setDiscRadius(const numeric_type discRadius) {
    rtiDevice->set_grid_spacing(discRadius);
  }

  void setPowerCosineDirection(const numeric_type exp) {
    auto direction = rti::ray::power_cosine_direction_z<numeric_type>{exp};
    rtiDevice->set(direction);
  }

  void setNumberOfRays(size_t num) { numberOfRaysMult = num; }

  void setDomain(lsSmartPointer<lsDomain<numeric_type, D>> passedlsDomain,
                 const numeric_type discRadius) {
    domain = passedlsDomain;
    rtiDevice->set_grid_spacing(discRadius);
  }

  void setBoundaryX(rtTraceBoundary bound) {
    switch (bound) {
    case rtTraceBoundary::REFLECTIVE:
      rtiDevice->set_x(rti::geo::bound_condition::REFLECTIVE);
      break;
    case rtTraceBoundary::PERIODIC:
      rtiDevice->set_x(rti::geo::bound_condition::PERIODIC);
      break;
    default:
      std::cout << "Invalid boundary condition" << std::endl;
      break;
    }
  }

  void setBoundaryY(rtTraceBoundary bound) {
    switch (bound) {
    case rtTraceBoundary::REFLECTIVE:
      rtiDevice->set_y(rti::geo::bound_condition::REFLECTIVE);
      break;
    case rtTraceBoundary::PERIODIC:
      rtiDevice->set_y(rti::geo::bound_condition::PERIODIC);
      break;
    default:
      std::cout << "Invalid boundary condition" << std::endl;
      break;
    }
  }
};

#endif // RT_TRACE_HPP