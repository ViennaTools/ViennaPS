#ifndef RT_TRACE_HPP
#define RT_TRACE_HPP

#include <iostream>
#include <lsToDiskMesh.hpp>
#include <psSmartPointer.hpp>
#include <rtVelocityField.hpp>
#include <rti/device.hpp>

enum struct rtTraceBoundary : unsigned { REFLECTIVE = 0, PERIODIC = 1 };

/*
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
  lsSmartPointer<rtVelocityField<numeric_type, D>> velocityField = nullptr;
  size_t numberOfRaysPerPoint = 1000;
  static constexpr numeric_type discFactor = 0.5 * 1.7320508 * (1 + 1e-5);

public:
  rtTrace() {
    rtiDevice = rtDeviceType::New();
    velocityField = lsSmartPointer<rtVelocityField<numeric_type, D>>::New();
  };

  rtTrace(lsSmartPointer<lsDomain<numeric_type, D>> passedlsDomain) {
    rtiDevice = rtDeviceType::New();
    velocityField = lsSmartPointer<rtVelocityField<numeric_type, D>>::New();
    setDomain(passedlsDomain);
  }

  /// Extract the mesh points, normals and pointId translator from the LS domain
  /// and then run the ray tracer
  void apply() {
    {
      auto translator = lsSmartPointer<
          std::unordered_map<unsigned long, unsigned long>>::New();
      auto mesh = lsSmartPointer<lsMesh<numeric_type>>::New();
      lsToDiskMesh<numeric_type, D>(domain, mesh, translator).apply();
      velocityField->setTranslator(translator);

      auto points = mesh->getNodes();
      auto normals = *mesh->getVectorData("Normals");
      rtiDevice->set_points(points);
      rtiDevice->set_normals(normals);
      rtiDevice->set_number_of_rays(numberOfRaysPerPoint * points.size());
    }

    rtiDevice->run();
    auto mcestimates = lsSmartPointer<std::vector<numeric_type>>::New(
        rtiDevice->get_mc_estimates());
    velocityField->setMcEstimates(mcestimates);
  }

  // Returns the particle hit counts for each grid point
  std::vector<size_t> getHitCounts() { return rtiDevice->get_hit_cnts(); }

  // Returns the particle hit counts for each grid point
  // normalized to the overall maximal hit count
  std::vector<numeric_type> getMcEstimates() {
    return rtiDevice->get_mc_estimates();
  }

  // Return the velocity field needed for advection
  lsSmartPointer<rtVelocityField<numeric_type, D>> getVelocityField() {
    return velocityField;
  }

  void setDiscRadius(const numeric_type discRadius) {
    rtiDevice->set_grid_spacing(discRadius);
  }

  void setCosinePower(const numeric_type exp) {
    auto direction =
        std::make_unique<rti::ray::power_cosine_direction_z<numeric_type>>(exp);
    rtiDevice->set(direction);
  }

  void setNumberOfRaysPerPoint(size_t num) { numberOfRaysPerPoint = num; }

  void setDomain(lsSmartPointer<lsDomain<numeric_type, D>> passedlsDomain) {
    domain = passedlsDomain;
    rtiDevice->set_grid_spacing(domain->getDomain().getGrid().getGridDelta() *
                                discFactor);

    auto boundaryConds = domain->getGrid().getBoundaryConditions();
    for (int i = 0; i < D - 1; i++) {
      switch (boundaryConds[i]) {
      case lsDomain<numeric_type, D>::BoundaryType::REFLECTIVE_BOUNDARY:
        setBoundary(i, rtTraceBoundary::REFLECTIVE);
        break;
      case lsDomain<numeric_type, D>::BoundaryType::PERIODIC_BOUNDARY:
        setBoundary(i, rtTraceBoundary::PERIODIC);
        break;
      default:
        lsMessage::getInstance().addWarning(
            "rtTrace: Non-compatible boundary condition in lsDomain.");
        break;
      }
    }
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
      lsMessage::getInstance().addWarning(
          "rtTrace: Non-compatible boundary condition.");
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
      lsMessage::getInstance().addWarning(
          "rtTrace: Non-compatible boundary condition");
      break;
    }
  }

  void setBoundary(int direction, rtTraceBoundary bound) {
    if (direction == 0) {
      setBoundaryX(bound);
    } else if (direction == 1) {
      setBoundaryY(bound);
    }
  }
};

#endif // RT_TRACE_HPP