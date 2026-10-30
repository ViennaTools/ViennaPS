#include <gds/psGDSGeometry.hpp>
#include <geometries/psMakeFin.hpp>
#include <geometries/psMakeTrench.hpp>
#include <models/psDirectionalProcess.hpp>
#include <models/psGeometricDistributionModels.hpp>
#include <models/psIsotropicProcess.hpp>
#include <models/psOxidation.hpp>
#include <models/psSF6O2Etching.hpp>
#include <models/psSelectiveEpitaxy.hpp>
#include <models/psSingleParticleALD.hpp>
#include <models/psSingleParticleProcess.hpp>
#include <models/psTEOSDeposition.hpp>
#include <models/psWetEtching.hpp>

#include <process/psProcess.hpp>
#include <psDomain.hpp>
#include <psReader.hpp>
#include <psWriter.hpp>

#include <lsBooleanOperation.hpp>
#include <lsMakeGeometry.hpp>
#include <lsToDualContouringMesh.hpp>
#include <lsToSurfaceMesh.hpp>

#include <lsTestAsserts.hpp>
#include <vcTestAsserts.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

/**
  Processes on a Euclidean domain, whose level sets keep exact distances with
  their closest points and normals and are advected with AdvectHermite. A
  substrate is etched through the opening of a mask (MakeTrench with a mask)
  isotropically, into a pit that undercuts the mask, and directionally, into
  a trench with vertical walls; the surface mesh is compared with the exact
  result. The flux of a single particle process, traced on the dual
  contouring mesh of the Euclidean surface, is compared with the view factor
  of the opening, and the process with the same process on the usual
  domain. A domain read from a file keeps the normalization of its level
  sets for the layers added later, and a rotated rectangle of a GDS layout
  becomes a Euclidean level set of the exact polygon (2D) or prism (3D).
  Geometric processes with sphere and box distributions, with and without
  the mask, give the exact profile where it has sharp edges. An atomic layer
  deposition with saturated coverage grows the conformal film, and models
  with coverages (SF6/O2 plasma etching, a deposition of two TEOS particles)
  etch and deposit as on the usual domain. Of the models whose speeds depend
  on the normal, wet etching of (100) silicon through the opening gives the
  groove of {111} walls through the edges of the mask, and selective
  epitaxy grows the film on the floor of the opening, flat up to the walls
  of the mask.
*/

namespace viennacore {

using namespace viennaps;

// the callback of a process without advection: a box joined to the top level
// set, made with the normalization of the top
template <class NumericType, int D>
class BoxCallback : public AdvectionCallback<NumericType, D> {
public:
  NumericType lo[D], hi[D];
  bool applyPreAdvect(NumericType) override {
    auto top = this->domain->getLevelSets().back();
    auto box = viennals::Domain<NumericType, D>::New(top->getGrid());
    box->setNormalization(top->getNormalization());
    viennals::MakeGeometry<NumericType, D>(
        box, viennals::Box<NumericType, D>::New(lo, hi))
        .apply();
    viennals::BooleanOperation<NumericType, D>(
        top, box, viennals::BooleanOperationEnum::UNION)
        .apply();
    return true;
  }
};

// distance of (x, y) below the substrate surface y = ys to the groove that
// wet etching of (100) silicon cuts through an opening |x| < a along <110>:
// {111} walls through the edges of the opening at atan(sqrt 2) = 54.74
// degrees, moved by w into the substrate, and the (100) floor sunk by d
double grooveDistance(double x, double y, double a, double ys, double d,
                      double w) {
  const double theta = std::atan(std::sqrt(2.));
  const double s = std::sin(theta), c = std::cos(theta);
  x = std::abs(x);
  // the wall from the substrate surface down to the floor
  auto wall = [&](double t) {
    return std::array<double, 2>{a - t * c + w * s, ys - t * s - w * c};
  };
  auto segment = [&](const std::array<double, 2> &p,
                     const std::array<double, 2> &q) {
    const double dx = q[0] - p[0], dy = q[1] - p[1];
    const double t = std::clamp(
        ((x - p[0]) * dx + (y - p[1]) * dy) / (dx * dx + dy * dy), 0., 1.);
    return std::hypot(x - p[0] - t * dx, y - p[1] - t * dy);
  };
  const auto top = wall(-w * c / s), corner = wall((d - w * c) / s);
  return std::min(segment(top, corner),
                  segment(corner, std::array<double, 2>{0., ys - d}));
}

// distance of (x, y) to the exact boundary after etching by r through an
// opening |x| < a in a mask of height h on the substrate surface y = ys: a
// pit of the points within r of the opening, undercutting the mask, or a
// trench with vertical walls (directional)
double exactDistance(double x, double y, double a, double ys, double h,
                     double r, bool directional) {
  x = std::abs(x);
  const double sd = x - a; // signed distance to the opening along x
  double d = std::hypot(y - (ys + h), std::max(-sd, 0.)); // mask top
  const double bottom = directional ? ys - r : ys;
  d = std::min(d, std::hypot(sd, std::max({bottom - y, y - (ys + h), 0.})));
  if (directional)
    return std::min(d, std::hypot(y - (ys - r), std::max(sd, 0.)));
  // underside of the mask over the undercut, 0 <= sd <= r
  d = std::min(d, std::hypot(y - ys, std::max({-sd, sd - r, 0.})));
  // pit surface; from above, its nearest points lie on the rim or straight
  // below on the flat bottom
  double pit;
  if (y <= ys) {
    pit = std::abs(std::hypot(y - ys, std::max(sd, 0.)) - r);
  } else {
    pit = std::hypot(y - ys, sd - r);
    if (sd <= 0.)
      pit = std::min(pit, y - ys + r);
  }
  return std::min(d, pit);
}

template <class NumericType, int D> void RunTest() {
  Logger::setLogLevel(LogLevel::WARNING);
  const NumericType gridDelta = 0.5, xExtent = 12., yExtent = 3.;
  const NumericType a = 2.37, base = 0.29, maskHeight = 1.13,
                    time = 2.25 * gridDelta;

  auto makeDomain = [&](bool euclidean, bool periodic = false) {
    auto domain = Domain<NumericType, D>::New();
    if (euclidean)
      domain->setNormalization(viennals::NormalizationEnum::EUCLIDEAN);
    MakeTrench<NumericType, D>(domain, gridDelta, xExtent, yExtent, 2. * a,
                               maskHeight, 0., base, periodic, true,
                               Material::Si)
        .apply();
    return domain;
  };

  for (bool directional : {false, true}) {
    auto domain = makeDomain(true);
    VC_TEST_ASSERT(domain->getSurface()->getNormalization() ==
                   viennals::NormalizationEnum::EUCLIDEAN);
    if (directional) {
      Vec3D<NumericType> direction{0., 0., 0.};
      direction[D - 1] = -1.;
      // negative rates etch, as in the other models
      auto model = SmartPointer<DirectionalProcess<NumericType, D>>::New(
          direction, -1., 0., Material::Mask);
      Process<NumericType, D>(domain, model, time).apply();
    } else {
      auto model = SmartPointer<IsotropicProcess<NumericType, D>>::New(
          -1., Material::Mask);
      Process<NumericType, D>(domain, model, time).apply();
    }
    VC_TEST_ASSERT(domain->getSurface()->getNormalization() ==
                   viennals::NormalizationEnum::EUCLIDEAN);
    LSTEST_ASSERT_VALID_LS(domain->getSurface(), NumericType, D);

    // vertices of the surface mesh against the exact profile, away from the
    // reflective boundaries along x
    auto mesh = domain->getSurfaceMesh(false);
    VC_TEST_ASSERT(!mesh->getNodes().empty());
    double deviation = 0.;
    for (const auto &n : mesh->getNodes()) {
      if (std::abs(n[0]) > xExtent / 2 - gridDelta)
        continue;
      deviation =
          std::max(deviation, exactDistance(n[0], n[D - 1], a, base, maskHeight,
                                            time, directional));
    }
    std::cout << D << "D Euclidean domain, "
              << (directional ? "directional" : "isotropic")
              << " etching through a mask opening: worst vertex "
              << deviation / gridDelta << " grid spacings" << std::endl;
    VC_TEST_ASSERT(deviation / gridDelta < (directional ? 0.05 : 0.3));
  }

  // flux of a single particle process (sticking 1, cosine source) on the
  // trench bottom before any motion, from the elements of the dual contouring
  // mesh the rays are traced on, against the view factor of the opening,
  // also with periodic boundaries along x
  for (bool periodic : {false, true}) {
    auto domain = makeDomain(true, periodic);
    auto model = SmartPointer<SingleParticleProcess<NumericType, D>>::New(
        -1., 1., 1., Material::Mask);
    Process<NumericType, D> process(domain, model);
    // enough rays that the noise stays well below the tolerance
    RayTracingParameters rayParams;
    rayParams.raysPerPoint = 10000;
    process.setParameters(rayParams);
    VC_TEST_ASSERT(process.calculateFlux() != nullptr);
    auto surface = process.getTriangleMesh();
    VC_TEST_ASSERT(surface != nullptr);
    auto *flux = surface->getCellData().getScalarData("particleFlux");
    VC_TEST_ASSERT(flux != nullptr);
    auto viewFactor = [&](double x) {
      return (std::sin(std::atan((a - x) / maskHeight)) +
              std::sin(std::atan((a + x) / maskHeight))) /
             2;
    };
    double traced = 0., exact = 0., area = 0.;
    for (std::size_t e = 0; e < surface->triangles.size(); ++e) {
      const auto &t = surface->triangles[e];
      const auto &p0 = surface->nodes[t[0]], &p1 = surface->nodes[t[1]],
                 &p2 = surface->nodes[t[2]];
      const double x = (p0[0] + p1[0] + p2[0]) / 3.,
                   y = (p0[D - 1] + p1[D - 1] + p2[D - 1]) / 3.;
      if (std::abs(y - base) > 0.01 * gridDelta || std::abs(x) > 1.)
        continue;
      const auto u = p1 - p0, v = p2 - p0;
      const double A = 0.5 * std::sqrt(std::pow(u[1] * v[2] - u[2] * v[1], 2) +
                                       std::pow(u[2] * v[0] - u[0] * v[2], 2) +
                                       std::pow(u[0] * v[1] - u[1] * v[0], 2));
      traced += A * (*flux)[e];
      exact += A * viewFactor(x);
      area += A;
    }
    VC_TEST_ASSERT(area > 0.);
    std::cout << D << "D flux on the trench bottom"
              << (periodic ? ", periodic boundaries: " : ": ") << traced / area
              << " traced, " << exact / area << " exact" << std::endl;
    VC_TEST_ASSERT(std::abs(traced - exact) / exact < 0.03);
  }

  // the process on a Euclidean and on the usual domain, each with the flux
  // engine chosen for it: deepest point of the etched trench below the opening
  NumericType depth[2] = {0., 0.};
  for (bool euclidean : {false, true}) {
    auto domain = makeDomain(euclidean);
    auto model = SmartPointer<SingleParticleProcess<NumericType, D>>::New(
        -1., 1., 1., Material::Mask);
    Process<NumericType, D> process(domain, model, time);
    process.apply();
    LSTEST_ASSERT_VALID_LS(domain->getSurface(), NumericType, D);
    auto mesh = domain->getSurfaceMesh(false);
    NumericType lowest = 1e30;
    for (const auto &n : mesh->getNodes())
      if (std::abs(n[0]) < a - gridDelta)
        lowest = std::min(lowest, NumericType(n[D - 1]));
    depth[euclidean] = base - lowest;
  }
  std::cout << D << "D single particle process, depth below the opening: "
            << depth[0] << " (Manhattan), " << depth[1] << " (Euclidean)"
            << std::endl;
  VC_TEST_ASSERT(depth[1] > 0.5 * time);
  VC_TEST_ASSERT(std::abs(depth[1] - depth[0]) < 0.25 * gridDelta);

  // a domain read from a file: the setup takes the normalization of the
  // level sets, which store it
  {
    const std::string file = "euclideanDomain" + std::to_string(D) + "D.vpsd";
    Writer<NumericType, D>(makeDomain(true), file).apply();
    auto read = Domain<NumericType, D>::New();
    Reader<NumericType, D>(read, file).apply();
    std::remove(file.c_str());
    VC_TEST_ASSERT(read->getSurface()->getNormalization() ==
                   viennals::NormalizationEnum::EUCLIDEAN);
    VC_TEST_ASSERT(read->getSetup().normalization() ==
                   viennals::NormalizationEnum::EUCLIDEAN);
  }

  // a rotated rectangle on layer 1 of a layout, extruded from 0.2 to 1.5 in
  // 3D, against the exact polygon or prism
  {
    const double angle = 0.4, hx = 2.1, hy = 1.3, cx = 0.37, cy = -0.21,
                 bottom = 0.2, height = 1.3;
    const double c = std::cos(angle), s = std::sin(angle);
    GDS::Element<NumericType> element;
    element.elementType = GDS::ElementType::elBoundary;
    element.layer = 1;
    const double corners[5][2] = {
        {-hx, -hy}, {hx, -hy}, {hx, hy}, {-hx, hy}, {-hx, -hy}};
    GDS::Structure<NumericType> structure;
    structure.elementBoundingBox = {{{1e30, 1e30}, {-1e30, -1e30}}};
    for (const auto &q : corners) {
      const NumericType x = cx + c * q[0] - s * q[1],
                        y = cy + s * q[0] + c * q[1];
      element.pointCloud.push_back({x, y, 0.});
      structure.elementBoundingBox[0][0] =
          std::min(structure.elementBoundingBox[0][0], x);
      structure.elementBoundingBox[0][1] =
          std::min(structure.elementBoundingBox[0][1], y);
      structure.elementBoundingBox[1][0] =
          std::max(structure.elementBoundingBox[1][0], x);
      structure.elementBoundingBox[1][1] =
          std::max(structure.elementBoundingBox[1][1], y);
    }
    structure.name = "rectangle";
    structure.elements.push_back(element);
    structure.containsLayers.insert(1);
    auto layout = SmartPointer<GDSGeometry<NumericType, D>>::New(gridDelta);
    BoundaryType boundaryConds[D];
    for (int i = 0; i < D; ++i)
      boundaryConds[i] = (i < 2) ? BoundaryType::REFLECTIVE_BOUNDARY
                                 : BoundaryType::INFINITE_BOUNDARY;
    layout->setBoundaryConditions(boundaryConds);
    layout->setBoundaryPadding(2., 2.);
    layout->setNormalization(viennals::NormalizationEnum::EUCLIDEAN);
    layout->insertNextStructure(structure);
    layout->finalize();
    SmartPointer<viennals::Domain<NumericType, D>> levelSet;
    if constexpr (D == 2)
      levelSet = layout->layerToLevelSet(1, false);
    else
      levelSet = layout->layerToLevelSet(1, bottom, height, false, false);
    VC_TEST_ASSERT(levelSet->getNormalization() ==
                   viennals::NormalizationEnum::EUCLIDEAN);
    auto mesh = viennals::Mesh<NumericType>::New();
    viennals::ToDualContouringMesh<NumericType, D>(levelSet, mesh).apply();
    VC_TEST_ASSERT(!mesh->getNodes().empty());
    double deviation = 0.;
    for (const auto &n : mesh->getNodes()) {
      const double u = c * (n[0] - cx) + s * (n[1] - cy),
                   v = -s * (n[0] - cx) + c * (n[1] - cy);
      double q[3] = {std::abs(u) - hx, std::abs(v) - hy,
                     D == 3 ? std::abs(n[2] - bottom - height / 2) - height / 2
                            : -1e30};
      double outside = 0., inside = -1e30;
      for (int i = 0; i < D; ++i) {
        outside += std::pow(std::max(q[i], 0.), 2);
        inside = std::max(inside, q[i]);
      }
      deviation = std::max(deviation,
                           std::abs(std::sqrt(outside) + std::min(inside, 0.)));
    }
    std::cout << D << "D Euclidean level set of a GDS layer: worst vertex "
              << deviation / gridDelta << " grid spacings" << std::endl;
    VC_TEST_ASSERT(deviation / gridDelta <
                   1e4 * std::numeric_limits<NumericType>::epsilon());
  }

  // geometric processes: the sum of a sphere or box with the surface, or
  // with the part of it not on the mask, against the exact profile; the
  // level set stays Euclidean with its width. MakeTrench lowers the mask by
  // 1e-4.
  {
    const double eps = 1e-4, top = base + maskHeight - eps, far = 100.;
    struct Segment {
      double x0, y0, x1, y1;
    };
    auto segmentDistance = [](const Segment &s, double x, double y) {
      const double dx = s.x1 - s.x0, dy = s.y1 - s.y0;
      const double t = std::clamp(
          ((x - s.x0) * dx + (y - s.y0) * dy) / (dx * dx + dy * dy), 0., 1.);
      return std::hypot(x - s.x0 - t * dx, y - s.y0 - t * dy);
    };
    struct Case {
      std::string name;
      SmartPointer<ProcessModelBase<NumericType, D>> model;
      bool deposition;
      std::function<double(double, double)> exact; // x >= 0
      double bound;                                // grid spacings
    };
    std::vector<Case> cases;
    {
      // a film of 2 grid spacings, rounded at the top corners of the mask
      const double r = 2. * gridDelta;
      cases.push_back(
          {"sphere deposition",
           SmartPointer<SphereDistribution<NumericType, D>>::New(r), true,
           [=](double x, double y) {
             double d = std::min(
                 segmentDistance({0., base + r, a - r, base + r}, x, y),
                 segmentDistance({a - r, base + r, a - r, top}, x, y));
             d = std::min(d, segmentDistance({a, top + r, far, top + r}, x, y));
             // arc around the top corner of the mask, from the wall up
             if (x <= a && y >= top)
               d = std::min(d, std::abs(std::hypot(x - a, y - top) - r));
             return d;
           },
           0.15});
    }
    {
      // the substrate etched by 2 grid spacings from the floor of the
      // opening, under the mask too, which stays
      const double r = 2. * gridDelta;
      auto model = SmartPointer<SphereDistribution<NumericType, D>>::New(-r);
      model->addMaskMaterial(Material::Mask);
      cases.push_back({"sphere etching, mask kept", model, false,
                       [=](double x, double y) {
                         return exactDistance(x, y, a, base, maskHeight, r,
                                              false);
                       },
                       0.15});
    }
    {
      // a box one grid spacing wide and two deep: a pit with vertical walls
      // one grid spacing under the mask
      const double bx = gridDelta, by = 2. * gridDelta;
      std::array<NumericType, 3> halfAxes{NumericType(-bx), NumericType(-by),
                                          NumericType(-by)};
      if constexpr (D == 3)
        halfAxes = {NumericType(-bx), NumericType(-bx), NumericType(-by)};
      auto model = SmartPointer<BoxDistribution<NumericType, D>>::New(halfAxes);
      model->addMaskMaterial(Material::Mask);
      cases.push_back(
          {"box etching, mask kept", model, false,
           [=](double x, double y) {
             double d = std::min(
                 segmentDistance({0., base - by, a + bx, base - by}, x, y),
                 segmentDistance({a + bx, base - by, a + bx, base - eps}, x,
                                 y));
             d = std::min(
                 d, segmentDistance({a, base - eps, a + bx, base - eps}, x, y));
             d = std::min(d, segmentDistance({a, base - eps, a, top}, x, y));
             return std::min(d, segmentDistance({a, top, far, top}, x, y));
           },
           0.01});
    }
    {
      // a film on the floor of the opening only
      const double r = 1.2 * gridDelta;
      auto model = SmartPointer<SphereDistribution<NumericType, D>>::New(r);
      model->addMaskMaterial(Material::Mask);
      cases.push_back(
          {"sphere deposition, mask excluded", model, true,
           [=](double x, double y) {
             return std::min(
                 {segmentDistance({0., base + r, a, base + r}, x, y),
                  segmentDistance({a, base + r, a, top}, x, y),
                  segmentDistance({a, top, far, top}, x, y)});
           },
           0.01});
    }
    for (const auto &c : cases) {
      auto domain = makeDomain(true);
      if (c.deposition)
        domain->duplicateTopLevelSet(Material::SiO2);
      Process<NumericType, D>(domain, c.model).apply();
      auto surface = domain->getSurface();
      VC_TEST_ASSERT(surface->getNormalization() ==
                     viennals::NormalizationEnum::EUCLIDEAN);
      VC_TEST_ASSERT(surface->getLevelSetWidth() == 12);
      LSTEST_ASSERT_VALID_LS(surface, NumericType, D);
      auto mesh = domain->getSurfaceMesh(false);
      VC_TEST_ASSERT(!mesh->getNodes().empty());
      double deviation = 0.;
      for (const auto &n : mesh->getNodes()) {
        if (std::abs(n[0]) > xExtent / 2 - gridDelta)
          continue;
        deviation =
            std::max(deviation, c.exact(std::abs(double(n[0])), n[D - 1]));
      }
      std::cout << D << "D Euclidean domain, geometric " << c.name
                << ": worst vertex " << deviation / gridDelta
                << " grid spacings" << std::endl;
      VC_TEST_ASSERT(deviation / gridDelta < c.bound);
    }
  }

  // an atomic layer deposition with saturated coverage grows the film of a
  // sphere deposition, 4 cycles of 0.25, rounded at the top corners of the
  // mask
  {
    const double r = 1., top = base + maskHeight - 1e-4, far = 100.;
    auto domain = makeDomain(true);
    domain->duplicateTopLevelSet(Material::Al2O3);
    SingleParticleALDParams params;
    params.stickingProbability = 1.;
    params.growthPerCycle = 0.25;
    params.incomingFlux = 100.;
    params.s0 = 1.;
    Process<NumericType, D> process(
        domain, SmartPointer<SingleParticleALD<NumericType, D>>::New(params));
    AtomicLayerProcessParameters cycles;
    cycles.numCycles = 4;
    cycles.pulseTime = 0.1;
    cycles.coverageTimeStep = 0.01;
    process.setParameters(cycles);
    process.apply();
    auto surface = domain->getSurface();
    VC_TEST_ASSERT(surface->getNormalization() ==
                   viennals::NormalizationEnum::EUCLIDEAN);
    VC_TEST_ASSERT(surface->getLevelSetWidth() == 12);
    LSTEST_ASSERT_VALID_LS(surface, NumericType, D);
    auto segment = [](double x0, double y0, double x1, double y1, double x,
                      double y) {
      const double dx = x1 - x0, dy = y1 - y0;
      const double t = std::clamp(
          ((x - x0) * dx + (y - y0) * dy) / (dx * dx + dy * dy), 0., 1.);
      return std::hypot(x - x0 - t * dx, y - y0 - t * dy);
    };
    double deviation = 0.;
    auto mesh = domain->getSurfaceMesh(false);
    for (const auto &n : mesh->getNodes()) {
      const double x = std::abs(double(n[0])), y = n[D - 1];
      if (x > xExtent / 2 - gridDelta)
        continue;
      double d = std::min({segment(0., base + r, a - r, base + r, x, y),
                           segment(a - r, base + r, a - r, top, x, y),
                           segment(a, top + r, far, top + r, x, y)});
      if (x <= a && y >= top)
        d = std::min(d, std::abs(std::hypot(x - a, y - top) - r));
      deviation = std::max(deviation, d);
    }
    std::cout << D << "D Euclidean domain, atomic layer deposition: worst "
              << "vertex " << deviation / gridDelta << " grid spacings"
              << std::endl;
    VC_TEST_ASSERT(deviation / gridDelta < 0.15);
  }

  // models with coverages: plasma etching (SF6/O2) and a deposition by two
  // particles (TEOS) on the Euclidean and on the usual domain, compared by
  // the depth of the opening or the thickness of the film on its floor
  for (bool deposition : {false, true}) {
    NumericType level[2] = {0., 0.};
    for (bool euclidean : {false, true}) {
      auto domain = makeDomain(euclidean);
      SmartPointer<ProcessModelBase<NumericType, D>> model;
      NumericType duration;
      if (deposition) {
        domain->duplicateTopLevelSet(Material::SiO2);
        model = SmartPointer<TEOSDeposition<NumericType, D>>::New(
            0.1, 0.5, 1., 1e-4, 0.5, 1.);
        duration = 2.;
      } else {
        model = SmartPointer<SF6O2Etching<NumericType, D>>::New(
            SF6O2Etching<NumericType, D>::defaultParameters());
        duration = 10.;
      }
      Process<NumericType, D> process(domain, model, duration);
      RayTracingParameters rays;
      rays.raysPerPoint = 2000;
      process.setParameters(rays);
      process.apply();
      if (euclidean) {
        VC_TEST_ASSERT(domain->getSurface()->getNormalization() ==
                       viennals::NormalizationEnum::EUCLIDEAN);
        VC_TEST_ASSERT(domain->getSurface()->getLevelSetWidth() == 12);
      }
      LSTEST_ASSERT_VALID_LS(domain->getSurface(), NumericType, D);
      NumericType lowest = 1e30;
      auto mesh = domain->getSurfaceMesh(false);
      for (const auto &n : mesh->getNodes())
        if (std::abs(n[0]) < a - gridDelta)
          lowest = std::min(lowest, NumericType(n[D - 1]));
      level[euclidean] = lowest;
    }
    std::cout << D << "D "
              << (deposition ? "TEOS deposition, film on the floor"
                             : "SF6/O2 etching, depth below the opening")
              << ": " << level[0] - base << " (Manhattan), " << level[1] - base
              << " (Euclidean)" << std::endl;
    VC_TEST_ASSERT(std::abs(level[1] - base) > 0.5 * gridDelta);
    VC_TEST_ASSERT(std::abs(level[1] - level[0]) < 0.25 * gridDelta);
  }

  // speeds that depend on the normal: wet etching with the rates of the model
  // (default crystal directions, the trench along <110> on (100)) until the
  // floor has sunk by 1.125, against the exact groove; selective epitaxy at
  // twice the rates of the model for 0.25, against the flat film of 0.5 on
  // the floor of the opening
  for (bool epitaxy : {false, true}) {
    auto domain = makeDomain(true);
    const NumericType r100 = 0.0166666666667, r111 = 0.000121666666667,
                      depth = 1.125, film = 0.5;
    if (epitaxy) {
      domain->duplicateTopLevelSet(Material::SiGe);
      auto model = SmartPointer<SelectiveEpitaxy<NumericType, D>>::New(
          std::vector<std::pair<Material, NumericType>>{{Material::Si, 2.},
                                                        {Material::SiGe, 2.}},
          0.4, 1.);
      Process<NumericType, D>(domain, model, 0.25).apply();
    } else {
      auto model = SmartPointer<WetEtching<NumericType, D>>::New(
          std::vector<std::pair<Material, NumericType>>{{Material::Si, -1.}});
      Process<NumericType, D>(domain, model, depth / r100).apply();
    }
    VC_TEST_ASSERT(domain->getSurface()->getNormalization() ==
                   viennals::NormalizationEnum::EUCLIDEAN);
    VC_TEST_ASSERT(domain->getSurface()->getLevelSetWidth() == 12);
    LSTEST_ASSERT_VALID_LS(domain->getSurface(), NumericType, D);
    auto mesh = domain->getSurfaceMesh(false);
    VC_TEST_ASSERT(!mesh->getNodes().empty());
    double deviation = 0.;
    for (const auto &n : mesh->getNodes()) {
      const double x = n[0], y = n[D - 1];
      if (epitaxy && std::abs(x) < a - 0.1 && y > base + 0.05)
        deviation = std::max(deviation, std::abs(y - (base + film)));
      if (!epitaxy && std::abs(x) < a + 0.3 && y < base - 0.05)
        deviation = std::max(deviation, grooveDistance(x, y, a, base, depth,
                                                       r111 * depth / r100));
    }
    std::cout << D << "D "
              << (epitaxy ? "selective epitaxy in" : "wet etching through")
              << " the mask opening: worst vertex " << deviation / gridDelta
              << " grid spacings" << std::endl;
    VC_TEST_ASSERT(deviation / gridDelta < (epitaxy ? 0.05 : 0.1));
  }

  // oxidation (wet, 1000 C, 0.1 h) of a silicon step under a native oxide of
  // one grid spacing, on a Euclidean and on the usual domain, compared by the
  // heights of the oxide surface and of the interface on the substrate; the
  // solver runs on Manhattan bands, and the Euclidean domain gets its level
  // sets back Euclidean at their width. The solver is 2D and double only here.
  if constexpr (D == 2 && std::is_same_v<NumericType, double>) {
    const NumericType delta = 0.02;
    NumericType surface[2] = {0., 0.}, interface[2] = {0., 0.};
    // the height of a level set's surface over x = 0.9
    auto heightAt =
        [](SmartPointer<viennals::Domain<NumericType, D>> levelSet) {
          auto mesh = SmartPointer<viennals::Mesh<NumericType>>::New();
          if (levelSet->getNormalization() ==
              viennals::NormalizationEnum::EUCLIDEAN)
            viennals::ToDualContouringMesh<NumericType, D>(levelSet, mesh)
                .apply();
          else
            viennals::ToSurfaceMesh<NumericType, D>(levelSet, mesh).apply();
          const NumericType x0 = 0.9;
          NumericType height = -1e30;
          const auto &nodes = mesh->getNodes();
          for (const auto &line : mesh->lines) {
            const auto &u = nodes[line[0]], &v = nodes[line[1]];
            if ((u[0] - x0) * (v[0] - x0) > 0. || u[0] == v[0])
              continue;
            const NumericType t = (x0 - u[0]) / (v[0] - u[0]);
            height = std::max(height, u[1] + t * (v[1] - u[1]));
          }
          return height;
        };
    for (bool euclidean : {false, true}) {
      auto domain = Domain<NumericType, D>::New(
          delta, 2., 2., BoundaryType::REFLECTIVE_BOUNDARY);
      if (euclidean)
        domain->setNormalization(viennals::NormalizationEnum::EUCLIDEAN);
      MakeFin<NumericType, D>(domain, 0.5, 1., 0., 0., 0., true).apply();
      domain->duplicateTopLevelSet(Material::SiO2);
      Process<NumericType, D>(
          domain, SmartPointer<SphereDistribution<NumericType, D>>::New(delta))
          .apply();
      auto model = SmartPointer<Oxidation<NumericType, D>>::New();
      model->setTemperature(1000.);
      model->setTime(0.1);
      model->setOxidant("wet");
      model->setPressure(1.);
      model->setOrientation("100");
      model->setInitialOxideThickness(delta);
      Process<NumericType, D>(domain, model, 0.).apply();
      for (auto &levelSet : domain->getLevelSets()) {
        if (euclidean) {
          VC_TEST_ASSERT(levelSet->getNormalization() ==
                         viennals::NormalizationEnum::EUCLIDEAN);
          VC_TEST_ASSERT(levelSet->getLevelSetWidth() == 12);
        }
        LSTEST_ASSERT_VALID_LS(levelSet, NumericType, D);
      }
      surface[euclidean] = heightAt(domain->getLevelSets().back());
      interface[euclidean] = heightAt(domain->getLevelSets().front());
    }
    std::cout << "2D oxidation of a step, oxide on the substrate from "
              << interface[0] << " to " << surface[0] << " (Manhattan), "
              << interface[1] << " to " << surface[1] << " (Euclidean)"
              << std::endl;
    VC_TEST_ASSERT(surface[1] - interface[1] > 2. * delta);
    VC_TEST_ASSERT(std::abs(surface[1] - surface[0]) < 0.25 * delta);
    VC_TEST_ASSERT(std::abs(interface[1] - interface[0]) < 0.25 * delta);
  }

  // a process of duration 0 with only a callback, which joins a box of height
  // 1 and half width 1 to the floor of the opening: the domain stays
  // Euclidean, and the top of the box is exact
  {
    auto domain = makeDomain(true);
    auto model = SmartPointer<ProcessModelCPU<NumericType, D>>::New();
    auto callback = SmartPointer<BoxCallback<NumericType, D>>::New();
    for (int i = 0; i < D - 1; ++i) {
      callback->lo[i] = (i == 0) ? -1. : -yExtent;
      callback->hi[i] = (i == 0) ? 1. : yExtent;
    }
    callback->lo[D - 1] = base - gridDelta;
    callback->hi[D - 1] = base + 1.;
    model->setAdvectionCallback(callback);
    Process<NumericType, D>(domain, model, 0.).apply();
    VC_TEST_ASSERT(domain->getSurface()->getNormalization() ==
                   viennals::NormalizationEnum::EUCLIDEAN);
    VC_TEST_ASSERT(domain->getSurface()->getLevelSetWidth() == 12);
    LSTEST_ASSERT_VALID_LS(domain->getSurface(), NumericType, D);
    auto mesh = domain->getSurfaceMesh(false);
    double deviation = 0.;
    int onTop = 0;
    for (const auto &n : mesh->getNodes())
      if (std::abs(n[0]) < 0.9 && n[D - 1] > base + 0.1) {
        deviation = std::max(deviation, std::abs(n[D - 1] - (base + 1.)));
        ++onTop;
      }
    std::cout << D << "D process with only a callback: " << onTop
              << " vertices on the box, worst " << deviation / gridDelta
              << " grid spacings" << std::endl;
    VC_TEST_ASSERT(onTop > 0);
    VC_TEST_ASSERT(deviation / gridDelta < 1e-5);
  }
}

} // namespace viennacore

int main() {
  // units of the plasma etching model
  viennaps::units::Length::setUnit(viennaps::units::Length::MICROMETER);
  viennaps::units::Time::setUnit(viennaps::units::Time::SECOND);
  viennacore::RunTest<double, 2>();
  viennacore::RunTest<double, 3>();
  viennacore::RunTest<float, 2>();
  viennacore::RunTest<float, 3>();
}
