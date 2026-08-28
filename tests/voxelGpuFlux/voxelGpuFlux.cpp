// The cell geometry on the GPU against the voxel flux on the CPU.
//
// Same fills, same source law, same deposit-full-weight-and-re-emit
// convention: what differs is where the transport runs and how a ray finds
// the surface -- the DDA or embree on the CPU, an OptiX BVH over cell boxes
// on the GPU. The flux profiles must agree statistically.
//
// Two known differences are tolerated and bounded, not hidden:
//   - the re-emission restart: the CPU displaces the origin past the
//     interface, the GPU stays put and flies blind for an arming distance.
//     The CPU code measured three restart schemes within a few percent of
//     one another, and the tolerance here covers that spread;
//   - low-weight rays: the CPU cuts hard at a relative weight, the GPU
//     plays unbiased Russian roulette. Both are unbiased conventions.

#include <models/psVoxelChemistry.hpp>
#include <psDomain.hpp>
#include <geometries/psMakeTrench.hpp>

#include <lsMakeGeometry.hpp>

#include <gpu/raygTraceCell.hpp>
#include <models/psChemicalMechanismIO.hpp>
#include <models/psVoxelFluxGPU.hpp>

#ifndef VIENNAPS_MECHANISM_DIR
#define VIENNAPS_MECHANISM_DIR "."
#endif

#include <iomanip>
#include <iostream>

namespace ls = viennals;
namespace cs = viennacs;
namespace ps = viennaps;
namespace rg = viennaray::gpu;

using T = double;
constexpr int D = 2;

int failures = 0;

void check(const std::string &name, bool ok, const std::string &detail = "") {
  std::cout << "  [" << (ok ? "PASS" : "FAIL") << "] " << name;
  if (!detail.empty())
    std::cout << "   " << detail;
  std::cout << "\n";
  if (!ok)
    ++failures;
}

int main() {
  std::cout << "Voxel flux: GPU cell geometry against the CPU engines\n\n";
  ps::Logger::setLogLevel(ps::LogLevel::WARNING);
  ps::units::Length::setUnit("nm");
  ps::units::Time::setUnit("s");

  // A masked flat substrate: the slot in the mask is the only opening, so
  // the flux profile has a shadowed floor, lit mask top, and sidewalls.
  const T GD = 1.0, W = 10., maskH = 8.;
  auto dom = ps::SmartPointer<ps::Domain<T, D>>::New(GD, T(4) * W, T(4) * W);
  ps::MakeTrench<T, D>(dom, W, T(0), T(0), maskH, T(0), false,
                       ps::Material::Si, ps::Material::Mask)
      .apply();

  auto topLS = dom->getLevelSets().back();
  auto deep = ls::SmartPointer<ls::Domain<T, D>>::New(topLS->getGrid());
  {
    T o[D] = {0., -T(8)}, n[D] = {0., 1.};
    ls::MakeGeometry<T, D>(deep, ls::SmartPointer<ls::Plane<T, D>>::New(o, n))
        .apply();
  }
  std::vector<ls::SmartPointer<ls::Domain<T, D>>> lss{deep};
  auto matMap = ls::SmartPointer<ls::MaterialMap>::New();
  matMap->insertNextMaterial(static_cast<int>(ps::Material::Si));
  for (size_t l = 0; l < dom->getLevelSets().size(); ++l) {
    lss.push_back(dom->getLevelSets()[l]);
    matMap->insertNextMaterial(
        static_cast<int>(dom->getMaterialMap()->getMaterialAtIdx(l)));
  }
  auto cellSet = viennacore::SmartPointer<cs::DenseCellSet<T, D>>::New();
  cellSet->setCellSetPosition(true);
  cellSet->setCoverMaterial(static_cast<int>(ps::Material::GAS));
  cellSet->fromLevelSets(lss, matMap, maskH + T(4));

  cs::LatticeMap<T, D> lat(*cellSet);
  const auto &mid = *cellSet->getScalarData("Material");
  std::vector<T> fill(cellSet->getNumberOfCells(), T(0));
  for (size_t c = 0; c < fill.size(); ++c)
    fill[c] = static_cast<int>(mid[c]) == static_cast<int>(ps::Material::GAS)
                  ? T(0)
                  : T(1);

  const T sticking = 0.5;
  const size_t numRays = 500000;
  const T sourceFlux = 1.0;

  // ---------------- the CPU reference ----------------
  cs::VoxelFlux<T, D> cpuFlux(lat, fill);
  cpuFlux.setTraversalEngine(cs::TraversalEngine::EmbreeBVH);
  const auto cpu = cpuFlux.trace(numRays, sourceFlux, sticking);

  // ---------------- the GPU arm ----------------
  // The band: every cell holding material that the 3^D neighbourhood opens.
  cs::VoxelAdvance<T, D> advance(lat);
  const auto band = advance.interfaceBand(fill);
  cs::VoxelInteraction<T, D> interaction(
      lat, fill, cs::NormalEstimator::FillGradientYoungs);

  const auto &dims = lat.dims();
  const auto &minC = lat.minCorner();
  size_t sites = 1;
  for (int d = 0; d < D; ++d)
    sites *= static_cast<size_t>(dims[d]);

  rg::CellGrid grid;
  grid.gridDelta = static_cast<float>(GD);
  for (int d = 0; d < 3; ++d) {
    grid.minimumExtent[d] = 0.f;
    grid.maximumExtent[d] = 0.f;
  }
  for (int d = 0; d < D; ++d) {
    grid.minimumExtent[d] = static_cast<float>(minC[d]);
    grid.maximumExtent[d] =
        static_cast<float>(minC[d] + GD * static_cast<T>(dims[d]));
  }
  std::vector<int> primCell; // primID -> cell id
  std::vector<std::array<int, D>> primIdx;
  for (size_t flat = 0; flat < sites; ++flat) {
    size_t rem = flat;
    std::array<int, D> idx{};
    for (int d = 0; d < D; ++d) {
      idx[d] = static_cast<int>(rem % static_cast<size_t>(dims[d]));
      rem /= static_cast<size_t>(dims[d]);
    }
    const int id = lat.cellId(idx);
    if (id < 0 || !band[id] || fill[id] <= T(0))
      continue;
    viennacore::Vec3Df p{0.f, 0.f, 0.f};
    for (int d = 0; d < D; ++d)
      p[d] = static_cast<float>(minC[d] + GD * static_cast<T>(idx[d]));
    grid.minPoints.push_back(p);
    grid.fills.push_back(static_cast<float>(fill[id]));
    // the same Youngs normal the CPU engines see; where it is degenerate the
    // surface points up, which is what a buried or isolated cell defaults to
    viennacore::Vec3D<T> up{0, 0, 0};
    up[D - 1] = 1;
    const auto n = interaction.gradientNormal(idx, up, true);
    viennacore::Vec3Df nf{0.f, 0.f, 0.f};
    for (int d = 0; d < D; ++d)
      nf[d] = static_cast<float>(n[d]);
    grid.normals.push_back(nf);
    primCell.push_back(id);
    primIdx.push_back(idx);
  }
  check("the band has primitives", !grid.minPoints.empty(),
        std::to_string(grid.minPoints.size()) + " cells");

  auto context = viennacore::DeviceContext::createContext();

  viennaray::gpu::Particle<float> particle;
  particle.name = "Particle";
  particle.sticking = static_cast<float>(sticking);
  particle.dataLabels = {"flux"};

  std::unordered_map<std::string, unsigned int> pMap = {{"Particle", 0}};
  std::vector<rg::CallableConfig> cMap = {
      {0, rg::CallableSlot::COLLISION,
       "__direct_callable__particleCollision"},
      {0, rg::CallableSlot::REFLECTION,
       "__direct_callable__particleReflectionConstSticking"},
  };

  rg::TraceCell<float, D> tracer(context);
  tracer.setGeometry(grid);
  tracer.setArmingDistance(3.f * grid.gridDelta);
  std::vector<float> materialIds(grid.minPoints.size(), 1.f);
  tracer.setMaterialIds(materialIds);
  tracer.setCallables("ViennaRayCallableWrapper", context->modulePath);
  tracer.setParticleCallableMap({pMap, cMap});
  tracer.insertNextParticle(particle);
  tracer.setNumberOfRaysFixed(numRays);
  tracer.prepareParticlePrograms();
  tracer.apply();
  const auto raw = tracer.getFlux(0, 0); // raw incident rate, per band cell

  // The CPU spreads each encounter over the interface neighbourhood and
  // divides by area; both are linear in the per-cell totals, so the same
  // operators applied to the GPU totals give the same convention.
  const T sourceArea = GD * static_cast<T>(dims[0]);
  const T rayRate = sourceFlux * sourceArea / static_cast<T>(numRays);
  std::vector<T> spread(fill.size(), T(0));
  for (size_t p = 0; p < raw.size(); ++p)
    if (raw[p] > 0)
      cpuFlux.deposit(spread, primIdx[p], static_cast<T>(raw[p]) * rayRate);
  std::vector<T> gpuF(fill.size(), T(0));
  for (size_t flat = 0; flat < sites; ++flat) {
    size_t rem = flat;
    std::array<int, D> idx{};
    for (int d = 0; d < D; ++d) {
      idx[d] = static_cast<int>(rem % static_cast<size_t>(dims[d]));
      rem /= static_cast<size_t>(dims[d]);
    }
    const int id = lat.cellId(idx);
    if (id < 0 || spread[id] <= T(0))
      continue;
    const T area = cpuFlux.areaAt(idx);
    if (area > T(1e-2) * GD)
      gpuF[id] = spread[id] / area;
  }
  cpuFlux.smooth(gpuF, 1);

  // ---------------- compare ----------------
  // Probes: the lit mask top (must see the source flux), and the shadowed
  // floor (where transport inside the feature decides the answer).
  auto probe = [&](const std::vector<T> &f, T xLo, T xHi, bool top) {
    T sum = 0;
    int n = 0;
    for (int i = 0; i < dims[0]; ++i) {
      const T x = minC[0] + GD * (T(i) + T(0.5));
      if (x < xLo || x > xHi)
        continue;
      for (int jj = 0; jj < dims[1]; ++jj) {
        const int j = top ? dims[1] - 1 - jj : jj;
        const int id = lat.cellId({i, j});
        if (id < 0 || f[id] <= T(0))
          continue;
        sum += f[id];
        ++n;
        break; // topmost (or bottommost) cell with flux in this column
      }
    }
    return n ? sum / T(n) : T(0);
  };

  const T cpuTop = probe(cpu.flux, -18., -12., true);
  const T gpuTop = probe(gpuF, -18., -12., true);
  const T cpuFloor = probe(cpu.flux, -3., 3., false);
  const T gpuFloor = probe(gpuF, -3., 3., false);

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "  mask top: cpu " << cpuTop << "  gpu " << gpuTop << "\n";
  std::cout << "  floor:    cpu " << cpuFloor << "  gpu " << gpuFloor << "\n";

  check("the lit mask top sees the source flux, both arms",
        std::abs(cpuTop - sourceFlux) < 0.05 * sourceFlux &&
            std::abs(gpuTop - sourceFlux) < 0.05 * sourceFlux);
  check("the shadowed floor agrees between CPU and GPU",
        std::abs(gpuFloor - cpuFloor) < 0.08 * cpuFloor,
        "difference " +
            std::to_string(100. * std::abs(gpuFloor - cpuFloor) /
                           std::max(cpuFloor, T(1e-12))) +
            " %");

  // ---------------- the driver, with sticking varying per cell ----------------
  // Sticking 0.9 above y=0 (the mask levels), 0.1 below: a strong contrast
  // that the per-primitive upload must reproduce.
  {
    std::vector<T> stick(fill.size(), T(0));
    for (size_t flat = 0; flat < sites; ++flat) {
      size_t rem = flat;
      std::array<int, D> idx{};
      for (int d = 0; d < D; ++d) {
        idx[d] = static_cast<int>(rem % static_cast<size_t>(dims[d]));
        rem /= static_cast<size_t>(dims[d]);
      }
      const int id = lat.cellId(idx);
      if (id < 0)
        continue;
      const T y = minC[1] + GD * (T(idx[1]) + T(0.5));
      stick[id] = y > T(0) ? T(0.9) : T(0.1);
    }
    const auto cpu2 = cpuFlux.trace(numRays, sourceFlux, stick, T(1), 7u);
    ps::VoxelFluxGPU<T, D> driver(lat, fill, context);
    driver.prepareGeometry(cs::NormalEstimator::FillGradientYoungs);
    const auto gpu2 = driver.trace(numRays, sourceFlux, stick, 7u);

    const T cpuFloor2 = probe(cpu2.flux, -3., 3., false);
    const T gpuFloor2 = probe(gpu2, -3., 3., false);
    std::cout << "  per-cell sticking floor: cpu " << cpuFloor2 << "  gpu "
              << gpuFloor2 << "\n";
    check("per-cell sticking floor agrees",
          std::abs(gpuFloor2 - cpuFloor2) < 0.08 * cpuFloor2,
          "difference " +
              std::to_string(100. * std::abs(gpuFloor2 - cpuFloor2) /
                             std::max(cpuFloor2, T(1e-12))) +
              " %");
  }

  // ---------------- a whole chemistry, GPU neutrals ----------------
  // Silane has no ion, so with the neutral transport on the GPU the entire
  // deposition runs. Same mechanism, same lattice, fresh fills: five steps on
  // each engine must deposit the same volume.
  {
    auto mech = ps::readChemicalMechanism<T>(
        std::string(VIENNAPS_MECHANISM_DIR) + "/silane.mechanism.json");
    auto runChem = [&](bool gpu) {
      std::vector<T> f(cellSet->getNumberOfCells(), T(0));
      std::vector<int> m(cellSet->getNumberOfCells());
      for (size_t c = 0; c < f.size(); ++c) {
        m[c] = static_cast<int>(mid[c]);
        f[c] = m[c] == static_cast<int>(ps::Material::GAS) ? T(0) : T(1);
      }
      ps::VoxelChemistry<T, D> vox(mech, lat, f, m);
      vox.setRaysPerStep(60000);
      vox.setUseGPU(gpu);
      auto cov = vox.makeCoverages();
      T moved = 0;
      for (int s = 0; s < 5; ++s)
        moved += vox.step(T(2.0), cov, 1 + s).volumeMoved;
      return moved;
    };
    const T cpuMoved = runChem(false);
    const T gpuMoved = runChem(true);
    std::cout << "  silane, 5 steps, volume moved: cpu " << cpuMoved
              << "  gpu " << gpuMoved << "\n";
    check("a whole silane deposition agrees, GPU neutrals",
          std::abs(gpuMoved - cpuMoved) < 0.05 * std::abs(cpuMoved),
          "difference " +
              std::to_string(100. * std::abs(gpuMoved - cpuMoved) /
                             std::max(std::abs(cpuMoved), T(1e-12))) +
              " %");
  }

  // ---------------- the whole masked etch, ion on the GPU ----------------
  // SF6/O2: neutrals AND the ion on the device, against the CPU arm. The
  // mask must stay, the floor must move the same distance.
  {
    auto mech = ps::readChemicalMechanism<T>(
        std::string(VIENNAPS_MECHANISM_DIR) + "/sf6o2.mechanism.json");
    struct Outcome {
      T maskLossPct, floorZ;
    };
    auto runEtch = [&](bool gpu) {
      std::vector<T> f(cellSet->getNumberOfCells(), T(0));
      std::vector<int> m(cellSet->getNumberOfCells());
      T maskVol0 = 0;
      for (size_t c = 0; c < f.size(); ++c) {
        m[c] = static_cast<int>(mid[c]);
        f[c] = m[c] == static_cast<int>(ps::Material::GAS) ? T(0) : T(1);
        if (m[c] == static_cast<int>(ps::Material::Mask))
          maskVol0 += f[c];
      }
      ps::VoxelChemistry<T, D> vox(mech, lat, f, m);
      vox.setRaysPerStep(60000);
      vox.setUseGPU(gpu);
      auto cov = vox.makeCoverages();
      const T dt = (T(3) / 47.74) / 8;
      for (int s = 0; s < 8; ++s)
        vox.step(dt, cov, 1 + s);
      const auto &labels = vox.materials();
      T maskVol1 = 0;
      for (size_t c = 0; c < f.size(); ++c)
        if (labels[c] == static_cast<int>(ps::Material::Mask))
          maskVol1 += f[c];
      T floorZ = std::numeric_limits<T>::lowest();
      const int i = dims[0] / 2;
      for (int j = dims[1] - 1; j >= 0; --j) {
        const int id = lat.cellId({i, j});
        if (id < 0 || f[id] <= T(1e-6))
          continue;
        floorZ = minC[1] + GD * T(j + 1) - (T(1) - f[id]) * GD;
        break;
      }
      return Outcome{100 * (maskVol0 - maskVol1) / maskVol0, floorZ};
    };
    const auto cpuE = runEtch(false);
    const auto gpuE = runEtch(true);
    std::cout << "  sf6o2 etch: mask loss cpu " << cpuE.maskLossPct
              << " %  gpu " << gpuE.maskLossPct << " %,  floor cpu "
              << cpuE.floorZ << "  gpu " << gpuE.floorZ << "\n";
    check("the mask is static on both engines",
          std::abs(cpuE.maskLossPct) < 0.05 && std::abs(gpuE.maskLossPct) < 0.05);
    check("the etched floor agrees, ion on the GPU",
          std::abs(gpuE.floorZ - cpuE.floorZ) < T(0.4),
          "cpu " + std::to_string(cpuE.floorZ) + " vs gpu " +
              std::to_string(gpuE.floorZ));
  }

  std::cout << "\n";
  if (failures) {
    std::cout << failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "all checks passed\n";
  return 0;
}
