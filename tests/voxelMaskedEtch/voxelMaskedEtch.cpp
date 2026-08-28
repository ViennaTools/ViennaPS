// A masked ion-driven etch on the voxel arm, small enough for CTest and built
// through the SAME construction path the examples ship: MakeTrench, level
// sets into a cell set with a material map, labels copied from the cell set.
//
// This is the regression for a day of failures that no unit test caught,
// because every one of them lived in the seams the units cannot see:
//
//   - the example's init loop discarded the cell set's material labels, so
//     the mask etched at the silicon rate (asserted: mask cells exist at t=0,
//     and mask volume is static under an etch that moves the floor);
//   - the gas cover stopped below the mask top, so the slot's gas column had
//     no cells (asserted: the floor actually etches);
//   - a material-blind advance drained the mask foot through the gas wildcard
//     (asserted: mask volume static to a tight tolerance);
//   - fills below one half kept solid labels, matter floated in gas
//     (asserted: the 0.5 labelling rule, and no unanchored partial cell).
//
// Runs with BOTH traversal engines: the physics may not depend on how a ray
// finds the surface.

#include <models/psChemicalMechanismIO.hpp>
#include <models/psVoxelChemistry.hpp>
#include <psDomain.hpp>
#include <geometries/psMakeTrench.hpp>

#include <lsMakeGeometry.hpp>

#include <iomanip>
#include <iostream>
#include <string>

namespace ls = viennals;
namespace cs = viennacs;
namespace ps = viennaps;

using T = double;
constexpr int D = 3;

#ifndef VIENNAPS_MECHANISM_DIR
#define VIENNAPS_MECHANISM_DIR "."
#endif

int failures = 0;

void check(const std::string &name, bool ok, const std::string &detail = "") {
  std::cout << "  [" << (ok ? "PASS" : "FAIL") << "] " << name;
  if (!detail.empty())
    std::cout << "   " << detail;
  std::cout << "\n";
  if (!ok)
    ++failures;
}

void run(cs::TraversalEngine engine, const std::string &label) {
  const T GD = 1.0, W = 10., maskH = 8., depth = 6.;
  auto dom = ps::SmartPointer<ps::Domain<T, D>>::New(GD, T(4) * W, T(4) * W);
  // flat substrate; the slot in the mask is the only opening
  ps::MakeTrench<T, D>(dom, W, T(0), T(0), maskH, T(0), false,
                       ps::Material::Si, ps::Material::Mask)
      .apply();

  auto topLS = dom->getLevelSets().back();
  auto deep = ls::SmartPointer<ls::Domain<T, D>>::New(topLS->getGrid());
  {
    T o[D] = {0., 0., -(depth + T(6))}, n[D] = {0., 0., 1.};
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
  cellSet->fromLevelSets(lss, matMap, maskH + T(4)); // cover clears the mask

  cs::LatticeMap<T, D> lat(*cellSet);
  const auto &mid = *cellSet->getScalarData("Material");
  std::vector<T> fill(cellSet->getNumberOfCells(), T(0));
  std::vector<int> material(cellSet->getNumberOfCells());
  T maskVol0 = 0;
  for (size_t c = 0; c < fill.size(); ++c) {
    material[c] = static_cast<int>(mid[c]); // KEEP the labels: the lesson
    fill[c] = material[c] == static_cast<int>(ps::Material::GAS) ? T(0) : T(1);
    if (material[c] == static_cast<int>(ps::Material::Mask))
      maskVol0 += fill[c];
  }
  check(label + ": the initial state contains the mask", maskVol0 > T(0),
        std::to_string(static_cast<long>(maskVol0)) + " mask cells");

  auto mech = ps::readChemicalMechanism<T>(
      std::string(VIENNAPS_MECHANISM_DIR) + "/sf6o2.mechanism.json");
  ps::VoxelChemistry<T, D> vox(mech, lat, fill, material);
  vox.setRaysPerStep(30000);
  vox.setTraversalEngine(engine);
  auto cov = vox.makeCoverages();
  const T dt = (T(3) / 47.74) / 8; // ~3 nm of floor over 8 steps
  for (int s = 0; s < 8; ++s)
    vox.step(dt, cov, 1 + s);

  const auto &labels = vox.materials();

  // The mask is static. Not approximately: the physical sputter dose at these
  // energies is a fraction of a percent, and anything beyond that is the
  // advance leaking mask matter -- the bug this test exists to catch.
  T maskVol1 = 0;
  for (size_t c = 0; c < fill.size(); ++c)
    if (labels[c] == static_cast<int>(ps::Material::Mask))
      maskVol1 += fill[c];
  const T maskLoss = 100 * (maskVol0 - maskVol1) / maskVol0;
  check(label + ": the mask volume is static",
        std::abs(maskLoss) < T(0.05),
        "changed " + std::to_string(maskLoss) + " %");

  // The floor moved: the slot's gas column has cells and the etch reaches it.
  const auto &dims = lat.dims();
  T floorZ = std::numeric_limits<T>::lowest();
  {
    const int i = dims[0] / 2, j = dims[1] / 2;
    for (int k = dims[2] - 1; k >= 0; --k) {
      const int id = lat.cellId({i, j, k});
      if (id < 0 || fill[id] <= T(1e-6))
        continue;
      floorZ = lat.minCorner()[2] + GD * T(k + 1) - (T(1) - fill[id]) * GD;
      break;
    }
  }
  check(label + ": the floor etched at least two cells", floorZ < -T(2),
        "floor z = " + std::to_string(floorZ));

  // The labelling rule: fill >= 0.5 is material, below is gas. And the
  // anchoring invariant: no partial cell fuller than every neighbour --
  // matter must not float.
  int badLabel = 0, floating = 0;
  size_t total = 1;
  for (int d = 0; d < D; ++d)
    total *= static_cast<size_t>(dims[d]);
  for (size_t flat = 0; flat < total; ++flat) {
    size_t rem = flat;
    std::array<int, D> idx{};
    for (int d = 0; d < D; ++d) {
      idx[d] = static_cast<int>(rem % static_cast<size_t>(dims[d]));
      rem /= static_cast<size_t>(dims[d]);
    }
    const int id = lat.cellId(idx);
    if (id < 0)
      continue;
    const bool gas = labels[id] == static_cast<int>(ps::Material::GAS);
    if ((fill[id] >= T(0.5)) == gas)
      ++badLabel;
    const T f = fill[id];
    if (f > T(1e-12) && f < T(1) - T(1e-12)) {
      T maxNbr = -1;
      for (int d = 0; d < D; ++d)
        for (int sgn = -1; sgn <= 1; sgn += 2) {
          auto nb = idx;
          nb[d] += sgn;
          const int nid = lat.cellId(nb);
          maxNbr = std::max(maxNbr, nid < 0 ? f : fill[nid]);
        }
      if (f > maxNbr + T(1e-9))
        ++floating;
    }
  }
  check(label + ": fill >= 0.5 is material, below is gas", badLabel == 0,
        std::to_string(badLabel) + " mislabelled");
  check(label + ": no partial cell fuller than all its neighbours",
        floating == 0, std::to_string(floating) + " floating");
}

int main() {
  std::cout << "Voxel masked etch: the seams the units cannot see\n\n";
  ps::Logger::setLogLevel(ps::LogLevel::WARNING);
  ps::units::Length::setUnit("nm");
  ps::units::Time::setUnit("s");

  run(cs::TraversalEngine::GridDDA, "DDA");
  run(cs::TraversalEngine::EmbreeBVH, "BVH");

  std::cout << "\n";
  if (failures) {
    std::cout << failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "all checks passed\n";
  return 0;
}
