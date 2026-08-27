#include <psVoxelChemistry.hpp>
#include <psChemicalMechanismIO.hpp>

#include <lsMakeGeometry.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// A chemical mechanism run on a voxel grid, against the answer it is known to
// have.
//
// The mechanism is the same object the level-set arm uses, read from the same
// file, so this does not test the chemistry -- that is already covered. What
// it tests is the coupling: whether the flux a cell receives, divided by the
// area it presents, is the flux density the rate law wants, and whether the
// velocity that comes back moves the surface by the right amount.
//
// On a blanket the answer is analytic: no shadowing, so every cell sees the
// incident flux, and the surface moves at the rate the mechanism reports for
// that flux. If the voxel arm cannot reproduce that, nothing it says about a
// trench is worth reading.

namespace ls = viennals;
namespace cs = viennacs;
namespace ps = viennaps;

using T = double;

int failures = 0;

void check(const std::string &name, bool ok, const std::string &detail = "") {
  std::cout << "  [" << (ok ? "PASS" : "FAIL") << "] " << name;
  if (!detail.empty())
    std::cout << "   " << detail;
  std::cout << "\n";
  if (!ok)
    ++failures;
}

template <int D> cs::DenseCellSet<T, D> makeDomain(T gridDelta, T lo, T hi) {
  ls::BoundaryConditionEnum bc[D];
  for (int i = 0; i < D - 1; ++i)
    bc[i] = ls::BoundaryConditionEnum::REFLECTIVE_BOUNDARY;
  bc[D - 1] = ls::BoundaryConditionEnum::INFINITE_BOUNDARY;
  T bounds[2 * D];
  for (int i = 0; i < D; ++i) {
    bounds[2 * i] = lo;
    bounds[2 * i + 1] = hi;
  }
  T origin[D] = {}, normal[D] = {};
  origin[D - 1] = hi;
  normal[D - 1] = 1.;
  auto plane = ls::SmartPointer<ls::Domain<T, D>>::New(bounds, bc, gridDelta);
  ls::MakeGeometry<T, D>(plane,
                         ls::SmartPointer<ls::Plane<T, D>>::New(origin, normal))
      .apply();
  std::vector<ls::SmartPointer<ls::Domain<T, D>>> lss{plane};
  cs::DenseCellSet<T, D> cellSet;
  cellSet.setCellSetPosition(false);
  cellSet.setCoverMaterial(0);
  cellSet.fromLevelSets(lss, nullptr, lo);
  return cellSet;
}

/// The height of the surface in one column, as the fractions report it.
T surfaceHeight(const std::vector<T> &fill, const cs::LatticeMap<T, 2> &lattice,
                int column) {
  T height = lattice.minCorner()[1];
  for (int j = 0; j < lattice.dims()[1]; ++j) {
    const int id = lattice.cellId({column, j});
    if (id >= 0)
      height += fill[id] * lattice.gridDelta();
  }
  return height;
}

void blanket(const std::string &file) {
  constexpr int D = 2;
  const T gridDelta = 1.0;

  auto mech = ps::readChemicalMechanism<T>(file);

  // The analytic answer: on a blanket every cell sees the incident flux.
  const auto gamma = mech.sourceFluxes(ps::Material(ps::BuiltInMaterial::Si));
  const auto k = mech.rateConstantsFor(ps::Material(ps::BuiltInMaterial::Si));
  std::vector<T> theta(mech.coverageNames.size(), T(0));
  mech.solveCoverages(gamma, k, theta);
  const T analytic =
      mech.growthRate(gamma, k, theta, ps::Material(ps::BuiltInMaterial::Si));

  auto cellSet = makeDomain<D>(gridDelta, -20., 20.);
  cs::LatticeMap<T, D> lattice(cellSet);
  std::vector<T> fill(cellSet.getNumberOfCells(), T(0));
  cs::fillFromSignedDistance<T, D>(lattice, fill,
                                   [](const viennacore::Vec3D<T> &p) {
                                     return p[1] - T(0);
                                   });
  std::vector<int> material(cellSet.getNumberOfCells(),
                            static_cast<int>(ps::BuiltInMaterial::Si));

  ps::VoxelChemistry<T, D> voxel(mech, lattice, fill, material);
  voxel.setRaysPerStep(200000);
  auto coverages = voxel.makeCoverages();

  const int column = lattice.dims()[0] / 2;
  const T start = surfaceHeight(fill, lattice, column);

  const T dt = std::abs(analytic) > 0 ? 0.2 / std::abs(analytic) : 0.1;
  const int steps = 10;
  ps::VoxelChemistry<T, D>::StepReport last{};
  for (int s = 0; s < steps; ++s)
    last = voxel.step(dt, coverages, 1 + s);

  const T moved = surfaceHeight(fill, lattice, column) - start;
  const T simulated = moved / (dt * steps);

  std::cout << "   " << file << "\n";
  std::cout << "     mechanism " << mech.name << ", " << mech.coverageNames.size()
            << " coverages\n";
  std::cout << "     analytic  " << std::fixed << std::setprecision(5)
            << analytic << " nm/s\n";
  std::cout << "     voxel     " << simulated << " nm/s   over "
            << last.surfaceCells << " surface cells\n";
  std::cout << "     spread    " << last.minVelocity << " to "
            << last.maxVelocity << " nm/s\n";

  const T relative =
      std::abs(analytic) > 1e-12 ? std::abs(simulated - analytic) / std::abs(analytic)
                                 : std::abs(simulated - analytic);
  check(mech.name + ": the blanket rate matches the analytic one",
        relative < 0.05,
        std::to_string(100 * relative) + "% from analytic");
  check(mech.name + ": nothing is lost off the grid",
        std::abs(last.volumeLost) < 1e-9,
        "lost " + std::to_string(last.volumeLost));
}

#ifndef VIENNAPS_MECHANISM_DIR
#define VIENNAPS_MECHANISM_DIR "."
#endif

int main(int argc, char **argv) {
  std::cout << "Voxel chemistry: a mechanism on a grid, against its analytic "
               "blanket rate\n\n";

  // Three mechanisms that between them exercise the paths a coupling can get
  // wrong: one with no coverages at all, which isolates geometry and advance;
  // one with an ion yield channel, whose flux the mechanism weights rather
  // than the tracer; and one whose sticking is 4e-4, where a ray keeps its
  // weight and every re-emission convention matters.
  const std::vector<std::string> shipped = {
      std::string(VIENNAPS_MECHANISM_DIR) + "/ar_sputter.mechanism.json",
      std::string(VIENNAPS_MECHANISM_DIR) + "/cf4ar_etch.mechanism.json",
      std::string(VIENNAPS_MECHANISM_DIR) + "/silane.mechanism.json"};

  if (argc > 1)
    for (int i = 1; i < argc; ++i)
      blanket(argv[i]);
  else
    for (const auto &file : shipped)
      blanket(file);

  std::cout << "\n";
  if (failures) {
    std::cout << failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "all checks passed\n";
  return 0;
}
