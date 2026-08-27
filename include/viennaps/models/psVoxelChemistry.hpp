#pragma once

#include "psSurfaceChemistry.hpp"

#include <csVoxelAdvance.hpp>
#include <csVoxelFlux.hpp>

namespace viennaps {

using namespace viennacore;

/// Runs a chemical mechanism on a voxel grid instead of a level set.
///
/// The mechanism is untouched -- the same `ChemicalMechanism`, read from the
/// same file, solving the same coverages and returning the same velocity as
/// the level-set path. That is the point: with the chemistry provably
/// identical, a difference between the two arms can only come from how the
/// surface is represented, how flux reaches it, and how it moves.
///
/// What differs, and only this:
///
///   level set   ViennaRay over a disk or triangle surface; velocities
///               extended into a field and advected.
///   voxel       a grid walk over filling fractions; velocities applied as
///               volume, cell by cell.
///
/// The coverage solve runs PER CELL, because flux varies over the surface and
/// the steady state is not linear in it. Cells with no interface are skipped:
/// they have no surface to react on.
template <class NumericType, int D> class VoxelChemistry {
  ChemicalMechanism<NumericType> &mech_;
  const viennacs::LatticeMap<NumericType, D> &lattice_;
  std::vector<NumericType> &fill_;
  std::vector<int> material_; ///< per cell, as the cell set reported it

  viennacs::VoxelAdvance<NumericType, D> advance_;

  size_t raysPerStep_ = 100000;
  NumericType minimumArea_ = 0; // set from the grid spacing in the constructor
  int coverageIterations_ = 500;
  NumericType coverageTolerance_ = 1e-13;

public:
  VoxelChemistry(ChemicalMechanism<NumericType> &mech,
                 const viennacs::LatticeMap<NumericType, D> &lattice,
                 std::vector<NumericType> &fill,
                 std::vector<int> material)
      : mech_(mech), lattice_(lattice), fill_(fill),
        material_(std::move(material)), advance_(lattice) {
    NumericType faceArea = 1;
    for (int d = 0; d < D - 1; ++d)
      faceArea *= lattice.gridDelta();
    minimumArea_ = NumericType(1e-2) * faceArea;
  }

  void setRaysPerStep(size_t n) { raysPerStep_ = n; }
  void setCoverageParameters(int iterations, NumericType tolerance) {
    coverageIterations_ = iterations;
    coverageTolerance_ = tolerance;
  }

  struct StepReport {
    size_t surfaceCells = 0;
    NumericType meanVelocity = 0;
    NumericType minVelocity = 0;
    NumericType maxVelocity = 0;
    NumericType volumeMoved = 0;
    NumericType volumeLost = 0;
  };

  /// The incident flux of every traced gas species, per cell.
  ///
  /// INCIDENT, not absorbed. The level-set particle deposits its full ray
  /// weight and lets the rate law apply the sticking, so this must do the
  /// same: depositing `weight * sticking` here would apply the sticking twice,
  /// and for silane, whose sticking is 4e-4 at 900 K, that is a rate three
  /// orders of magnitude too small. The sticking still governs how much weight
  /// survives to be re-emitted -- it is the deposit that must be unweighted.
  ///
  /// Untraced species are left empty here; `step` fills them from
  /// `sourceFluxes`, which knows to weight an ion channel by its yield.
  std::vector<std::vector<NumericType>> traceFluxes(unsigned seed) const {
    const size_t nGas = mech_.gas.size();
    std::vector<std::vector<NumericType>> gamma(
        nGas, std::vector<NumericType>(fill_.size(), NumericType(0)));

    for (size_t g = 0; g < nGas; ++g) {
      const auto &species = mech_.gas[g];
      if (!species.traced || species.isIonChannel)
        continue; // filled per cell from sourceFluxes, which weights ion yields
      viennacs::VoxelFlux<NumericType, D> flux(lattice_, fill_);
      const auto result =
          flux.trace(raysPerStep_, species.sourceFlux,
                     mech_.stickingOf(static_cast<int>(g),
                                      Material(BuiltInMaterial::Si)),
                     NumericType(1), seed + static_cast<unsigned>(g));
      gamma[g] = result.flux;
    }
    return gamma;
  }

  /// One step: trace, solve the chemistry on every surface cell, move the
  /// surface.
  StepReport step(NumericType dt, std::vector<std::vector<NumericType>> &theta,
                  unsigned seed = 1) {
    const auto gamma = traceFluxes(seed);
    const size_t nCov = mech_.coverageNames.size();
    const size_t nGas = mech_.gas.size();

    std::vector<NumericType> velocity(fill_.size(), NumericType(0));

    StepReport report;
    report.minVelocity = std::numeric_limits<NumericType>::max();
    report.maxVelocity = std::numeric_limits<NumericType>::lowest();

    const auto &dims = lattice_.dims();
    size_t sites = 1;
    for (int d = 0; d < D; ++d)
      sites *= static_cast<size_t>(dims[d]);

    std::array<int, D> idx{};
    for (size_t flat = 0; flat < sites; ++flat) {
      size_t rem = flat;
      for (int d = 0; d < D; ++d) {
        idx[d] = static_cast<int>(rem % static_cast<size_t>(dims[d]));
        rem /= static_cast<size_t>(dims[d]);
      }
      const int id = lattice_.cellId(idx);
      if (id < 0)
        continue;
      // A cell has to hold enough interface to be a surface. Flux density is
      // a rate over an area, so a cell with a sliver of area divides a small
      // number by a smaller one and reports a flux far above anything
      // incident -- which the chemistry then believes. On silane that put a
      // cell at 2.5 nm/s against a blanket rate of 0.07.
      const NumericType area = advance_.interfaceArea(fill_, idx);
      if (area <= minimumArea_)
        continue;

      const Material material = MaterialMap::mapToMaterial(material_[id]);
      // Start from the analytic incident fluxes, which carry the ion yield at
      // normal incidence for any yield channel, then replace the entries a
      // trace actually resolved.
      auto cellGamma = mech_.sourceFluxes(material);
      for (size_t g = 0; g < nGas; ++g)
        if (mech_.gas[g].traced && !mech_.gas[g].isIonChannel)
          cellGamma[g] = gamma[g][id];

      const auto k = mech_.rateConstantsFor(material);
      if (nCov > 0) {
        // Continue from this cell's previous coverages: they are close to the
        // new steady state, so Newton converges in a few steps instead of
        // climbing from a bare surface every time.
        mech_.solveCoverages(cellGamma, k, theta[id], coverageIterations_,
                             coverageTolerance_);
      }
      velocity[id] = mech_.growthRate(cellGamma, k, theta[id], material);

      ++report.surfaceCells;
      report.meanVelocity += velocity[id];
      report.minVelocity = std::min(report.minVelocity, velocity[id]);
      report.maxVelocity = std::max(report.maxVelocity, velocity[id]);
    }
    if (report.surfaceCells)
      report.meanVelocity /= static_cast<NumericType>(report.surfaceCells);
    else
      report.minVelocity = report.maxVelocity = 0;

    const auto moved = advance_.apply(fill_, velocity, dt);
    report.volumeMoved = moved.volumeRequested;
    report.volumeLost = moved.volumeLost;
    return report;
  }

  /// Coverages for every cell, starting from a bare surface.
  std::vector<std::vector<NumericType>> makeCoverages() const {
    return std::vector<std::vector<NumericType>>(
        fill_.size(),
        std::vector<NumericType>(mech_.coverageNames.size(), NumericType(0)));
  }
};

} // namespace viennaps
