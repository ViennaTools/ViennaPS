#pragma once

#include "psVoxelChemistry.hpp"

#include <lsMakeGeometry.hpp>

#include <string>
#include <vector>

namespace viennaps {

using namespace viennacore;

/// The voxel arm as one self-contained object: hand it a Domain and a
/// mechanism, and it owns everything the loose parts of the C++ examples
/// wire by hand -- the cell set built from the domain's level sets WITH
/// their materials, the deep solid margin, the gas cover above the tallest
/// surface, the lattice, the fills, the labels, the chemistry, and the
/// coverages. This is also the Python face of the voxel method, where
/// keeping reference-held vectors alive by hand is not an option.
///
/// The hard-won construction rules live here so no caller can miss them:
/// the material map is REQUIRED (without it the cover material is never
/// applied and gas reads as solid); the cover must clear the TALLEST
/// surface or a mask's slot has no gas cells; the labels are COPIED from
/// the cell set (discarding them once etched a mask at the silicon rate);
/// and a masked run whose initial state holds no Mask cell refuses to run.
template <class NumericType, int D> class VoxelProcess {
  ChemicalMechanism<NumericType> mech_; ///< owned: the chemistry keeps a
                                        ///< reference, and Python must not
                                        ///< have to keep the argument alive
  SmartPointer<viennacs::DenseCellSet<NumericType, D>> cellSet_;
  viennacs::LatticeMap<NumericType, D> lattice_;
  std::vector<NumericType> fill_;
  std::vector<int> material_;
  std::unique_ptr<VoxelChemistry<NumericType, D>> chemistry_;
  std::vector<std::vector<NumericType>> coverages_;

public:
  using StepReport = typename VoxelChemistry<NumericType, D>::StepReport;

  /// `depthBelow`: solid margin under the deepest surface point the process
  /// will reach -- an etch that runs out of lattice starves its floor.
  /// `coverAbove`: gas above the TALLEST surface point -- a deposit grows
  /// into it, and a masked etch needs its slot filled with gas cells.
  VoxelProcess(SmartPointer<Domain<NumericType, D>> domain,
               const ChemicalMechanism<NumericType> &mech,
               NumericType depthBelow, NumericType coverAbove)
      : mech_(mech) {
    auto topLS = domain->getLevelSets().back();
    auto deep =
        SmartPointer<viennals::Domain<NumericType, D>>::New(topLS->getGrid());
    {
      NumericType o[D] = {};
      NumericType n[D] = {};
      o[D - 1] = -depthBelow;
      n[D - 1] = 1.;
      viennals::MakeGeometry<NumericType, D>(
          deep, SmartPointer<viennals::Plane<NumericType, D>>::New(o, n))
          .apply();
    }
    std::vector<SmartPointer<viennals::Domain<NumericType, D>>> lss{deep};
    auto matMap = SmartPointer<viennals::MaterialMap>::New();
    matMap->insertNextMaterial(static_cast<int>(Material::Si));
    for (size_t l = 0; l < domain->getLevelSets().size(); ++l) {
      lss.push_back(domain->getLevelSets()[l]);
      matMap->insertNextMaterial(
          static_cast<int>(domain->getMaterialMap()->getMaterialAtIdx(l)));
    }
    cellSet_ = SmartPointer<viennacs::DenseCellSet<NumericType, D>>::New();
    cellSet_->setCellSetPosition(true);
    cellSet_->setCoverMaterial(static_cast<int>(Material::GAS));
    cellSet_->fromLevelSets(lss, matMap, coverAbove);

    lattice_.build(*cellSet_);
    const auto &mid = *cellSet_->getScalarData("Material");
    fill_.assign(cellSet_->getNumberOfCells(), NumericType(0));
    material_.resize(cellSet_->getNumberOfCells());
    size_t maskCells = 0;
    for (size_t c = 0; c < fill_.size(); ++c) {
      material_[c] = static_cast<int>(mid[c]); // KEEP the labels
      fill_[c] =
          material_[c] == static_cast<int>(Material::GAS) ? NumericType(0)
                                                          : NumericType(1);
      if (material_[c] == static_cast<int>(Material::Mask))
        ++maskCells;
    }
    // a masked domain whose voxel state has no mask is wrong at t = 0
    if (domain->getLevelSets().size() > 1 && maskCells == 0) {
      Logger::getInstance()
          .addError("VoxelProcess: the domain carries a mask level set but "
                    "the cell set holds no Mask cell -- the cover or the "
                    "material map lost it.")
          .print();
    }

    chemistry_ = std::make_unique<VoxelChemistry<NumericType, D>>(
        mech_, lattice_, fill_, material_);
    coverages_ = chemistry_->makeCoverages();
  }

  void setRaysPerStep(size_t rays) { chemistry_->setRaysPerStep(rays); }
  void setNormalEstimator(viennacs::NormalEstimator e) {
    chemistry_->setNormalEstimator(e);
  }
  void setTraversalEngine(viennacs::TraversalEngine e) {
    chemistry_->setTraversalEngine(e);
  }
  void setUseGPU(bool use) { chemistry_->setUseGPU(use); }

  /// One step of `dt`. The coverages carry over between calls.
  StepReport step(NumericType dt, unsigned seed = 1) {
    return chemistry_->step(dt, coverages_, seed);
  }

  /// `duration` split into `steps` equal steps; returns the totals, with
  /// the per-phase seconds summed for benchmarking.
  StepReport apply(NumericType duration, int steps) {
    StepReport total;
    for (int s = 0; s < steps; ++s) {
      const auto r = step(duration / static_cast<NumericType>(steps),
                          static_cast<unsigned>(1 + s));
      total.surfaceCells = r.surfaceCells;
      total.meanVelocity = r.meanVelocity;
      total.minVelocity = r.minVelocity;
      total.maxVelocity = r.maxVelocity;
      total.volumeMoved += r.volumeMoved;
      total.volumeLost += r.volumeLost;
      total.secondsTransport += r.secondsTransport;
      total.secondsChemistry += r.secondsChemistry;
      total.secondsAdvance += r.secondsAdvance;
      total.secondsRelabel += r.secondsRelabel;
    }
    return total;
  }

  /// The current state as a VTU: filling fractions and the EVOLVED labels.
  void writeCells(const std::string &fileName) {
    auto &ff = *cellSet_->getFillingFractions();
    auto &mm = *cellSet_->getScalarData("Material");
    const auto &labels = chemistry_->materials();
    for (size_t c = 0; c < fill_.size(); ++c) {
      ff[c] = fill_[c];
      mm[c] = static_cast<NumericType>(labels[c]);
    }
    cellSet_->writeVTU(fileName);
  }

  // Enough state for any probe a script wants to write itself.
  const std::vector<NumericType> &fills() const { return fill_; }
  const std::vector<int> &materials() const { return chemistry_->materials(); }
  std::array<int, D> dims() const { return lattice_.dims(); }
  std::array<NumericType, D> minCorner() const { return lattice_.minCorner(); }
  NumericType gridDelta() const { return lattice_.gridDelta(); }
  int cellId(const std::array<int, D> &idx) const {
    return lattice_.cellId(idx);
  }
};

} // namespace viennaps
