#pragma once

#include <FiDiTi/FDTD.hpp>

#include <csDenseCellSet.hpp>
#include <lsFromMesh.hpp>
#include <lsToMesh.hpp>

#include <cmath>

template <class Scalar, int D>
auto changeGridSpacing(
    viennals::SmartPointer<viennals::Domain<Scalar, D>> levelSet,
    Scalar newGridDelta) {
  auto mesh = viennals::SmartPointer<viennals::Mesh<Scalar>>::New();
  viennals::ToSurfaceMesh<Scalar, D>(levelSet, mesh).apply();

  const auto &grid = levelSet->getGrid();
  auto min = grid.getMinBounds(), max = grid.getMaxBounds();
  auto bcs = grid.getBoundaryConditions();
  for (int i = 0; i < D; i++) {
    // Undo -1 change done in Grid constructor...
    if (bcs[i] == viennahrle::BoundaryType::PERIODIC_BOUNDARY)
      min[i]++;

    min[i] = std::floor(min[i] * grid.getGridDelta() / newGridDelta);
    max[i] = std::ceil(max[i] * grid.getGridDelta() / newGridDelta);
  }
  viennahrle::Grid<D> newGrid(&min[0], &max[0], newGridDelta, &bcs[0]);

  auto newLevelSet = decltype(levelSet)::New(newGrid);
  viennals::FromSurfaceMesh<Scalar, D>(newLevelSet, mesh).apply();

  return newLevelSet;
}

template <class Scalar, int D>
auto changeGridSpacing(
    const std::vector<viennals::SmartPointer<viennals::Domain<Scalar, D>>>
        &levelSets,
    Scalar newGridDelta) {
  std::decay_t<decltype(levelSets)> newLevelSets;
  for (const auto &ls : levelSets)
    newLevelSets.push_back(changeGridSpacing(ls, newGridDelta));
  return newLevelSets;
}

template <class Scalar, int D>
auto extractMaterials(viennacs::DenseCellSet<Scalar, D> &cellSet,
                      fidi::Vec<D, int> N, fidi::Rect<D, int> innerBounds) {
  const auto &csMatIds = *cellSet.getScalarData("Material");

  std::vector<int> csMatIdsInt(csMatIds.begin(), csMatIds.end());

  return fidi::resizeField(csMatIdsInt, innerBounds.size(),
                           fidi::Rect{-innerBounds.min, N - innerBounds.min});
}

template <class Scalar, int D>
auto shrinkField(const std::vector<Scalar> &field, fidi::Vec<D, int> fieldSize,
                 int width) {
  fidi::Rect bounds(fieldSize - width);
  bounds.min += width;

  return fidi::resizeField(field, fieldSize, bounds);
}

template <class Scalar, int D>
fidi::Vec<D, int>
getGridSize(const viennacs::DenseCellSet<Scalar, D> &cellSet) {
  auto dx = cellSet.getGridDelta();
  auto [min, max] = cellSet.getBoundingBox();

  fidi::Vec<D, int> csGridSize;
  for (int i = 0; i < D; ++i)
    csGridSize[i] = std::round((max[i] - min[i]) / dx);

  return csGridSize;
}

template <class Scalar, int D>
fidi::Rect<D, Scalar>
getBounds(const viennacs::DenseCellSet<Scalar, D> &cellSet) {
  auto [min, max] = cellSet.getBoundingBox();

  return {fidi::toVec<D>(&min[0]), fidi::toVec<D>(&max[0])};
}

template <class Scalar, int D>
void setSphereMaterial(std::vector<Scalar> &field, fidi::VecNi<D> fieldSize,
                       fidi::Rect<D, Scalar> bounds,
                       fidi::Vec<D, Scalar> center, Scalar radius,
                       Scalar grid_dx, int material, int oldMaterial) {
  fidi::RectNi<D> box(fidi::VecNi<D>(bounds.min / grid_dx + Scalar(0.5)),
                      fidi::VecNi<D>(bounds.max / grid_dx + Scalar(0.5)));
  auto idxStride = indexStride(fieldSize);

  forEachCell(clamp(box, fidi::RectNi<D>(fieldSize)), [&](fidi::VecNi<D> pos) {
    fidi::Vec<D, Scalar> p(pos);
    p = p * grid_dx + grid_dx / 2;
    auto &x = field[dot(pos, idxStride)];
    if (len_sq(p - center) <= radius * radius && x == oldMaterial)
      x = material;
  });
}

template <class Scalar, int D>
void print(const fidi::FDTD<D> &sim, viennacs::DenseCellSet<Scalar, D> &cellSet,
           fidi::Rect<D, int> innerBounds, int step) {
  fidi::Vec<D, int> N = sim.gridSize();

  auto printField = [&](const auto &field, std::string name) {
    if (field.empty())
      return;
    auto *data = cellSet.addScalarData(name);
    *data = resizeField(field, N, innerBounds);
  };
  for (int i = 0; i < 3; ++i) {
    printField(sim.fieldA(i), "E_" + std::to_string(i + 1));
    printField(sim.fieldB(i), "H_" + std::to_string(i + 1));
  }
  cellSet.writeVTU("FDTD_cellSet_" + std::to_string(step) + ".vtu");
}

template <class Scalar, int D>
void printEnergy(const fidi::FDTD<D> &sim, int diodeHeight, Scalar dx,
                 Scalar cellDepth, int step) {
  Scalar cellVolume = 1;
  for (int i = 0; i < 3; i++)
    cellVolume *= (i < D) ? dx : cellDepth;

  // Region above diode (incoming wave)
  fidi::Rect rect(sim.gridSize());
  rect.min[D - 1] = diodeHeight;

  std::cout << "FDTD energy (" << step
            << "): " << sim.fieldEnergy(rect) * cellVolume << " (top), ";

  // Diode region (transmitted wave)
  rect = fidi::Rect(sim.gridSize());
  rect.max[D - 1] = diodeHeight;

  std::cout << sim.fieldEnergy(rect) * cellVolume << " (diode)\n";
}

template <class Scalar, int D>
Scalar getEnergyFlow(const fidi::FDTD<D> &sim, fidi::RectNi<D> innerBounds,
                     int diodeHeight, Scalar dx, Scalar cellDepth) {
  int gap = 5;
  double scale = 0.5;

  Scalar cellFaceArea = 1;
  for (int i = 0; i < 2; i++)
    cellFaceArea *= (i < D - 1) ? dx : cellDepth;

  // Plane surface below reflective layers
  fidi::Rect plane(innerBounds.min + innerBounds.size() * ((1 - scale) / 2),
                   innerBounds.max - innerBounds.size() * ((1 - scale) / 2));
  plane.max[D - 1] = diodeHeight - gap;
  plane.min[D - 1] = diodeHeight - gap - 1;

  return -sim.surfacePower(plane, D - 1) * cellFaceArea;
}

template <class Scalar, int D>
void runFDTD(viennacs::DenseCellSet<Scalar, D> &cellSet,
             fidi::fdtd::MaterialMap matMap, Scalar time, Scalar outputInterval,
             Scalar wavelength, Scalar diodeHeight) {
  using namespace fidi;

  auto dx = cellSet.getGridDelta();

  Vec<D, int> csGridSize = getGridSize(cellSet);

  // Manually extend grid on all sides for absorbing boundary + TF/SF interface
  int extend = 5 + 2;
  Vec<D, int> N = csGridSize + 2 * extend;
  Rect<D, int> innerBounds = {Vec<D, int>(extend), N - extend};

  FDTD<D> sim(N);
  Scalar energy = 0;

  Scalar dt = dx * sim.timeSpaceStepRatio();
  Scalar pulseWidth = wavelength / constants::c / dt;
  Scalar delay = pulseWidth * 1.3;

  fdtd::Source src;
  src.f = fn::makePulse(pulseWidth, delay);
  // src.duration = 2 * delay;

  sim.addTfsfSource(1, -1, src);
  // sim.addHardSource(innerBounds.max - project(innerBounds.size(), D-1) / 2.0,
  // src); sim.addPlaneSource(fdtd::Boundary::YMax, src);
  sim.addAbsorbingBoundary(2);

  sim.setMaterials(std::move(matMap));
  sim.setCellMaterials(extractMaterials(cellSet, N, innerBounds));

  int numSteps = std::ceil(time / dt);
  int frameInterval =
      std::max<int>(1, std::floor(numSteps * outputInterval / time));

  for (int q = 0; q < numSteps; ++q) {
    sim.step();

    // 2D: Cell depth = diode width (square shape in 3D)
    energy += getEnergyFlow(sim, innerBounds, diodeHeight / dx, dx,
                            csGridSize[0] * dx) *
              dt;

    if (!(q % frameInterval)) {
      print(sim, cellSet, innerBounds, q / frameInterval + 1);
      // printEnergy(sim, diodeHeight / dx, dx, csGridSize[0] * dx, q+1);
    }
  }
  std::cout << "FDTD output interval: " << frameInterval * dt * 1e9 << " fs ("
            << numSteps << " steps)\n";
  std::cout << "Energy flow: " << energy << '\n';
}
