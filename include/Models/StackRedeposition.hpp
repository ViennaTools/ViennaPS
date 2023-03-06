#pragma once

#include <csDenseCellSet.hpp>

#include <lsToDiskMesh.hpp>

#include <psAdvectionCallback.hpp>
#include <psDomain.hpp>

// The selective etching model works in accordance with the geometry generated
// by psMakeStack
template <typename NumericType>
class SelectiveEtchingSurfaceModel : public psSurfaceModel<NumericType> {
public:
  psSmartPointer<std::vector<NumericType>> calculateVelocities(
      psSmartPointer<psPointData<NumericType>> Rates,
      const std::vector<std::array<NumericType, 3>> &coordinates,
      const std::vector<NumericType> &materialIds) override {

    const auto numPoints = materialIds.size();
    std::vector<NumericType> etchRate(numPoints, 0.);

    for (size_t i = 0; i < numPoints; ++i) {
      int matId = static_cast<int>(materialIds[i]);
      if (matId == 0) {
        etchRate[i] = 0;
        continue;
      }

      if (matId % 2 == 0) {
        etchRate[i] = -0.1;
      } else {
        etchRate[i] = 0;
      }
    }

    return psSmartPointer<std::vector<NumericType>>::New(etchRate);
  }
};

template <class T, int D>
class RedepositionDynamics : public psAdvectionCalback<T, D> {
  using psAdvectionCalback<T, D>::domain;

  psSmartPointer<csDenseCellSet<T, D>> cellSet = nullptr;
  int plasmaMaterial = 0;
  T diffusionCoefficient = 1.;
  T sink = 1;
  T scallopStreamVel = 1.;
  T holeStreamVel = 1.;
  T top;
  T holeRadius;
  SelectiveEtchingSurfaceModel<T> surfModel;
  lsSmartPointer<std::vector<T>> velocities;
  std::vector<std::array<T, 3>> nodes;
  static constexpr T eps = 1e-4;
  bool redepoRun = false;

public:
  RedepositionDynamics(psSmartPointer<psDomain<T, D>> passedDomain,
                       const T passedDiffCoeff, const T passedSink,
                       const T passedScallopVel, const T passedHoleVel,
                       const int passedPlasmaMat, const T passedTop,
                       const T passedRadius)
      : diffusionCoefficient(passedDiffCoeff), sink(passedSink),
        scallopStreamVel(passedScallopVel), holeStreamVel(passedHoleVel),
        plasmaMaterial(passedPlasmaMat), top(passedTop),
        holeRadius(passedRadius) {
    domain = passedDomain;
    assert(domain->getUseCellSet());
    cellSet = domain->getCellSet();
    cellSet->addScalarData("byproductSum", 0.);
  }

  bool applyPreAdvect(const T processTime) override final {
    prepare();
    return true;
  }

  bool applyPostAdvect(const T advectionTime) override final {
    cellSet->updateMaterials();
    addByproducts(advectionTime);
    diffuseByproducts(advectionTime);
    return true;
  }

private:
  void diffuseByproducts(const T timeStep) {
    auto data = cellSet->getFillingFractions();
    auto materialIds = cellSet->getScalarData("Material");
    auto elems = cellSet->getElements();
    auto nodes = cellSet->getNodes();
    const auto gridDelta = cellSet->getGridDelta();
    // calculate time discretization
    const T dt =
        std::min(gridDelta * gridDelta / diffusionCoefficient * 0.245, 1.);
    const int numSteps = static_cast<int>(timeStep / dt);
    const T C = dt * diffusionCoefficient / (gridDelta * gridDelta);

    for (int iter = 0; iter < numSteps; iter++) {
      std::vector<T> solution(data->size(), 0.);

      // std::cout << (*std::max_element(materialIds->begin(),
      // materialIds->end()))
      //           << std::endl;

#pragma omp parallel for
      for (int e = 0; e < data->size(); e++) {
        if (materialIds->at(e) != plasmaMaterial) {
          continue;
        }

        int numNeighbors = 0;
        auto coord = nodes[elems[e][0]];
        std::array<csPair<int>, D> gridCoords = {};
        for (int i = 0; i < D; i++) {
          coord[i] += gridDelta / 2.;
          gridCoords[i][0] = -1;
          gridCoords[i][1] = -1;
        }

        for (const auto n : cellSet->getNeighbors(e)) {
          if (materialIds->at(n) != plasmaMaterial)
            continue;

          auto neighborCoord = nodes[elems[n][0]];
          for (int i = 0; i < D; i++)
            neighborCoord[i] += gridDelta / 2.;

          if (csUtil::distance(coord, neighborCoord) < gridDelta + 1e-4) {

            for (int d = 0; d < D; d++) {
              if (coord[d] - neighborCoord[d] > eps) {
                gridCoords[d][0] = n;
              } else if (coord[d] - neighborCoord[d] < -eps) {
                gridCoords[d][1] = n;
              }
            }

            solution[e] += data->at(n);
            numNeighbors++;
          }
        }

        // diffusion
        solution[e] =
            data->at(e) +
            C * (solution[e] - static_cast<T>(numNeighbors) * data->at(e));

        // sink at the top
        if (coord[1] > top - gridDelta) {
          solution[e] -= sink;
          solution[e] = std::max(solution[e], 0.);
          continue;
        }

        // convection
        if (std::abs(coord[0]) < holeRadius) {
          // in hole
          assert(gridCoords[1][1] != -1 && "holeStream up neighbor wrong");
          solution[e] -= dt / gridDelta * holeStreamVel *
                         (data->at(gridCoords[1][1]) - data->at(e));
        } else {
          if (coord[0] < 0) {
            // left side scallop - use forward difference
            assert(gridCoords[0][1] != -1 &&
                   "scallopStream right neighbor wrong");
            solution[e] -= dt / gridDelta * scallopStreamVel *
                           (data->at(gridCoords[0][1]) - data->at(e));
          } else {
            // right side scallop - use backward difference
            assert(gridCoords[0][0] != -1 &&
                   "scallopStream left neighbor wrong");
            solution[e] += dt / gridDelta * scallopStreamVel *
                           (data->at(e) - data->at(gridCoords[0][0]));
          }
        }
      }
      *data = std::move(solution);
    }
    auto sum = cellSet->getScalarData("byproductSum");

#pragma omp parallel for
    for (int e = 0; e < data->size(); e++) {
      if (materialIds->at(e) != plasmaMaterial) {
        continue;
      }

      sum->at(e) += data->at(e);
    }
  }

  void prepare() {
    lsToDiskMesh<T, D> meshConv;
    for (auto ls : *domain->getLevelSets())
      meshConv.insertNextLevelSet(ls);

    auto mesh = lsSmartPointer<lsMesh<T>>::New();
    meshConv.setMesh(mesh);
    meshConv.apply();

    nodes = mesh->getNodes();

    velocities = surfModel.calculateVelocities(
        nullptr, nodes, *mesh->getCellData().getScalarData("MaterialIds"));
  }

  void addByproducts(const T timeStep) {
    for (size_t j = 0; j < nodes.size(); j++) {
      cellSet->addFillingFraction(nodes[j], -velocities->at(j) * timeStep);
    }
  }
};

template <typename NumericType, int D>
class RedepositionSurfaceModel : public psSurfaceModel<NumericType> {
public:
  RedepositionSurfaceModel(
      psSmartPointer<csDenseCellSet<NumericType, D>> passedCellSet,
      const NumericType passedFactor, const NumericType passedTop,
      const int passedPlasmaMat)
      : cellSet(passedCellSet), top(passedTop), plasmaMaterial(passedPlasmaMat),
        depoMaterial(passedPlasmaMat - 1), depositionFactor(passedFactor) {}

  psSmartPointer<std::vector<NumericType>> calculateVelocities(
      psSmartPointer<psPointData<NumericType>> Rates,
      const std::vector<std::array<NumericType, 3>> &coordinates,
      const std::vector<NumericType> &materialIds) override {

    const auto numPoints = materialIds.size();
    std::vector<NumericType> depoRate(numPoints, 0.);
    auto ff = cellSet->getScalarData("byproductSum");
    auto cellMatIds = cellSet->getScalarData("Material");

    for (size_t i = 0; i < numPoints; ++i) {
      int matId = static_cast<int>(materialIds[i]);
      if (matId == 0 || matId == plasmaMaterial)
        continue;

      if (matId % 2 != 0 || matId == depoMaterial) {
        auto node = coordinates[i];

        if (node[D - 1] >= top)
          continue;

        auto cellIdx = cellSet->getIndex(node);
        int n = 0;
        for (const auto ni : cellSet->getNeighbors(cellIdx)) {
          const int neighborMatId = cellMatIds->at(ni);

          if (neighborMatId == plasmaMaterial) {
            depoRate[i] += ff->at(ni);
            n++;
          }
        }
        if (n > 1)
          depoRate[i] /= static_cast<NumericType>(n);
        depoRate[i] *= depositionFactor;
      }
    }

    return psSmartPointer<std::vector<NumericType>>::New(depoRate);
  }

private:
  psSmartPointer<csDenseCellSet<NumericType, D>> cellSet;
  const NumericType top = 0.;
  const int plasmaMaterial = 0;
  const int depoMaterial = 0;
  const NumericType depositionFactor = 1.;
};