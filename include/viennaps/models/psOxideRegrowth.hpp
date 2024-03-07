#pragma once

#include "../cellSet/csDenseCellSet.hpp"

#include "../psAdvectionCallback.hpp"
#include "../psProcessModel.hpp"
#include "../psToDiskMesh.hpp"

#include <lsAdvect.hpp>

namespace OxideRegrowthImplementation {
template <class NumericType>
class SelectiveEtchingVelocityField : public psVelocityField<NumericType> {
public:
  SelectiveEtchingVelocityField(const NumericType pRate,
                                const NumericType pOxideRate)
      : rate(pRate), oxide_rate(pOxideRate) {}

  NumericType getScalarVelocity(const std::array<NumericType, 3> &coordinate,
                                int matId,
                                const std::array<NumericType, 3> &normalVector,
                                unsigned long pointId) override {
    auto material = psMaterialMap::mapToMaterial(matId);
    if (material == psMaterial::Si3N4) {
      return -rate;
    } else if (material == psMaterial::SiO2) {
      return -oxide_rate;
    } else {
      return 0.;
    }
  }

  int getTranslationFieldOptions() const override { return 0; }

private:
  const NumericType rate;
  const NumericType oxide_rate;
};

template <class NumericType>
class RedepositionVelocityField : public lsVelocityField<NumericType> {
public:
  RedepositionVelocityField(
      const std::vector<NumericType> &passedVelocities,
      const std::vector<std::array<NumericType, 3>> &points)
      : velocities(passedVelocities), kdTree(points) {
    assert(points.size() == passedVelocities.size());
    kdTree.build();
  }

  NumericType getScalarVelocity(const std::array<NumericType, 3> &coordinate,
                                int matId,
                                const std::array<NumericType, 3> &normalVector,
                                unsigned long pointId) override {
    auto nearest = kdTree.findNearest(coordinate);
    assert(nearest->first < velocities.size());
    return velocities[nearest->first];
  }

private:
  const std::vector<NumericType> &velocities;
  psKDTree<NumericType, std::array<NumericType, 3>> kdTree;
};

template <class T, int D>
class ByproductDynamics : public psAdvectionCallback<T, D> {
  using psAdvectionCallback<T, D>::domain;

  const T diffusionCoefficient = 1.;
  const T sink = 1;
  const T scallopStreamVel = 1.;
  const T holeStreamVel = 1.;
  const T top;
  const T holeRadius;
  const T etchRate;
  const T reDepositionFactor;
  const T reDepositionThreshold = 0.1;
  const T reDepoTimeInt = 60;
  const T timeStabilityFactor = 0.245;
  std::vector<std::array<T, 3>> nodes;
  T prevProcTime = 0.;
  unsigned counter = 0;

public:
  ByproductDynamics(const T passedDiffCoeff, const T passedSink,
                    const T passedScallopVel, const T passedHoleVel,
                    const T passedTop, const T passedRadius,
                    const T passedEtchRate, const T passedRedepoFactor,
                    const T passedRedepoThreshold, const T passedRedepoTimeInt,
                    const T passedTimeStabilityFactor)
      : diffusionCoefficient(passedDiffCoeff), sink(passedSink),
        scallopStreamVel(passedScallopVel), holeStreamVel(passedHoleVel),
        top(passedTop), holeRadius(passedRadius), etchRate(passedEtchRate),
        reDepositionFactor(passedRedepoFactor),
        reDepositionThreshold(passedRedepoThreshold),
        reDepoTimeInt(passedRedepoTimeInt),
        timeStabilityFactor(passedTimeStabilityFactor) {}

  bool applyPreAdvect(const T processTime) override {
    assert(domain->getCellSet());
    auto &cellSet = domain->getCellSet();

    // redeposition
    auto mesh = psSmartPointer<lsMesh<T>>::New();
    psToDiskMesh<T, D>(domain, mesh).apply();

    const auto &points = mesh->nodes;
    auto materialIds = mesh->getCellData().getScalarData("MaterialIds");

    // save points where the surface is etched before advection
    nodes.clear();
    nodes.reserve(points.size());
    for (size_t i = 0; i < points.size(); i++) {
      auto material = psMaterialMap::mapToMaterial(materialIds->at(i));
      if (material == psMaterial::Si3N4)
        nodes.push_back(points[i]);
    }
    nodes.shrink_to_fit();

    // redeposit oxide
    if (processTime - reDepoTimeInt * (counter + 1) > -1) {
      const auto numPoints = points.size();
      std::vector<T> depoRate(numPoints, 0.);
      auto ff = cellSet->getScalarData("byproductSum");
      auto cellMatIds = cellSet->getScalarData("Material");

      for (size_t i = 0; i < numPoints; ++i) {
        auto surfaceMaterial = psMaterialMap::mapToMaterial(materialIds->at(i));
        const auto &node = points[i];

        // redeposit only on oxide
        if ((surfaceMaterial == psMaterial::SiO2 ||
             surfaceMaterial == psMaterial::Polymer) &&
            node[D - 1] < top) {
          auto cellIdx = cellSet->getIndex(node);
          int n = 0;
          if (cellIdx == -1)
            continue;
          if (psMaterialMap::mapToMaterial(cellMatIds->at(cellIdx)) ==
              psMaterial::GAS) {
            depoRate[i] = ff->at(cellIdx);
            n++;
          }
          for (const auto ni : cellSet->getNeighbors(cellIdx)) {
            if (ni != -1 && psMaterialMap::mapToMaterial(cellMatIds->at(ni)) ==
                                psMaterial::GAS) {
              depoRate[i] += ff->at(ni);
              n++;
            }
          }

          if (n > 1)
            depoRate[i] /= static_cast<T>(n);

          depoRate[i] /= processTime;

          if (depoRate[i] < reDepositionThreshold)
            depoRate[i] = 0.;

          depoRate[i] *= reDepositionFactor;
        }
      }

      // advect surface
      auto redepoVelField =
          psSmartPointer<RedepositionVelocityField<T>>::New(depoRate, points);

      lsAdvect<T, D> advectionKernel;
      advectionKernel.insertNextLevelSet(domain->getLevelSets()->back());
      advectionKernel.setVelocityField(redepoVelField);
      advectionKernel.setAdvectionTime(processTime - prevProcTime);
      advectionKernel.apply();

      prevProcTime = processTime;
      counter++;
    }

    return true;
  }

  bool applyPostAdvect(const T advectedTime) override {
    auto &cellSet = domain->getCellSet();
    cellSet->updateMaterials();
    const auto gridDelta = cellSet->getGridDelta();

    // add byproducts
    for (size_t j = 0; j < nodes.size(); j++) {
      cellSet->addFillingFraction(nodes[j],
                                  etchRate * advectedTime / gridDelta);
    }

    diffuseByproducts(cellSet, advectedTime);

    return true;
  }

private:
  void diffuseByproducts(psSmartPointer<csDenseCellSet<T, D>> cellSet,
                         const T timeStep) {
    auto data = cellSet->getFillingFractions();
    auto materialIds = cellSet->getScalarData("Material");
    auto elems = cellSet->getElements();
    auto nodes = cellSet->getNodes();
    const auto gridDelta = cellSet->getGridDelta();
    // calculate time discretization
    const T dt = std::min(gridDelta * gridDelta / diffusionCoefficient *
                              timeStabilityFactor,
                          T(1.));
    const int numSteps = static_cast<int>(timeStep / dt);
    const T C = dt * diffusionCoefficient / (gridDelta * gridDelta);
    const T holeC = dt / gridDelta * holeStreamVel;
    const T scallopC = dt / gridDelta * scallopStreamVel;

    for (int ts = 0; ts < numSteps; ts++) {
      std::vector<T> solution(data->size(), 0.);

#pragma omp parallel for
      for (int e = 0; e < data->size(); e++) {
        if (!psMaterialMap::isMaterial(materialIds->at(e), psMaterial::GAS)) {
          continue;
        }

        int numNeighbors = 0;
        auto coord = nodes[elems[e][0]];
        for (int i = 0; i < D; i++) {
          coord[i] += gridDelta / 2.;
        }

        auto cellNeighbors = cellSet->getNeighbors(e);
        for (const auto &n : cellNeighbors) {
          if (n == -1 ||
              !psMaterialMap::isMaterial(materialIds->at(n), psMaterial::GAS))
            continue;

          solution[e] += data->at(n);
          numNeighbors++;
        }

        // diffusion
        solution[e] =
            data->at(e) +
            C * (solution[e] - static_cast<T>(numNeighbors) * data->at(e));

        // sink at the top
        if (coord[D - 1] > top - gridDelta) {
          solution[e] = std::max(solution[e] - sink, T(0.));
          continue;
        }

        // convection
        if (std::abs(coord[0]) < holeRadius) {
          // in hole
          assert((cellNeighbors[2] != -1 && D == 2) ||
                 (cellNeighbors[4] != -1 && D == 3) &&
                     "holeStream up neighbor wrong");
          if (psMaterialMap::isMaterial(
                  materialIds->at(cellNeighbors[2 * (D - 1)]),
                  psMaterial::GAS)) {
            solution[e] -= holeC * (((coord[D - 1] - gridDelta) / top) *
                                        data->at(cellNeighbors[2 * (D - 1)]) -
                                    (coord[D - 1] / top) * data->at(e));
          }
        } else {
          if (coord[0] < 0) {
            // left side scallop - use forward difference
            assert(cellNeighbors[1] != -1 &&
                   "scallopStream right neighbor wrong");
            if (psMaterialMap::isMaterial(materialIds->at(cellNeighbors[1]),
                                          psMaterial::GAS)) {
              solution[e] -=
                  scallopC * (data->at(cellNeighbors[1]) - data->at(e));
            }
          } else {
            // right side scallop - use backward difference
            assert(cellNeighbors[0] != -1 &&
                   "scallopStream left neighbor wrong");
            if (psMaterialMap::isMaterial(materialIds->at(cellNeighbors[0]),
                                          psMaterial::GAS)) {
              solution[e] +=
                  scallopC * (data->at(e) - data->at(cellNeighbors[0]));
            }
          }
        }
      }
      *data = std::move(solution);
    }
    auto sum = cellSet->getScalarData("byproductSum");

#pragma omp parallel for shared(sum)
    for (int e = 0; e < data->size(); e++) {
      if (!psMaterialMap::isMaterial(materialIds->at(e), psMaterial::GAS)) {
        continue;
      }

      assert(data->at(e) >= 0. && "Negative concentration");
      sum->at(e) += data->at(e) * timeStep;
    }
  }
};
} // namespace OxideRegrowthImplementation

// The selective etching model works in accordance with the geometry generated
// by psMakeStack
template <class NumericType, int D>
class psOxideRegrowth : public psProcessModel<NumericType, D> {
public:
  psOxideRegrowth(const NumericType nitrideEtchRate,
                  const NumericType oxideEtchRate,
                  const NumericType redepositionRate,
                  const NumericType reDepositionThreshold,
                  const NumericType redepositionTimeInt,
                  const NumericType diffusionCoefficient,
                  const NumericType sinkStrength,
                  const NumericType scallopVelocity,
                  const NumericType centerVelocity, const NumericType topHeight,
                  const NumericType centerWidth, // 11 parameters
                  const NumericType timeStabilityFactor = 0.245) {

    auto velocityField = psSmartPointer<
        OxideRegrowthImplementation::SelectiveEtchingVelocityField<
            NumericType>>::New(nitrideEtchRate, oxideEtchRate);

    auto surfModel = psSmartPointer<psSurfaceModel<NumericType>>::New();

    auto dynamics =
        psSmartPointer<OxideRegrowthImplementation::ByproductDynamics<
            NumericType, D>>::New(diffusionCoefficient, sinkStrength,
                                  scallopVelocity, centerVelocity, topHeight,
                                  centerWidth / 2., nitrideEtchRate,
                                  redepositionRate, reDepositionThreshold,
                                  redepositionTimeInt, timeStabilityFactor);

    this->setVelocityField(velocityField);
    this->setSurfaceModel(surfModel);
    this->setAdvectionCallback(dynamics);
    this->setProcessName("OxideRegrowth");
  }
};
