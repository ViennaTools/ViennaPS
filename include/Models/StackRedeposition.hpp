#pragma once

#include <csDenseCellSet.hpp>

#include <lsToDiskMesh.hpp>

#include <psAdvectionCallback.hpp>
#include <psDomain.hpp>
#include <psProcessModel.hpp>

// The selective etching model works in accordance with the geometry generated
// by psMakeStack
template <class NumericType>
class SelectiveEtchingVelocityField : public psVelocityField<NumericType> {
public:
  SelectiveEtchingVelocityField(const NumericType pRate,
                                const NumericType pOxideRate,
                                const int pDepoMat)
      : rate(pRate), oxide_rate(pOxideRate), depoMat(pDepoMat) {}

  NumericType getScalarVelocity(const std::array<NumericType, 3> &coordinate,
                                int matId,
                                const std::array<NumericType, 3> &normalVector,
                                unsigned long pointId) override {
    if (matId == 0 || matId == depoMat) {
      return 0.;
    }

    if (matId % 2 == 0) {
      return -rate;
    } else {
      return -oxide_rate;
    }
  }

  bool useTranslationField() const override { return false; }

private:
  const NumericType rate;
  const NumericType oxide_rate;
  const int depoMat;
};

template <class T, int D>
class ByproductDynamics : public psAdvectionCallback<T, D> {
  using psAdvectionCalback<NumericType, D>::domain;

  psSmartPointer<csDenseCellSet<T, D>> cellSet = nullptr;
  const int plasmaMaterial = 0;
  const T diffusionCoefficient = 1.;
  const T sink = 1;
  const T scallopStreamVel = 1.;
  const T holeStreamVel = 1.;
  const T top;
  const T holeRadius;
  lsSmartPointer<std::vector<T>> velocities;
  std::vector<std::array<T, 3>> nodes;
  static constexpr T eps = 1e-4;

public:
  ByproductDynamics(const T passedDiffCoeff, const T passedSink,
                    const T passedScallopVel, const T passedHoleVel,
                    const int passedPlasmaMat, const T passedTop,
                    const T passedRadius)
      : diffusionCoefficient(passedDiffCoeff), sink(passedSink),
        scallopStreamVel(passedScallopVel), holeStreamVel(passedHoleVel),
        plasmaMaterial(passedPlasmaMat), top(passedTop),
        holeRadius(passedRadius) {}

  void bool applyPreAdvect(const NumericType processTime) override {
    assert(domain->getUseCellSet());
    cellSet = domain->getCellSet();

    // TODO redeposition

    // TODO: save points
  }

  void bool applyPostAdvect(const NumericType advectionTime) override {
    cellSet->updateMaterials();

    // TODO add byproducts

    diffuseByproducts(advectionTime);
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
    const T holeC = dt / gridDelta * holeStreamVel;
    const T scallopC = dt / gridDelta * scallopStreamVel;

    for (int ts = 0; ts < numSteps; ts++) {
      std::vector<T> solution(data->size(), 0.);

#pragma omp parallel for
      for (int e = 0; e < data->size(); e++) {
        if (materialIds->at(e) != plasmaMaterial) {
          continue;
        }

        int numNeighbors = 0;
        auto coord = nodes[elems[e][0]];
        for (int i = 0; i < D; i++) {
          coord[i] += gridDelta / 2.;
        }

        auto cellNeighbors = cellSet->getNeighbors(e);
        for (const auto &n : cellNeighbors) {
          if (n == -1 || materialIds->at(n) != plasmaMaterial)
            continue;

          solution[e] += data->at(n);
          numNeighbors++;
        }

        // diffusion
        solution[e] =
            data->at(e) +
            C * (solution[e] - static_cast<T>(numNeighbors) * data->at(e));

        // sink at the top
        if (coord[1] > top - gridDelta) {
          solution[e] = std::max(solution[e] - sink, 0.);
          continue;
        }

        // convection
        if (std::abs(coord[0]) < holeRadius) {
          // in hole
          assert(cellNeighbors[2] != -1 && "holeStream up neighbor wrong");
          if (materialIds->at(cellNeighbors[2]) == plasmaMaterial) {
            solution[e] -= holeC * (((coord[1] - gridDelta) / top) *
                                        data->at(cellNeighbors[2]) -
                                    (coord[1] / top) * data->at(e));
          }
        } else {
          if (coord[0] < 0) {
            // left side scallop - use forward difference
            assert(cellNeighbors[1] != -1 &&
                   "scallopStream right neighbor wrong");
            if (materialIds->at(cellNeighbors[1]) == plasmaMaterial) {
              solution[e] -=
                  scallopC * (data->at(cellNeighbors[1]) - data->at(e));
            }
          } else {
            // right side scallop - use backward difference
            assert(cellNeighbors[0] != -1 &&
                   "scallopStream left neighbor wrong");
            if (materialIds->at(cellNeighbors[0]) == plasmaMaterial) {
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
      if (materialIds->at(e) != plasmaMaterial) {
        continue;
      }

      assert(data->at(e) >= 0. && "Negative concentration");
      sum->at(e) += data->at(e) * timeStep;
    }
  }
};

template <class NumericType, int D>
class OxideRegrowthModel : public psProcessModel<NumericType, D> {
public:
  OxideRegrowthModel(
      const int depoMaterialId, const NumericType nitrideEtchRate,
      const NumericType oxideEtchRate, const NumericType diffusionCoefficient,
      const NumericType sinkStrength, const NumericType scallopVelocitiy,
      const NumericType centerVelocity, const int plasmaMaterial,
      const NumericType topHeight, const NumericType centerWidth) {

    auto veloField =
        psSmartPointer<SelectiveEtchingVelocityField<NumericType>>::New(
            nitrideEtchRate, oxideEtchRate, depoMaterialId);

    auto surfModel = psSmartPointer<psSurfaceModel<NumericType>>::New();

    auto dynamics = psSmartPointer<ByproductDynamics<NumericType, D>>::New(
        diffusionCoefficient, sinkStrength, scallopVelocitiy, centerVelocity,
        plasmaMaterial, topHeight, centerWidth / 2.);

    this->setVelocityField(veloField);
    this->setSurfaceModel(surfModel);
    this->setProcessName("OxideRegrowth");
  }
};