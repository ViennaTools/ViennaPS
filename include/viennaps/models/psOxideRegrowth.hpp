#pragma once

#include "../psAdvectionCallback.hpp"
#include "../psProcessModel.hpp"
#include "../psToDiskMesh.hpp"

#include <csDenseCellSet.hpp>

#include <lsAdvect.hpp>

namespace viennaps {

using namespace viennacore;

namespace impl {
template <class NumericType, int D>
class SelectiveEtchingVelocityField : public VelocityField<NumericType, D> {
public:
  SelectiveEtchingVelocityField(const NumericType pRate,
                                const NumericType pOxideRate)
      : rate(pRate), oxide_rate(pOxideRate) {}

  NumericType getScalarVelocity(const Vec3D<NumericType> &coordinate, int matId,
                                const Vec3D<NumericType> &normalVector,
                                unsigned long pointId) override {
    auto material = MaterialMap::mapToMaterial(matId);
    if (material == Material::Si3N4) {
      return -rate;
    } else if (material == Material::SiO2) {
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
class RedepositionVelocityField : public viennals::VelocityField<NumericType> {
public:
  RedepositionVelocityField(const std::vector<NumericType> &passedVelocities,
                            const std::vector<Vec3D<NumericType>> &points)
      : velocities(passedVelocities), kdTree(points) {
    assert(points.size() == passedVelocities.size());
    kdTree.build();
  }

  NumericType getScalarVelocity(const Vec3D<NumericType> &coordinate, int matId,
                                const Vec3D<NumericType> &normalVector,
                                unsigned long pointId) override {
    auto nearest = kdTree.findNearest(coordinate);
    assert(nearest->first < velocities.size());
    return velocities[nearest->first];
  }

private:
  const std::vector<NumericType> &velocities;
  KDTree<NumericType, Vec3D<NumericType>> kdTree;
};

template <class T, int D>
class ByproductDynamics : public AdvectionCallback<T, D> {
  using AdvectionCallback<T, D>::domain;

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
  std::vector<Vec3D<T>> nodes;
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
    auto mesh = viennals::Mesh<T>::New();
    ToDiskMesh<T, D>(domain, mesh).apply();

    const auto &points = mesh->nodes;
    auto materialIds = mesh->getCellData().getScalarData("MaterialIds");

    // save points where the surface is etched before advection
    nodes.clear();
    nodes.reserve(points.size());
    for (size_t i = 0; i < points.size(); i++) {
      auto material = MaterialMap::mapToMaterial(materialIds->at(i));
      if (material == Material::Si3N4)
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
        auto surfaceMaterial = MaterialMap::mapToMaterial(materialIds->at(i));
        const auto &node = points[i];

        // redeposit only on oxide
        if ((surfaceMaterial == Material::SiO2 ||
             surfaceMaterial == Material::Polymer) &&
            node[D - 1] < top) {
          auto cellIdx = cellSet->getIndex(node);
          int n = 0;
          if (cellIdx == -1)
            continue;
          if (MaterialMap::mapToMaterial(cellMatIds->at(cellIdx)) ==
              Material::GAS) {
            depoRate[i] = ff->at(cellIdx);
            n++;
          }
          for (const auto ni : cellSet->getNeighbors(cellIdx)) {
            if (ni != -1 && MaterialMap::mapToMaterial(cellMatIds->at(ni)) ==
                                Material::GAS) {
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
          SmartPointer<RedepositionVelocityField<T>>::New(depoRate, points);

      viennals::Advect<T, D> advectionKernel;
      advectionKernel.insertNextLevelSet(domain->getLevelSets().back());
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
  void diffuseByproducts(SmartPointer<viennacs::DenseCellSet<T, D>> cellSet,
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
        if (!MaterialMap::isMaterial(materialIds->at(e), Material::GAS)) {
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
              !MaterialMap::isMaterial(materialIds->at(n), Material::GAS))
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
          if (MaterialMap::isMaterial(
                  materialIds->at(cellNeighbors[2 * (D - 1)]), Material::GAS)) {
            solution[e] -= holeC * (((coord[D - 1] - gridDelta) / top) *
                                        data->at(cellNeighbors[2 * (D - 1)]) -
                                    (coord[D - 1] / top) * data->at(e));
          }
        } else {
          if (coord[0] < 0) {
            // left side scallop - use forward difference
            assert(cellNeighbors[1] != -1 &&
                   "scallopStream right neighbor wrong");
            if (MaterialMap::isMaterial(materialIds->at(cellNeighbors[1]),
                                        Material::GAS)) {
              solution[e] -=
                  scallopC * (data->at(cellNeighbors[1]) - data->at(e));
            }
          } else {
            // right side scallop - use backward difference
            assert(cellNeighbors[0] != -1 &&
                   "scallopStream left neighbor wrong");
            if (MaterialMap::isMaterial(materialIds->at(cellNeighbors[0]),
                                        Material::GAS)) {
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
      if (!MaterialMap::isMaterial(materialIds->at(e), Material::GAS)) {
        continue;
      }

      assert(data->at(e) >= 0. && "Negative concentration");
      sum->at(e) += data->at(e) * timeStep;
    }
  }
};
} // namespace impl

// The selective etching model works in accordance with the geometry generated
// by psMakeStack
template <class NumericType, int D>
class OxideRegrowth : public ProcessModel<NumericType, D> {
public:
  OxideRegrowth(const NumericType nitrideEtchRate,
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

    auto velocityField =
        SmartPointer<impl::SelectiveEtchingVelocityField<NumericType, D>>::New(
            nitrideEtchRate, oxideEtchRate);

    auto surfModel = SmartPointer<SurfaceModel<NumericType>>::New();

    auto dynamics = SmartPointer<impl::ByproductDynamics<NumericType, D>>::New(
        diffusionCoefficient, sinkStrength, scallopVelocity, centerVelocity,
        topHeight, centerWidth / 2., nitrideEtchRate, redepositionRate,
        reDepositionThreshold, redepositionTimeInt, timeStabilityFactor);

    this->setVelocityField(velocityField);
    this->setSurfaceModel(surfModel);
    this->setAdvectionCallback(dynamics);
    this->setProcessName("OxideRegrowth");
  }
};

} // namespace viennaps
