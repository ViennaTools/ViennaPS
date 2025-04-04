// LaplacianSmoothing.hpp
#pragma once

#include <lsDomain.hpp>
#include <lsFiniteDifferences.hpp>
#include <lsPreCompileMacros.hpp>
#include <vcSmartPointer.hpp>

namespace viennaps {

template <class T, int D> class LaplacianSmoothing {
  using LevelSetType = SmartPointer<Domain<T, D>>;

  LevelSetType levelSet = nullptr;
  unsigned iterations = 10;
  T timeStep = 0.1;

public:
  LaplacianSmoothing(LevelSetType ls) : levelSet(ls) {}

  void setIterations(unsigned iters) { iterations = iters; }
  void setTimeStep(T dt) { timeStep = dt; }

  void apply() {
    if (!levelSet)
      return;

    auto &domain = levelSet->getDomain();
    const auto &grid = levelSet->getGrid();

    for (unsigned iter = 0; iter < iterations; ++iter) {
      std::vector<std::pair<hrleVectorType<hrleIndexType, D>, T>> updates;

      for (unsigned seg = 0; seg < domain.getNumberOfSegments(); ++seg) {
        auto it = domain.getDomainSegment(seg).getIterator();
        lsInternal::FiniteDifferences<T, D> fd(domain, grid);

        while (!it.isFinished()) {
          if (it.isDefined()) {
            T laplacian = fd.calculateLaplacian(it);
            T newValue = it.getValue() + timeStep * laplacian;
            updates.emplace_back(it.getIndices(), newValue);
          }
          ++it;
        }
      }

      for (const auto &[idx, value] : updates) {
        domain.getNode(idx).setValue(value);
      }
    }

    levelSet->finalize();
  }
};

} // namespace viennaps
