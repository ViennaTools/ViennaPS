#ifndef CS_CALCULATE_NORMAL_VECTORS_HPP
#define CS_CALCULATE_NORMAL_VECTORS_HPP

#include <algorithm>

#include <hrleSparseStarIterator.hpp>
#include <hrleVectorType.hpp>

#include <csDomain.hpp>
#include <lsMessage.hpp>

/// This algorithm is used to compute the normal vectors for all points
/// with level set values <= 0.5. The result is saved in the csDomain and
/// can be retrieved with csDomain.getNormalVectors().
/// Since neighbors in each cartesian direction are necessary for
/// the calculation, the levelset width must be >=3.
template <class T, int D> class csCalculateNormalVectors {
  csDomain<T, D> *domain = nullptr;

public:
  csCalculateNormalVectors() {}

  csCalculateNormalVectors(csDomain<T, D> &passedDomain)
      : domain(&passedDomain) {}

  void setCellSet(csDomain<T, D> &passedDomain) { domain = &passedDomain; }

  void apply() {
    if (domain == nullptr) {
      lsMessage::getInstance()
          .addWarning("No cell set was passed to csCalculateNormalVectors.")
          .print();
    }

    std::vector<std::vector<std::array<double, D>>> normalVectorsVector(
        domain->getNumberOfSegments());
    double pointsPerSegment =
        double(2 * domain->getDomain().getNumberOfPoints()) /
        double(domain->getNumberOfSegments());

    auto grid = domain->getGrid();

    //! Calculate Normalvectors
#pragma omp parallel num_threads(domain->getNumberOfSegments())
    {
      int p = 0;
#ifdef _OPENMP
      p = omp_get_thread_num();
#endif

      std::vector<std::array<double, D>> &normalVectors = normalVectorsVector[p];
      normalVectors.reserve(pointsPerSegment);

      hrleVectorType<hrleIndexType, D> startVector =
          (p == 0) ? grid.getMinGridPoint()
                   : domain->getDomain().getSegmentation()[p - 1];

      hrleVectorType<hrleIndexType, D> endVector =
          (p != static_cast<int>(domain->getNumberOfSegments() - 1))
              ? domain->getDomain().getSegmentation()[p]
              : grid.incrementIndices(grid.getMaxGridPoint());

      for (hrleConstSparseStarIterator<typename csDomain<T, D>::DomainType>
               neighborIt(domain->getDomain(), startVector);
           neighborIt.getIndices() < endVector; neighborIt.next()) {

        auto &center = neighborIt.getCenter();
        if (!center.isDefined())
          continue;

        std::array<double, D> n;

        double denominator = 0;
        for (int i = 0; i < D; i++) {
          double pos = neighborIt.getNeighbor(i).getValue().getFillingFraction() - center.getValue().getFillingFraction();
          double neg = center.getValue().getFillingFraction() - neighborIt.getNeighbor(i + D).getValue().getFillingFraction();
          n[i] = (pos + neg) * 0.5;
          denominator += n[i] * n[i];
        }

        denominator = std::sqrt(denominator);
        if (std::abs(denominator) < 1e-12) {
          for (unsigned i = 0; i < D; ++i)
            n[i] = 0.;
        } else {
          for (unsigned i = 0; i < D; ++i) {
            n[i] /= denominator;
          }
        }

        normalVectors.push_back(n);
      }
    }

    // copy all normals
    auto &normals = domain->getNormalVectors();
    normals.clear();
    unsigned numberOfNormals = 0;
    for (unsigned i = 0; i < domain->getNumberOfSegments(); ++i) {
      numberOfNormals += normalVectorsVector[i].size();
    }
    normals.reserve(numberOfNormals);

    for (unsigned i = 0; i < domain->getNumberOfSegments(); ++i) {
      normals.insert(normals.end(), normalVectorsVector[i].begin(),
                     normalVectorsVector[i].end());
    }
  }
};

#endif // CS_CALCULATE_NORMAL_VECTORS_HPP