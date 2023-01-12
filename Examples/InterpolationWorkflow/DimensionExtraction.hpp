#pragma once

#include <algorithm>
#include <vector>

#include <lsToDiskMesh.hpp>

#include <psDomain.hpp>
#include <psKDTree.hpp>

template <typename NumericType, int D> class DimensionExtraction {
public:
  DimensionExtraction()
      : verticalSampleLocations(
            distributeSampleLocations(numberOfSamples, -edgeAffinity)) {}

  DimensionExtraction(psSmartPointer<psDomain<NumericType, D>> passedDomain)
      : domain(passedDomain), verticalSampleLocations(distributeSampleLocations(
                                  numberOfSamples, -edgeAffinity)) {}

  void setDomain(psSmartPointer<psDomain<NumericType, D>> passedDomain) {
    domain = passedDomain;
  }

  void setNumberOfSamples(int passedNumberOfSamples) {
    assert(numberOfSamples > 1);
    numberOfSamples = passedNumberOfSamples;
    verticalSampleLocations =
        distributeSampleLocations(numberOfSamples - 1, -edgeAffinity);
  }

  void setEdgeAffinity(NumericType passedEdgeAffinity) {
    edgeAffinity = passedEdgeAffinity;
  }

  psSmartPointer<std::vector<NumericType>> getDimensions() {
    return dimensions;
  }

  const std::vector<NumericType> &getSampleLocations() {
    return verticalSampleLocations;
  }

  void apply() {
    if (!domain)
      return;

    // Re-initialize the dimensions vector with a value of zero. The first
    // element will be the depth.
    dimensions = psSmartPointer<std::vector<NumericType>>::New(
        verticalSampleLocations.size() + 1, 0.);

    auto mesh = psSmartPointer<lsMesh<>>::New();

    lsToDiskMesh<NumericType, D>(domain->getLevelSets()->back(), mesh).apply();

    auto nodes = mesh->getNodes();
    auto [min, max] = getMinMax(nodes, 1 /* axis */);
    NumericType depth = max - min;
    dimensions->at(0) = depth;

    // Only use the vertical trench axis for the kDtree
    std::vector<std::array<NumericType, 1>> ys;
    ys.reserve(nodes.size());
    std::transform(
        nodes.begin(), nodes.end(), std::back_inserter(ys),
        [=](auto &node) { return std::array<NumericType, 1>{node[D - 1]}; });

    psKDTree<NumericType, 1> tree;
    tree.setPoints(ys);
    tree.build();

    // The extract the diameters along its depth at the relative coordinates
    // given by depths
    int i = 1;
    const NumericType gridDelta = domain->getGrid().getGridDelta();
    for (auto sl : verticalSampleLocations) {
      std::array<NumericType, 1> loc{max - depth * sl};

      auto neighbors = tree.findNearestWithinRadius(loc, gridDelta);

      // Here we assume that the trench is centered at zero and symmetric with
      // its vertical axis being the axis of symmetry
      int idxL = -1;
      int idxR = -1;
      for (auto &nb : *neighbors) {
        // if the point is on the left trench sidewall
        if (idxL < 0 && nodes[nb.first][0] < 0) {
          idxL = nb.first;
        }

        // if the point is on the right trench sidewall
        if (idxR < 0 && nodes[nb.first][0] >= 0) {
          idxR = nb.first;
        }

        // if both indices were found
        if (idxL >= 0 && idxR >= 0)
          break;
      }

      if (idxL >= 0 && idxR >= 0) {
        const auto d = std::abs(nodes[idxL][0]) + std::abs(nodes[idxR][0]);
        dimensions->at(i) = d;
      }
      ++i;
    }
  }

private:
  static std::vector<NumericType>
  distributeSampleLocations(int n, NumericType alpha = 1.0) {
    std::vector<NumericType> x;
    x.reserve(n);
    for (unsigned i = 0; i < n; ++i)
      x.emplace_back(-1.0 + 2.0 * i / (n - 1));

    if (alpha != 0) {
      std::transform(x.begin(), x.end(), x.begin(), [alpha](NumericType xi) {
        return xi < 0 ? 1 - std::exp(-alpha * xi) : std::exp(alpha * xi) - 1;
      });
      std::transform(
          x.begin(), x.end(), x.begin(),
          [maxVal = x.back()](NumericType xi) { return xi / maxVal; });
    }
    std::transform(x.begin(), x.end(), x.begin(),
                   [](NumericType xi) { return (xi + 1) / 2; });
    if (alpha < 0) {
      std::reverse(x.begin(), x.end());
      return x;
    } else {
      return x;
    }
  }

  std::tuple<NumericType, NumericType>
  getMinMax(const std::vector<std::array<NumericType, 3>> &nodes, int axis) {
    if (nodes.empty())
      return {};

    const auto [minposIter, maxposIter] = std::minmax_element(
        nodes.cbegin(), nodes.cend(),
        [&](const auto &a, const auto &b) { return a.at(axis) < b.at(axis); });

    return {minposIter->at(axis), maxposIter->at(axis)};
  }

  psSmartPointer<psDomain<NumericType, D>> domain = nullptr;
  psSmartPointer<std::vector<NumericType>> dimensions = nullptr;

  int numberOfSamples = 10;
  NumericType edgeAffinity = 3.;

  std::vector<NumericType> verticalSampleLocations;
};