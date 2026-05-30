#include <process/psSurfaceDiffusion.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

viennaps::PointCloud<double> makePlane(int n, double spacing) {
  viennaps::PointCloud<double> cloud;
  cloud.positions.reserve(static_cast<std::size_t>(n * n));
  cloud.normals.reserve(static_cast<std::size_t>(n * n));
  const int half = n / 2;
  for (int iy = 0; iy < n; ++iy) {
    for (int ix = 0; ix < n; ++ix) {
      cloud.positions.push_back(
          {(ix - half) * spacing, (iy - half) * spacing, 0.});
      cloud.normals.push_back({0., 0., 1.});
    }
  }
  return cloud;
}

double maxAbs(const std::vector<double> &values) {
  double result = 0.;
  for (const auto value : values) {
    result = std::max(result, std::abs(value));
  }
  return result;
}

} // namespace

int main() {
  using GraphSolver = viennaps::SurfaceDiffusionSolver<double>;
  using GraphStencil = viennaps::SurfaceDiffusionStencil<double>;

  const int n = 21;
  const auto cloud = makePlane(n, 0.2);

  GraphStencil::Parameters params;
  params.kNeighbors = 24;
  params.symmetrizeWeights = true;
  GraphSolver solver(GraphStencil(cloud, params));

  const std::vector<double> constant(cloud.size(), 3.14);
  require(maxAbs(solver.applyLaplacian(constant)) < 1e-12,
          "graph diffusion must preserve constants");

  std::vector<double> peak(cloud.size(), 0.);
  peak[static_cast<std::size_t>((n / 2) * n + (n / 2))] = 1.;
  const auto beforeMass = std::accumulate(peak.begin(), peak.end(), 0.);
  const auto beforeMax = *std::max_element(peak.begin(), peak.end());

  const auto diffused = solver.stepExplicit(peak, 1e-4, 1.);
  const auto afterMass = std::accumulate(diffused.begin(), diffused.end(), 0.);
  const auto afterMax = *std::max_element(diffused.begin(), diffused.end());

  require(afterMax < beforeMax, "graph diffusion should smooth a peak");
  require(std::abs(afterMass - beforeMass) < 1e-12,
          "symmetric graph diffusion should conserve mass");

  return 0;
}
