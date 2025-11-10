#include <geometries/psMakeHole.hpp>
#include <psCreateSurfaceMesh.hpp>
#include <psDomain.hpp>

using namespace viennaps;

void findMinElementSize(SmartPointer<viennals::Mesh<float>> mesh) {
  float minSize = std::numeric_limits<float>::max();
  for (const auto &tri : mesh->triangles) {
    const auto &v0 = mesh->nodes[tri[0]];
    const auto &v1 = mesh->nodes[tri[1]];
    const auto &v2 = mesh->nodes[tri[2]];
    const float a = Norm(v1 - v0);
    const float b = Norm(v2 - v1);
    const float c = Norm(v0 - v2);
    const float s = 0.5f * (a + b + c);
    const float area = std::sqrt(s * (s - a) * (s - b) * (s - c));
    const float height = 2.f * area / a;
    if (height < minSize)
      minSize = height;
  }
  std::cout << "Minimum element size: " << minSize << std::endl;
}

int main() {

  auto domain =
      Domain<float, 3>::New(1.0, 200., 200., BoundaryType::REFLECTIVE_BOUNDARY);
  MakeHole<float, 3>(domain, 75., 50.).apply();

  // Create a surface mesh
  auto surfaceMesh = viennals::Mesh<float>::New();
  Timer timer;
  timer.start();
  CreateSurfaceMesh<float, float, 3>(domain->getLevelSets().back(), surfaceMesh,
                                     nullptr, 1e-12, 0.0)
      .apply();
  timer.finish();
  std::cout << "Surface mesh creation (no check node dist) took: "
            << timer.currentDuration / 1e6 << " milliseconds." << std::endl;
  std::cout << "Number of surface elements: " << surfaceMesh->triangles.size()
            << std::endl;
  findMinElementSize(surfaceMesh);
  viennals::VTKWriter<float>(surfaceMesh, "surfaceMeshNoCheck.vtp").apply();
  surfaceMesh->clear();

  timer.start();
  CreateSurfaceMesh<float, float, 3>(domain->getLevelSets().back(), surfaceMesh,
                                     nullptr, 1e-12, 0.05)
      .apply();
  timer.finish();
  std::cout << "Surface mesh creation (with check node dist) took: "
            << timer.currentDuration / 1e6 << " milliseconds." << std::endl;
  std::cout << "Number of surface elements: " << surfaceMesh->triangles.size()
            << std::endl;
  findMinElementSize(surfaceMesh);
  viennals::VTKWriter<float>(surfaceMesh, "surfaceMeshCheck.vtp").apply();

  surfaceMesh->clear();
  auto kdTree = SmartPointer<KDTree<float, std::array<float, 3>>>::New();
  timer.start();
  CreateSurfaceMesh<float, float, 3>(domain->getLevelSets().back(), surfaceMesh,
                                     kdTree, 1e-12, 0.05)
      .apply();
  timer.finish();
  std::cout << "Surface mesh creation (with kd-tree) took: "
            << timer.currentDuration / 1e6 << " milliseconds." << std::endl;
  std::cout << "Number of surface elements: " << surfaceMesh->triangles.size()
            << std::endl;
  findMinElementSize(surfaceMesh);

  return 0;
}