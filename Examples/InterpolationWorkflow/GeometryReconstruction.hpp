#pragma once

#include <lsBooleanOperation.hpp>
#include <lsFromSurfaceMesh.hpp>
#include <lsGeometries.hpp>
#include <lsMakeGeometry.hpp>
#include <lsMesh.hpp>
#include <lsVTKWriter.hpp>

template <typename NumericType, int D> class GeometryReconstruction {

  psSmartPointer<lsDomain<NumericType, D>> &levelset;
  const NumericType (&origin)[D];
  const std::vector<NumericType> &sampleLocations;
  const std::vector<NumericType> &dimensions;
  NumericType eps;

public:
  GeometryReconstruction(
      psSmartPointer<lsDomain<NumericType, D>> &passedLevelset,
      const NumericType (&passedOrigin)[D],
      const std::vector<NumericType> &passedSampleLocations,
      const std::vector<NumericType> &passedDimensions)
      : levelset(passedLevelset), origin(passedOrigin),
        sampleLocations(passedSampleLocations), dimensions(passedDimensions),
        eps(1e-4) {}

  void apply() {
    // First generate an initial plane from which we will remove the trench
    // geometry later on
    {
      NumericType normal[D] = {0.};
      normal[D - 1] = 1.;

      auto plane = psSmartPointer<lsPlane<NumericType, D>>::New(origin, normal);
      lsMakeGeometry<NumericType, D>(levelset, plane).apply();
    }

    // Now create a mesh that reconstructs the trench profile by using the
    // extracted dimensions. Use this mesh to generate a new levelset, which
    // will be subtracted from the plane.
    {
      NumericType depth = dimensions.at(0);

      // Manually create a surface mesh based on the extracted dimensions
      auto mesh = psSmartPointer<lsMesh<>>::New();

      for (unsigned i = 1; i < dimensions.size(); ++i) {
        std::array<NumericType, 3> point{0.};
        point[0] = origin[0];
        point[1] = origin[1];
        if constexpr (D == 3)
          point[2] = origin[2];

        point[0] -= std::max(dimensions.at(i) / 2, eps);
        point[D - 1] -= depth * sampleLocations.at(i - 1);

        mesh->insertNextNode(point);
      }

      for (unsigned i = dimensions.size() - 1; i >= 1; --i) {
        std::array<NumericType, 3> point{0.};
        point[0] = origin[0];
        point[1] = origin[1];
        if constexpr (D == 3)
          point[2] = origin[2];

        point[0] += std::max(dimensions.at(i) / 2, eps);
        point[D - 1] -= depth * sampleLocations.at(i - 1);

        mesh->insertNextNode(point);
      }

      for (unsigned i = 0; i < mesh->nodes.size() - 1; ++i)
        mesh->lines.emplace_back(std::array<unsigned, 2>{i, i + 1});

      mesh->lines.emplace_back(std::array<unsigned, 2>{
          static_cast<unsigned>(mesh->lines.size()), 0U});

      lsVTKWriter<NumericType>(mesh, "hullMesh.vtp").apply();

      // Create the new levelset based on the mesh and substract it from the
      // plane
      auto hull =
          psSmartPointer<lsDomain<NumericType, D>>::New(levelset->getGrid());

      lsFromSurfaceMesh<NumericType, D>(hull, mesh).apply();

      lsBooleanOperation<NumericType, D>(
          levelset, hull, lsBooleanOperationEnum::RELATIVE_COMPLEMENT)
          .apply();
    }
  }
};