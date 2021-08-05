#include <psSurfaceModel.hpp>
#include <psSmartPointer.hpp>
#include <psProcessModel.hpp>
#include <psProcess.hpp>
#include <lsToDiskMesh.hpp>
#include <lsVTKWriter.hpp>

#include "particles.hpp"
#include "surfaceModel.hpp"
#include "geometryFactory.hpp"
#include "velocityField.hpp"

class myCellType : public cellBase
{
    using cellBase::cellBase;
};

int main()
{
    omp_set_num_threads(12);
    using NumericType = double;
    constexpr int D = 3;

    // particles
    auto ionParticle = std::make_unique<Ion<NumericType>>();
    auto polyParticle = std::make_unique<Polymer<NumericType, D>>();
    auto etchantParticle = std::make_unique<Etchant<NumericType, D>>();
    auto etchantPolyParticle = std::make_unique<EtchantPoly<NumericType, D>>();

    // surface model
    auto surfModel = psSmartPointer<viaEtchingSurfaceModel<NumericType>>::New();

    // velocity field
    auto velField = psSmartPointer<velocityField<NumericType>>::New();

    /* ------------- Geometry setup ------------ */
    // domain
    NumericType extent = 8;
    NumericType gridDelta = 0.5;
    double bounds[2 * D] = {-extent, extent, -extent, extent, -extent, extent};
    lsDomain<NumericType, D>::BoundaryType boundaryCons[D];
    boundaryCons[0] = lsDomain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY;
    boundaryCons[1] = lsDomain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY;
    boundaryCons[2] = lsDomain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY;

    // Create via mask on top
    auto mask = lsSmartPointer<lsDomain<NumericType, D>>::New(bounds, boundaryCons, gridDelta);
    {
        std::array<NumericType, D> maskOrigin = {0., 0., 0.};
        MakeMask<NumericType, D> makeMask(mask);
        makeMask.setMaskOrigin(maskOrigin);
        makeMask.setMaskRadius(5);
        makeMask.apply();
    }
    auto domain = psSmartPointer<psDomain<myCellType, NumericType, D>>::New(mask);

    // Create SiO2/SiNx layers
    constexpr int numLayers = 20;
    NumericType layerSize = 2;
    {
        std::vector<lsSmartPointer<lsDomain<NumericType, D>>> layers;
        for (int i = 0; i < numLayers; ++i)
        {
            layers.push_back(lsSmartPointer<lsDomain<NumericType, D>>::New(bounds, boundaryCons, gridDelta));
            NumericType layerHeight = -(numLayers - i - 1) * layerSize + 1e-3;
            NumericType origin[D] = {0};
            origin[D - 1] = layerHeight;

            NumericType normal[D] = {0};
            normal[D - 1] = 1.;

            auto plane = lsSmartPointer<lsPlane<NumericType, D>>::New(origin, normal);
            lsMakeGeometry<NumericType, D>(layers[i], plane).apply();
        }
        for (auto &l : layers)
            domain->insertNextLevelSet(l);
        // polymer is equivalent to uppermost layer
        auto polymerLayer = lsSmartPointer<lsDomain<NumericType, D>>::New(layers.back());
        domain->insertNextLevelSet(polymerLayer);
    }

    auto model = psSmartPointer<psProcessModel<NumericType>>::New();
    model->insertNextParticleType(ionParticle);
    model->insertNextParticleType(polyParticle);
    model->insertNextParticleType(etchantParticle);
    model->insertNextParticleType(etchantPolyParticle);
    model->setSurfaceModel(surfModel);
    model->setVelocityField(velField);

    psProcess<myCellType, NumericType, D> process;
    process.setDomain(domain);
    process.setProcessModel(model);
    process.setSourceDirection(rayTraceDirection::POS_Z);
    process.setProcessDuration(100);
    process.apply();

    auto coverages = model->getSurfaceModel()->getCoverages();
    auto pCov = *coverages->getScalarData("pCoverage");
    auto peCov = *coverages->getScalarData("peCoverage");
    auto eCov = *coverages->getScalarData("eCoverage");

    // auto topLayer = domain->getLevelSets()->back();
    // auto mesh = lsSmartPointer<lsMesh<NumericType>>::New();
    // lsToDiskMesh<NumericType, D>(topLayer, mesh).apply();
    // mesh->getCellData().insertNextScalarData(pCov, "pCoverage");
    // mesh->getCellData().insertNextScalarData(peCov, "peCoverage");
    // mesh->getCellData().insertNextScalarData(eCov, "eCoverage");

    // lsVTKWriter<NumericType>(mesh, "test.vtp").apply();

    return 0;
}