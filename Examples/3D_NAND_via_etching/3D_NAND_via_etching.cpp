#include <psSurfaceModel.hpp>
#include <psSmartPointer.hpp>
#include <psProcessModel.hpp>
#include <psProcess.hpp>
#include <lsToDiskMesh.hpp>
#include <lsToSurfaceMesh.hpp>
#include <lsVTKWriter.hpp>

#include "particles.hpp"
#include "surfaceModel.hpp"
#include "geometryFactory.hpp"
#include "velocityField.hpp"

class myCellType : public cellBase
{
    using cellBase::cellBase;
};

template <typename T, int D>
void printLS(lsSmartPointer<lsDomain<T, D>> dom, std::string name)
{
    auto mesh = lsSmartPointer<lsMesh<T>>::New();
    lsToSurfaceMesh<T, D>(dom, mesh).apply();
    lsVTKWriter<T>(mesh, name).apply();
}

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
    NumericType gridDelta = 0.25;
    double bounds[2 * D] = {0};
    for (int i = 0; i < 2 * D; ++i)
        bounds[i] = i % 2 == 0 ? -extent : extent;
    lsDomain<NumericType, D>::BoundaryType boundaryCons[D];
    for (int i = 0; i < D - 1; ++i)
        boundaryCons[i] = lsDomain<NumericType, D>::BoundaryType::REFLECTIVE_BOUNDARY;
    boundaryCons[D - 1] = lsDomain<NumericType, D>::BoundaryType::INFINITE_BOUNDARY;

    // Create via mask on top
    auto mask = psSmartPointer<lsDomain<NumericType, D>>::New(bounds, boundaryCons, gridDelta);
    {
        std::array<NumericType, 3> maskOrigin = {0.};
        MakeMask<NumericType, D> makeMask(mask);
        makeMask.setMaskOrigin(maskOrigin);
        makeMask.setMaskRadius(5);
        makeMask.apply();
    }
    auto domain = psSmartPointer<psDomain<myCellType, NumericType, D>>::New(mask);

    // Create SiO2/SiNx layers
    {
        MakeLayers<NumericType, D> makeLayers(mask);
        makeLayers.setLayerHeight(2.);
        makeLayers.setNumberOfLayers(20);
        auto layers = makeLayers.apply();
        for (auto &l : layers)
            domain->insertNextLevelSet(l);
        auto polymerLayer = psSmartPointer<lsDomain<NumericType, D>>::New(layers.back()); // polymer is equivalent to uppermost layer
        domain->insertNextLevelSet(polymerLayer);
    }

    // print initial layers
    {
        int n = 0;
        for (auto &layer : *domain->getLevelSets())
        {
            printLS(layer, "layer_" + std::to_string(n++) + ".vtp");
        }
    }

    // auto model = psSmartPointer<psProcessModel<NumericType>>::New();
    // model->insertNextParticleType(ionParticle);
    // model->insertNextParticleType(polyParticle);
    // model->insertNextParticleType(etchantParticle);
    // model->insertNextParticleType(etchantPolyParticle);
    // model->setSurfaceModel(surfModel);
    // model->setVelocityField(velField);

    // psProcess<myCellType, NumericType, D> process;
    // process.setDomain(domain);
    // process.setProcessModel(model);
    // process.setSourceDirection(rayTraceDirection::POS_Z);
    // process.setProcessDuration(100);
    // process.apply();

    // auto coverages = model->getSurfaceModel()->getCoverages();
    // auto pCov = *coverages->getScalarData("pCoverage");
    // auto peCov = *coverages->getScalarData("peCoverage");
    // auto eCov = *coverages->getScalarData("eCoverage");

    // auto topLayer = domain->getLevelSets()->back();
    // auto mesh = lsSmartPointer<lsMesh<NumericType>>::New();
    // lsToDiskMesh<NumericType, D>(topLayer, mesh).apply();
    // mesh->getCellData().insertNextScalarData(pCov, "pCoverage");
    // mesh->getCellData().insertNextScalarData(peCov, "peCoverage");
    // mesh->getCellData().insertNextScalarData(eCov, "eCoverage");

    // lsVTKWriter<NumericType>(mesh, "test.vtp").apply();

    return 0;
}