// #include <psProcess.hpp>
#include <psSmartPointer.hpp>
#include <psSurfaceModel.hpp>
#include <psProcessModel.hpp>
#include <vector>
#include <iostream>
#include <cellBase.hpp>
#include <psProcess.hpp>

// needs to be derived for interacting particles, if only non-interacting can
// use base implementation with predefined sticky/non-sticky particles if
// interacting particles should be used, particle types may be defined as
// standalone classes or as subclasses of this class In any case, this class
// must hold references to particleType objects which are returned on a call to
// getParticleTypes()
// template <typename NumericType>
// class myModel : public psProcessModel<NumericType> {
//   // implementation of particle types and surface model here or somewhere
//   // outside the class std::vector<std::unique_ptr<rayParticle>>
//   // getParticleTypes() override
//   // {
//   // }
//   psSmartPointer<psSurfaceModel<NumericType>> getSurfaceModel() override {}
// };

template <typename NumericType>
class mySurfaceModel : public psSurfaceModel<NumericType>
{
    using psSurfaceModel<NumericType>::Coverages;

public:
    void initializeCoverages(unsigned numGeometryPoints) override
    {
        Coverages = psSmartPointer<psPointData<NumericType>>::New();
        std::vector<NumericType> cov(numGeometryPoints);
        Coverages->insertNextScalarData(cov, "coverage");
    }

    psSmartPointer<std::vector<NumericType>>
    calculateVelocities(psSmartPointer<psPointData<NumericType>> Rates,
                        const std::vector<NumericType> &materialIDs, const long numRaysPerPoint) override
    {
        const auto numPoints = Rates->getScalarData(0)->size();
        std::vector<NumericType> velocities(numPoints);
        auto vel = psSmartPointer<std::vector<NumericType>>::New(velocities);
        return velocities;
    }

    void updateCoverages(psSmartPointer<psPointData<NumericType>> Rates,
                         const long numRaysPerPoint) override
    {
        const auto numPoints = Rates->getScalarData(0)->size();
    }
};

class myCellType : public cellBase
{
    using cellBase::cellBase;
};

int main()
{
    constexpr int D = 3;
    using NumericType = double;
    auto surfaceModel = psSmartPointer<mySurfaceModel<NumericType>>::New();
    auto processModel = psSmartPointer<psProcessModel<NumericType>>::New();

    processModel->setSurfaceModel(surfaceModel);
    auto particle = std::make_unique<rayTestParticle<NumericType>>();
    processModel->insertNextParticleType(particle);

    psProcess<myCellType, NumericType, D> process;
    process.setProcessModel(processModel);
    return 0;
}
