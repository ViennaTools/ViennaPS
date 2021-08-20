#pragma once

#include <psSurfaceModel.hpp>

#define POLYMER 2
#define SUBSTRATE 1
#define MASK 0

template <typename NumericType>
class Remove : public psSurfaceModel<NumericType>
{
public:
    psSmartPointer<std::vector<NumericType>>
    calculateVelocities(psSmartPointer<psPointData<NumericType>> Rates,
                        const std::vector<NumericType> &materialIds,
                        const long numRaysPerPoint) override
    {
        std::vector<NumericType> etchRate(materialIds.size(), 0.);

        auto ionSputteringRate = Rates->getScalarData("ionSputteringRate");
        auto ionEnhancedRate = Rates->getScalarData("ionEnhancedRate");

        for (size_t i = 0; i < materialIds.size(); ++i)
        {
            if (materialIds[i] != MASK)
                etchRate[i] = -(ionEnhancedRate->at(i) + ionSputteringRate->at(i)) / numRaysPerPoint;
        }

        return psSmartPointer<std::vector<NumericType>>::New(etchRate);
    }
};

template <typename NumericType>
class Clear : public psSurfaceModel<NumericType>
{
public:
    psSmartPointer<std::vector<NumericType>>
    calculateVelocities(psSmartPointer<psPointData<NumericType>> Rates,
                        const std::vector<NumericType> &materialIds,
                        const long numRaysPerPoint) override
    {
        std::vector<NumericType> etchRate(materialIds.size(), 0.);

        for (size_t i = 0; i < materialIds.size(); ++i)
        {
            if (materialIds[i] == POLYMER)
                etchRate[i] = -1;
        }

        return psSmartPointer<std::vector<NumericType>>::New(etchRate);
    }
};

template <typename NumericType>
class Oxidize : public psSurfaceModel<NumericType>
{
public:
    psSmartPointer<std::vector<NumericType>>
    calculateVelocities(psSmartPointer<psPointData<NumericType>> Rates,
                        const std::vector<NumericType> &materialIds,
                        const long numRaysPerPoint) override
    {
        std::vector<NumericType> etchRate(materialIds.size(), 1.);

        return psSmartPointer<std::vector<NumericType>>::New(etchRate);
    }
};

template <typename NumericType>
class Etch : public psSurfaceModel<NumericType>
{
public:
    psSmartPointer<std::vector<NumericType>>
    calculateVelocities(psSmartPointer<psPointData<NumericType>> Rates,
                        const std::vector<NumericType> &materialIds,
                        const long numRaysPerPoint) override
    {
        std::vector<NumericType> etchRate(materialIds.size(), 0.);

        // auto ionSputteringRate = Rates->getScalarData("ionSputteringRate");
        auto ionEnhancedRate = Rates->getScalarData("ionEnhancedRate");
        auto etchantRate = Rates->getScalarData("etchantRate");
        std::cout << materialIds.size() << std::endl;
        for (size_t i = 0; i < materialIds.size(); ++i)
        {
            std::cout << materialIds[i] << " ";
            if (materialIds[i] == SUBSTRATE)
            {
                etchRate[i] = -(etchantRate->at(i) + ionEnhancedRate->at(i) / 2) / numRaysPerPoint;
                std::cout << "substrate " << etchRate[i] << std::endl;
            }
        }

        return psSmartPointer<std::vector<NumericType>>::New(etchRate);
    }
};