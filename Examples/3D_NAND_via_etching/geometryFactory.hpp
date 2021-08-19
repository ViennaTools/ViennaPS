#ifndef GEOMFAC_HPP
#define GEOMFAC_HPP

#include <lsDomain.hpp>
#include <lsMakeGeometry.hpp>
#include <lsBooleanOperation.hpp>

template <class T, int D>
class MakeMask
{
    using LSPtrType = lsSmartPointer<lsDomain<T, D>>;
    LSPtrType mask;

    std::array<T, 3> maskOrigin = {};
    T maskRadius = 0;
    T maskHeight = 5.;

public:
    MakeMask(LSPtrType passedMask)
        : mask(passedMask) {}

    void setMaskOrigin(std::array<T, 3> &origin) { maskOrigin = origin; }

    void setMaskRadius(T radius) { maskRadius = radius; }

    void apply()
    {
        auto &grid = mask->getGrid();
        auto &boundaryCons = grid.getBoundaryConditions();
        auto gridDelta = grid.getGridDelta();
        T extent = grid.getGridExtent(0) * gridDelta;

        // create mask
        {
            T normal[3] = {static_cast<T>(0.), (D == 2) ? static_cast<T>(1.) : static_cast<T>(0.), (D == 3) ? static_cast<T>(1.) : static_cast<T>(0.)};
            T origin[3] = {static_cast<T>(0.), (D == 2) ? maskHeight : static_cast<T>(0.),
                           (D == 3) ? maskHeight : static_cast<T>(0.)};

            lsMakeGeometry<T, D>(mask,
                                 lsSmartPointer<lsPlane<T, D>>::New(origin, normal))
                .apply();
            normal[D - 1] = static_cast<T>(-1.0);
            origin[D - 1] = static_cast<T>(0.);
            auto maskBottom = LSPtrType::New(grid);
            lsMakeGeometry<T, D>(maskBottom,
                                 lsSmartPointer<lsPlane<T, D>>::New(origin, normal))
                .apply();
            lsBooleanOperation<T, D>(mask, maskBottom,
                                     lsBooleanOperationEnum::INTERSECT)
                .apply();
            auto maskHole = LSPtrType::New(grid);

            if constexpr (D == 3)
            {
                maskOrigin[2] = origin[2] - gridDelta;
                T axis[3] = {0.0, 0.0, 1.0};
                lsMakeGeometry<T, D>(maskHole,
                                     lsSmartPointer<lsCylinder<T, D>>::New(
                                         maskOrigin.data(), axis,
                                         maskHeight + 2 * gridDelta, maskRadius))
                    .apply();
            }
            else
            {
                T min[3] = {-maskRadius, -gridDelta};
                T max[3] = {maskRadius, maskHeight + 2 * gridDelta};
                lsMakeGeometry<T, D>(maskHole,
                                     lsSmartPointer<lsBox<T, D>>::New(min, max))
                    .apply();
            }
            lsBooleanOperation<T, D>(mask, maskHole,
                                     lsBooleanOperationEnum::RELATIVE_COMPLEMENT)
                .apply();
        }
    }
};

template <typename T, int D>
class MakeLayers
{
    using LSPtrType = lsSmartPointer<lsDomain<T, D>>;
    LSPtrType mask;
    int numLayers = 20;
    T layerSize = 2.;

public:
    MakeLayers(lsSmartPointer<lsDomain<T, D>> passedMask) : mask(passedMask) {}

    void setNumberOfLayers(int num) { numLayers = num; }
    void setLayerHeight(T height) { layerSize = height; }
    std::vector<LSPtrType> apply()
    {
        auto &grid = mask->getGrid();
        std::vector<LSPtrType> layers;
        for (int i = 0; i < numLayers; ++i)
        {
            layers.push_back(LSPtrType::New(grid));
            T layerHeight = -(numLayers - i - 1) * layerSize + 1e-3;
            T origin[D] = {0};
            origin[D - 1] = layerHeight;

            T normal[D] = {0};
            normal[D - 1] = 1.;

            auto plane = lsSmartPointer<lsPlane<T, D>>::New(origin, normal);
            lsMakeGeometry<T, D>(layers[i], plane).apply();
        }

        return std::move(layers);
    }
};

#endif