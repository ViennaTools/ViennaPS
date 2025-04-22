#pragma once

#include <string>
// #include <vector>
// #include <memory>

#include <psDomain.hpp>
#include <psProcessModel.hpp>
#include <psSurfaceModel.hpp>
#include <psVelocityFieldFromFile.hpp>

namespace viennaps {

template <typename NumericType, int D>
class CSVFileProcess : public ProcessModel<NumericType, D> {
public:
  CSVFileProcess(const std::string &ratesFile, const Vec3D<NumericType> &dir,
                 const Vec2D<NumericType> &off, const NumericType isoScale = 0.,
                 const NumericType dirScale = 1.,
                 const std::vector<Material> &masks = {Material::Mask},
                 bool calcVis = true) {

    auto velField =
        SmartPointer<impl::velocityFieldFromFile<NumericType, D>>::New(
            ratesFile, dir, off, isoScale, dirScale, masks, calcVis);

    auto surfModel = SmartPointer<SurfaceModel<NumericType>>::New();

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("CSVFileProcess");
  }

  void setIDWNeighbors(const int k = 4) {
    auto velField =
        std::dynamic_pointer_cast<impl::velocityFieldFromFile<NumericType, D>>(
            this->getVelocityField());
    if (velField)
      velField->setIDWNeighbors(k);
  }

  void setInterpolationMode(const std::string &str) {
    auto velField =
        std::dynamic_pointer_cast<impl::velocityFieldFromFile<NumericType, D>>(
            this->getVelocityField());
    if (velField)
      velField->setInterpolationMode(str);
  }

  void setInterpolationMode(
      typename impl::velocityFieldFromFile<NumericType, D>::Interpolation
          mode) {
    auto velField =
        std::dynamic_pointer_cast<impl::velocityFieldFromFile<NumericType, D>>(
            this->getVelocityField());
    if (velField)
      velField->setInterpolationMode(mode);
  }

  void setCustomInterpolator(
      std::function<NumericType(const Vec3D<NumericType> &)> func) {
    auto velField =
        std::dynamic_pointer_cast<impl::velocityFieldFromFile<NumericType, D>>(
            this->getVelocityField());
    if (velField)
      velField->setCustomInterpolator(func);
  }

  void setOffset(const Vec2D<NumericType> &off) {
    auto velField =
        std::dynamic_pointer_cast<impl::velocityFieldFromFile<NumericType, D>>(
            this->getVelocityField());
    if (velField)
      velField->setOffset(off);
  }
};

} // namespace viennaps
