#include <geometries/psMakePlane.hpp>
#include <models/psDirectionalEtching.hpp>
#include <models/psIonBeamEtching.hpp>
#include <models/psSingleParticleProcess.hpp>
#include <psProcess.hpp>

template <typename NumericType, int D>
class FaradayCageSource : public raySource<NumericType> {
public:
  FaradayCageSource(const NumericType xExtent, const NumericType yExtent,
                    const NumericType zPos, const NumericType gridDelta,
                    const NumericType tiltAngle, const NumericType cageAngle,
                    const NumericType extendFac = 1.)
      : minPoint_{-xExtent / 2., -yExtent / 2.}, extent_{xExtent, yExtent},
        zPos_(zPos), gridDelta_(gridDelta), extendFac_(extendFac),
        cosTilt(std::cos(tiltAngle * M_PI / 180.)),
        sinTilt(std::sin(tiltAngle * M_PI / 180.)),
        cage_x(std::cos(cageAngle * M_PI / 180.)),
        cage_y(std::sin(cageAngle * M_PI / 180.)) {}

  rayPair<rayTriple<NumericType>>
  getOriginAndDirection(const size_t idx, rayRNG &RngState) const {
    std::uniform_real_distribution<NumericType> dist(0., 1.);

    rayTriple<NumericType> origin;
    origin[0] = (minPoint_[0] - (extendFac_ * extent_[0])) +
                (extent_[0] + extent_[0] * extendFac_ * 2.) * dist(RngState);
    origin[1] = (minPoint_[1] - (extendFac_ * extent_[1])) +
                (extent_[1] + extent_[1] * extendFac_ * 2.) * dist(RngState);
    // origin[0] = minPoint_[0] + extent_[0] * dist(RngState);
    // origin[1] = minPoint_[1] + extent_[1] * dist(RngState);
    origin[2] = zPos_;

    rayTriple<NumericType> direction;
    if (cage_x * origin[1] - cage_y * origin[0] < 0) {
      direction[0] = -cosTilt * cage_y;
      direction[1] = cosTilt * cage_x;
      direction[2] = -sinTilt;
    } else {
      direction[0] = cosTilt * cage_y;
      direction[1] = -cosTilt * cage_x;
      direction[2] = -sinTilt;
    }
    rayInternal::Normalize(direction);

    return {origin, direction};
  }

  size_t getNumPoints() const {
    return extent_[0] * extent_[1] / (gridDelta_ * gridDelta_);
  }

private:
  std::array<NumericType, 2> const minPoint_;
  std::array<NumericType, 2> const extent_;
  NumericType const zPos_;
  NumericType const gridDelta_;
  NumericType const extendFac_;
  const NumericType cosTilt;
  const NumericType sinTilt;
  const NumericType cage_x;
  const NumericType cage_y;
};

int main(int argc, char *argv[]) {
  using NumericType = double;
  constexpr int D = 3;

  psLogger::setLogLevel(psLogLevel::INTERMEDIATE);
  omp_set_num_threads(16);

  // Parse the parameters
  psUtils::Parameters params;
  if (argc > 1) {
    params.readConfigFile(argv[1]);
  } else {
    std::cout << "Usage: " << argv[0] << " <parameter file>" << std::endl;
    return 1;
  }

  // geometry setup
  auto geometry = psSmartPointer<psDomain<NumericType, D>>::New();
  psMakePlane<NumericType, D>(geometry, params.get("gridDelta"),
                              params.get("xExtent"), params.get("yExtent"),
                              0. /* base height */,
                              false /* periodic boundary */, psMaterial::Si)
      .apply();
  {
    auto box = psSmartPointer<lsDomain<NumericType, D>>::New(
        geometry->getLevelSets()->back());
    NumericType minPoint[D] = {-params.get("boxWidth") / 2.,
                               -params.get("boxWidth") / 2., 0.};
    NumericType maxPoint[D] = {params.get("boxWidth") / 2.,
                               params.get("boxWidth") / 2.,
                               params.get("boxHeight")};
    lsMakeGeometry<NumericType, D>(
        box, lsSmartPointer<lsBox<NumericType, D>>::New(minPoint, maxPoint))
        .apply();
    geometry->insertNextLevelSetAsMaterial(box, psMaterial::Mask);
  }

  {
    std::array<NumericType, 3> direction = {0., 0., -1.};
    auto directionalEtch =
        psSmartPointer<psDirectionalEtching<NumericType, D>>::New(direction);
    psProcess<NumericType, D>(geometry, directionalEtch, 8.).apply();
  }

  // use pre-defined IBE etching model
  auto model = psSmartPointer<psSingleParticleProcess<NumericType, D>>::New(
      -1., 1., 1., psMaterial::Mask);
  // auto model = psSmartPointer<psIonBeamEtching<NumericType, D>>::New(
  //     std::vector<psMaterial>{psMaterial::Mask});
  // auto &modelParams = model->getParameters();
  // modelParams.meanEnergy = params.get("meanEnergy");
  // modelParams.sigmaEnergy = params.get("sigmaEnergy");
  // const NumericType yieldFac = params.get("yieldFac");
  // modelParams.yieldFunction = [yieldFac](NumericType cosTheta) {
  //   return (yieldFac * cosTheta - 1.55 * cosTheta * cosTheta +
  //           0.65 * cosTheta * cosTheta * cosTheta) /
  //          (yieldFac - 1.55 + 0.65);
  // };

  // faraday cage source setup
  auto source = psSmartPointer<FaradayCageSource<NumericType, D>>::New(
      params.get("xExtent"), params.get("yExtent"), params.get("zPos"),
      params.get("gridDelta"), params.get("tiltAngle"), params.get("cageAngle"),
      params.get("extendFac"));
  model->setSource(source);
  // modelParams.tiltAngle = params.get("angle");

  // process setup
  psProcess<NumericType, D> process;
  process.setDomain(geometry);
  process.setProcessModel(model);
  process.setIntegrationScheme(
      lsIntegrationSchemeEnum::LOCAL_LAX_FRIEDRICHS_1ST_ORDER);
  process.setNumberOfRaysPerPoint(params.get<int>("raysPerPoint"));
  process.setProcessDuration(params.get("processTime"));
  process.disableFluxBoundaries();

  // print initial surface
  geometry->saveSurfaceMesh("initial.vtp");

  // run the process
  process.apply();

  // print final surface
  geometry->saveSurfaceMesh("final.vtp");
}
