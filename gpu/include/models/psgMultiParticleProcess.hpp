#pragma once

#include <models/psMultiParticleProcess.hpp>
#include <models/psgPipelineParameters.hpp>
#include <process/psProcessModel.hpp>
#include <psConstants.hpp>
#include <psMaterials.hpp>

namespace viennaps::gpu {

using namespace viennacore;

template <typename NumericType, int D>
class MultiParticleProcess final : public ProcessModelGPU<NumericType, D> {
public:
  MultiParticleProcess() {
    // surface model
    auto surfModel = SmartPointer<viennaps::impl::MultiParticleSurfaceModel<
        NumericType, D>>::New(fluxDataLabels_);

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New(2);

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("MultiParticleProcess");
    // this->setPipelineFileName("MultiParticlePipeline");
    this->setPipelineFileName("GeneralPipeline");
  }

  void addNeutralParticle(NumericType stickingProbability,
                          const std::string &label = "neutralFlux") {
    std::string dataLabel = label + std::to_string(fluxDataLabels_.size());
    fluxDataLabels_.push_back(dataLabel);
    viennaray::gpu::Particle<NumericType> particle;
    particle.name = "Neutral";
    particle.sticking = stickingProbability;
    particle.dataLabels.push_back(dataLabel);
    setDirection(particle);
    particle.materialSticking[static_cast<int>(Material::Undefined)] =
        1.; // this will initialize all to default sticking

    this->insertNextParticleType(particle);
    this->setUseMaterialIds(true);

    addStickingData(stickingProbability);
  }

  void
  addNeutralParticle(std::unordered_map<Material, NumericType> materialSticking,
                     NumericType defaultStickingProbability = 1.,
                     const std::string &label = "neutralFlux") {
    std::string dataLabel = label + std::to_string(fluxDataLabels_.size());
    fluxDataLabels_.push_back(dataLabel);

    viennaray::gpu::Particle<NumericType> particle;
    particle.name = "Neutral";
    particle.sticking = defaultStickingProbability;
    particle.dataLabels.push_back(dataLabel);
    setDirection(particle);
    for (auto &mat : materialSticking) {
      particle.materialSticking[static_cast<int>(mat.first)] = mat.second;
    }

    this->insertNextParticleType(particle);
    this->setUseMaterialIds(true);

    addStickingData(defaultStickingProbability);
  }

  void addIonParticle(NumericType sourcePower, NumericType thetaRMin = 0.,
                      NumericType thetaRMax = 90., NumericType minAngle = 80.,
                      NumericType B_sp = -1., NumericType meanEnergy = 0.,
                      NumericType sigmaEnergy = 0.,
                      NumericType thresholdEnergy = 0.,
                      NumericType inflectAngle = 0., NumericType n = 1,
                      const std::string &label = "ionFlux") {
    std::string dataLabel = label + std::to_string(fluxDataLabels_.size());
    fluxDataLabels_.push_back(dataLabel);

    viennaray::gpu::Particle<NumericType> particle;
    particle.name = "Ion";
    particle.dataLabels.push_back(dataLabel);
    setDirection(particle);
    particle.cosineExponent = sourcePower;

    impl::IonParams params;
    params.thetaRMin = constants::degToRad(thetaRMin);
    params.thetaRMax = constants::degToRad(thetaRMax);
    params.minAngle = constants::degToRad(minAngle);
    params.B_sp = B_sp;
    params.meanEnergy = meanEnergy;
    params.sigmaEnergy = sigmaEnergy;
    params.thresholdEnergy = thresholdEnergy;
    params.inflectAngle = constants::degToRad(inflectAngle);
    params.n = n;
    this->processData.allocUploadSingle(params);
    this->insertNextParticleType(particle);

    addIonData({{"SourcePower", sourcePower},
                {"MeanEnergy", meanEnergy},
                {"SigmaEnergy", sigmaEnergy},
                {"ThresholdEnergy", thresholdEnergy},
                {"B_sp", B_sp},
                {"ThetaRMin", thetaRMin},
                {"ThetaRMax", thetaRMax},
                {"InflectAngle", inflectAngle},
                {"MinAngle", minAngle},
                {"n", n}});
  }

  void
  setRateFunction(std::function<NumericType(const std::vector<NumericType> &,
                                            const Material &)>
                      rateFunction) {
    auto surfModel = std::dynamic_pointer_cast<
        viennaps::impl::MultiParticleSurfaceModel<NumericType, D>>(
        this->getSurfaceModel());
    surfModel->rateFunction_ = rateFunction;
  }

private:
  std::vector<std::string> fluxDataLabels_;
  using ProcessModelBase<NumericType, D>::processMetaData;

  void setDirection(viennaray::gpu::Particle<NumericType> &particle) {
    auto direction = this->getPrimaryDirection();
    if (direction.has_value()) {
      particle.direction = direction.value();
    }
  }

  void addStickingData(NumericType stickingProbability) {
    if (processMetaData.find("StickingProbability") == processMetaData.end()) {
      processMetaData["StickingProbability"] =
          std::vector<double>{stickingProbability};
    } else {
      processMetaData["StickingProbability"].push_back(stickingProbability);
    }
  }

  void addIonData(std::vector<std::pair<std::string, NumericType>> data) {
    for (const auto &pair : data) {
      if (processMetaData.find(pair.first) == processMetaData.end()) {
        processMetaData[pair.first] = std::vector<double>{pair.second};
      } else {
        processMetaData[pair.first].push_back(pair.second);
      }
    }
  }
};

} // namespace viennaps::gpu
