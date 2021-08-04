#ifndef PS_VELOCITY_FIELD
#define PS_VELOCITY_FIELD

#include <lsVelocityField.hpp>
#include <vector>
#include <unordered_map>

template <typename NumericType>
class psVelocityField : lsVelocityField<NumericType> {
private:
  using TranslatorType = std::unordered_map<unsigned long, unsigned long>;

  psSmartPointer<TranslatorType> translator = nullptr;
  std::vector<NumericType> velocities;

public:
  psVelocityField(){}

  void setVelocities(std::vector<NumericType> &passedVelocities) {
    velocities = passedVelocities;
  }
  void setTranslator(lsSmartPointer<TranslatorType> passedTranslator) {
    translator = passedTranslator;
  }
};

#endif