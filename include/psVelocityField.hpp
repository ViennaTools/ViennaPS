#ifndef PS_VELOCITY_FIELD
#define PS_VELOCITY_FIELD

#include <lsVelocityField.hpp>
#include <lsMessage.hpp>
#include <vector>
#include <unordered_map>

template <typename NumericType>
class psVelocityField : public lsVelocityField<NumericType>
{
private:
  using TranslatorType = std::unordered_map<unsigned long, unsigned long>;

  psSmartPointer<TranslatorType> translator = nullptr;
  psSmartPointer<std::vector<NumericType>> velocities = nullptr;

public:
  psVelocityField() {}

  long getVelocityId(unsigned long lsId)
  {
    if (auto it = translator->find(lsId); it != translator->end())
    {
      return it->second;
    }
    else
    {
      return -1;
    }
  }

  NumericType getVelocity(long velId)
  {
    if (velId < velocities->size())
    {
      return velocities->at(velId);
    }
    else
    {
      lsMessage::getInstance().addError("Velocity ID out of range.").print();
      return 0;
    }
  }

  void setVelocities(psSmartPointer<std::vector<NumericType>> passedVelocities)
  {
    velocities = passedVelocities;
  }
  void setTranslator(lsSmartPointer<TranslatorType> passedTranslator)
  {
    translator = passedTranslator;
  }
};

#endif