#ifndef RT_VELOCITY_FIELD_HPP
#define RT_VELOCITY_FIELD_HPP

#include <lsSmartPointer.hpp>
#include <lsVelocityField.hpp>
#include <unordered_map>

template <class T> class rtVelocityField : public lsVelocityField<T> {
private:
  typedef std::unordered_map<unsigned long, unsigned long> TranslatorType;

  lsSmartPointer<std::vector<T>> mcestimates = nullptr;
  lsSmartPointer<TranslatorType> translator = nullptr;

public:
  rtVelocityField() {}

  rtVelocityField(lsSmartPointer<std::vector<T>> passedMcEstimates,
                  lsSmartPointer<TranslatorType> passedTranslator)
      : mcestimates(passedMcEstimates), translator(passedTranslator) {}

  T getScalarVelocity(const std::array<T, 3> & /*coordinate*/, int /*material*/,
                      const std::array<T, 3> & /*normalVector*/,
                      unsigned long pointID) {

    assert(translator->find(pointID) != translator->end() && "Invalid pointId");

    if (auto it = translator->find(pointID); it != translator->end())
    {
      return mcestimates->operator[](it->second);
    }
    else 
    {
      lsMessage::getInstance().addError("rtVelocityField: Invalid pointId");
      return 0.;
    }
  }

  void setTranslator(lsSmartPointer<TranslatorType> passedTranslator) {
    translator = passedTranslator;
  }

  void setMcEstimates(lsSmartPointer<std::vector<T>> passedMcEstimates) {
    mcestimates = passedMcEstimates;
  }
};

#endif // RT_VELOCITY_FIELD_HPP