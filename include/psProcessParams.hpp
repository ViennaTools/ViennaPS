
#ifndef PS_PROCESS_PARAMS
#define PS_PROCESS_PARAMS

#include <psSmartPointer.hpp>
#include <rayMessage.hpp>
#include <vector>

// TODO: Implement ViennaPS messaging system

template <typename NumericType> class psProcessParams {
private:
  std::vector<NumericType> scalarData;
  std::vector<std::string> scalarDataLabels;

public:
  void insertNextScalar(NumericType value, std::string label = "scalarData") {
    scalarData.push_back(value);
    scalarDataLabels.push_back(label);
  }

  NumericType &getScalarData(int i) { return scalarData[i]; }

  const NumericType &getScalarData(int i) const { return scalarData[i]; }

  NumericType &getScalarData(std::string label) {
    int idx = getScalarDataIndex(label);
    return scalarData[idx];
  }

  int getScalarDataIndex(std::string label) {
    for (int i = 0; i < scalarDataLabels.size(); ++i) {
      if (scalarDataLabels[i] == label) {
        return i;
      }
    }
    rayMessage::getInstance()
        .addError("Can not find scalar data label in psProcessParams.")
        .print();
    return -1;
  }

  std::vector<NumericType> &getScalarData() { return scalarData; }

  const std::vector<NumericType> &getScalarData() const { return scalarData; }
  std::string getScalarDataLabel(int i) const {
    if (i >= scalarDataLabels.size())
      rayMessage::getInstance()
          .addError(
              "Getting scalar data label in psProcessParams out of range.")
          .print();
    return scalarDataLabels[i];
  }
};

#endif