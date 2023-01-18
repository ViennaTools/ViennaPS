#ifndef PS_VALUE_ESTIMATOR_HPP
#define PS_VALUE_ESTIMATOR_HPP

#include <array>
#include <optional>
#include <tuple>

#include <psDataSource.hpp>
#include <psSmartPointer.hpp>

template <typename NumericType, int InputDim, int OutputDim,
          typename... FeedbackType>
class psValueEstimator {
public:
  using InputType = std::array<NumericType, InputDim>;
  using OutputType = std::array<NumericType, OutputDim>;

  static constexpr int DataDim = InputDim + OutputDim;

  using DataVector = std::vector<std::array<NumericType, DataDim>>;
  using DataPtr = psSmartPointer<DataVector>;

protected:
  DataPtr data = nullptr;
  bool dataChanged = true;

public:
  void setData(DataPtr passedData) {
    data = passedData;
    dataChanged = true;
  }

  virtual bool initialize() { return true; }

  virtual std::optional<std::tuple<OutputType, FeedbackType...>>
  estimate(const InputType &input) = 0;
};

#endif