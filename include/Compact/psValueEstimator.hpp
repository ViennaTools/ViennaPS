#ifndef PS_VALUE_ESTIMATOR_HPP
#define PS_VALUE_ESTIMATOR_HPP

#include <array>
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

protected:
  psSmartPointer<psDataSource<NumericType, DataDim>> dataSource = nullptr;

public:
  void setDataSource(
      psSmartPointer<psDataSource<NumericType, DataDim>> passedDataSource) {
    dataSource = passedDataSource;
  }

  virtual bool initialize() { return true; }

  virtual std::tuple<OutputType, FeedbackType...>
  estimate(const InputType &input) = 0;
};

#endif