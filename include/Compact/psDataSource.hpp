#ifndef PS_DATA_SOURCE_HPP
#define PS_DATA_SOURCE_HPP

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

#include <psSmartPointer.hpp>

template <typename NumericType, int D> class psDataSource {
public:
  using DataItemType = std::array<NumericType, D>;
  using DataVectorType = std::vector<DataItemType>;

  // Returns a smart pointer to the in-memory copy of the data. If the in-memory
  // copy of the data is empty, the read function is called on the data source.
  const DataVectorType &get() {
    if (data.empty()) {
      auto d = read();
      if (d)
        data = *d;
    }
    return data;
  };

  // Synchronizes the in-memory copy of the data with the underlying data source
  // (e.g. CSV file)
  bool sync() {
    if (modified) {
      if (!write(psSmartPointer<DataVectorType>::New(data)))
        return false;
    }
    modified = false;
    return true;
  };

  // Adds an item to the in-memory copy of the data
  void add(const DataItemType &item) {
    modified = true;
    data.push_back(item);
  }

  // Optional: the data source can also expose additional parameters that are
  // stored alongside the actual data (e.g. depth at which trench diameters were
  // captured)
  virtual std::vector<NumericType> getPositionalParameters() { return {}; }

  // These optional parameters can also be named
  virtual std::unordered_map<std::string, NumericType> getNamedParameters() {
    return {};
  }

private:
  // An in-memory copy of the data
  DataVectorType data;

  // Flag that specifies whether the in-memory copy of the data has been
  // modified (i.e. whether the append function has been called)
  bool modified = false;

  // Each implementing class has to implement the read function, which reads the
  // complete data of the underlying datasource (e.g. CSV file)
  virtual psSmartPointer<DataVectorType> read() = 0;

  // Each implementing class has to implement the write function, which writes
  // the content of the in-memory data into the underlying datasource (e.g. CSV
  // file)
  virtual bool write(psSmartPointer<DataVectorType>) = 0;
};

#endif