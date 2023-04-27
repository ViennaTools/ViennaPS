#ifndef PS_DATA_SOURCE_HPP
#define PS_DATA_SOURCE_HPP

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

#include <psLogger.hpp>
#include <psSmartPointer.hpp>

template <typename NumericType> class psDataSource {
public:
  using ItemType = std::vector<NumericType>;
  using VectorType = std::vector<ItemType>;
  using ConstPtr = psSmartPointer<const VectorType>;

  // Returns a smart pointer to the in-memory copy of the data. If the in-memory
  // copy of the data is empty, the read function is called on the data source.
  ConstPtr getData() {
    // Refresh the data
    if (!modified)
      data = read();

    return ConstPtr::New(data);
  };

  void setData(const VectorType &passedData) {
    modified = true;
    data = passedData;
  }

  // Synchronizes the in-memory copy of the data with the underlying data source
  // (e.g. CSV file)
  bool sync() {
    if (modified) {
      // If the data was modified write it to the underlying data source
      if (!write(data))
        return false;
      modified = false;
    } else {
      // If it was not modified, read from the underlying source
      data = read();
      modified = true;
    }
    return true;
  };

  // Adds an item to the in-memory copy of the data
  void add(const ItemType &item) {
    modified = true;
    data.push_back(item);
  }

  // Optional: the data source can also expose additional parameters that are
  // stored alongside the actual data (e.g. depth at which trench diameters were
  // captured)
  virtual std::vector<NumericType> getPositionalParameters() {
    psLogger::getInstance()
        .addWarning("This data source does not support positional parameters.")
        .print();
    return {};
  }
  void setPositionalParameters(
      const std::vector<NumericType> &passedPositionalParameters) {
    positionalParameters = passedPositionalParameters;
  }

  // These optional parameters can also be named
  virtual std::unordered_map<std::string, NumericType> getNamedParameters() {
    psLogger::getInstance()
        .addWarning("This data source does not support named parameters.")
        .print();
    return {};
  }

  void setNamedParameters(const std::unordered_map<std::string, NumericType>
                              &passedNamedParameters) {
    namedParameters = passedNamedParameters;
  }

protected:
  // Each implementing class has to implement the read function, which reads the
  // complete data of the underlying datasource (e.g. CSV file)
  virtual VectorType read() = 0;

  // Each implementing class has to implement the write function, which writes
  // the content of the in-memory data into the underlying datasource (e.g. CSV
  // file)
  virtual bool write(const VectorType &) = 0;

  std::unordered_map<std::string, NumericType> namedParameters;
  std::vector<NumericType> positionalParameters;

private:
  // An in-memory copy of the data
  VectorType data;

  // Flag that specifies whether the in-memory copy of the data has been
  // modified (i.e. whether the append function has been called)
  bool modified = false;
};

#endif